# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "onnxruntime"]
# ///
"""로봇 런타임 검사 (ROS·Isaac Lab 없이).

    uv run deploy/runtime/test_runtime.py [deploy/mixed_phase_v1]

1. 순기구학: 기준 기록의 관절각·중력으로 계산한 feet_below_body가 시뮬레이션 값과 같은지
2. RobotPolicy: 기준 기록의 센서 값을 넣으면 관측 순서·단계 신경망 입력·행동이 기록과 같은지
   (기준 기록은 실제 관절각으로 움직였으므로, 여기서는 목표각 자리에 기록된 관절각을 넣어 순서만 맞춰 본다)
3. PhaseRule: 기록된 측정값을 넣으면 규칙 단계·규칙 특징(40)·머문 시간이 시뮬레이션과 같은지
4. L2Fusion: 합성 계단 점군과 Point-LIO 자세로 만든 높이맵·body_x·각속도·중력·계단 좌표 몸통이 참값과 같은지
   (+ RobotPolicy.rule_measurement가 계단 좌표 몸통과 목표각으로 발끝을 계산하는지)
5. joint_map: URDF 기하로 잰 실제 관절 각도가 sim_to_true와 같은지, 역변환, (HexaCore가 옆에 있으면) leg_model.fk와 발끝 비교
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from hexapod_kinematics import HexapodKinematics  # noqa: E402
import joint_map  # noqa: E402
from l2_sensors import L2Fusion  # noqa: E402
from phase_rule import Measurement, PhaseRule  # noqa: E402
from robot_policy import Readings, RobotPolicy  # noqa: E402
from heightmap import GX, GY, Stairs, rpy_to_matrix  # noqa: E402  (l2_sensors가 경로를 넣는다)

failures = []


def check(name, value, tol):
    ok = value <= tol
    print(f"{'통과' if ok else '실패'}  {name}: {value:.2e} (허용 {tol:.0e})")
    if not ok:
        failures.append(name)


def test_policy(folder: Path):
    policy = RobotPolicy(folder)
    ref = dict(np.load(folder / "reference.npz"))  # npz는 키로 읽을 때마다 압축을 푼다
    terms = policy.terms
    get = lambda obs, name: obs[terms[name]["start"]:terms[name]["start"] + terms[name]["size"]]  # noqa: E731
    kin = HexapodKinematics(joint_names=policy.joint_names)

    so, nx = ref["student_obs"], ref["net_x"]
    T, N = so.shape[:2]
    fk = max(np.abs(kin.feet_below_body(get(so[t, e], "joint_pos") + policy.default, get(so[t, e], "projected_gravity"))
                    - nx[t, e, 279:285]).max() for t in range(0, T, 5) for e in range(N))
    check("순기구학 feet_below_body (m)", fk, 1e-5)

    groups = policy.io["phase_net"]["x_groups"]
    rule = groups["phase_rule"][0]
    offset = 275 + sum(t["size"] for t in groups["phase_net"])
    worst_obs = worst_actions = worst_logits = 0.0
    for e in range(N):
        policy.reset()
        for t in range(T):
            o = so[t, e]
            # 기록된 관절각·속도가 나오도록 목표각 기억을 맞춘다
            policy.last_target = (get(o, "joint_pos") + policy.default).astype(np.float32)
            policy.prev_target = policy.last_target - get(o, "joint_vel") * policy.dt
            policy.last_actions = get(o, "last_action").astype(np.float32)
            r = Readings(get(o, "base_ang_vel"), get(o, "projected_gravity"), get(o, "feet_contact"),
                         get(o, "height_map"), get(o, "command"), nx[t, e, 275:279])
            out = policy.step(r, nx[t, e, offset + rule["start"]:offset + rule["start"] + rule["size"]], nx[t, e, -1])
            worst_obs = max(worst_obs, float(np.abs(out.student_obs - o).max()))
            worst_logits = max(worst_logits, float(np.abs(out.logits - ref["logits"][t, e]).max()))
            worst_actions = max(worst_actions, float(np.abs(out.actions - ref["actions"][t, e]).max()))
            if ref["done"][t, e]:
                policy.reset()
    check("RobotPolicy student 관측", worst_obs, 1e-4)
    check("RobotPolicy 단계 logits", worst_logits, 1e-3)
    check("RobotPolicy actions", worst_actions, 1e-3)


def test_rule(folder: Path):
    import json

    io = json.loads((folder / "robot_io.json").read_text(encoding="utf-8"))
    ref = dict(np.load(folder / "reference.npz"))
    if "m_tips" not in ref:
        print("건너뜀  PhaseRule: 이 기준 기록에는 측정값이 없습니다 (export_robot_policy.py로 다시 내보낼 것)")
        return
    groups = io["phase_net"]["x_groups"]
    offset = 275 + sum(t["size"] for t in groups["phase_net"])
    size = groups["phase_rule"][0]["size"]
    nx = ref["net_x"]
    T, N = nx.shape[:2]
    worst_f = worst_time = 0.0
    move_bad = 0
    for e in range(N):
        rule = PhaseRule(io)
        for t in range(T):
            m = Measurement(float(ref["m_body_x"][t, e]), ref["m_tips"][t, e], ref["m_contact"][t, e],
                            ref["m_front_coxa"][t, e], float(ref["m_speed"][t, e]))
            if t == 0 or ref["done"][t - 1, e]:
                rule.reset(int(ref["est_move"][t, e]))  # 계단 env는 단계 0 또는 1에서 시작한다
            else:
                rule.update(m)
            move_bad += int(rule.move != ref["est_move"][t, e])
            worst_f = max(worst_f, float(np.abs(rule.features(m) - nx[t, e, offset:offset + size]).max()))
            worst_time = max(worst_time, abs(rule.time_feature() - float(nx[t, e, -1])))
    check("PhaseRule 단계 번호 불일치 (step 수)", move_bad, 0)
    check("PhaseRule 규칙 특징", worst_f, 1e-5)
    check("PhaseRule 머문 시간", worst_time, 1e-3)


def stair_cloud(stairs: Stairs, first_riser_world: float, density=40000, seed=0) -> np.ndarray:
    """계단 윗면·바닥 점 (월드, 1 m²당 density개를 무작위로). 수직면 점도 넣어 실제 스캔처럼 만든다."""
    rng = np.random.default_rng(seed)
    n = int(density * 3.6 * 1.0)
    gx, gy = rng.uniform(-1.0, 2.6, n), rng.uniform(-0.5, 0.5, n)
    pts = [np.stack([gx + first_riser_world, gy, stairs.height(gx, gy)], 1)]
    for rx, top in zip(stairs.riser_x, stairs.top_z):
        m = int(density * top * 1.0)
        pts.append(np.stack([np.full(m, rx + first_riser_world - 1e-3), rng.uniform(-0.5, 0.5, m),
                             rng.uniform(0.0, top, m)], 1))
    return np.concatenate(pts)


def test_fusion():
    stairs = Stairs.uniform()
    base_in_imu = (-0.06, 0.01, -0.09, 0.0, np.radians(30.0), np.radians(2.0))  # 앞으로 30° 숙인 장착 (예시)
    fusion = L2Fusion(base_in_imu, stairs)
    r_ib, p_ib = fusion.r_ib, fusion.p_ib

    def set_base(p_wb, r_wb, t):
        r_wi = r_wb @ r_ib.T
        p_wi = p_wb - r_wi @ p_ib
        w, x, y, z = _matrix_to_quat_wxyz(r_wi)
        fusion.on_pose(p_wi, (x, y, z, w), t)

    first_riser = 0.8
    cloud = stair_cloud(stairs, first_riser)
    base_z = 0.24
    for k, bx in enumerate(np.linspace(0.0, 0.25, 6)):  # 앞으로 걸으며 쌓기
        set_base(np.array([bx, 0.0, base_z]), np.eye(3), float(k))
        fusion.on_cloud(cloud, float(k))
    s = fusion.snapshot()
    sx = bx - first_riser + GX
    truth = (base_z - stairs.height(sx, GY)).ravel()
    err = np.abs(s["height_map"] - truth)
    # heightmap.py는 지도를 3×3칸(3cm) 평균으로 읽어, 수직면 바로 앞 칸은 아래·윗단이 섞인다 (시뮬레이션은 날카로운 모서리)
    near_edge = (np.min(np.abs(sx[..., None] - np.asarray(stairs.edges)), axis=-1) <= 0.015).ravel()
    check("높이맵, 모서리 1.5cm 밖 (m)", float(err[~near_edge].max()), 0.011)
    check("높이맵, 모서리 1.5cm 안 (m, 반 단까지 섞임)", float(err[near_edge].max()), stairs.top_z[0] / 2 + 0.011)
    true_body_x = bx - first_riser
    want = np.concatenate([[true_body_x], np.clip(true_body_x - np.array(stairs.riser_x), -0.3, 0.6)])
    check("body_x (m)", float(np.abs(s["body_x"] - want).max()), 0.011)
    x0 = 0.5
    pos, rot = fusion.stairs_pose(x0)
    check("계단 좌표 몸통 위치 (m)", float(np.abs(pos - [x0 + true_body_x, 0.0, base_z]).max()), 0.011)
    check("계단 좌표 몸통 회전", float(np.abs(rot - np.eye(3)).max()), 1e-9)
    return fusion

    r_wb = rpy_to_matrix(np.radians(4.0), np.radians(-7.0), np.radians(10.0))
    set_base(np.array([0.1, 0.0, base_z]), r_wb, 10.0)
    w_b = np.array([0.3, -0.2, 0.5])
    fusion.on_imu(r_ib @ w_b, 10.0)
    s = fusion.snapshot()
    check("각속도 (rad/s)", float(np.abs(s["gyro"] - w_b).max()), 1e-9)
    check("중력 방향", float(np.abs(s["gravity"] - r_wb.T @ np.array([0, 0, -1.0])).max()), 1e-9)


def _matrix_to_quat_wxyz(r):
    w = np.sqrt(max(1e-12, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2.0
    return w, (r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w)


def test_measurement(folder: Path):
    """rule_measurement: 발끝 = 계단 좌표 몸통 위치 + 회전 × FK(목표각)."""
    policy = RobotPolicy(folder)
    rng = np.random.default_rng(1)
    policy.last_target = (policy.default + rng.uniform(-0.3, 0.3, 18)).astype(np.float32)
    rot = rpy_to_matrix(0.05, -0.1, 0.2)
    pos = np.array([0.4, 0.02, 0.25])
    m = policy.rule_measurement(pos, rot, np.ones(6), 0.1)
    want = pos + (rot @ policy.kin.tips(policy.last_target).T).T
    check("rule_measurement 발끝 (m)", float(np.abs(m.tips - want).max()), 1e-9)
    check("rule_measurement 앞다리 coxa", float(np.abs(
        m.front_coxa - policy.last_target[[policy.joint_names.index("L4_coxa"), policy.joint_names.index("L3_coxa")]]).max()), 1e-9)


def _suffix(leg):
    return "" if leg == "L1" else "_" + leg[1]


def test_joint_map(folder: Path, hexacore: Path | None = None):
    """sim 관절각을 흔들어 URDF 기하로 hip·femur·tibia 실제 각도를 재고 sim_to_true와 비교한다."""
    import json

    io = json.loads((folder / "robot_io.json").read_text(encoding="utf-8"))["action"]
    names, lim = io["joint_names"], np.array(io["joint_limits_rad"])
    default = np.array(io["default_joint_pos_rad"])
    kin = HexapodKinematics()
    mounts = [35.0, 90.0, 145.2, -144.0, -90.0, -34.8]  # calibration.json kinematics.mounts heading_deg (펌웨어 순서)
    lm = None
    if hexacore is not None and (hexacore / "software" / "raspberrypi" / "leg_model.py").exists():
        sys.path.insert(0, str(hexacore / "software" / "raspberrypi"))
        import leg_model as lm  # noqa: PLC0415
    rng = np.random.default_rng(0)
    err_geom = err_inv = err_fk = 0.0
    for _ in range(200):
        q = np.clip(default + rng.uniform(-0.6, 0.6, 18), lim[:, 0], lim[:, 1])
        t = joint_map.sim_to_true(q, names)
        err_inv = max(err_inv, np.abs(joint_map.true_to_sim(t, names) - q).max())
        qd = dict(zip(names, q))
        tips = kin.tips(qd)
        for fw, leg in enumerate(joint_map.FW_TO_URDF):
            _, H = kin._link_pose("coxa" + _suffix(leg), qd)
            rf, F = kin._link_pose("femur" + _suffix(leg), qd)
            _, K = kin._link_pose("tibia" + _suffix(leg), qd)
            n = rf[:, 2]  # femur 관절축 = 다리 평면의 법선
            plane = lambda v: v - n * (v @ n)  # noqa: E731
            out = np.cross(n, [0.0, 0.0, 1.0])
            out = out if out @ (F - H) > 0 else -out  # 다리 평면의 수평 바깥 방향
            dyaw = (np.degrees(np.arctan2(out[1], out[0])) - mounts[fw] + 180) % 360 - 180
            hip = 180 - joint_map.SIDE[fw] * dyaw  # 왼다리: 시계 방향(yaw 감소) = 앞 = +
            u = K - F
            femur = 180 - np.degrees(np.arcsin(u[2] / np.linalg.norm(u)))
            v = plane(tips[kin_index(leg)] - K)
            # 부호 있는 꺾임 (arccos는 일자를 넘어 펴진 무릎을 접힌 쪽으로 읽는다). tibia +(펴짐) = 꺾임 감소
            rt, _ = kin._link_pose("tibia" + _suffix(leg), qd)
            bend = -np.degrees(np.arctan2(np.cross(u, v) @ rt[:, 2], u @ v))
            err_geom = max(err_geom, np.abs(np.array([hip, femur, 180 - bend]) - t[fw]).max())
            if lm is not None:  # HexaCore 평면 모델: 다리 평면 안에서만 비교 (femur가 coxa 선에서 2 cm 옆에 붙은 것은 모델에 없다)
                tip = (K + 0.1375 * v / np.linalg.norm(v)) * 1000
                e = np.array(lm.fk(fw, list(t[fw]))) - tip
                err_fk = max(err_fk, np.linalg.norm(plane(e)) / 1000)
    check("joint_map 기하 [도] (heading_deg가 0.1° 단위)", err_geom, 0.1)
    check("joint_map 역변환 [rad]", err_inv, 1e-9)
    if lm is not None:
        check("joint_map ↔ HexaCore leg_model.fk 다리 평면 안 발끝 [m]", err_fk, 1e-3)
    else:
        print("건너뜀  HexaCore leg_model.fk 비교 (--hexacore 경로 없음)")


def kin_index(leg):
    from hexapod_kinematics import FOOT_ORDER  # noqa: PLC0415

    return FOOT_ORDER.index("tibia" + _suffix(leg))


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "mixed_phase_v1"
    hexacore = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parents[3] / "HexaCore-clone"
    test_joint_map(folder, hexacore)
    test_policy(folder)
    test_rule(folder)
    test_measurement(folder)
    test_fusion()
    print("모두 통과" if not failures else f"실패: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
