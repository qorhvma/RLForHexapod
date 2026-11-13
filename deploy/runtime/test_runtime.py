# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "onnxruntime"]
# ///
"""로봇 런타임 검사 (ROS·Isaac Lab 없이).

    uv run deploy/runtime/test_runtime.py [deploy/mixed_phase_v1]

1. 순기구학: 기준 기록의 관절각·중력으로 계산한 feet_below_body가 시뮬레이션 값과 같은지
2. RobotPolicy: 기준 기록의 센서 값을 넣으면 관측 순서·단계 신경망 입력·행동이 기록과 같은지
   (기준 기록은 실제 관절각으로 움직였으므로, 여기서는 목표각 자리에 기록된 관절각을 넣어 순서만 맞춰 본다)
3. L2Fusion: 합성 계단 점군과 Point-LIO 자세로 만든 높이맵·body_x·각속도·중력이 참값과 같은지
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from hexapod_kinematics import HexapodKinematics  # noqa: E402
from l2_sensors import L2Fusion  # noqa: E402
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
    ref = np.load(folder / "reference.npz")
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


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "mixed_phase_v1"
    test_policy(folder)
    test_fusion()
    print("모두 통과" if not failures else f"실패: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
