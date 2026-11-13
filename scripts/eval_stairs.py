"""계단 평가: 저장된 정책을 계단 앞에서 출발시켜 몇 %가 꼭대기까지 오르는지 잰다. 학습하지 않는다.

teacher와 student를 같은 조건으로 비교한다.
- 저장 버퍼를 쓰지 않는다. 0번 또는 1번 시작 위치에서만 출발한다.
- 노이즈 없는 평균 행동을 쓴다. student의 높이맵 노이즈는 그대로 둔다.
- 환경마다 첫 에피소드만 센다. 짧은 에피소드가 더 많이 집계되지 않는다.

    uv run python scripts/eval_stairs.py --task Hexapod-Stair --checkpoint <teacher.pt>
    uv run python scripts/eval_stairs.py --task Hexapod-Stair-Student-A --checkpoint <student.pt>

걷기·계단 통합 task(Hexapod-Student-*)는 그룹별로 나눠 보고한다. 걷기는 첫 에피소드의 유지 비율과 명령 추종 오차를 잰다.
--teacher를 주면 같은 조건에서 teacher 묶음으로 움직여 기준 성능을 잰다.

    uv run python scripts/eval_stairs.py --task Hexapod-Student-A --checkpoint <teachers.pt> --teacher
    uv run python scripts/eval_stairs.py --task Hexapod-Student-A --checkpoint <student.pt>

단계 보조 출력 student(Hexapod-Stair-Student-A-PhaseAux)는 단계 예측 정답률도 함께 보고한다 (단계별, 틀렸을 때 어느 단계로).

    uv run python scripts/eval_stairs.py --task Hexapod-Stair-Student-A-PhaseAux --checkpoint <student.pt>

student는 센서 오차를 넣어 잴 수 있다 (mdp/sensor_model.py). 오차는 센서별 원인으로 한 번만 뽑고,
그 센서 값을 쓰는 모든 곳(단계 계산과 student 관측)에 같은 값으로 들어간다. --err_* 플래그는 --help 참고.
--error_sweep은 원인별로 차례대로 재서 표로 보고한다. 설정마다 첫 에피소드 전체를 돌리므로 --num_envs 1024 정도를 권한다.
추정 단계를 안 보는 student(A, A-PhaseAux)는 단계 계산에만 쓰는 원인을 건너뛴다.

    uv run python scripts/eval_stairs.py --task Hexapod-Stair-Student-A-Phase --checkpoint <student.pt> --error_sweep --num_envs 1024
    uv run python scripts/eval_stairs.py --task Hexapod-Stair-Student-A-Phase --checkpoint <student.pt> --err_lidar_x 2 --phase_delay 40

높이맵 방식: --map_mode accumulated(기본, 지도에 쌓아 잘라 씀: x 위치 오차만큼 높이맵도 밀림) 또는 per_frame.
단계 판정 방식은 task 설정을 따른다 (A-Phase, A-PhaseNoise 모두 높이 3cm, 건너뛰기 끔). 옵션으로 바꿀 수 있다.
- --phase_judge contact: 높이 3cm 또는 (발끝 스위치 + 높이 --phase_contact_window cm, --phase_contact_confirm ms 유지)
- --phase_catch_up: 전진 단계에서 다음 옮기기 단계가 이미 끝났으면 건너뛴다
평가에서는 오차 크기를 항상 지정한 값 그대로 쓴다 (학습용 무작위 크기는 쓰지 않는다).
sweep의 각 줄 아래에 뒤처진 시간이 어느 추정 단계에서 생겼는지 나온다.

    uv run python scripts/eval_stairs.py --task Hexapod-Stair-Student-A-Phase --checkpoint <student.pt> --error_sweep --phase_judge contact --phase_catch_up

--phase_net을 주면 A-Phase 계열의 단계 입력을 규칙 추정기 대신 단계 추정 신경망(agents/phase_net.py) 출력으로 바꾼다.
신경망 예측과 참값의 일치/앞섬/뒤처짐, 단계별 정답률, 0.5초 넘게 계속 틀린 비율을 함께 보고한다.
sweep에서는 단계 계산에만 쓰는 원인(발끝 높이 계산, 계산 지연)을 건너뛴다.
규칙 판정을 입력으로 받는 신경망(phase_rule)은 지연만 건너뛰고, 줄마다 입력으로 받은 규칙 추정기의 일치도 보여 준다.

    uv run python scripts/eval_stairs.py --task Hexapod-Stair-Student-A-Phase --checkpoint <A_phase.pt> --phase_net <phase_net.pt> --error_sweep --num_envs 1024

--fall_probe: 뒷발 단계에서 넘어진 에피소드와 통과한 에피소드의 자세(기울기, 발 위치, 지지 범위)를 단별로 비교한다.

    uv run python scripts/eval_stairs.py --task Hexapod-Stair --checkpoint <teacher.pt> --fall_probe --num_envs 4096
"""

import argparse
import contextlib
import dataclasses
import importlib.metadata
import math
import sys

import torch

START_NAMES = {0: "0번 (계단 앞)", 1: "1번 (앞발이 수직면 앞)"}
FAIL_TERMS = ("base_low", "bad_ori", "sim_nan", "sim_blowup")  # 걷기에서 넘어지거나 물리가 터진 종료

# 센서 오차 플래그: (플래그, SensorErrorCfg 필드, 단위 변환, 설명)
SENSOR_FLAGS = [
    ("--err_lidar_x", "lidar_x_bias", 0.01, "라이다: 수직면 기준 몸통 x 편향 [cm]"),
    ("--err_lidar_x_noise", "lidar_x_noise", 0.01, "라이다: 몸통 x 노이즈 std [cm]"),
    ("--err_lidar_tilt", "lidar_mount_tilt", math.pi / 180.0, "라이다: 장착 각도 오차 (높이맵 기울기) [도]"),
    ("--err_lidar_range", "lidar_range_bias", 0.01, "라이다: 거리 편향 (높이맵 높이) [cm]"),
    ("--err_lidar_period", "lidar_period", 0.001, "라이다: 높이맵 갱신 주기 [ms]"),
    ("--err_lidar_dropout", "lidar_dropout", 0.01, "라이다: 높이맵 칸이 직전 값으로 남을 확률 [%%]"),
    ("--err_imu_tilt", "imu_tilt", math.pi / 180.0, "IMU: 중력 방향 오차 (중력 관측, 높이맵, 발끝 높이) [도]"),
    ("--err_gyro_noise", "gyro_noise", 1.0, "IMU: 각속도 노이즈 std [rad/s]"),
    ("--err_encoder", "encoder_bias", math.pi / 180.0, "엔코더: 관절각 편향 (관절 관측, 발끝, 앞다리 접힘) [도]"),
    ("--err_encoder_noise", "encoder_noise", math.pi / 180.0, "엔코더: 관절각 노이즈 std [도]"),
    ("--err_contact_flip", "contact_flip", 0.01, "발끝 스위치: 오판 확률 (접촉 관측, 단계 계산) [%%]"),
    ("--err_step_height", "step_height_error", 0.01, "발끝 높이 계산: 가정한 단 높이 오차 [cm]"),
    ("--err_foot_height", "foot_height_bias", 0.01, "발끝 높이 계산: 나머지 편향 (기준 발 등) [cm]"),
    ("--err_foot_height_noise", "foot_height_noise", 0.01, "발끝 높이 계산: 나머지 노이즈 std [cm]"),
]

# --error_sweep의 시험 목록. 측정값이 없어 잡은 시작값이다. 값은 SensorErrorCfg / PhaseEstimatorCfg 단위 (m, rad, 확률, 초)
# 한 줄 = 한 원인. 그 원인은 단계 계산과 student 관측 중 그 센서를 쓰는 모든 곳에 같은 값으로 들어간다
ERROR_SWEEP = [
    ("오차 없음", {}),
    ("라이다 x 편향 1cm", {"lidar_x_bias": 0.01}),
    ("라이다 x 편향 2cm", {"lidar_x_bias": 0.02}),
    ("라이다 x 편향 4cm", {"lidar_x_bias": 0.04}),
    ("라이다 x 노이즈 0.5cm", {"lidar_x_noise": 0.005}),
    ("라이다 x 노이즈 1cm", {"lidar_x_noise": 0.01}),
    ("라이다 장착 각도 0.5도", {"lidar_mount_tilt": math.radians(0.5)}),
    ("라이다 장착 각도 1도", {"lidar_mount_tilt": math.radians(1.0)}),
    ("라이다 장착 각도 2도", {"lidar_mount_tilt": math.radians(2.0)}),
    ("라이다 거리 편향 1cm", {"lidar_range_bias": 0.01}),
    ("라이다 거리 편향 2cm", {"lidar_range_bias": 0.02}),
    ("라이다 주기 50ms", {"lidar_period": 0.05}),
    ("라이다 주기 100ms", {"lidar_period": 0.1}),
    ("높이맵 구멍 5%", {"lidar_dropout": 0.05}),
    ("높이맵 구멍 20%", {"lidar_dropout": 0.2}),
    ("IMU 기울기 0.5도", {"imu_tilt": math.radians(0.5)}),
    ("IMU 기울기 1도", {"imu_tilt": math.radians(1.0)}),
    ("IMU 기울기 2도", {"imu_tilt": math.radians(2.0)}),
    ("자이로 노이즈 0.05", {"gyro_noise": 0.05}),
    ("엔코더 편향 1도", {"encoder_bias": math.radians(1.0)}),
    ("엔코더 편향 2도", {"encoder_bias": math.radians(2.0)}),
    ("엔코더 편향 4도", {"encoder_bias": math.radians(4.0)}),
    ("엔코더 노이즈 0.5도", {"encoder_noise": math.radians(0.5)}),
    ("스위치 오판 2%", {"contact_flip": 0.02}),
    ("스위치 오판 5%", {"contact_flip": 0.05}),
    ("단 높이 가정 1cm", {"step_height_error": 0.01}),
    ("단 높이 가정 2cm", {"step_height_error": 0.02}),
    ("발끝 높이 나머지 1cm", {"foot_height_bias": 0.01}),
    ("발끝 높이 나머지 2cm", {"foot_height_bias": 0.02}),
    ("발끝 높이 노이즈 0.5cm", {"foot_height_noise": 0.005}),
    ("계산 지연 40ms", {"delay": 0.04}),
    ("계산 지연 100ms", {"delay": 0.1}),
    ("전체 (예상 수준)", {
        "lidar_x_bias": 0.02, "lidar_x_noise": 0.005, "lidar_mount_tilt": math.radians(1.0), "lidar_range_bias": 0.01,
        "lidar_period": 0.1, "lidar_dropout": 0.05, "imu_tilt": math.radians(0.5), "gyro_noise": 0.05,
        "encoder_bias": math.radians(2.0), "encoder_noise": math.radians(0.5), "contact_flip": 0.02,
        "step_height_error": 0.01, "foot_height_bias": 0.01, "foot_height_noise": 0.003, "delay": 0.04,
    }),
]
# 단계 계산에만 들어가는 원인. 추정 단계를 관측으로 받지 않는 student에는 영향이 없다
ESTIMATE_ONLY = {"step_height_error", "foot_height_bias", "foot_height_noise", "delay"}


def collect_first_episodes(env, policy, obs, tracker, termination_manager, max_steps, on_step=None):
    """환경마다 첫 에피소드가 끝날 때까지 돌리고 결과를 모은다. on_step(dones, finished)는 step마다 불린다."""
    term_names = termination_manager.active_terms
    start = tracker.move.clone()
    ended = torch.zeros_like(start)
    reasons = torch.zeros(start.shape[0], len(term_names), dtype=torch.bool, device=start.device)
    finished = torch.zeros_like(start, dtype=torch.bool)
    length = torch.zeros_like(start)  # 에피소드 길이 [step]
    top_step = torch.full_like(start, -1)  # 꼭대기에 처음 도달한 step. -1 = 도달 못 함

    with torch.inference_mode():
        for step in range(1, max_steps + 1):
            obs, _, dones, _ = env.step(policy(obs))
            dones = dones.bool()
            policy.reset(dones)  # student의 기억을 에피소드마다 비운다
            if on_step is not None:
                on_step(dones, finished)

            # step 안에서 리셋까지 끝났으므로, 끝난 환경의 tracker.move는 이미 다음 에피소드 값이다
            move = torch.where(dones, tracker.ended_move, tracker.move)
            top_step[(move >= tracker.final_advance) & (top_step < 0) & ~finished] = step

            newly = dones & ~finished
            ended[newly] = tracker.ended_move[newly]
            reasons[newly] = torch.stack([termination_manager.get_term(n) for n in term_names], dim=1)[newly]
            length[newly] = step
            finished |= newly
            if finished.all():
                break

    if not finished.all():
        raise RuntimeError(f"{max_steps} step 안에 끝나지 않은 환경이 {(~finished).sum().item()}개 있습니다.")
    return {
        "start": start,
        "ended": ended,
        "reasons": reasons,
        "term_names": term_names,
        "length": length,
        "top_step": top_step,
    }


def summarize(data, mask, final_advance, done_move, step_dt):
    """mask로 고른 에피소드들의 결과."""
    ended = data["ended"][mask]
    length = data["length"][mask]
    top_step = data["top_step"][mask]
    count = len(ended)
    reached_top = (ended >= final_advance).float().mean().item()
    result = {
        "count": count,
        "reached_top": reached_top,
        "margin": 1.96 * (reached_top * (1.0 - reached_top) / count) ** 0.5,  # 95% 신뢰구간 반폭
        "success": (ended == done_move).float().mean().item(),
        "mean_move": ended.float().mean().item(),
        "move_counts": torch.bincount(ended, minlength=done_move + 1).tolist(),
        "reasons": dict(zip(data["term_names"], data["reasons"][mask].float().mean(dim=0).tolist(), strict=True)),
        "top_time": None,
        "top_share": 0.0,
    }
    reached = top_step >= 0
    if reached.any():
        seconds = top_step[reached].float() * step_dt
        result["top_time"] = (seconds.mean().item(), seconds.median().item(), seconds.quantile(0.9).item())
        # 전체 시뮬레이션 시간 중 꼭대기에 도달한 뒤의 비중
        result["top_share"] = ((length[reached] - top_step[reached]).sum() / length.sum()).item()
    return result


def format_report(data, final_advance, done_move, step_dt):
    """전체와 시작 위치별 결과를 글로 만든다."""
    start = data["start"]
    lines = [f"꼭대기 도달 = 단계 {final_advance} 이상, 정지 성공 = 단계 {done_move}"]
    groups = [("전체", torch.ones_like(start, dtype=torch.bool))]
    for value in start.unique().tolist():
        groups.append((START_NAMES.get(value, f"{value}번"), start == value))

    for name, mask in groups:
        result = summarize(data, mask, final_advance, done_move, step_dt)
        reasons_text = ", ".join(f"{k} {v:.1%}" for k, v in result["reasons"].items() if v > 0)
        moves_text = " ".join(f"{k}:{v}" for k, v in enumerate(result["move_counts"]) if v > 0)
        if result["top_time"] is None:
            time_text = "도달한 로봇 없음"
        else:
            time_text = "평균 {:.1f}초, 중앙값 {:.1f}초, 90%가 {:.1f}초 안에 도달".format(*result["top_time"])
        lines += [
            "",
            f"[{name}] {result['count']}대",
            f"  꼭대기 도달: {result['reached_top']:.1%} (±{result['margin']:.1%})",
            f"  정지 성공:   {result['success']:.1%}",
            f"  평균 단계:   {result['mean_move']:.2f} / {done_move}",
            f"  끝난 단계:   {moves_text}",
            f"  종료 이유:   {reasons_text}",
            f"  도달 시간:   {time_text}",
            f"  꼭대기 체류: 전체 시간의 {result['top_share']:.1%}",
        ]
    return "\n".join(lines)


def walk_meter(env, mask):
    """걷기 env의 첫 에피소드 동안 명령 추종 오차를 누적한다. (on_step, 합계 (N, 3), step 수 (N,))"""
    robot = env.scene["robot"]
    commands = env.command_manager
    totals = torch.zeros(env.num_envs, 3, device=env.device)
    steps = torch.zeros(env.num_envs, device=env.device)

    def on_step(dones, finished):
        # 리셋된 env의 상태는 다음 에피소드 값이므로 세지 않는다
        active = mask & ~finished & ~dones
        velocity = commands.get_command("base_velocity")
        height = commands.get_command("base_height")[:, 0]
        data = robot.data
        base_z = data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
        errors = torch.stack([
            (velocity[:, :2] - data.root_lin_vel_b.torch[:, :2]).norm(dim=1),
            (velocity[:, 2] - data.root_ang_vel_b.torch[:, 2]).abs(),
            (height - base_z).abs(),
        ], dim=1)
        totals[active] += errors[active]
        steps[active] += 1.0

    return on_step, totals, steps


def format_walk_report(data, mask, totals, steps):
    """걷기 env의 결과를 글로 만든다."""
    reasons = data["reasons"][mask]
    names = data["term_names"]
    survived = reasons[:, names.index("walk_time_out")].float().mean().item()
    fail_ids = [names.index(n) for n in FAIL_TERMS if n in names]
    failed = reasons[:, fail_ids].any(dim=1).float().mean().item()
    xy, yaw, height = (totals[mask].sum(dim=0) / steps[mask].sum().clamp_min(1.0)).tolist()
    return "\n".join([
        f"[걷기] {mask.sum().item()}대",
        f"  20초 유지:   {survived:.1%}",
        f"  넘어짐/오류: {failed:.1%}",
        f"  속도 오차:   xy {xy:.3f} m/s, yaw {yaw:.3f} rad/s",
        f"  높이 오차:   {height * 100:.2f} cm",
    ])


PHASE_NAMES = ["전진", "앞발", "중간발", "접기 전진", "뒷발1", "뒷발2", "마지막 전진", "정지"]  # PHASES 순서 + 2


def phase_meter(tracker, mask):
    """첫 에피소드 동안 추정 단계와 참값을 비교한다.
    (on_step, step 수 (같음, 앞섬, 뒤처짐), 뒤처진 step 수를 추정 단계 종류별로 (len(PHASE_NAMES),))"""
    device = tracker.move.device
    counts = torch.zeros(3, device=device)
    behind_at = torch.zeros(len(PHASE_NAMES), device=device)
    num_phases = tracker.final_advance // tracker.stairs.num_steps

    def on_step(dones, finished):
        # 리셋된 env의 상태는 다음 에피소드 값이므로 세지 않는다
        active = mask & ~finished & ~dones
        estimate = tracker.estimator.delayed_move[active]
        difference = estimate - tracker.move[active]
        counts[0] += (difference == 0).sum()
        counts[1] += (difference > 0).sum()
        counts[2] += (difference < 0).sum()
        kind = torch.where(estimate < tracker.final_advance, estimate % num_phases, num_phases)
        kind = torch.where(estimate > tracker.final_advance, num_phases + 1, kind)
        behind_at.add_(torch.bincount(kind[difference < 0], minlength=len(PHASE_NAMES)).float())

    return on_step, counts, behind_at


AUX_PHASE_NAMES = ["전진", "앞발", "중간발", "접기 전진", "뒷발1", "뒷발2", "정지"]  # stair_phase_obs 순서 (마지막 전진 = 전진)


class PhaseAuxMeter:
    """단계 보조 출력 student(Hexapod-Stair-Student-A-PhaseAux)의 단계 예측을 첫 에피소드 동안 참값과 비교한다.
    정책을 감싸서, 정책에 들어간 관측의 phase_label(그 순간의 참값 단계)과 예측을 비교한다."""

    def __init__(self, policy, mask):
        self.policy, self.mask = policy, mask
        n = len(AUX_PHASE_NAMES)
        self.confusion = torch.zeros(n, n, device=mask.device)  # [참값, 예측]
        self.label = self.predicted = None

    def __call__(self, obs):
        actions = self.policy(obs)
        self.label = obs["phase_label"].argmax(dim=-1)
        self.predicted = self.policy.last_phase_logits.argmax(dim=-1)
        return actions

    def reset(self, dones):
        self.policy.reset(dones)

    def chain(self, on_step):
        def step(dones, finished):
            # 방금 step의 관측은 아직 첫 에피소드였던 env만 센다 (끝난 step 포함)
            active = self.mask & ~finished
            index = self.label[active] * len(AUX_PHASE_NAMES) + self.predicted[active]
            self.confusion.view(-1).add_(torch.bincount(index, minlength=self.confusion.numel()).float())
            if on_step is not None:
                on_step(dones, finished)

        return step

    def report(self):
        return confusion_report(self.confusion, "단계 예측 (보조 출력)")


def confusion_report(confusion, title):
    """[참값, 예측] 표를 단계별 정답률과 틀린 예측으로 글로 만든다."""
    total = confusion.sum().clamp_min(1.0)
    lines = [f"\n{title}: 정답률 {confusion.diag().sum() / total:.1%}"]
    per_label = confusion.sum(dim=1)
    for k, name in enumerate(AUX_PHASE_NAMES):
        if per_label[k] == 0:
            continue
        row = confusion[k] / per_label[k]
        wrong = ", ".join(
            f"{AUX_PHASE_NAMES[j]} {row[j]:.0%}" for j in range(len(AUX_PHASE_NAMES)) if j != k and row[j] >= 0.01
        )
        lines.append(f"  {name}: {row[k]:.1%} (시간 비중 {per_label[k] / total:.0%})" + (f" | 틀린 예측: {wrong}" if wrong else ""))
    return "\n".join(lines)


STUCK_STEPS = 25  # 0.5초. 이보다 오래 계속 틀리면 "굳음"으로 센다


class FallProbe:
    """뒷발 단계(접기 전진·뒷발1·뒷발2)에서 넘어진 에피소드와 통과한 에피소드의 자세를 단별로 비교한다 (--fall_probe).
    teacher 실패가 3번째 단의 뒷발2(move 17)에 몰려 있어, 아래 단(move 5, 11)과 무엇이 다른지 보려고 만들었다.
    넘어짐: 그 단계에서 bad_ori로 끝남. 기록 시점은 끝나기 직전 step과 0.5초 전.
    통과: 그 단계를 끝내고 다음 단계로 넘어감. 기록 시점은 넘어가기 직전 step과 0.5초 전."""

    PHASES = (3, 4, 5)  # 접기 전진, 뒷발1, 뒷발2
    HISTORY = 25  # 0.5초
    # 몸통 기준: 앞 = +x, 왼쪽 = +y
    FEATURES = [
        ("앞으로 숙임 [도]", "pitch"), ("왼쪽 기울어짐 [도]", "roll"), ("|기울어짐| [도]", "tilt"),
        ("이 단계에 머문 시간 [s]", "time"),
        ("몸통 x - 수직면 [cm]", "body_x"), ("몸통 z - 윗면 [cm]", "body_z"),
        ("앞발 x - 다음 수직면 [cm]", "front_x"), ("앞발 높이 - 지면 [cm]", "front_z"),
        ("가운데발 x - 수직면 [cm]", "mid_x"),
        ("L6 x - 수직면 [cm]", "l6_x"), ("L6 높이 - 지면 [cm]", "l6_z"),
        ("L1 x - 수직면 [cm]", "l1_x"), ("L1 높이 - 지면 [cm]", "l1_z"),
        ("닿은 발 수", "contacts"),
        ("지지 뒤 끝 - 몸통 x [cm]", "support_back"), ("지지 앞 끝 - 몸통 x [cm]", "support_front"),
        ("앞다리 접힘 (작은 쪽)", "fold"),
    ]
    PERCENT = {"body_x", "body_z", "front_x", "front_z", "mid_x", "l6_x", "l6_z", "l1_x", "l1_z",
               "support_back", "support_front"}

    def __init__(self, env, tracker, mask):
        from RLforHexapod.tasks.walk.mdp import stairs_terrain_height

        self.env, self.tracker, self.mask = env, tracker, mask
        self.terrain = stairs_terrain_height
        n, device = env.num_envs, env.device
        self.names = [key for _, key in self.FEATURES]
        self.buf = torch.zeros(self.HISTORY, n, len(self.names), device=device)
        self.ptr = 0
        self.filled = 0
        self.phase_time = torch.zeros(n, device=device)
        self.prev_move = tracker.move.clone()
        steps = tracker.stairs.num_steps
        shape = (steps, len(self.PHASES), 2, 2, len(self.names))  # (단, 단계, 통과/넘어짐, 0.5초 전/직전, 값)
        self.sums = torch.zeros(shape, device=device)
        self.counts = torch.zeros(shape[:3], device=device)
        self.snapshot()

    def snapshot(self):
        t, env = self.tracker, self.env
        phase, level = t.phase_and_level()
        riser, top = t.riser_and_top(level)
        riser, top = riser.unsqueeze(1), top.unsqueeze(1)
        body, _ = t.body_state()
        tips = t.tips()
        ground = self.terrain(t.stairs, tips[..., 0], tips[..., 1])
        air_time = t.sensor.data.current_air_time.torch[:, t.sensor_ids]
        contact = air_time == 0
        gravity = t.asset.data.projected_gravity_b.torch
        rel_x = tips[..., 0] - body[:, 0:1]
        big = torch.full_like(rel_x, 1e3)
        back = torch.where(contact, rel_x, big).min(1).values
        front = torch.where(contact, rel_x, -big).max(1).values
        none = ~contact.any(1)
        fold = (t.asset.data.joint_pos.torch[:, t.front_coxa_ids] / t.front_parallel).clamp(max=1.0)
        x_from = tips[..., 0] - riser
        z_from = tips[..., 2] - ground - t.gait.tip_height
        values = {
            "pitch": torch.rad2deg(torch.asin(gravity[:, 0].clamp(-1, 1))),
            "roll": torch.rad2deg(torch.asin(gravity[:, 1].clamp(-1, 1))),
            "tilt": torch.rad2deg(torch.acos((-gravity[:, 2]).clamp(-1, 1))),
            "time": self.phase_time,
            "body_x": body[:, 0] - riser[:, 0],
            "body_z": body[:, 2] - top[:, 0],
            "front_x": (tips[:, [0, 5], 0] - riser - t.stairs.tread).mean(1),
            "front_z": z_from[:, [0, 5]].mean(1),
            "mid_x": x_from[:, [1, 4]].mean(1),
            "l6_x": x_from[:, 2], "l6_z": z_from[:, 2],
            "l1_x": x_from[:, 3], "l1_z": z_from[:, 3],
            "contacts": contact.float().sum(1),
            "support_back": torch.where(none, torch.zeros_like(back), back),
            "support_front": torch.where(none, torch.zeros_like(front), front),
            "fold": fold.min(1).values,
        }
        self.buf[self.ptr] = torch.stack([values[k] for k in self.names], dim=1)
        self.ptr = (self.ptr + 1) % self.HISTORY
        self.filled = min(self.filled + 1, self.HISTORY)

    def _add(self, ids, move, outcome):
        """ids의 env를 move 단계의 outcome(0 = 통과, 1 = 넘어짐)으로 기록. 버퍼의 마지막 칸 = 직전 step."""
        if len(ids) == 0:
            return
        last = self.buf[(self.ptr - 1) % self.HISTORY, ids]
        early = self.buf[(self.ptr - self.filled) % self.HISTORY, ids]
        phase, level = self.tracker.phase_and_level(move)
        slot = torch.full_like(phase, -1)
        for i, p in enumerate(self.PHASES):
            slot = torch.where(phase == p, i, slot)
        keep = slot >= 0
        level, slot, last, early = level[keep], slot[keep], last[keep], early[keep]
        result = torch.full_like(level, outcome)
        for when, data in ((0, early), (1, last)):
            self.sums.index_put_((level, slot, result, torch.full_like(level, when)), data, accumulate=True)
        self.counts.index_put_((level, slot, result), torch.ones_like(level, dtype=self.counts.dtype), accumulate=True)

    def chain(self, on_step):
        manager = self.env.termination_manager

        def step(dones, finished):
            t = self.tracker
            active = self.mask & ~finished
            in_stairs = self.prev_move < t.final_advance
            # 넘어짐: 이번 step에 bad_ori로 끝남. 끝난 env의 tracker.move는 이미 다음 에피소드 값이다
            fell = dones & active & manager.get_term("bad_ori").bool() & in_stairs
            self._add(fell.nonzero().squeeze(-1), t.ended_move[fell], 1)
            # 통과: 끝나지 않고 단계가 올라감
            passed = ~dones & active & (t.move > self.prev_move) & in_stairs
            self._add(passed.nonzero().squeeze(-1), self.prev_move[passed], 0)
            self.phase_time = torch.where(dones | (t.move != self.prev_move), torch.zeros_like(self.phase_time),
                                          self.phase_time + self.env.step_dt)
            self.prev_move = t.move.clone()
            self.snapshot()
            if on_step is not None:
                on_step(dones, finished)

        return step

    def report(self):
        lines = ["\n===== 뒷발 단계 넘어짐 진단 (--fall_probe) ====="]
        lines.append("값은 평균. 앞으로 숙임 + = 머리가 아래, 왼쪽 기울어짐 + = 왼쪽(L1~L3)이 아래. 발 높이 0 = 딛고 있음")
        names = [PHASE_NAMES[p] for p in self.PHASES]
        steps = self.counts.shape[0]
        for i, name in enumerate(names):
            header = f"\n[{name}] " + " | ".join(
                f"{k + 1}번째 단: 통과 {int(self.counts[k, i, 0])}, 넘어짐 {int(self.counts[k, i, 1])}" for k in range(steps)
            )
            lines.append(header)
            cols = []
            for k in range(steps):
                for outcome, label in ((0, "통과"), (1, "넘어짐")):
                    if self.counts[k, i, outcome] > 0:
                        cols.append((k, outcome, f"{k + 1}단 {label}"))
            if not cols:
                continue
            lines.append(f"{'':28s}" + "".join(f"{c[2] + ' -0.5s':>14s}{c[2] + ' 직전':>13s}" for c in cols))
            for j, (label, key) in enumerate(self.FEATURES):
                scale = 100.0 if key in self.PERCENT else 1.0
                row = f"{label:28s}"
                for k, outcome, _ in cols:
                    mean = self.sums[k, i, outcome, :, j] / self.counts[k, i, outcome] * scale
                    row += f"{mean[0].item():14.2f}{mean[1].item():13.2f}"
                lines.append(row)
        return "\n".join(lines)


class NetPhaseMeter:
    """단계 추정 신경망(--phase_net)의 예측을 첫 에피소드 동안 참값 단계와 비교한다. PhaseNetPolicy를 감싼다.
    단계는 단마다 반복되는 7가지(stair_phase_obs)라서, 앞섬/뒤처짐은 한 단 안의 순환 차이로 정한다
    (-3 ~ +2칸, 정확히 반 바퀴는 뒤처짐). 정지를 미리 예측하면 앞섬, 정지인데 다른 단계로 예측하면 뒤처짐."""

    def __init__(self, policy, tracker, mask):
        self.policy, self.tracker, self.mask = policy, tracker, mask
        k, device = len(AUX_PHASE_NAMES), mask.device
        self.confusion = torch.zeros(k, k, device=device)  # [참값, 예측]
        self.counts = torch.zeros(3, device=device)  # 같음, 앞섬, 뒤처짐
        self.behind_at = torch.zeros(k, device=device)  # 뒤처진 step을 예측 단계별로
        self.streak = torch.zeros(mask.shape[0], dtype=torch.long, device=device)
        self.longest = torch.zeros_like(self.streak)  # env별 가장 길게 계속 틀린 step 수
        self.stuck = torch.zeros((), device=device)
        self.steps = torch.zeros((), device=device)
        self.label = self.predicted = None

    def __call__(self, obs):
        # 관측이 만들어진 순간의 참값 (step 전)
        self.label = self.tracker.phase_one_hot(self.tracker.move).argmax(dim=-1)
        actions = self.policy(obs)
        self.predicted = self.policy.last_logits.argmax(dim=-1)
        return actions

    def reset(self, dones):
        self.policy.reset(dones)

    def chain(self, on_step):
        k = len(AUX_PHASE_NAMES)
        stop = k - 1

        def step(dones, finished):
            active = self.mask & ~finished
            label, predicted = self.label, self.predicted
            index = label[active] * k + predicted[active]
            self.confusion.view(-1).add_(torch.bincount(index, minlength=self.confusion.numel()).float())
            difference = (predicted - label + stop // 2) % stop - stop // 2
            difference = torch.where(predicted == stop, torch.ones_like(difference), difference)
            difference = torch.where(label == stop, -torch.ones_like(difference), difference)
            difference = torch.where(predicted == label, torch.zeros_like(difference), difference)
            d = difference[active]
            self.counts[0] += (d == 0).sum()
            self.counts[1] += (d > 0).sum()
            self.counts[2] += (d < 0).sum()
            behind = active & (difference < 0)
            self.behind_at.add_(torch.bincount(predicted[behind], minlength=k).float())
            wrong = active & (predicted != label)
            self.streak = torch.where(wrong, self.streak + 1, torch.zeros_like(self.streak))
            self.longest = torch.maximum(self.longest, self.streak)
            self.stuck += (self.streak >= STUCK_STEPS).sum()
            self.steps += active.sum()
            if on_step is not None:
                on_step(dones, finished)

        return step

    def stuck_text(self, step_dt):
        """0.5초 넘게 계속 틀린 step 비율과 env별 최장 연속 틀림."""
        longest = self.longest[self.mask].float() * step_dt
        return (
            f"0.5초+ 연속 틀림 {(self.stuck / self.steps.clamp_min(1)).item():.1%} | "
            f"최장 연속 틀림 평균 {longest.mean().item():.2f}초, 90% {longest.quantile(0.9).item():.2f}초"
        )

    def report(self, step_dt):
        same, ahead, behind = phase_shares(self.counts)
        lines = [confusion_report(self.confusion, "단계 추정 신경망")]
        lines.append(f"  일치 {same:.1%}, 앞섬 {ahead:.1%}, 뒤처짐 {behind:.1%}")
        lines.append("  " + self.stuck_text(step_dt))
        return "\n".join(lines)


def phase_shares(counts):
    """step 수를 (같음, 앞섬, 뒤처짐) 비율로."""
    return (counts / counts.sum().clamp_min(1.0)).tolist()


def run_error_sweep(env, policy, tracker, is_stair, make_error_cfgs, observes_estimate, phase_net=False,
                    net_uses_rule=False):
    """ERROR_SWEEP의 오차마다 첫 에피소드를 돌려 결과를 한 줄씩 출력한다. 정책과 env는 재사용한다.
    추정 단계를 관측으로 받지 않는 student는 영향이 없는 줄을 건너뛰고, 단계 일치 칸을 비운다.
    phase_net: policy가 PhaseNetPolicy다. 일치 칸은 신경망 예측으로 잰다. 단계 계산에만 쓰는 원인은 건너뛴다.
    net_uses_rule: 신경망이 규칙 판정(phase_rule)을 입력으로 받는다. 발끝 높이 계산 오차가 영향을 주므로 지연만 건너뛴다."""
    base_env = env.unwrapped
    print("\n===== 센서 오차 시험 =====")
    print("오차 | 꼭대기 도달 | 추정 단계 일치 / 앞섬 / 뒤처짐 | 평균 단계 | 종료 이유")
    for name, error in ERROR_SWEEP:
        skipped = {"delay"} if net_uses_rule else ESTIMATE_ONLY
        if (not observes_estimate or phase_net) and error and set(error) <= skipped:
            continue
        estimator_cfg, sensor_cfg = make_error_cfgs(**error)
        # 이전 평가에서 만든 상태는 inference tensor라서 리셋도 inference_mode 안에서 해야 한다
        with torch.inference_mode():
            tracker.sensors.configure(sensor_cfg)
            if tracker.estimator is not None:
                tracker.estimator.configure(estimator_cfg)
            env.reset()  # 여기서 새 오차의 편향을 뽑는다
            policy.reset(torch.ones(base_env.num_envs, dtype=torch.bool, device=base_env.device))
        driver, net_meter, rule_counts = policy, None, None
        if phase_net:
            net_meter = NetPhaseMeter(policy, tracker, is_stair)
            rule_step = None
            if net_uses_rule:  # 신경망이 입력으로 받는 규칙 추정기의 일치도 함께 잰다 (신경망이 고친 만큼 비교)
                rule_step, rule_counts, _ = phase_meter(tracker, is_stair)
            driver, on_step = net_meter, net_meter.chain(rule_step)
            counts, behind_at, names = net_meter.counts, net_meter.behind_at, AUX_PHASE_NAMES
        elif not observes_estimate:
            on_step, counts, behind_at, names = None, torch.zeros(3), torch.zeros(len(PHASE_NAMES)), PHASE_NAMES
        else:
            (on_step, counts, behind_at), names = phase_meter(tracker, is_stair), PHASE_NAMES
        data = collect_first_episodes(
            env,
            driver,
            env.get_observations(),
            tracker,
            base_env.termination_manager,
            base_env.max_episode_length + 1,
            on_step,
        )
        stair_data = {k: v[is_stair] if torch.is_tensor(v) else v for k, v in data.items()}
        everyone = torch.ones_like(stair_data["start"], dtype=torch.bool)
        result = summarize(stair_data, everyone, tracker.final_advance, tracker.done_move, base_env.step_dt)
        same, ahead, behind = phase_shares(counts)
        agreement = f"{same:.1%} / {ahead:.1%} / {behind:.1%}" if observes_estimate else "-"
        reasons = ", ".join(f"{k} {v:.1%}" for k, v in result["reasons"].items() if v > 0)
        print(
            f"{name} | {result['reached_top']:.1%} (±{result['margin']:.1%}) | "
            f"{agreement} | {result['mean_move']:.2f} | {reasons}",
            flush=True,
        )
        if not observes_estimate:
            continue
        # 뒤처진 시간이 어느 추정 단계에서 생겼는지 (멈춘 위치). 건너뛰기는 그 오차에서 일어난 횟수
        details = []
        if behind_at.sum() > 0:
            shares = (behind_at / behind_at.sum()).tolist()
            details.append("뒤처진 위치: " + ", ".join(
                f"{n} {s:.0%}" for n, s in zip(names, shares, strict=True) if s >= 0.005
            ))
        if net_meter is not None:
            details.append(net_meter.stuck_text(base_env.step_dt))
            if rule_counts is not None:
                rule = phase_shares(rule_counts)
                details.append(f"규칙 추정기 일치 {rule[0]:.1%} / 앞섬 {rule[1]:.1%} / 뒤처짐 {rule[2]:.1%}")
        elif tracker.estimator.cfg.catch_up:
            details.append(f"건너뛰기 {tracker.estimator.catch_ups.item()}회")
        if details:
            print("    " + " | ".join(details), flush=True)


def main():
    import gymnasium as gym

    from isaaclab.app import add_launcher_args, launch_simulation

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

    import RLforHexapod.tasks  # noqa: F401
    from RLforHexapod.tasks.walk.mdp import PhaseEstimatorCfg, SensorErrorCfg, get_sensor_model, stair_env_mask

    parser = argparse.ArgumentParser(description="저장된 정책의 계단 오르기 평가")
    parser.add_argument("--task", required=True, help="Hexapod-Stair 또는 Hexapod-Stair-Student-*")
    parser.add_argument("--checkpoint", required=True, help="불러올 .pt 파일")
    parser.add_argument("--teacher", action="store_true", help="student 대신 teacher로 움직여 기준 성능을 잰다")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--error_sweep", "--phase_sweep", dest="error_sweep", action="store_true",
        help="센서 오차 원인별로 차례대로 재서 표로 보고한다",
    )
    for flag, _, _, help_text in SENSOR_FLAGS:
        parser.add_argument(flag, type=float, default=0.0, help=help_text)
    parser.add_argument(
        "--map_mode", choices=("accumulated", "per_frame"), default="accumulated",
        help="높이맵 방식. accumulated = 지도에 쌓아 잘라 씀 (x 위치 오차만큼 높이맵도 밀림), per_frame = 매 프레임 스캔",
    )
    parser.add_argument("--phase_delay", type=float, default=None, help="단계 계산 지연 [ms]. 기본은 task 설정")
    parser.add_argument(
        "--phase_judge", choices=("height", "contact"), default=None,
        help="딛고 있음 판정: height = 발 높이 (참값 tracker와 같음), contact = 높이 또는 발끝 스위치. 기본은 task 설정",
    )
    parser.add_argument(
        "--phase_contact_window", type=float, default=None,
        help="contact 판정에서 스위치로 인정할 발끝 높이 차 [cm]. 기본은 task 설정 (8)",
    )
    parser.add_argument(
        "--phase_contact_confirm", type=float, default=None,
        help="contact 판정에서 스위치로만 통과하는 상태가 이 시간 이어져야 인정한다 [ms]. 0 = 바로. 기본은 task 설정",
    )
    parser.add_argument(
        "--phase_catch_up", action=argparse.BooleanOptionalAction, default=None,
        help="전진 단계에서 다음 옮기기 단계가 이미 끝났으면 건너뛴다. 기본은 task 설정",
    )
    parser.add_argument(
        "--phase_net", default=None,
        help="단계 추정 신경망 (.pt, scripts/train_phase_net.py). A-Phase 계열의 단계 입력을 이 신경망 출력으로 바꾼다",
    )
    parser.add_argument(
        "--fall_probe", action="store_true",
        help="뒷발 단계(접기 전진·뒷발1·뒷발2)에서 넘어진 에피소드와 통과한 에피소드의 자세를 단별로 비교한다",
    )
    add_launcher_args(parser)
    args, remaining = setup_preset_cli(parser, sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining

    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")

    sensor_values = {field: getattr(args, flag[2:]) * scale for flag, field, scale, _ in SENSOR_FLAGS}
    judge_options = (args.phase_delay, args.phase_judge, args.phase_contact_window, args.phase_contact_confirm,
                     args.phase_catch_up)
    wants_errors = args.error_sweep or any(sensor_values.values()) or args.map_mode != "accumulated"
    is_student = hasattr(env_cfg.observations, "student")
    if wants_errors and (args.teacher or not is_student):
        parser.error("오차 평가는 student용입니다. teacher는 참값 관측을 봅니다.")
    # 추정 단계를 관측으로 받는 task (A-Phase, A-PhaseNoise)
    observes_estimate = hasattr(env_cfg, "phase_estimator")
    if any(o is not None for o in judge_options) and not observes_estimate:
        parser.error(f"{args.task}는 추정 단계를 쓰지 않습니다. --phase_* 판정 옵션은 A-Phase 계열에만 있습니다.")
    if args.phase_net is not None and (not observes_estimate or args.teacher):
        parser.error("--phase_net은 단계 입력을 받는 A-Phase 계열 student에만 쓸 수 있습니다.")

    # 평가는 오차를 지정한 크기 그대로 쓴다 (학습용 무작위 크기는 쓰지 않는다). 판정 방식은 task 설정을 따른다
    estimator_settings = {}
    if observes_estimate:
        task = env_cfg.phase_estimator

        def pick(value, default, scale=1.0):
            return default if value is None else value * scale

        estimator_settings = {
            "by_contact": task.by_contact if args.phase_judge is None else args.phase_judge == "contact",
            "contact_window": pick(args.phase_contact_window, task.contact_window, 0.01),
            "contact_confirm": pick(args.phase_contact_confirm, task.contact_confirm, 0.001),
            "catch_up": pick(args.phase_catch_up, task.catch_up),
        }
        estimator_delay = pick(args.phase_delay, 0.0, 0.001)
        env_cfg.phase_estimator = PhaseEstimatorCfg(delay=estimator_delay, **estimator_settings)
    if is_student and not args.teacher:
        env_cfg.sensor_error = SensorErrorCfg(**sensor_values, map_mode=args.map_mode)

    estimator_fields = {f.name for f in dataclasses.fields(PhaseEstimatorCfg)}

    def make_error_cfgs(**error):
        """sweep 한 줄의 오차를 센서 오차 설정과 단계 추정기 설정으로 나눈다."""
        estimator = {k: v for k, v in error.items() if k in estimator_fields}
        sensor = {k: v for k, v in error.items() if k not in estimator_fields}
        return PhaseEstimatorCfg(**estimator, **estimator_settings), SensorErrorCfg(**sensor, map_mode=args.map_mode)

    with launch_simulation(env_cfg, args), contextlib.ExitStack() as cleanup:
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = agent_cfg.seed = args.seed
        if getattr(args, "device", None) is not None:
            env_cfg.sim.device = agent_cfg.device = args.device
        env_cfg.events.stair_reset.params["buffer_ratio"] = 0.0

        env = gym.make(args.task, cfg=env_cfg)
        cleanup.callback(env.close)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)  # 여기서 첫 리셋이 일어난다
        # 센서 오차와 판정 설정을 env를 만든 뒤 직접 넣고 다시 리셋한다 (오차 시험과 같은 방식)
        if observes_estimate:
            env.unwrapped.stair_tracker.estimator.configure(
                PhaseEstimatorCfg(delay=estimator_delay, **estimator_settings))
        if is_student and not args.teacher:
            get_sensor_model(env.unwrapped).configure(SensorErrorCfg(**sensor_values, map_mode=args.map_mode))
        env.reset()

        runner = create_rsl_rl_runner(env, agent_cfg)
        print(f"[INFO] 체크포인트: {args.checkpoint}")
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        if args.teacher:
            policy = runner.alg.teacher

        base_env = env.unwrapped
        tracker = base_env.stair_tracker
        is_stair = stair_env_mask(base_env)
        is_walk = ~is_stair
        if args.phase_net is not None:
            from RLforHexapod.tasks.walk.config.hexapod.agents.phase_net import PhaseNet, PhaseNetPolicy, term_slice

            net, _ = PhaseNet.load(args.phase_net, base_env.device)
            net.eval()
            if not net.uses_rule and any(o is not None for o in judge_options):
                parser.error("이 신경망은 규칙 판정을 입력으로 받지 않으므로 --phase_* 판정 옵션은 의미가 없습니다.")
            policy = PhaseNetPolicy(policy, net, term_slice(base_env.observation_manager), base_env.num_envs,
                                    base_env.device)
            print(f"[INFO] 단계 입력: 단계 추정 신경망 {args.phase_net} ({net.config['input_mode']}), "
                  f"추가 입력 {net.extra_groups}")
        if observes_estimate and (args.phase_net is None or net.uses_rule):
            s = estimator_settings
            judge = "높이 3cm"
            if s["by_contact"]:
                judge += f" 또는 (스위치 + 높이 {s['contact_window'] * 100:.1f}cm, {s['contact_confirm'] * 1000:.0f}ms 유지)"
            print(f"[INFO] 딛고 있음 판정: {judge} | 전진 건너뛰기: {'켬' if s['catch_up'] else '끔'}")
        if is_student and not args.teacher:
            print(f"[INFO] 높이맵 방식: {args.map_mode}")

        if args.error_sweep:
            run_error_sweep(env, policy, tracker, is_stair, make_error_cfgs, observes_estimate,
                            phase_net=args.phase_net is not None,
                            net_uses_rule=args.phase_net is not None and net.uses_rule)
            return

        on_step, walk_totals, walk_steps = walk_meter(base_env, is_walk) if is_walk.any() else (None, None, None)
        phase_counts = behind_at = None
        if observes_estimate and not is_walk.any() and args.phase_net is None:
            on_step, phase_counts, behind_at = phase_meter(tracker, is_stair)
        obs = env.get_observations()
        aux_meter = net_meter = fall_probe = None
        if args.fall_probe:
            fall_probe = FallProbe(base_env, tracker, is_stair)
            on_step = fall_probe.chain(on_step)
        if args.phase_net is not None:
            net_meter = NetPhaseMeter(policy, tracker, is_stair)
            policy, on_step = net_meter, net_meter.chain(on_step)
        if hasattr(policy, "last_phase_logits") and "phase_label" in obs.keys():
            aux_meter = PhaseAuxMeter(policy, is_stair)
            policy, on_step = aux_meter, aux_meter.chain(on_step)
        data = collect_first_episodes(
            env,
            policy,
            obs,
            tracker,
            base_env.termination_manager,
            base_env.max_episode_length + 1,
            on_step,
        )
        print(f"\n===== {args.task} ({'teacher' if args.teacher else 'student'}) =====")
        if is_stair.any():
            stair_data = {k: v[is_stair] if torch.is_tensor(v) else v for k, v in data.items()}
            print(format_report(stair_data, tracker.final_advance, tracker.done_move, base_env.step_dt))
        if phase_counts is not None:
            same, ahead, behind = phase_shares(phase_counts)
            print(f"\n단계 추정: 일치 {same:.1%}, 앞섬 {ahead:.1%}, 뒤처짐 {behind:.1%}")
            if behind_at.sum() > 0:
                shares = (behind_at / behind_at.sum()).tolist()
                print("  뒤처진 위치: " + ", ".join(
                    f"{n} {s:.0%}" for n, s in zip(PHASE_NAMES, shares, strict=True) if s >= 0.005
                ))
        if aux_meter is not None:
            print(aux_meter.report())
        if net_meter is not None:
            print(net_meter.report(base_env.step_dt))
        if fall_probe is not None:
            print(fall_probe.report())
        if is_walk.any():
            print(format_walk_report(data, is_walk, walk_totals, walk_steps))


if __name__ == "__main__":
    main()
