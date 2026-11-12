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

추정 단계를 보는 student(Hexapod-Stair-Student-A-Phase)는 단계 계산의 오차를 넣어 잴 수 있다.
--phase_sweep은 오차별로 차례대로 재서 표로 보고한다. 설정마다 첫 에피소드 전체를 돌리므로 --num_envs 1024 정도를 권한다.

    uv run python scripts/eval_stairs.py --task Hexapod-Stair-Student-A-Phase --checkpoint <student.pt> --phase_sweep --num_envs 1024
    uv run python scripts/eval_stairs.py --task Hexapod-Stair-Student-A-Phase --checkpoint <student.pt> --phase_pos_bias 2 --phase_delay 40

위치 오차는 x(라이다, 수직면 기준)와 높이(지지 발 기준 발끝 높이)로 나눠 넣는다.
판정 방식은 task 설정을 따른다 (A-Phase, A-PhaseNoise 모두 높이 3cm, 건너뛰기 끔). 옵션으로 바꿀 수 있다.
- --phase_judge contact: 높이 3cm 또는 (발끝 스위치 + 높이 --phase_contact_window cm, --phase_contact_confirm ms 유지)
- --phase_catch_up: 전진 단계에서 다음 옮기기 단계가 이미 끝났으면 건너뛴다
평가에서는 오차 크기를 항상 지정한 값 그대로 쓴다 (학습용 무작위 크기는 쓰지 않는다).
sweep의 각 줄 아래에 뒤처진 시간이 어느 추정 단계에서 생겼는지 나온다.

    uv run python scripts/eval_stairs.py --task Hexapod-Stair-Student-A-Phase --checkpoint <student.pt> --phase_sweep --phase_judge contact --phase_catch_up
"""

import argparse
import contextlib
import importlib.metadata
import math
import sys

import torch

START_NAMES = {0: "0번 (계단 앞)", 1: "1번 (앞발이 수직면 앞)"}
FAIL_TERMS = ("base_low", "bad_ori", "sim_nan", "sim_blowup")  # 걷기에서 넘어지거나 물리가 터진 종료

# --phase_sweep의 시험 목록. 측정값이 없어 잡은 시작값이다. 값은 PhaseErrorCfg 단위 (m, rad, 확률, 초)
PHASE_SWEEP = [
    ("오차 없음", {}),
    ("x 편향 1cm", {"pos_bias": 0.01}),
    ("x 편향 2cm", {"pos_bias": 0.02}),
    ("x 편향 4cm", {"pos_bias": 0.04}),
    ("x 노이즈 0.5cm", {"pos_noise": 0.005}),
    ("x 노이즈 1cm", {"pos_noise": 0.01}),
    ("높이 편향 0.5cm", {"height_bias": 0.005}),
    ("높이 편향 1cm", {"height_bias": 0.01}),
    ("높이 편향 2cm", {"height_bias": 0.02}),
    ("높이 노이즈 0.5cm", {"height_noise": 0.005}),
    ("관절 편향 1도", {"joint_bias": math.radians(1.0)}),
    ("관절 편향 2도", {"joint_bias": math.radians(2.0)}),
    ("관절 편향 4도", {"joint_bias": math.radians(4.0)}),
    ("접촉 오판 2%", {"contact_flip": 0.02}),
    ("접촉 오판 5%", {"contact_flip": 0.05}),
    ("지연 40ms", {"delay": 0.04}),
    ("지연 100ms", {"delay": 0.1}),
    ("전체 (예상 수준)", {
        "pos_bias": 0.02, "pos_noise": 0.005, "height_bias": 0.01, "height_noise": 0.003,
        "joint_bias": math.radians(2.0), "contact_flip": 0.02, "delay": 0.04,
    }),
]


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


def phase_shares(counts):
    """step 수를 (같음, 앞섬, 뒤처짐) 비율로."""
    return (counts / counts.sum().clamp_min(1.0)).tolist()


def run_phase_sweep(env, policy, tracker, is_stair, make_error_cfg):
    """PHASE_SWEEP의 오차마다 첫 에피소드를 돌려 결과를 한 줄씩 출력한다. 정책과 env는 재사용한다."""
    base_env = env.unwrapped
    print("\n===== 단계 추정 오차 시험 =====")
    print("오차 | 꼭대기 도달 | 추정 단계 일치 / 앞섬 / 뒤처짐 | 평균 단계 | 종료 이유")
    for name, error in PHASE_SWEEP:
        # 이전 평가에서 만든 상태는 inference tensor라서 리셋도 inference_mode 안에서 해야 한다
        with torch.inference_mode():
            tracker.estimator.configure(make_error_cfg(**error))
            env.reset()  # 여기서 새 오차의 편향을 뽑는다
            policy.reset(torch.ones(base_env.num_envs, dtype=torch.bool, device=base_env.device))
        on_step, counts, behind_at = phase_meter(tracker, is_stair)
        data = collect_first_episodes(
            env,
            policy,
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
        reasons = ", ".join(f"{k} {v:.1%}" for k, v in result["reasons"].items() if v > 0)
        print(
            f"{name} | {result['reached_top']:.1%} (±{result['margin']:.1%}) | "
            f"{same:.1%} / {ahead:.1%} / {behind:.1%} | {result['mean_move']:.2f} | {reasons}",
            flush=True,
        )
        # 뒤처진 시간이 어느 추정 단계에서 생겼는지 (멈춘 위치). 건너뛰기는 그 오차에서 일어난 횟수
        details = []
        if behind_at.sum() > 0:
            shares = (behind_at / behind_at.sum()).tolist()
            details.append("뒤처진 위치: " + ", ".join(
                f"{n} {s:.0%}" for n, s in zip(PHASE_NAMES, shares, strict=True) if s >= 0.005
            ))
        if tracker.estimator.cfg.catch_up:
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
    from RLforHexapod.tasks.walk.mdp import PhaseErrorCfg, stair_env_mask

    parser = argparse.ArgumentParser(description="저장된 정책의 계단 오르기 평가")
    parser.add_argument("--task", required=True, help="Hexapod-Stair 또는 Hexapod-Stair-Student-*")
    parser.add_argument("--checkpoint", required=True, help="불러올 .pt 파일")
    parser.add_argument("--teacher", action="store_true", help="student 대신 teacher로 움직여 기준 성능을 잰다")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase_sweep", action="store_true", help="단계 추정 오차별로 차례대로 재서 표로 보고한다")
    parser.add_argument("--phase_pos_bias", type=float, default=0.0, help="수직면에 대한 x 위치 편향 [cm]")
    parser.add_argument("--phase_pos_noise", type=float, default=0.0, help="x 위치 노이즈 std [cm]")
    parser.add_argument("--phase_height_bias", type=float, default=0.0, help="발끝 높이 편향 [cm]")
    parser.add_argument("--phase_height_noise", type=float, default=0.0, help="발끝 높이 노이즈 std [cm]")
    parser.add_argument("--phase_joint_bias", type=float, default=0.0, help="관절각 편향 [도]")
    parser.add_argument("--phase_contact_flip", type=float, default=0.0, help="접촉 판정이 뒤집힐 확률 [%%]")
    parser.add_argument("--phase_delay", type=float, default=0.0, help="단계 계산 지연 [ms]")
    parser.add_argument(
        "--phase_judge", choices=("height", "contact"), default=None,
        help="딛고 있음 판정: height = 발 높이 (참값 tracker와 같음), contact = 접촉 센서. 기본은 task 설정",
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
    add_launcher_args(parser)
    args, remaining = setup_preset_cli(parser, sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining

    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")

    judge_options = (args.phase_judge, args.phase_contact_window, args.phase_contact_confirm, args.phase_catch_up)
    wants_phase_error = args.phase_sweep or any(o is not None for o in judge_options) or any(
        (args.phase_pos_bias, args.phase_pos_noise, args.phase_height_bias, args.phase_height_noise,
         args.phase_joint_bias, args.phase_contact_flip, args.phase_delay)
    )
    if wants_phase_error and not hasattr(env_cfg, "phase_error"):
        parser.error(f"{args.task}는 추정 단계를 쓰지 않습니다. Hexapod-Stair-Student-A-Phase를 쓰세요.")
    if args.phase_sweep and args.teacher:
        parser.error("--phase_sweep은 student 평가용입니다. teacher는 추정 단계를 보지 않습니다.")

    # 평가는 오차를 지정한 크기 그대로 쓴다. 판정 방식(접촉 범위, 확인 시간, 건너뛰기)은 task 설정을 따른다
    phase_settings = {}
    if hasattr(env_cfg, "phase_error"):
        task_error = env_cfg.phase_error

        def pick(value, default, scale=1.0):
            return default if value is None else value * scale

        phase_settings = {
            "by_contact": task_error.by_contact if args.phase_judge is None else args.phase_judge == "contact",
            "contact_window": pick(args.phase_contact_window, task_error.contact_window, 0.01),
            "contact_confirm": pick(args.phase_contact_confirm, task_error.contact_confirm, 0.001),
            "catch_up": pick(args.phase_catch_up, task_error.catch_up),
        }
        env_cfg.phase_error = PhaseErrorCfg(
            pos_bias=args.phase_pos_bias / 100.0,
            pos_noise=args.phase_pos_noise / 100.0,
            height_bias=args.phase_height_bias / 100.0,
            height_noise=args.phase_height_noise / 100.0,
            joint_bias=math.radians(args.phase_joint_bias),
            contact_flip=args.phase_contact_flip / 100.0,
            delay=args.phase_delay / 1000.0,
            **phase_settings,
        )

    def make_error_cfg(**error):
        return PhaseErrorCfg(**error, **phase_settings)

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
        if phase_settings:
            s = phase_settings
            judge = "높이 3cm"
            if s["by_contact"]:
                judge += f" 또는 (스위치 + 높이 {s['contact_window'] * 100:.1f}cm, {s['contact_confirm'] * 1000:.0f}ms 유지)"
            print(f"[INFO] 딛고 있음 판정: {judge} | 전진 건너뛰기: {'켬' if s['catch_up'] else '끔'}")

        if args.phase_sweep:
            run_phase_sweep(env, policy, tracker, is_stair, make_error_cfg)
            return

        on_step, walk_totals, walk_steps = walk_meter(base_env, is_walk) if is_walk.any() else (None, None, None)
        phase_counts = None
        if tracker.estimator is not None and not is_walk.any():
            on_step, phase_counts = phase_meter(tracker, is_stair)
        data = collect_first_episodes(
            env,
            policy,
            env.get_observations(),
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
        if is_walk.any():
            print(format_walk_report(data, is_walk, walk_totals, walk_steps))


if __name__ == "__main__":
    main()
