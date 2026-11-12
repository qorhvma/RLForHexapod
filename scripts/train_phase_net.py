"""단계 추정 신경망 학습 (agents/phase_net.py). 계단 정책은 학습하지 않는다.

A-Phase student(참값 단계로 증류한 체크포인트)가 계단을 오르는 동안, 신경망이 student 센서 관측만 보고
참값 단계를 맞히도록 cross-entropy로 학습한다.

    uv run python scripts/train_phase_net.py --policy <A_phase model_3550.pt> --headless

데이터를 만드는 방식 (DAgger와 같은 생각):
- 처음에는 A-Phase가 참값 단계를 보며 움직인다. 신경망이 아직 엉터리라 그 출력으로 움직이면 첫 단에서 넘어진다.
- --dagger_iters 동안 참값으로 움직이는 에피소드 비율을 1 → 0으로 줄인다. 그 뒤는 모두 신경망 출력으로 움직인다.
  실제로 쓸 때처럼 신경망이 틀린 단계를 넣었을 때 생기는 상태에서도 학습하기 위해서다.
- 정답은 언제나 참값 단계다 (StairTracker).

센서 오차 (mdp/sensor_model.py): 학습 중에는 에피소드마다 0 ~ 최댓값에서 뽑는다 (TRAIN_ERRORS × --error_scale).
신경망이 실물 같은 관측 오차에서도 단계를 맞히게 하기 위해서다. 같은 오차가 A-Phase의 관측에도 들어간다.
단계 계산에만 쓰는 원인(발끝 높이 계산, 계산 지연)은 신경망과 관계없어 넣지 않는다.

--resume으로 이어 학습할 때는 --dagger_iters 0을 주면 처음부터 신경망 출력으로 움직인다.

단계별 가중치 (--class_weight_power): 단계마다 시간 비중이 크게 다르다 (꼭대기에 서 있는 정지가 약 50%).
가중치가 없으면 애매할 때 정지로 답하는 쪽으로 학습돼, 마지막 단의 접기 전진·뒷발에서 정지를 일찍 말한다.
그래서 손실에서 단계 c의 가중치를 (시간 비중)^(-power)로 준다. 1 = 시간 비중에 반비례 (단계마다 손실 몫이 같음),
0 = 가중치 없음 (기본). 시간 비중은 정답 단계를 iteration마다 세어 지수 평균한다. 가중치는 WEIGHT_RANGE로 자른다.
1로 학습해 보니 꼭대기 도달이 58.9% → 47.6%로 나빠졌다: 정지를 일찍 말하는 대신 꼭대기에서 접기 전진·뒷발이라고 말했다.

추가 입력 (phase_net 관측 그룹, 기본으로 넣음, --no_extra로 뺌): 마지막 단은 앞쪽 높이맵이 평평해서 꼭대기와 같아 보인다.
- 라이다로 잰 몸통 x (mdp.lidar_body_x): 몇 번째 단 위인지. 라이다 x 오차가 높이맵 밀림과 같은 값으로 들어간다
- 몸통 기준 발끝 높이 (mdp.feet_below_body, IMU + 관절각 FK): 뒷발이 아직 아래 단에 있는지.
  마지막 단의 접기 전진·뒷발은 몸통 x가 꼭대기 정지와 거의 같아서 이것이 주된 구분 정보다

결과는 logs/phase_net/<시각>/model_<iter>.pt. TensorBoard:
- Phase/accuracy_net: 신경망 출력으로 움직인 env에서의 정답률
- Phase/stuck_share: 신경망으로 움직인 env에서 0.5초 넘게 계속 틀리고 있는 step의 비율 (규칙 추정기의 "뒤처져 굳음"에 해당)
- Metrics/*: env 로그 (stair_reached_top 등, 참값으로 움직인 에피소드도 섞여 있다)
"""

import argparse
import importlib.metadata
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

# 학습용 센서 오차 최댓값. eval_stairs.py --error_sweep의 "전체 (예상 수준)"에서 관측에 들어가는 원인만, 조금 넉넉하게
TRAIN_ERRORS = {
    "lidar_x_bias": 0.02,
    "lidar_x_noise": 0.005,
    "lidar_mount_tilt": math.radians(1.0),
    "lidar_range_bias": 0.01,
    "lidar_dropout": 0.05,
    "imu_tilt": math.radians(1.0),
    "gyro_noise": 0.05,
    "encoder_bias": math.radians(3.0),
    "encoder_noise": math.radians(0.5),
    "contact_flip": 0.05,
}
STUCK_STEPS = 25  # 0.5초 (step 0.02초)
WEIGHT_RANGE = (0.1, 10.0)  # 단계별 가중치 하한·상한 (평균 1 기준). 아주 드문 단계가 손실을 독차지하지 않게
SHARE_DECAY = 0.95  # 단계별 시간 비중의 지수 평균


def class_weights(share: torch.Tensor, power: float) -> torch.Tensor:
    """단계별 가중치 = share^(-power), 데이터 전체의 평균 가중치가 1이 되게 맞춘 뒤 WEIGHT_RANGE로 자른다."""
    share = share / share.sum().clamp_min(1e-12)
    weight = share.clamp_min(1e-4) ** (-power)
    weight = weight / (share * weight).sum().clamp_min(1e-12)
    return weight.clamp(*WEIGHT_RANGE)


def main():
    import gymnasium as gym

    from isaaclab.app import add_launcher_args, launch_simulation

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

    import RLforHexapod.tasks  # noqa: F401
    from RLforHexapod.tasks.walk.config.hexapod.agents.phase_net import (
        EXTRA_GROUP,
        PhaseNet,
        PhaseNetPolicy,
        net_input,
        term_slice,
    )
    from RLforHexapod.tasks.walk.mdp import PhaseEstimatorCfg, SensorErrorCfg

    parser = argparse.ArgumentParser(description="단계 추정 신경망 학습")
    parser.add_argument("--task", default="Hexapod-Stair-Student-A-Phase", help="단계 입력을 받는 student task")
    parser.add_argument("--policy", required=True, help="A-Phase student 체크포인트 (.pt)")
    parser.add_argument("--resume", default=None, help="이어서 학습할 단계 신경망 (.pt)")
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--steps", type=int, default=64, help="iteration마다 env당 step 수 (= 역전파 길이)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--mini_batches", type=int, default=4, help="env를 나눠 한 번에 학습하는 묶음 수")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128, help="GRU 크기")
    parser.add_argument("--input_mode", choices=("onehot", "probs"), default="onehot",
                        help="A-Phase에 넣는 형태. onehot = 가장 그럴듯한 단계 (A-Phase가 학습한 형태)")
    parser.add_argument("--dagger_iters", type=int, default=300, help="참값으로 움직이는 비율을 1 → 0으로 줄이는 iteration 수")
    parser.add_argument("--error_scale", type=float, default=1.0, help="학습용 센서 오차 크기 배율. 0 = 오차 없음")
    parser.add_argument("--class_weight_power", type=float, default=0.0,
                        help="단계별 손실 가중치 = (시간 비중)^(-power). 1 = 반비례, 0 = 가중치 없음 (1은 역효과였음)")
    parser.add_argument("--no_extra", action="store_true",
                        help="phase_net 관측 그룹(몸통 x, 발끝 높이)을 입력에 넣지 않는다 (Student-A 관측만)")
    parser.add_argument("--save_interval", type=int, default=100)
    add_launcher_args(parser)
    args, remaining = setup_preset_cli(parser, sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining

    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
    if not hasattr(env_cfg, "phase_estimator"):
        parser.error(f"{args.task}는 단계 입력을 받지 않습니다. A-Phase 계열 task를 쓰세요.")
    # 규칙 추정기의 출력은 신경망 출력으로 바꿔 넣으므로 쓰지 않는다
    env_cfg.phase_estimator = PhaseEstimatorCfg()
    errors = {k: v * args.error_scale for k, v in TRAIN_ERRORS.items()}
    env_cfg.sensor_error = SensorErrorCfg(**errors, randomize=True)

    with launch_simulation(env_cfg, args):
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = agent_cfg.seed = args.seed
        if getattr(args, "device", None) is not None:
            env_cfg.sim.device = agent_cfg.device = args.device
        torch.manual_seed(args.seed)

        env = gym.make(args.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        base_env = env.unwrapped
        device = base_env.device
        tracker = base_env.stair_tracker
        n = base_env.num_envs

        runner = create_rsl_rl_runner(env, agent_cfg)
        print(f"[INFO] A-Phase 정책: {args.policy}")
        runner.load(args.policy)
        policy = runner.get_inference_policy(device=device)

        cut = term_slice(base_env.observation_manager)
        obs = env.get_observations()
        num_classes = cut.stop - cut.start

        start_iter = 0
        if args.resume:
            net, saved = PhaseNet.load(args.resume, device)
            start_iter = saved.get("iteration", 0) + 1
            print(f"[INFO] 이어서 학습: {args.resume} (iteration {start_iter}부터)")
            extra_groups = net.extra_groups
        else:
            extra_groups = [] if args.no_extra else [EXTRA_GROUP]
            missing = [g for g in extra_groups if g not in obs.keys()]
            if missing:
                parser.error(f"{args.task} 관측에 {missing} 그룹이 없습니다. --no_extra를 쓰거나 A-Phase 계열 task를 쓰세요.")
        input_dim = net_input(obs, cut, extra_groups).shape[1]
        if not args.resume:
            net = PhaseNet(input_dim, args.hidden, num_classes, args.input_mode, extra_groups).to(device)
            saved = {}
        print(f"[INFO] 신경망 입력: student {obs['student'].shape[1] - num_classes} + {extra_groups} = {input_dim}")
        if net.config["input_dim"] != input_dim or net.config["num_classes"] != num_classes:
            raise ValueError(f"관측 크기가 다릅니다: 신경망 {net.config}, env 입력 {input_dim} / 단계 {num_classes}")
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        if "optimizer" in saved:
            optimizer.load_state_dict(saved["optimizer"])

        wrapped = PhaseNetPolicy(policy, net, cut, n, device)
        wrapped.forced = torch.ones(n, dtype=torch.bool, device=device)  # 처음엔 모두 참값으로 움직인다

        log_dir = os.path.join("logs", "phase_net", time.strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "args.json"), "w") as f:
            json.dump({**vars(args), "train_errors": errors}, f, indent=2, ensure_ascii=False)
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir)
        except ImportError:
            writer = None
        print(f"[INFO] 저장 위치: {log_dir}")

        T = args.steps
        buf_x = torch.zeros(T, n, input_dim, device=device)
        buf_y = torch.zeros(T, n, dtype=torch.long, device=device)
        buf_done = torch.zeros(T, n, dtype=torch.bool, device=device)
        buf_forced = torch.zeros(T, n, dtype=torch.bool, device=device)
        h0 = torch.zeros(n, net.config["hidden_dim"], device=device)
        share = saved.get("class_share")  # 단계별 시간 비중 (지수 평균). 이어 학습이면 저장된 값부터
        share = None if share is None else share.to(device)
        wrong_streak = torch.zeros(n, dtype=torch.long, device=device)

        def true_phase():
            return tracker.phase_one_hot(tracker.move)

        for it in range(start_iter, start_iter + args.iterations):
            started = time.time()
            done_frac = (it - start_iter) / max(1, args.dagger_iters)
            beta = max(0.0, 1.0 - done_frac)
            correct_net = torch.zeros((), device=device)
            count_net = torch.zeros((), device=device)
            stuck = torch.zeros((), device=device)
            env_logs: dict[str, list[float]] = {}

            # ----- 데이터 모으기 -----
            net.eval()
            with torch.inference_mode():
                h0.copy_(wrapped.hidden)
                for t in range(T):
                    label_hot = true_phase()
                    label = label_hot.argmax(dim=-1)
                    wrapped.forced_input = label_hot
                    actions = wrapped(obs)
                    buf_x[t].copy_(wrapped.last_input)
                    buf_y[t].copy_(label)
                    buf_forced[t].copy_(wrapped.forced)

                    wrong = wrapped.last_logits.argmax(dim=-1) != label
                    by_net = ~wrapped.forced
                    correct_net += (~wrong & by_net).sum()
                    count_net += by_net.sum()
                    wrong_streak = torch.where(wrong, wrong_streak + 1, torch.zeros_like(wrong_streak))
                    stuck += ((wrong_streak >= STUCK_STEPS) & by_net).sum()

                    obs, _, dones, extras = env.step(actions)
                    dones = dones.bool()
                    wrapped.reset(dones)
                    wrong_streak[dones] = 0
                    buf_done[t].copy_(dones)
                    # 끝난 env는 다음 에피소드를 참값으로 움직일지 새로 뽑는다
                    redraw = torch.rand(n, device=device) < beta
                    wrapped.forced = torch.where(dones, redraw, wrapped.forced)
                    for key, value in extras.get("log", {}).items():
                        env_logs.setdefault(key, []).append(float(value))

            # ----- 학습 -----
            net.train()
            net.norm.update(buf_x)
            counts = torch.bincount(buf_y.flatten(), minlength=num_classes).float()
            counts = counts / counts.sum()
            share = counts if share is None else SHARE_DECAY * share + (1.0 - SHARE_DECAY) * counts
            weight = class_weights(share, args.class_weight_power)
            total_loss, total_correct, batches = 0.0, 0.0, 0
            for _ in range(args.epochs):
                for ids in torch.randperm(n, device=device).chunk(args.mini_batches):
                    hidden = h0[ids].clone()
                    loss = 0.0
                    correct = torch.zeros((), device=device)
                    for t in range(T):
                        logits, hidden = net.step(buf_x[t, ids], hidden)
                        loss = loss + F.cross_entropy(logits, buf_y[t, ids], weight=weight)
                        correct += (logits.argmax(dim=-1) == buf_y[t, ids]).float().mean().detach()
                        hidden = hidden * (~buf_done[t, ids]).unsqueeze(1).float()
                    loss = loss / T
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()
                    total_correct += (correct / T).item()
                    batches += 1

            # ----- 기록 -----
            stats = {
                "Phase/loss": total_loss / batches,
                "Phase/accuracy_train": total_correct / batches,
                "Phase/accuracy_net": (correct_net / count_net.clamp_min(1)).item() if count_net > 0 else float("nan"),
                "Phase/stuck_share": (stuck / count_net.clamp_min(1)).item(),
                "Phase/beta": beta,
            }
            for c in range(num_classes):
                stats[f"Phase/share_{c}"] = share[c].item()
                stats[f"Phase/weight_{c}"] = weight[c].item()
            for key, values in env_logs.items():
                if key.startswith("Metrics/"):
                    stats[key] = sum(values) / len(values)
            if writer is not None:
                for key, value in stats.items():
                    if not math.isnan(value):
                        writer.add_scalar(key, value, it)
            if it % 10 == 0:
                top = stats.get("Metrics/stair_reached_top", float("nan"))
                print(
                    f"[{it}] loss {stats['Phase/loss']:.3f} | 정답률 (학습) {stats['Phase/accuracy_train']:.1%} "
                    f"(신경망으로 움직인 env) {stats['Phase/accuracy_net']:.1%} | 0.5초+ 틀림 {stats['Phase/stuck_share']:.1%} "
                    f"| 참값 비율 {beta:.2f} | 꼭대기 {top:.2f} | {time.time() - started:.1f}s",
                    flush=True,
                )
            if it % 100 == 0:
                print("    단계별 시간 비중 / 가중치: " + ", ".join(
                    f"{c}: {share[c].item():.0%}/{weight[c].item():.2f}" for c in range(num_classes)), flush=True)
            last = it == start_iter + args.iterations - 1
            if it % args.save_interval == 0 or last:
                path = os.path.join(log_dir, f"model_{it}.pt")
                net.save(path, {"iteration": it, "optimizer": optimizer.state_dict(), "policy": args.policy, "class_share": share.cpu(),
                                "train_errors": errors})
        if writer is not None:
            writer.close()
        env.close()


if __name__ == "__main__":
    main()
