"""A-Phase student를 단계 추정 신경망의 출력으로 이어 증류한다. 신경망은 고정하고 학습하지 않는다.

A-Phase(model_3550 등)는 참값 단계로 증류돼서, 단계 입력이 틀리면 그대로 따라 한다. 신경망이 앞서면 다음 단계 동작을 해서
참값 단계가 끝나지 않고 굳는다. 여기서는 student가 신경망 단계를 보며 움직이고, teacher(참값 단계를 앎)의 행동을 정답으로
배운다. 그래서 "단계 입력이 앞서 있어도 아직 끝나지 않은 동작을 마저 한다"를 배울 수 있다.

    uv run python scripts/finetune_phase_policy.py --policy <A_phase model_3550.pt> --phase_net <phase_net.pt>

- 단계 칸: env가 관측을 만든 뒤 신경망 출력(one-hot)으로 바꾼다 (agents/phase_net.PhaseNetObsHook, HexapodEnv.phase_net_hook).
  rsl_rl 러너는 그 관측을 저장하고 학습하므로 student는 처음부터 끝까지 신경망 단계만 본다.
- 센서 오차: 기본으로 신경망 체크포인트에 저장된 학습 오차(train_errors, bias_free_ratio 포함)를 그대로 쓴다.
  신경망이 학습 때와 같은 분포의 입력을 보게 하기 위해서다. 같은 오차가 student 관측에도 들어간다.
- 규칙 추정기: 신경망 입력(phase_rule, phase_timer)에만 쓴다. 학습 때와 같게 높이 3cm, 건너뛰기 끔, 지연 없음.
- 증류: --policy 체크포인트에서 student, teacher, iteration을 불러와 --iterations만큼 더 돈다 (optimizer는 새로).
  보통 학습처럼 에피소드 시작 시각을 env마다 흩어 놓는다 (init_at_random_ep_len).
  결과는 logs/rsl_rl/<experiment>/<시각>_<run_name>/model_<iter>.pt (보통 학습과 같은 위치).

정책이 바뀌면 신경망이 보는 움직임도 바뀌므로, 끝나면 신경망을 새 정책으로 이어 학습한다:
    uv run python scripts/train_phase_net.py --policy <새 model.pt> --resume <phase_net.pt> --dagger_iters 0 --iterations 500
평가: eval_stairs.py --task Hexapod-Stair-Student-A-Phase --checkpoint <새 model.pt> --phase_net <phase_net.pt> --error_sweep
"""

import argparse
import dataclasses
import importlib.metadata
import json
import os
import sys
import time

import torch


def nonzero_fields(cfg) -> dict:
    """설정에서 0/False가 아닌 값만 (로그용)."""
    return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg) if getattr(cfg, f.name) not in (0, 0.0, False)}


def main():
    import gymnasium as gym

    from isaaclab.app import add_launcher_args, launch_simulation

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

    import RLforHexapod.tasks  # noqa: F401
    from RLforHexapod.tasks.walk.config.hexapod.agents.phase_net import PhaseNet, PhaseNetObsHook, term_slice
    from RLforHexapod.tasks.walk.mdp import PhaseEstimatorCfg, SensorErrorCfg

    parser = argparse.ArgumentParser(description="A-Phase student를 단계 추정 신경망 출력으로 이어 증류")
    parser.add_argument("--task", default="Hexapod-Stair-Student-A-Phase", help="단계 입력을 받는 student task")
    parser.add_argument("--policy", required=True, help="이어 학습할 A-Phase student 체크포인트 (.pt, teacher 포함)")
    parser.add_argument("--phase_net", required=True, help="단계 추정 신경망 (.pt, scripts/train_phase_net.py)")
    parser.add_argument("--iterations", type=int, default=500, help="더 돌 iteration 수")
    parser.add_argument("--num_envs", type=int, default=2048,
                        help="A-Phase 원래 학습과 같은 2048. task 설정값은 4라서 그대로 쓰면 정책이 무너진다")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", default="A_phase_netft")
    parser.add_argument("--error_scale", type=float, default=1.0, help="신경망 학습 오차에 곱하는 배율. 0 = 오차 없음")
    parser.add_argument("--bias_free_ratio", type=float, default=None,
                        help="편향을 0으로 하는 에피소드 비율. 기본: 신경망 학습 때 값")
    parser.add_argument("--resume_optimizer", action="store_true",
                        help="체크포인트의 optimizer 상태도 불러온다. 기본은 새로 시작 (보통 학습의 이어 학습과 같음)")
    parser.add_argument("--no_net", action="store_true",
                        help="대조 실험: 단계 칸을 바꾸지 않고 규칙 추정기 단계를 그대로 쓴다 (오차 0이면 보통 이어 학습과 같음)")
    add_launcher_args(parser)
    args, remaining = setup_preset_cli(parser, sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining

    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
    if not hasattr(env_cfg, "phase_estimator"):
        parser.error(f"{args.task}는 단계 입력을 받지 않습니다. A-Phase 계열 task를 쓰세요.")

    saved_net = torch.load(args.phase_net, map_location="cpu", weights_only=False)
    net_errors = dict(saved_net.get("train_errors") or {})
    if not net_errors:
        parser.error(f"{args.phase_net}에 학습 오차(train_errors)가 없습니다. train_phase_net.py로 만든 체크포인트를 쓰세요.")
    saved_ratio = net_errors.pop("bias_free_ratio", 0.0)
    bias_free_ratio = saved_ratio if args.bias_free_ratio is None else args.bias_free_ratio
    errors = {k: v * args.error_scale for k, v in net_errors.items()}
    errors["bias_free_ratio"] = bias_free_ratio
    env_cfg.phase_estimator = PhaseEstimatorCfg()
    env_cfg.sensor_error = SensorErrorCfg(**errors, randomize=True)

    with launch_simulation(env_cfg, args):
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = agent_cfg.seed = args.seed
        if getattr(args, "device", None) is not None:
            env_cfg.sim.device = agent_cfg.device = args.device
        agent_cfg.run_name = args.run_name
        torch.manual_seed(args.seed)

        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        log_dir = os.path.join(log_root, time.strftime("%Y-%m-%d_%H-%M-%S") + f"_{args.run_name}")
        env_cfg.log_dir = log_dir

        env = gym.make(args.task, cfg=env_cfg)
        base_env = env.unwrapped
        device = base_env.device
        net, _ = PhaseNet.load(args.phase_net, device)
        for p in net.parameters():
            p.requires_grad_(False)
        cut = term_slice(base_env.observation_manager)
        if net.config["num_classes"] != cut.stop - cut.start:
            raise ValueError(f"단계 크기가 다릅니다: 신경망 {net.config['num_classes']}, 관측 {cut.stop - cut.start}")
        # 래퍼가 처음 리셋할 때부터 단계 칸이 신경망 출력이 되도록 감싸기 전에 넣는다
        if not args.no_net:
            base_env.phase_net_hook = PhaseNetObsHook(net, cut, base_env.num_envs, device)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # 센서 오차와 판정 설정을 env에 직접 넣고 다시 리셋한다 (train_phase_net.py, eval_stairs.py와 같은 방식)
        tracker = base_env.stair_tracker
        sensors = tracker.sensors
        tracker.estimator.configure(PhaseEstimatorCfg())
        sensors.configure(SensorErrorCfg(**errors, randomize=True))
        env.reset()  # 여기서 새 설정으로 오차를 뽑는다
        print(f"[INFO] 학습 센서 오차 (에피소드마다 0 ~ 이 값): {nonzero_fields(sensors.cfg)}")
        print(f"[INFO] 편향 없는 에피소드 {sensors.bias_free.float().mean().item():.0%} (bias_free_ratio {bias_free_ratio})")

        obs = env.get_observations()
        input_dim = obs["student"].shape[1] - (cut.stop - cut.start) + sum(obs[g].shape[1] for g in net.extra_groups)
        if input_dim != net.config["input_dim"]:
            raise ValueError(f"신경망 입력 크기가 다릅니다: 신경망 {net.config['input_dim']}, env {input_dim}")
        shown = obs["student"][:, cut]
        if not args.no_net and not torch.equal(shown, net.to_phase_input(base_env.phase_net_hook.last_logits)):
            raise RuntimeError("student 관측의 단계 칸이 신경망 출력으로 바뀌지 않았습니다.")
        truth = tracker.phase_one_hot(tracker.move)
        source = "규칙 추정기 (--no_net 대조 실험)" if args.no_net else f"단계 추정 신경망 {args.phase_net} (입력 그룹 {net.extra_groups})"
        print(f"[INFO] 단계 입력: {source}. "
              f"리셋 직후 참값 일치 {(shown.argmax(-1) == truth.argmax(-1)).float().mean().item():.1%}")

        runner = create_rsl_rl_runner(env, agent_cfg, log_dir=log_dir)
        runner.add_git_repo_to_log(__file__)
        # 보통 학습(isaaclab_rl train)과 같게: optimizer는 기본으로 새로 시작한다
        load_cfg = None if args.resume_optimizer else {"student": True, "teacher": True, "optimizer": False, "iteration": True}
        print(f"[INFO] 이어 학습할 정책: {args.policy}")
        runner.load(args.policy, load_cfg=load_cfg, map_location=agent_cfg.device)
        start = runner.current_learning_iteration
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "finetune_args.json"), "w") as f:
            json.dump({**vars(args), "train_errors": errors, "phase_net_config": net.config, "start_iteration": start},
                      f, indent=2, ensure_ascii=False)
        try:
            from isaaclab_rl.entrypoints.common import dump_train_configs

            dump_train_configs(log_dir, env_cfg, agent_cfg)
        except Exception as error:  # 설정 파일 저장은 없어도 학습에는 지장이 없다
            print(f"[WARN] 설정 파일을 저장하지 못했습니다: {error}")
        print(f"[INFO] 저장 위치: {log_dir} (env {base_env.num_envs}대, iteration {start}부터 {args.iterations}회)")
        # 에피소드 시작 시각을 env마다 흩어 놓는다 (보통 학습과 같음). 안 하면 모든 env가 같은 순간을 같이 지나서
        # 한 번의 업데이트가 에피소드의 한 구간(예: 출발 직후)만 배우고 나머지를 잊는다
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=agent_cfg.init_at_random_ep_len)
        env.close()


if __name__ == "__main__":
    main()
