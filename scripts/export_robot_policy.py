"""통합 A-Phase student와 단계 추정 신경망을 실물 로봇용 파일로 꺼낸다 (Isaac Lab이 있는 PC에서 실행).

    uv run python scripts/export_robot_policy.py \
        --checkpoint logs/rsl_rl/hexapod_student/<run>/model_1499.pt \
        --phase_net logs/phase_net/<run>/model_1499.pt --out deploy/mixed_phase_v1

만드는 파일 (--out 폴더):
    student.onnx     입력 obs (1, 282), h_in (1, 1, 256) → 출력 actions (1, 18), h_out. 관측 정규화 포함
    phase_net.onnx   입력 x (1, 326), h_in (1, 128) → 출력 logits (1, 7), h_out. 관측 정규화 포함
    robot_io.json    관측 항의 순서·크기·뜻, 행동 → 관절 목표각 식, 관절 이름·기본 자세·범위, 단계 이름
    reference.npz    시뮬레이션에서 신경망 단계로 움직인 기록 (센서 오차 없음). 로봇 쪽 구현을 맞춰 볼 정답
                     student_obs, net_x, logits, actions, true_phase, done, is_stair (모두 step 순서)

ONNX 파일이 PyTorch와 같은 값을 내는지는 로컬에서 scripts/verify_robot_policy.py로 검사한다
(이 PC의 Isaac Lab 환경에는 onnxruntime이 없다).
"""

import argparse
import importlib.metadata
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

PHASE_NAMES = ["전진", "앞발", "중간발", "접기 전진", "뒷발1", "뒷발2", "정지"]  # stair_phase_obs 순서
OPSET = 17

# 관측 항의 뜻. 실물에서 이 값을 같은 단위·좌표계로 만들어 넣어야 한다
TERM_DOCS = {
    "joint_pos": "관절각 - 기본 자세 [rad], joint_names 순서",
    "joint_vel": "관절 각속도 [rad/s], joint_names 순서",
    "base_ang_vel": "몸통 각속도, 몸통 좌표 (x 앞 = 라이다, y 왼쪽, z 위) [rad/s]",
    "projected_gravity": "몸통 좌표로 본 중력 방향 단위 벡터 (평평하게 서면 (0, 0, -1))",
    "last_action": "직전 step에 정책이 낸 행동 (자르기 전 값, 시작은 0)",
    "feet_contact": "발끝 스위치 (1 = 눌림), feet_order 순서",
    "height_map": "몸통 원점 높이 - 지형 높이 [m]. 격자는 height_map_grid (몸통 위치·yaw를 따라감, 높이는 중력 방향)",
    "command": "조종 명령 (vx [m/s], vy [m/s], wz [rad/s], 몸통 높이 [m]). 계단을 오를 때는 (0.2, 0, 0, 0.24)로 학습",
    "phase": "단계 one-hot (7). 단계 추정 신경망 출력의 argmax를 one-hot으로 넣는다",
    "body_x": "몸통 x - 첫 수직면 x [m], 이어서 수직면마다 (몸통 x - 수직면 x)를 (-0.3, 0.6)으로 자른 값 (단 수)",
    "feet_below_body": "발끝 높이 - 몸통 원점 높이, 중력 방향 [m] (IMU 자세 + 관절각 FK), feet_order 순서",
    "rule": "규칙 단계 추정기의 판정과 판정 값 (mdp.stair_rule_features). 로봇에서 PhaseEstimator를 그대로 옮겨 계산해야 한다",
    "time": "규칙 추정기가 지금 단계에 머문 시간 / 2초, 10초에서 자름",
}


class PhaseNetOnnx(nn.Module):
    """PhaseNet.step을 ONNX로: (x, h) → (logits, h_out). 정규화 포함."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, h_in):
        logits, h = self.net.step(x, h_in)
        return logits, h


def group_layout(observation_manager, group: str) -> list[dict]:
    names = observation_manager.active_terms[group]
    dims = observation_manager.group_obs_term_dim[group]
    out, start = [], 0
    for name, dim in zip(names, dims, strict=True):
        size = int(np.prod(dim))
        out.append({"name": name, "start": start, "size": size, "doc": TERM_DOCS.get(name, "")})
        start += size
    return out


def main():
    import gymnasium as gym

    from isaaclab.app import add_launcher_args, launch_simulation

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

    import RLforHexapod.tasks  # noqa: F401
    from RLforHexapod.tasks.walk.config.hexapod.agents.phase_net import PhaseNet, PhaseNetPolicy, term_slice
    from RLforHexapod.tasks.walk.mdp import FOOT_BODIES, stair_env_mask

    parser = argparse.ArgumentParser(description="실물 로봇용 정책 파일 만들기")
    parser.add_argument("--task", default="Hexapod-Student-A-Phase")
    parser.add_argument("--checkpoint", required=True, help="통합 A-Phase student model_*.pt")
    parser.add_argument("--phase_net", required=True, help="단계 추정 신경망 model_*.pt")
    parser.add_argument("--out", required=True, help="저장할 폴더")
    parser.add_argument("--num_envs", type=int, default=8, help="기록할 env 수 (절반 계단, 절반 걷기)")
    parser.add_argument("--steps", type=int, default=500, help="기록할 step 수")
    parser.add_argument("--seed", type=int, default=7)
    add_launcher_args(parser)
    args, remaining = setup_preset_cli(parser, sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    os.makedirs(args.out, exist_ok=True)

    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
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
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(args.checkpoint)
        student = runner.alg.get_policy()
        policy = runner.get_inference_policy(device=device)
        net, net_saved = PhaseNet.load(args.phase_net, device)
        net.eval()
        om = base_env.observation_manager
        cut = term_slice(om)

        # ----- ONNX -----
        onnx_student = student.as_onnx().to("cpu").eval()
        student_path = os.path.join(args.out, "student.onnx")
        torch.onnx.export(
            onnx_student, onnx_student.get_dummy_inputs(), student_path,
            input_names=onnx_student.input_names, output_names=onnx_student.output_names,
            opset_version=OPSET, dynamo=False,
        )
        net_cpu = PhaseNet(**net.config)
        net_cpu.load_state_dict(net.state_dict())
        net_onnx = PhaseNetOnnx(net_cpu.eval())
        net_path = os.path.join(args.out, "phase_net.onnx")
        torch.onnx.export(
            net_onnx, (torch.zeros(1, net.config["input_dim"]), torch.zeros(1, net.config["hidden_dim"])), net_path,
            input_names=["x", "h_in"], output_names=["logits", "h_out"], opset_version=OPSET, dynamo=False,
        )
        print(f"[INFO] 저장: {student_path}, {net_path}")

        # ----- 행동 → 관절 목표각 -----
        action_term = base_env.action_manager.get_term("joint_pos")
        asset = base_env.scene["robot"]
        joint_ids = action_term._joint_ids
        joint_ids = list(range(asset.num_joints)) if isinstance(joint_ids, slice) else list(joint_ids)
        joint_names = [asset.joint_names[i] for i in joint_ids]
        scale = action_term._scale
        scale = (scale[0] if torch.is_tensor(scale) and scale.dim() == 2 else torch.as_tensor(scale)).flatten()
        scale = scale.expand(len(joint_ids)) if scale.numel() == 1 else scale
        as_torch = lambda x: x.torch if hasattr(x, "torch") else x  # noqa: E731
        default = as_torch(asset.data.default_joint_pos)[0, joint_ids]
        limits = as_torch(asset.data.joint_pos_limits)[0, joint_ids]

        # 높이맵 격자 (stairs_height_map과 같은 순서: x가 바깥, y가 안쪽)
        hm_params = env_cfg.observations.student.height_map.params
        (x0, x1), (y0, y1) = hm_params["x_range"], hm_params["y_range"]
        sx, sy = hm_params["spacing"]
        nx, ny = int((x1 - x0) / sx + 1e-6) + 1, int((y1 - y0) / sy + 1e-6) + 1

        stairs = env_cfg.stairs
        net_groups = ["student (phase 칸 제외)"] + list(net.extra_groups)
        io = {
            "task": args.task,
            "student_checkpoint": args.checkpoint,
            "phase_net_checkpoint": args.phase_net,
            "policy_dt": base_env.step_dt,
            "student": {
                "file": "student.onnx",
                "inputs": {"obs": [1, om.group_obs_dim["student"][0]], "h_in": [1, 1, onnx_student.hidden_size]},
                "outputs": {"actions": [1, len(joint_ids)], "h_out": [1, 1, onnx_student.hidden_size]},
                "obs_terms": group_layout(om, "student"),
                "hidden_reset": "에피소드(동작) 시작 때 h_in = 0, 그 뒤 매 step h_in = 직전 h_out",
            },
            "phase_net": {
                "file": "phase_net.onnx",
                "inputs": {"x": [1, net.config["input_dim"]], "h_in": [1, net.config["hidden_dim"]]},
                "outputs": {"logits": [1, net.config["num_classes"]], "h_out": [1, net.config["hidden_dim"]]},
                "x_layout": net_groups,
                "x_groups": {g: group_layout(om, g) for g in net.extra_groups},
                "phase_slice_in_student_obs": [cut.start, cut.stop],
                "to_student": "phase = one_hot(argmax(logits)); student obs의 phase 칸에 넣는다",
                "train_errors": net_saved.get("train_errors"),
            },
            "phase_names": PHASE_NAMES,
            "action": {
                "joint_names": joint_names,
                "default_joint_pos_rad": default.tolist(),
                "scale": scale.tolist(),
                "clip": agent_cfg.clip_actions,
                "target_formula": "target = default_joint_pos_rad + scale * clip(actions, -clip, clip), 이어서 joint_limits_rad로 자름",
                "joint_limits_rad": limits.tolist(),
                "last_action": "student obs의 last_action에는 자르기 전 actions를 넣는다",
            },
            "feet_order": FOOT_BODIES,
            "height_map_grid": {
                "x": [x0 + sx * i for i in range(nx)], "y": [y0 + sy * j for j in range(ny)],
                "order": "index = ix * len(y) + iy",
                "frame": "몸통 위치 기준, 몸통 yaw로만 돌림 (roll·pitch 무시), 높이 = 몸통 원점 z - 지형 z",
                "train_noise_std_m": 0.01,
            },
            "stairs_trained": {"x0": stairs.x0, "tread": stairs.tread, "step_height": stairs.step_height,
                               "num_steps": stairs.num_steps, "width": stairs.width},
        }
        with open(os.path.join(args.out, "robot_io.json"), "w", encoding="utf-8") as f:
            json.dump(io, f, indent=2, ensure_ascii=False)

        # ----- 기준 기록 (신경망 단계로 움직임, 센서 오차 없음) -----
        wrapped = PhaseNetPolicy(policy, net, cut, base_env.num_envs, device)
        tracker = base_env.stair_tracker
        obs = env.get_observations()
        rec = {k: [] for k in ("student_obs", "net_x", "logits", "actions", "true_phase", "done")}
        with torch.inference_mode():
            for _ in range(args.steps):
                rec["true_phase"].append(tracker.phase_one_hot(tracker.move).argmax(-1).cpu())
                actions = wrapped(obs)
                rec["student_obs"].append(obs["student"].cpu().clone())  # 단계 칸이 신경망 출력으로 바뀐 관측
                rec["net_x"].append(wrapped.last_input.cpu().clone())
                rec["logits"].append(wrapped.last_logits.cpu().clone())
                rec["actions"].append(actions.cpu().clone())
                obs, _, dones, _ = env.step(actions)
                dones = dones.bool()
                wrapped.reset(dones)
                rec["done"].append(dones.cpu())
        data = {k: torch.stack(v).numpy() for k, v in rec.items()}
        data["is_stair"] = stair_env_mask(base_env).cpu().numpy()
        np.savez_compressed(os.path.join(args.out, "reference.npz"), **data)
        steps_done = data["done"].sum()
        print(f"[INFO] 기준 기록: {args.steps} step × {args.num_envs} env, 끝난 에피소드 {steps_done}")
        print(f"[INFO] 저장: {args.out}")
        env.close()


if __name__ == "__main__":
    main()
