"""로봇에서 통합 정책을 한 step씩 돌린다 (numpy + onnxruntime). 센서 읽기와 서보 출력은 밖에서 한다.

    policy = RobotPolicy("deploy/mixed_phase_cmd_v1")
    policy.reset()                                  # 서기 자세를 보낸 뒤, 동작을 시작할 때
    out = policy.step(readings, rule_features, rule_time)
    send(out.target)                                # 시뮬레이션 관절각 [rad], robot_io.json의 joint_names 순서

관절각을 재지 못하는 로봇용이다 (학습 task Hexapod-Student-A-Phase-Cmd, SensorErrorCfg.commanded_joints):
    joint_pos 관측 = 마지막으로 보낸 목표각 - 기본 자세
    joint_vel 관측 = (마지막 목표각 - 그 전 목표각) / 0.02초
    발끝 높이(feet_below_body)도 마지막 목표각으로 계산한다
리셋 직후는 목표각 = 기본 자세, 속도 0 (시뮬레이션 리셋과 같음).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from hexapod_kinematics import HexapodKinematics
from phase_rule import Measurement, PhaseRule


@dataclass
class Readings:
    """한 step의 센서 값. 좌표는 몸통 기준 (x 앞 = 라이다, y 왼쪽, z 위)."""

    gyro: np.ndarray        # (3,) 몸통 각속도 [rad/s]
    gravity: np.ndarray     # (3,) 몸통 좌표의 중력 단위 벡터 (평평하면 (0, 0, -1))
    contacts: np.ndarray    # (6,) 발끝 스위치 0/1, feet_order 순서 (FOOT_ORDER)
    height_map: np.ndarray  # (205,) 몸통 z - 지형 z [m], robot_io.json height_map_grid 순서
    command: np.ndarray     # (4,) vx, vy, wz, 몸통 높이. 계단은 (0.2, 0, 0, 0.24)
    body_x: np.ndarray      # (4,) 단계 신경망 입력 (첫 수직면 기준 몸통 x + 수직면마다 지난 거리)


@dataclass
class Step:
    actions: np.ndarray     # (18,) 정책 출력 (자르기 전)
    target: np.ndarray      # (18,) 서보에 보낼 관절 목표각 [rad] (관절 범위 안으로 자름)
    phase: int              # 단계 번호 (robot_io.json phase_names)
    logits: np.ndarray      # (7,)
    student_obs: np.ndarray  # (282,) 이번 step에 student에 넣은 관측 (기록·디버깅용)


class RobotPolicy:
    def __init__(self, folder: str | Path):
        folder = Path(folder)
        self.io = io = json.loads((folder / "robot_io.json").read_text(encoding="utf-8"))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.student = ort.InferenceSession(str(folder / io["student"]["file"]), opts, providers=["CPUExecutionProvider"])
        self.net = ort.InferenceSession(str(folder / io["phase_net"]["file"]), opts, providers=["CPUExecutionProvider"])
        a = io["action"]
        self.joint_names = a["joint_names"]
        self.default = np.array(a["default_joint_pos_rad"], np.float32)
        self.scale = np.array(a["scale"], np.float32)
        self.clip = float(a["clip"])
        self.limits = np.array(a["joint_limits_rad"], np.float32)
        self.dt = float(io["policy_dt"])
        self.kin = HexapodKinematics(joint_names=self.joint_names)
        self.terms = {t["name"]: t for t in io["student"]["obs_terms"]}
        self.obs_dim = io["student"]["inputs"]["obs"][1]
        self.phase_slice = slice(*io["phase_net"]["phase_slice_in_student_obs"])
        groups = io["phase_net"]["x_groups"]
        self.net_groups = list(groups)  # phase_net, phase_rule, phase_timer 순서
        self.h_student_shape = io["student"]["inputs"]["h_in"]
        self.h_net_shape = io["phase_net"]["inputs"]["h_in"]
        self.rule = PhaseRule(io)
        self.x0 = io["stairs_trained"]["x0"]
        self.front_coxa_ids = [self.joint_names.index(n) for n in io["phase_rule"]["front_coxa_joints"]]
        self.reset()

    def reset(self, start_move: int = 0):
        """동작 시작: 두 신경망 기억을 비우고, 목표각을 기본 자세로, 규칙 단계를 start_move로 둔다."""
        self.rule.reset(start_move)
        self._fresh = True
        self.h_student = np.zeros(self.h_student_shape, np.float32)
        self.h_net = np.zeros(self.h_net_shape, np.float32)
        self.last_target = self.default.copy()
        self.prev_target = self.default.copy()
        self.last_actions = np.zeros(len(self.joint_names), np.float32)

    def student_obs(self, r: Readings) -> np.ndarray:
        """student 관측 (282,). 단계 칸은 0으로 비워 둔다."""
        obs = np.zeros(self.obs_dim, np.float32)
        values = {
            "joint_pos": self.last_target - self.default,
            "joint_vel": (self.last_target - self.prev_target) / self.dt,
            "base_ang_vel": r.gyro,
            "projected_gravity": r.gravity,
            "last_action": self.last_actions,
            "feet_contact": r.contacts,
            "height_map": r.height_map,
            "command": r.command,
        }
        for name, term in self.terms.items():
            if name == "phase":
                continue
            v = np.asarray(values[name], np.float32).ravel()
            if v.shape[0] != term["size"]:
                raise ValueError(f"{name}: 크기 {term['size']}이어야 하는데 {v.shape[0]}")
            obs[term["start"]:term["start"] + term["size"]] = v
        return obs

    def net_input(self, obs: np.ndarray, r: Readings, rule_features: np.ndarray, rule_time: float) -> np.ndarray:
        extra = {
            "phase_net": np.concatenate([np.asarray(r.body_x, np.float32),
                                         self.kin.feet_below_body(self.last_target, r.gravity).astype(np.float32)]),
            "phase_rule": np.asarray(rule_features, np.float32),
            "phase_timer": np.array([rule_time], np.float32),
        }
        parts = [np.delete(obs, np.arange(self.phase_slice.start, self.phase_slice.stop))]
        parts += [extra[g] for g in self.net_groups]
        return np.concatenate(parts)[None]

    def rule_measurement(self, base_pos: np.ndarray, base_rot: np.ndarray, contacts, speed: float) -> Measurement:
        """규칙 추정기 측정값. base_pos·base_rot = 계단 좌표의 몸통 (L2Fusion.stairs_pose). 발끝·앞다리는 마지막 목표각."""
        tips = np.asarray(base_pos, float) + self.kin.tips(self.last_target) @ np.asarray(base_rot, float).T
        return Measurement(float(base_pos[0]), tips, np.asarray(contacts, bool),
                           self.last_target[self.front_coxa_ids].astype(float), float(speed))

    def step_with_rule(self, r: Readings, base_pos, base_rot, speed: float) -> Step:
        """규칙 추정기까지 돌리는 한 step. 시뮬레이션 순서와 같다: 측정 → 단계 진행(리셋 직후 제외) → 규칙 특징 → 신경망."""
        m = self.rule_measurement(base_pos, base_rot, r.contacts, speed)
        if not self._fresh:
            self.rule.update(m)
        self._fresh = False
        return self.step(r, self.rule.features(m), self.rule.time_feature())

    def step(self, r: Readings, rule_features: np.ndarray, rule_time: float) -> Step:
        obs = self.student_obs(r)
        x = self.net_input(obs, r, rule_features, rule_time)
        logits, self.h_net = self.net.run(None, {"x": x, "h_in": self.h_net})
        phase = int(np.argmax(logits[0]))
        obs[self.phase_slice] = 0.0
        obs[self.phase_slice.start + phase] = 1.0
        actions, self.h_student = self.student.run(None, {"obs": obs[None], "h_in": self.h_student})
        actions = actions[0]
        clipped = np.clip(actions, -self.clip, self.clip).astype(np.float32)
        # 관측에는 관절 범위로 자르기 전 목표각을 쓴다 (시뮬레이션 행동 항의 processed_actions와 같음)
        command = (self.default + self.scale * clipped).astype(np.float32)
        self.prev_target, self.last_target = self.last_target, command
        self.last_actions = clipped  # 시뮬레이션 last_action = 래퍼가 ±clip으로 자른 행동
        target = np.clip(command, self.limits[:, 0], self.limits[:, 1]).astype(np.float32)
        return Step(actions, target, phase, logits[0], obs)
