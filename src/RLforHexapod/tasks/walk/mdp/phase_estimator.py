"""실물 센서로 계단 단계를 계산할 때의 오차를 흉내 내는 추정기.

실물에서는 Point-LIO의 위치, 지도에서 찾은 수직면, 관절각으로 계산한 발끝, 발 접촉으로 단계를 계산한다.
여기서는 참값에 오차를 더한 측정값을 StairTracker와 같은 조건(check_phase)에 넣는다.
오차가 전부 0이고 by_contact=False, catch_up=False이면 StairTracker와 항상 같은 단계가 나온다.

편향이 있으면 추정이 한 단계에서 영영 멈출 수 있다 (참값은 기준을 겨우 넘기고 다음 단계로 가 버린다).
그래서 실물에서도 쓸 두 가지 복구를 둔다.
- by_contact: 높이 3cm 조건에 "발끝 스위치가 눌리고 높이 contact_window 안"을 OR로 더한다 (높이 편향 대책)
- catch_up: 전진 단계에서 다음 옮기기 단계가 이미 끝났으면 두 칸 진행한다 (x 편향, 앞다리 접힘 대책)

teacher의 정답과 성공 판정은 계속 참값(StairTracker)을 쓴다. student만 이 추정값을 본다.
"""

import torch
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply

from .observations import stairs_terrain_height


@configclass
class PhaseErrorCfg:
    """randomize=False(평가): 편향은 정확히 이 크기이고 부호만 무작위다. 값이 곧 "이만큼 틀렸을 때"를 뜻한다.
    randomize=True(학습): 이 값은 최댓값이다. 에피소드마다 env별로 0부터 이 값 사이에서 고르게 뽑는다."""

    # x: 라이다(Point-LIO + 지도에서 찾은 수직면)로 잰 수직면 기준 몸통·발끝 앞뒤 위치
    pos_bias: float = 0.0      # x 편향 [m]. 에피소드마다 고정
    pos_noise: float = 0.0     # x 노이즈 std [m]. step마다
    # z: 발끝 높이는 라이다 z가 아니라 IMU + 관절각 + 지지 발의 디딤면 기준으로 계산한다 (몸통 z 오차가 상쇄된다)
    # 남는 오차: 실제 단 높이와 가정(16cm)의 차이, IMU 기울기, 기준 발이 모서리에 걸침. 관절각 오차는 joint_bias가 따로 맡는다
    height_bias: float = 0.0   # 발끝 높이 편향 [m]. 에피소드마다 고정, 모든 발 공통
    height_noise: float = 0.0  # 발끝 높이 노이즈 std [m]. step마다, 모든 발 공통
    joint_bias: float = 0.0    # 관절각 편향 [rad]. 관절마다 에피소드 동안 고정. 발끝 계산과 앞다리 접힘에 반영
    contact_flip: float = 0.0  # 발 접촉 판정이 뒤집힐 확률. 발마다 step마다
    delay: float = 0.0         # 단계 계산 지연 [s]
    randomize: bool = False
    # 딛고 있음 = 높이 3cm 안 OR (by_contact일 때) 발끝 스위치가 눌리고 높이 contact_window 안
    # 높이 제한 없이 스위치만 OR하면 오차 0에서 teacher보다 16% 앞섰다. 공중의 발은 스위치가 안 눌려 걸러진다
    by_contact: bool = False
    contact_window: float = 0.08   # 스위치로 인정할 발끝 높이 차 [m]. 단 높이(16cm)의 절반
    # 스위치로만 통과하는 상태가 이 시간 이어져야 인정한다 [s]. 0 = 바로. teacher가 잠깐 지나가는 상태를 거른다
    contact_confirm: float = 0.0
    # 전진 단계가 안 끝났는데 다음 옮기기(마지막 전진이면 정지) 단계가 이미 끝났으면 두 칸 진행한다
    catch_up: bool = False
    # 같은 센서 값이 student 관측에도 들어가게 한다 (sensor_error.py): x 오차 → 높이맵 위치, 관절각 편향 → joint_pos,
    # 접촉 오판 → feet_contact. 높이 편향·지연은 단계 계산에만 쓰는 값이라 공유하지 않는다
    share_obs: bool = False
    # 높이맵을 지도에서 잘라 쓰면 x 오차가 높이맵에도 들어간다. 매 프레임 스캔으로 몸통 기준으로 만들면 거의 안 들어간다
    map_shares_x: bool = True


class PhaseEstimator:
    """오차가 들어간 측정값으로 env별 단계 번호를 따로 진행시킨다."""

    def __init__(self, tracker, cfg: PhaseErrorCfg):
        self.tracker = tracker
        self.env = tracker.env
        self.asset = tracker.asset
        n, device = self.env.num_envs, self.env.device
        self.n, self.device = n, device

        # 발마다 (coxa, femur, tibia) 바디. URDF의 관절은 모두 자식 링크 원점에 있고 축은 자식 링크의 +z다
        foot_names = [self.asset.body_names[i] for i in tracker.foot_ids]
        chain_names = [name.replace("tibia", part) for name in foot_names for part in ("coxa", "femur", "tibia")]
        self.chain_ids, _ = self.asset.find_bodies(chain_names, preserve_order=True)
        self.joint_axis = torch.zeros(n, len(chain_names), 3, device=device)
        self.joint_axis[..., 2] = 1.0

        self.move = tracker.move.clone()
        self.best_body_x = torch.zeros(n, device=device)
        self.need_init = torch.ones(n, dtype=torch.bool, device=device)
        self.steady_time = torch.zeros(n, device=device)
        self.pressed_time = torch.zeros(n, len(foot_names), device=device)  # 스위치로만 딛고 있다고 본 시간 [s]
        self.ahead_steady_time = torch.zeros(n, device=device)  # catch_up: 다음 단계(정지)를 내다본 멈춘 시간 [s]
        self.catch_ups = torch.zeros((), dtype=torch.long, device=device)  # catch_up으로 건너뛴 횟수 (평가 로그용)

        # env별 오차. 리셋 때 뽑는다
        self.pos_bias = torch.zeros(n, 2, device=device)                     # (x, z) [m]
        self.joint_bias = torch.zeros(n, len(foot_names), 3, device=device)  # (coxa, femur, tibia) [rad]
        self.pos_noise = torch.zeros(n, 2, device=device)                    # (x, z) std [m]
        self.contact_flip = torch.zeros(n, device=device)                    # 확률
        self.delay_steps = torch.zeros(n, dtype=torch.long, device=device)
        # 이번 step에 뽑은 값. student 관측이 같은 값을 읽는다 (share_obs)
        self.pos_error = torch.zeros(n, 2, device=device)                    # (x, z) 편향 + 노이즈 [m]
        self.flips = torch.zeros(n, len(foot_names), dtype=torch.bool, device=device)
        # 관절각 편향을 asset 관절 순서로 옮길 번호. 관절 이름은 L{다리}_{coxa|femur|tibia}, 바디는 coxa / coxa_{다리}
        joint_names = []
        for name in chain_names:
            part, _, leg = name.partition("_")
            joint_names.append(f"L{leg or '1'}_{part}")
        self.chain_joint_ids, _ = self.asset.find_joints(joint_names, preserve_order=True)
        self.configure(cfg)

    def configure(self, cfg: PhaseErrorCfg):
        """오차 설정을 바꾼다. 다음 리셋부터 새 오차가 적용된다 (평가 스크립트의 오차별 반복에서 쓴다)."""
        self.cfg = cfg
        self.max_delay_steps = round(cfg.delay / self.env.step_dt)
        # 0번이 가장 최근. 지연이 0이면 칸이 하나뿐이다
        self.history = self.move.unsqueeze(1).repeat(1, self.max_delay_steps + 1)
        self.delay_steps = self.delay_steps.clamp(max=self.max_delay_steps)
        self.catch_ups = torch.zeros_like(self.catch_ups)

    def joint_bias_by_joint(self) -> torch.Tensor:
        """관절각 편향을 asset 관절 순서로 (N, 관절 수). joint_pos 관측에 더한다."""
        out = torch.zeros(self.n, self.asset.num_joints, device=self.device)
        out[:, self.chain_joint_ids] = self.joint_bias.reshape(self.n, -1)
        return out

    @property
    def delayed_move(self) -> torch.Tensor:
        """지연을 반영해 student에게 보여 줄 단계 번호 (N,)."""
        return self.history.gather(1, self.delay_steps.unsqueeze(1)).squeeze(1)

    def _signed(self, shape, size: float) -> torch.Tensor:
        """편향: 평가는 크기 고정 ±size, 학습은 -size ~ +size 균등."""
        if self.cfg.randomize:
            return (torch.rand(shape, device=self.device) * 2.0 - 1.0) * size
        return (torch.randint(2, shape, device=self.device).float() * 2.0 - 1.0) * size

    def _scale(self, count: int, size: float) -> torch.Tensor:
        """크기: 평가는 size 그대로, 학습은 0 ~ size 균등."""
        if self.cfg.randomize:
            return torch.rand(count, device=self.device) * size
        return torch.full((count,), size, device=self.device)

    def reset(self, env_ids, moves):
        cfg, count = self.cfg, len(env_ids)
        self.move[env_ids] = moves
        self.best_body_x[env_ids] = 0.0
        self.need_init[env_ids] = True
        self.steady_time[env_ids] = 0.0
        self.pressed_time[env_ids] = 0.0
        self.ahead_steady_time[env_ids] = 0.0
        self.history[env_ids] = moves.unsqueeze(1).repeat(1, self.history.shape[1])

        self.pos_bias[env_ids, 0] = self._signed((count,), cfg.pos_bias)
        self.pos_bias[env_ids, 1] = self._signed((count,), cfg.height_bias)
        self.joint_bias[env_ids] = self._signed((count, *self.joint_bias.shape[1:]), cfg.joint_bias)
        self.pos_noise[env_ids, 0] = self._scale(count, cfg.pos_noise)
        self.pos_noise[env_ids, 1] = self._scale(count, cfg.height_noise)
        self.contact_flip[env_ids] = self._scale(count, cfg.contact_flip)
        if cfg.randomize:
            self.delay_steps[env_ids] = torch.randint(self.max_delay_steps + 1, (count,), device=self.device)
        else:
            self.delay_steps[env_ids] = self.max_delay_steps
        # 다음 update 전에 계산되는 첫 관측이 새 편향을 쓰게 한다
        self.pos_error[env_ids] = self.pos_bias[env_ids]
        self.flips[env_ids] = False

    def _tip_error_from_joints(self, tips: torch.Tensor) -> torch.Tensor:
        """관절각 편향만큼 발끝 계산이 틀어지는 양 (N, 6, 3). 각이 작으므로 축 × 팔 길이로 근사한다."""
        data = self.asset.data
        joint_pos = data.body_pos_w.torch[:, self.chain_ids] - self.env.scene.env_origins.unsqueeze(1)
        axis = quat_apply(data.body_quat_w.torch[:, self.chain_ids], self.joint_axis)
        num_feet = tips.shape[1]
        joint_pos = joint_pos.reshape(self.n, num_feet, 3, 3)
        axis = axis.reshape(self.n, num_feet, 3, 3)
        lever = tips.unsqueeze(2) - joint_pos
        return (self.joint_bias.unsqueeze(-1) * torch.cross(axis, lever, dim=-1)).sum(dim=2)

    def update(self, truth):
        """참값 측정에 오차를 더해 단계를 진행시킨다. truth: stair_reward.PhaseMeasurement."""
        cfg = self.cfg
        pos_error = self.pos_bias + torch.randn(self.n, 2, device=self.device) * self.pos_noise
        self.pos_error = pos_error
        error_x, error_z = pos_error.unbind(dim=1)

        tips = truth.tips
        if cfg.joint_bias > 0.0:
            tips = tips + self._tip_error_from_joints(tips)
        tips = torch.stack([tips[..., 0] + error_x.unsqueeze(1), tips[..., 1], tips[..., 2] + error_z.unsqueeze(1)], dim=-1)

        # 접촉은 참값 tracker와 같은 값에 같은 오판을 넣는다. 오판은 발끝 스위치에도 똑같이 들어간다
        flips = torch.rand(truth.in_contact.shape, device=self.device) < self.contact_flip.unsqueeze(1)
        self.flips = flips
        contact = truth.in_contact ^ flips
        contact_ground = None
        if cfg.by_contact:
            # 스위치가 눌리고 발끝 높이가 발 밑 지형(발끝 x로 고른 단)에서 contact_window 안
            gait = self.tracker.gait
            terrain = stairs_terrain_height(self.tracker.stairs, tips[..., 0], tips[..., 1])
            height_gap = (tips[..., 2] - terrain - gait.tip_height).abs()
            pressed = (truth.sole_contact ^ flips) & (height_gap < cfg.contact_window)
            # 높이 3cm 조건은 실패하고 스위치로만 통과하는 상태가 이어진 시간. 높이로 통과하면 다시 잰다
            # (눌린 시간으로 재면 오래 딛고 있던 발이 잠깐 높게 읽힐 때 바로 통과해서 거르지 못한다)
            switch_only = pressed & (height_gap >= gait.contact_tol)
            self.pressed_time = torch.where(
                switch_only, self.pressed_time + self.env.step_dt, torch.zeros_like(self.pressed_time)
            )
            contact_ground = switch_only & (self.pressed_time >= cfg.contact_confirm + self.env.step_dt - 1e-6)
        measured = truth._replace(
            body_x=truth.body_x + error_x,
            tips=tips,
            in_contact=contact,
            front_coxa=truth.front_coxa + self.joint_bias[:, self.tracker.front_feet, 0],
            contact_ground=contact_ground,
        )

        tracker = self.tracker
        check = tracker.check_phase(
            self.move, self.best_body_x, self.need_init, self.steady_time, measured, by_contact=cfg.by_contact
        )
        steps = check.done.long()
        if cfg.catch_up:
            # 다음 단계를 방금 시작한 것처럼 판정한다. 정지 단계는 1초를 채워야 끝나므로 내다본 쪽 멈춘 시간을 따로 잰다
            phase, _ = tracker.phase_and_level(self.move)
            advancing = (tracker.table_kind[phase] == 0) & (self.move <= tracker.final_advance)  # 0 = ADVANCE
            ahead = tracker.check_phase(
                self.move + 1, self.best_body_x, torch.ones_like(self.need_init), self.ahead_steady_time, measured,
                by_contact=cfg.by_contact,
            )
            jump = advancing & ~check.done & ahead.done
            self.ahead_steady_time = torch.where(advancing, ahead.steady_time, torch.zeros_like(self.ahead_steady_time))
            steps = torch.where(jump, 2, steps)
            self.catch_ups = self.catch_ups + jump.sum()
        self.best_body_x = check.best_body_x
        self.steady_time = check.steady_time
        self.move = self.move + steps
        self.need_init = steps > 0
        self.ahead_steady_time = torch.where(steps > 0, torch.zeros_like(self.ahead_steady_time), self.ahead_steady_time)
        self.history = torch.cat([self.move.unsqueeze(1), self.history[:, :-1]], dim=1)
