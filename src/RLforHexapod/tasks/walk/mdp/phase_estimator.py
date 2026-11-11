"""실물 센서로 계단 단계를 계산할 때의 오차를 흉내 내는 추정기.

실물에서는 Point-LIO의 위치, 지도에서 찾은 수직면, 관절각으로 계산한 발끝, 발 접촉으로 단계를 계산한다.
여기서는 참값에 오차를 더한 측정값을 StairTracker와 같은 조건(check_phase)에 넣는다.
오차가 전부 0이고 by_contact=False이면 StairTracker와 항상 같은 단계가 나온다.

teacher의 정답과 성공 판정은 계속 참값(StairTracker)을 쓴다. student만 이 추정값을 본다.
"""

import torch
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply


@configclass
class PhaseErrorCfg:
    """randomize=False(평가): 편향은 정확히 이 크기이고 부호만 무작위다. 값이 곧 "이만큼 틀렸을 때"를 뜻한다.
    randomize=True(학습): 이 값은 최댓값이다. 에피소드마다 env별로 0부터 이 값 사이에서 고르게 뽑는다."""

    pos_bias: float = 0.0      # 수직면에 대한 몸통 위치 편향 [m]. x와 z에 에피소드마다 고정
    pos_noise: float = 0.0     # 몸통 위치 노이즈 std [m]. x와 z에 step마다
    joint_bias: float = 0.0    # 관절각 편향 [rad]. 관절마다 에피소드 동안 고정. 발끝 계산과 앞다리 접힘에 반영
    contact_flip: float = 0.0  # 발 접촉 판정이 뒤집힐 확률. 발마다 step마다
    delay: float = 0.0         # 단계 계산 지연 [s]
    randomize: bool = False
    # 판정 방식. True면 딛고 있음을 접촉 센서로, 발 밑 단을 발끝 높이로 판정한다 (StairTracker.check_phase 참고)
    by_contact: bool = False
    # by_contact에서 접촉이 이 시간 안에 있었으면 딛고 있다고 본다 [s]. 0 = 지금 접촉만
    # 0.1초로 두면 막 들어 올린 발도 딛고 있다고 봐서 단계가 teacher보다 앞섰다 (오차 0에서 일치 66%, 0이면 100%)
    contact_hold: float = 0.0


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
        self.since_contact = torch.zeros(n, len(foot_names), device=device)  # 마지막 접촉 뒤 지난 시간 [s]

        # env별 오차. 리셋 때 뽑는다
        self.pos_bias = torch.zeros(n, 2, device=device)                     # (x, z) [m]
        self.joint_bias = torch.zeros(n, len(foot_names), 3, device=device)  # (coxa, femur, tibia) [rad]
        self.pos_noise = torch.zeros(n, device=device)                       # std [m]
        self.contact_flip = torch.zeros(n, device=device)                    # 확률
        self.delay_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.configure(cfg)

    def configure(self, cfg: PhaseErrorCfg):
        """오차 설정을 바꾼다. 다음 리셋부터 새 오차가 적용된다 (평가 스크립트의 오차별 반복에서 쓴다)."""
        self.cfg = cfg
        self.max_delay_steps = round(cfg.delay / self.env.step_dt)
        # 0번이 가장 최근. 지연이 0이면 칸이 하나뿐이다
        self.history = self.move.unsqueeze(1).repeat(1, self.max_delay_steps + 1)
        self.delay_steps = self.delay_steps.clamp(max=self.max_delay_steps)

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
        self.since_contact[env_ids] = 0.0
        self.history[env_ids] = moves.unsqueeze(1).repeat(1, self.history.shape[1])

        self.pos_bias[env_ids] = self._signed((count, 2), cfg.pos_bias)
        self.joint_bias[env_ids] = self._signed((count, *self.joint_bias.shape[1:]), cfg.joint_bias)
        self.pos_noise[env_ids] = self._scale(count, cfg.pos_noise)
        self.contact_flip[env_ids] = self._scale(count, cfg.contact_flip)
        if cfg.randomize:
            self.delay_steps[env_ids] = torch.randint(self.max_delay_steps + 1, (count,), device=self.device)
        else:
            self.delay_steps[env_ids] = self.max_delay_steps

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
        pos_error = self.pos_bias + torch.randn(self.n, 2, device=self.device) * self.pos_noise.unsqueeze(1)
        error_x, error_z = pos_error.unbind(dim=1)

        tips = truth.tips
        if cfg.joint_bias > 0.0:
            tips = tips + self._tip_error_from_joints(tips)
        tips = torch.stack([tips[..., 0] + error_x.unsqueeze(1), tips[..., 1], tips[..., 2] + error_z.unsqueeze(1)], dim=-1)

        flips = torch.rand(truth.in_contact.shape, device=self.device) < self.contact_flip.unsqueeze(1)
        contact = truth.in_contact ^ flips
        self.since_contact = torch.where(contact, torch.zeros_like(self.since_contact), self.since_contact + self.env.step_dt)
        measured = truth._replace(
            body_x=truth.body_x + error_x,
            tips=tips,
            in_contact=contact,
            front_coxa=truth.front_coxa + self.joint_bias[:, self.tracker.front_feet, 0],
            recent_contact=self.since_contact <= cfg.contact_hold + 1e-6,
        )

        check = self.tracker.check_phase(
            self.move, self.best_body_x, self.need_init, self.steady_time, measured, by_contact=cfg.by_contact
        )
        self.best_body_x = check.best_body_x
        self.steady_time = check.steady_time
        self.move = self.move + check.done.long()
        self.need_init = check.done.clone()
        self.history = torch.cat([self.move.unsqueeze(1), self.history[:, :-1]], dim=1)
