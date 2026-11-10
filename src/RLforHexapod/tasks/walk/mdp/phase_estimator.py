"""실물 센서로 계단 단계를 계산할 때의 오차를 흉내 내는 추정기.

실물에서는 Point-LIO의 위치, 지도에서 찾은 수직면, 관절각으로 계산한 발끝, 발 접촉으로 단계를 계산한다.
여기서는 참값에 오차를 더한 측정값을 StairTracker와 같은 조건(check_phase)에 넣는다.
오차가 전부 0이면 StairTracker와 항상 같은 단계가 나온다.

teacher의 정답과 성공 판정은 계속 참값(StairTracker)을 쓴다. student만 이 추정값을 본다.
"""

import torch
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply


@configclass
class PhaseErrorCfg:
    """편향의 크기는 정확히 이 값이고 부호만 무작위다. 그래서 값이 곧 "이만큼 틀렸을 때"를 뜻한다."""

    pos_bias: float = 0.0      # 수직면에 대한 몸통 위치 편향 [m]. x와 z에 에피소드마다 고정
    pos_noise: float = 0.0     # 몸통 위치 노이즈 std [m]. x와 z에 step마다
    joint_bias: float = 0.0    # 관절각 편향 [rad]. 관절마다 에피소드 동안 고정. 발끝 계산과 앞다리 접힘에 반영
    contact_flip: float = 0.0  # 발 접촉 판정이 뒤집힐 확률. 발마다 step마다
    delay: float = 0.0         # 단계 계산 지연 [s]


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
        self.pos_bias = torch.zeros(n, 2, device=device)            # (x, z)
        self.joint_bias = torch.zeros(n, len(foot_names), 3, device=device)  # (coxa, femur, tibia)
        self.configure(cfg)

    def configure(self, cfg: PhaseErrorCfg):
        """오차 설정을 바꾼다. 다음 리셋부터 새 편향이 적용된다 (평가 스크립트의 오차별 반복에서 쓴다)."""
        self.cfg = cfg
        delay_steps = round(cfg.delay / self.env.step_dt)
        # 0번이 가장 최근. 지연이 0이면 칸이 하나뿐이다
        self.history = self.move.unsqueeze(1).repeat(1, delay_steps + 1)

    @property
    def delayed_move(self) -> torch.Tensor:
        """지연을 반영해 student에게 보여 줄 단계 번호 (N,)."""
        return self.history[:, -1]

    def _random_sign(self, shape) -> torch.Tensor:
        return torch.randint(2, shape, device=self.device).float() * 2.0 - 1.0

    def reset(self, env_ids, moves):
        count = len(env_ids)
        self.move[env_ids] = moves
        self.best_body_x[env_ids] = 0.0
        self.need_init[env_ids] = True
        self.steady_time[env_ids] = 0.0
        self.history[env_ids] = moves.unsqueeze(1).repeat(1, self.history.shape[1])
        self.pos_bias[env_ids] = self._random_sign((count, 2)) * self.cfg.pos_bias
        self.joint_bias[env_ids] = self._random_sign((count, *self.joint_bias.shape[1:])) * self.cfg.joint_bias

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
        pos_error = self.pos_bias + torch.randn(self.n, 2, device=self.device) * cfg.pos_noise
        error_x, error_z = pos_error.unbind(dim=1)

        tips = truth.tips
        if cfg.joint_bias > 0.0:
            tips = tips + self._tip_error_from_joints(tips)
        tips = torch.stack([tips[..., 0] + error_x.unsqueeze(1), tips[..., 1], tips[..., 2] + error_z.unsqueeze(1)], dim=-1)

        flips = torch.rand(truth.in_contact.shape, device=self.device) < cfg.contact_flip
        measured = truth._replace(
            body_x=truth.body_x + error_x,
            tips=tips,
            in_contact=truth.in_contact ^ flips,
            front_coxa=truth.front_coxa + self.joint_bias[:, self.tracker.front_feet, 0],
        )

        check = self.tracker.check_phase(self.move, self.best_body_x, self.need_init, self.steady_time, measured)
        self.best_body_x = check.best_body_x
        self.steady_time = check.steady_time
        self.move = self.move + check.done.long()
        self.need_init = check.done.clone()
        self.history = torch.cat([self.move.unsqueeze(1), self.history[:, :-1]], dim=1)
