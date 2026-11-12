"""실물 센서로 계단 단계를 계산하는 추정기 (판정 알고리즘만 둔다. 센서 오차는 sensor_model.py).

실물에서는 Point-LIO의 위치, 지도에서 찾은 수직면, 관절각으로 계산한 발끝, 발 접촉으로 단계를 계산한다.
여기서는 SensorModel의 읽기 값(참값 + 센서 오차)으로 측정값을 만들어 StairTracker와 같은 조건(check_phase)에 넣는다.
센서 오차가 모두 0이고 by_contact=False, catch_up=False, delay=0이면 StairTracker와 항상 같은 단계가 나온다.

편향이 있으면 추정이 한 단계에서 영영 멈출 수 있다 (참값은 기준을 겨우 넘기고 다음 단계로 가 버린다).
그래서 실물에서도 쓸 두 가지 복구를 둔다.
- by_contact: 높이 3cm 조건에 "발끝 스위치가 눌리고 높이 contact_window 안"을 OR로 더한다 (높이 오차 대책)
- catch_up: 전진 단계에서 다음 옮기기 단계가 이미 끝났으면 두 칸 진행한다 (x 오차, 앞다리 접힘 대책)

teacher의 정답과 성공 판정은 계속 참값(StairTracker)을 쓴다. student만 이 추정값을 본다.
"""

import torch
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply

from .observations import stairs_terrain_height
from .sensor_model import get_sensor_model


@configclass
class PhaseEstimatorCfg:
    """단계 판정 알고리즘의 설정. 오차 크기는 SensorErrorCfg에 있다."""

    delay: float = 0.0            # 단계 계산 지연 [s] (계산·통신 시간)
    randomize_delay: bool = False  # True(학습): 에피소드마다 0 ~ delay에서 고른다
    # 딛고 있음 = 높이 3cm 안 OR (by_contact일 때) 발끝 스위치가 눌리고 높이 contact_window 안
    # 높이 제한 없이 스위치만 OR하면 오차 0에서 teacher보다 16% 앞섰다. 공중의 발은 스위치가 안 눌려 걸러진다
    by_contact: bool = False
    contact_window: float = 0.08   # 스위치로 인정할 발끝 높이 차 [m]. 단 높이(16cm)의 절반
    # 스위치로만 통과하는 상태가 이 시간 이어져야 인정한다 [s]. 0 = 바로. teacher가 잠깐 지나가는 상태를 거른다
    contact_confirm: float = 0.0
    # 전진 단계가 안 끝났는데 다음 옮기기(마지막 전진이면 정지) 단계가 이미 끝났으면 두 칸 진행한다
    catch_up: bool = False


class PhaseEstimator:
    """센서 읽기 값으로 env별 단계 번호를 따로 진행시킨다."""

    def __init__(self, tracker, cfg: PhaseEstimatorCfg):
        self.tracker = tracker
        self.env = tracker.env
        self.asset = tracker.asset
        self.sensors = get_sensor_model(self.env)
        n, device = self.env.num_envs, self.env.device
        self.n, self.device = n, device

        # 발마다 (coxa, femur, tibia) 바디. URDF의 관절은 모두 자식 링크 원점에 있고 축은 자식 링크의 +z다
        foot_names = [self.asset.body_names[i] for i in tracker.foot_ids]
        chain_names = [name.replace("tibia", part) for name in foot_names for part in ("coxa", "femur", "tibia")]
        self.chain_ids, _ = self.asset.find_bodies(chain_names, preserve_order=True)
        self.joint_axis = torch.zeros(n, len(chain_names), 3, device=device)
        self.joint_axis[..., 2] = 1.0
        # 같은 순서의 관절 번호. 관절 이름은 L{다리}_{coxa|femur|tibia}, 바디는 coxa / coxa_{다리}
        joint_names = []
        for name in chain_names:
            part, _, leg = name.partition("_")
            joint_names.append(f"L{leg or '1'}_{part}")
        self.chain_joint_ids, _ = self.asset.find_joints(joint_names, preserve_order=True)

        self.move = tracker.move.clone()
        self.best_body_x = torch.zeros(n, device=device)
        self.need_init = torch.ones(n, dtype=torch.bool, device=device)
        self.steady_time = torch.zeros(n, device=device)
        self.phase_time = torch.zeros(n, device=device)  # 지금 단계에 머문 시간 [s] (단계 추정 신경망 입력)
        self.pressed_time = torch.zeros(n, len(foot_names), device=device)  # 스위치로만 딛고 있다고 본 시간 [s]
        self.ahead_steady_time = torch.zeros(n, device=device)  # catch_up: 다음 단계(정지)를 내다본 멈춘 시간 [s]
        self.catch_ups = torch.zeros((), dtype=torch.long, device=device)  # catch_up으로 건너뛴 횟수 (평가 로그용)
        self.delay_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.configure(cfg)

    def configure(self, cfg: PhaseEstimatorCfg):
        """설정을 바꾼다. 다음 리셋부터 적용된다 (평가 스크립트의 오차별 반복에서 쓴다)."""
        self.cfg = cfg
        self.max_delay_steps = round(cfg.delay / self.env.step_dt)
        # 0번이 가장 최근. 지연이 0이면 칸이 하나뿐이다
        self.history = self.move.unsqueeze(1).repeat(1, self.max_delay_steps + 1)
        self.delay_steps = self.delay_steps.clamp(max=self.max_delay_steps)
        self.catch_ups = torch.zeros_like(self.catch_ups)

    @property
    def delayed_move(self) -> torch.Tensor:
        """지연을 반영해 student에게 보여 줄 단계 번호 (N,)."""
        return self.history.gather(1, self.delay_steps.unsqueeze(1)).squeeze(1)

    def reset(self, env_ids, moves):
        count = len(env_ids)
        self.move[env_ids] = moves
        self.best_body_x[env_ids] = 0.0
        self.need_init[env_ids] = True
        self.steady_time[env_ids] = 0.0
        self.phase_time[env_ids] = 0.0
        self.pressed_time[env_ids] = 0.0
        self.ahead_steady_time[env_ids] = 0.0
        self.history[env_ids] = moves.unsqueeze(1).repeat(1, self.history.shape[1])
        if self.cfg.randomize_delay:
            self.delay_steps[env_ids] = torch.randint(self.max_delay_steps + 1, (count,), device=self.device)
        else:
            self.delay_steps[env_ids] = self.max_delay_steps

    def _tip_error_from_joints(self, tips: torch.Tensor, chain_error: torch.Tensor) -> torch.Tensor:
        """관절각 오차만큼 발끝 계산이 틀어지는 양 (N, 6, 3). 각이 작으므로 축 × 팔 길이로 근사한다."""
        data = self.asset.data
        joint_pos = data.body_pos_w.torch[:, self.chain_ids] - self.env.scene.env_origins.unsqueeze(1)
        axis = quat_apply(data.body_quat_w.torch[:, self.chain_ids], self.joint_axis)
        num_feet = tips.shape[1]
        joint_pos = joint_pos.reshape(self.n, num_feet, 3, 3)
        axis = axis.reshape(self.n, num_feet, 3, 3)
        lever = tips.unsqueeze(2) - joint_pos
        return (chain_error.unsqueeze(-1) * torch.cross(axis, lever, dim=-1)).sum(dim=2)

    def read(self, truth):
        """참값 측정을 센서 읽기 값으로 바꾼다 (몸통 x, 발끝, 접촉, 앞다리 coxa). 상태를 바꾸지 않는다.
        단계 추정 신경망의 규칙 특징(stair_rule_features)도 이 값을 쓴다. truth: stair_reward.PhaseMeasurement."""
        sensors, tracker = self.sensors, self.tracker
        tips = truth.tips
        front_coxa = truth.front_coxa
        if sensors.has_encoder_error:
            encoder = sensors.read_encoder_error()
            chain_error = encoder[:, self.chain_joint_ids].reshape(self.n, -1, 3)
            tips = tips + self._tip_error_from_joints(tips, chain_error)
            front_coxa = front_coxa + encoder[:, tracker.front_coxa_ids]
        error_x = sensors.read_x_error()
        tips_x = tips[..., 0] + error_x.unsqueeze(1)
        # 발끝 높이는 지지 발 디딤면 기준으로 계산한다. 기준 단 = 지금 추정한 단계의 단
        _, level = tracker.phase_and_level(self.move)
        foot_level = stairs_terrain_height(tracker.stairs, tips_x, tips[..., 1]) / tracker.stairs.step_height
        tips_z = tips[..., 2] + sensors.tip_height_error(tips, foot_level - level.unsqueeze(1).float())
        tips = torch.stack([tips_x, tips[..., 1], tips_z], dim=-1)

        # 접촉: 참값 tracker와 같은 접촉에 발끝 스위치의 오판을 넣는다 (스위치 값과 같은 오판)
        contact = truth.in_contact ^ sensors.read_flips()
        return truth._replace(body_x=truth.body_x + error_x, tips=tips, in_contact=contact, front_coxa=front_coxa)

    def measure(self, truth):
        """판정에 넣을 측정값 = read() + (by_contact일 때) 스위치로 딛고 있다고 확인된 발."""
        tracker, cfg = self.tracker, self.cfg
        measured = self.read(truth)
        contact_ground = None
        if cfg.by_contact:
            # 스위치가 눌리고 발끝 높이가 발 밑 지형(발끝 x로 고른 단)에서 contact_window 안
            gait = tracker.gait
            tips = measured.tips
            terrain = stairs_terrain_height(tracker.stairs, tips[..., 0], tips[..., 1])
            height_gap = (tips[..., 2] - terrain - gait.tip_height).abs()
            pressed = (truth.sole_contact ^ self.sensors.read_flips()) & (height_gap < cfg.contact_window)
            # 높이 3cm 조건은 실패하고 스위치로만 통과하는 상태가 이어진 시간. 높이로 통과하면 다시 잰다
            # (눌린 시간으로 재면 오래 딛고 있던 발이 잠깐 높게 읽힐 때 바로 통과해서 거르지 못한다)
            switch_only = pressed & (height_gap >= gait.contact_tol)
            self.pressed_time = torch.where(
                switch_only, self.pressed_time + self.env.step_dt, torch.zeros_like(self.pressed_time)
            )
            contact_ground = switch_only & (self.pressed_time >= cfg.contact_confirm + self.env.step_dt - 1e-6)
        return measured._replace(contact_ground=contact_ground)

    def update(self, truth):
        """센서 읽기 값으로 단계를 진행시킨다."""
        cfg, tracker = self.cfg, self.tracker
        measured = self.measure(truth)
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
        self.phase_time = torch.where(steps > 0, torch.zeros_like(self.phase_time), self.phase_time + self.env.step_dt)
        self.ahead_steady_time = torch.where(steps > 0, torch.zeros_like(self.ahead_steady_time), self.ahead_steady_time)
        self.history = torch.cat([self.move.unsqueeze(1), self.history[:, :-1]], dim=1)
