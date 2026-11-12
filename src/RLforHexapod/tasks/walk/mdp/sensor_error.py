"""student 관측에 실물 센서 오차를 넣는다. 지금은 평가(scripts/eval_stairs.py)에서 무엇이 student를 무너뜨리는지 보는 데 쓴다.

오차는 두 종류다.
1. 단계 추정기와 공유하는 오차 (PhaseErrorCfg.share_obs): 실물에서 같은 센서 값이 단계 계산과 관측에 함께 들어간다.
   - x 위치 오차: 수직면을 찾는 점군과 높이맵이 같은 지도에서 나오므로 높이맵의 모서리도 같은 만큼 밀려 보인다
   - 관절각 편향: 같은 엔코더 값이 발끝 계산과 joint_pos 관측에 들어간다
   - 접촉 오판: 같은 스위치 값이 단계 계산과 feet_contact 관측에 들어간다
   값은 PhaseEstimator가 env별로 뽑은 것을 그대로 읽는다 (같은 값이어야 실물처럼 함께 틀린다).
2. 관측에만 들어가는 오차 (ObsErrorCfg): 높이맵 기울기·높이 편향·갱신 주기·구멍, IMU 기울기, 관절·자이로 노이즈.

오차 설정이 없으면 모든 함수가 원래 관측과 똑같은 값을 낸다 (학습 task는 기본으로 꺼져 있다).
"""

import torch
from isaaclab.envs import mdp as base_mdp
from isaaclab.utils.configclass import configclass


@configclass
class ObsErrorCfg:
    """randomize=False(평가): 각도·편향은 정확히 이 크기이고 부호만 무작위. randomize=True(학습): 최댓값, 0부터 균등."""

    map_tilt: float = 0.0     # 라이다 장착 각도 오차 [rad]. pitch와 roll에 각각. 높이맵만 기울어진다
    imu_tilt: float = 0.0     # 중력 방향 추정 오차 [rad]. pitch와 roll에 각각. projected_gravity와 높이맵이 같은 각도로 틀린다
    map_z_bias: float = 0.0   # 높이맵 전체 높이 편향 [m]
    map_period: float = 0.0   # 높이맵 갱신 주기 [s]. 그 사이는 마지막 값을 유지한다. 0 = 매 step
    map_dropout: float = 0.0  # 칸마다 step마다 새 값 대신 직전 값을 유지할 확률 (가려짐, 점 부족)
    joint_noise: float = 0.0  # 관절각 노이즈 std [rad]. step마다
    gyro_noise: float = 0.0   # 각속도 노이즈 std [rad/s]. step마다
    randomize: bool = False


def _signed(shape, size: float, randomize: bool, device) -> torch.Tensor:
    if randomize:
        return (torch.rand(shape, device=device) * 2.0 - 1.0) * size
    return (torch.randint(2, shape, device=device).float() * 2.0 - 1.0) * size


class ObsErrorState:
    """env별로 뽑은 관측 오차. StairTracker가 만들고 리셋 때 다시 뽑는다."""

    def __init__(self, env, cfg: ObsErrorCfg):
        self.env = env
        n, device = env.num_envs, env.device
        self.n, self.device = n, device
        self.map_tilt = torch.zeros(n, 2, device=device)  # (roll, pitch) [rad]
        self.imu_tilt = torch.zeros(n, 2, device=device)  # (roll, pitch) [rad]
        self.map_z = torch.zeros(n, device=device)
        self.period_offset = torch.zeros(n, dtype=torch.long, device=device)  # env마다 갱신 시점을 어긋나게
        self.configure(cfg)

    def configure(self, cfg: ObsErrorCfg):
        """설정을 바꾼다. 다음 리셋부터 새 오차가 적용된다."""
        self.cfg = cfg
        self.period_steps = max(1, round(cfg.map_period / self.env.step_dt))

    def reset(self, env_ids):
        cfg, count = self.cfg, len(env_ids)
        self.map_tilt[env_ids] = _signed((count, 2), cfg.map_tilt, cfg.randomize, self.device)
        self.imu_tilt[env_ids] = _signed((count, 2), cfg.imu_tilt, cfg.randomize, self.device)
        self.map_z[env_ids] = _signed((count,), cfg.map_z_bias, cfg.randomize, self.device)
        self.period_offset[env_ids] = torch.randint(self.period_steps, (count,), device=self.device)

    @property
    def holds_map(self) -> bool:
        return self.period_steps > 1 or self.cfg.map_dropout > 0.0

    def map_tilt_total(self) -> torch.Tensor:
        """높이맵이 기울어지는 각도 (N, 2) = 장착 각도 + 중력 방향 오차."""
        return self.map_tilt + self.imu_tilt


def shared_errors(env):
    """(관측과 공유하는 PhaseEstimator 또는 None, ObsErrorState 또는 None)."""
    tracker = getattr(env, "stair_tracker", None)
    if tracker is None:
        return None, None
    estimator = tracker.estimator
    if estimator is not None and not estimator.cfg.share_obs:
        estimator = None
    return estimator, tracker.obs_error


# ===== student 관측 (오차가 없으면 원래 관측과 같다) =====
def student_joint_pos(env) -> torch.Tensor:
    """joint_pos_rel + (공유) 관절각 편향 + 관절각 노이즈."""
    out = base_mdp.joint_pos_rel(env)
    estimator, obs = shared_errors(env)
    if estimator is not None and estimator.cfg.joint_bias > 0.0:
        out = out + estimator.joint_bias_by_joint()
    if obs is not None and obs.cfg.joint_noise > 0.0:
        out = out + torch.randn_like(out) * obs.cfg.joint_noise
    return out


def student_base_ang_vel(env) -> torch.Tensor:
    out = base_mdp.base_ang_vel(env)
    _, obs = shared_errors(env)
    if obs is not None and obs.cfg.gyro_noise > 0.0:
        out = out + torch.randn_like(out) * obs.cfg.gyro_noise
    return out


def student_projected_gravity(env) -> torch.Tensor:
    """중력 방향 추정이 (roll, pitch)만큼 틀린 값. 작은 각이라 몸통 좌표에서 g + e × g로 돌린다."""
    out = base_mdp.projected_gravity(env)
    _, obs = shared_errors(env)
    if obs is None or obs.cfg.imu_tilt <= 0.0:
        return out
    tilt = torch.zeros_like(out)
    tilt[:, :2] = obs.imu_tilt
    out = out + torch.cross(tilt, out, dim=-1)
    return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def student_feet_contact(env, sensor_cfg, threshold: float = 1.0) -> torch.Tensor:
    """발끝 스위치 + (공유) 접촉 오판. 오판은 PhaseEstimator가 이번 step에 뽑은 것과 같다."""
    from .observations import feet_contact

    out = feet_contact(env, sensor_cfg, threshold)
    estimator, _ = shared_errors(env)
    if estimator is not None and estimator.cfg.contact_flip > 0.0:
        out = (out.bool() ^ estimator.flips).float()
    return out
