"""실물 센서 모델: 센서마다 오차 원인을 두고, 그 센서 값을 쓰는 모든 곳이 같은 읽기 값을 쓴다.

실물에서는 같은 센서 값이 단계 계산(PhaseEstimator)과 student 관측에 함께 들어간다.
그래서 오차는 "누가 쓰는지"가 아니라 "어느 센서의 어떤 원인인지"로 한 번만 뽑고, 쓰는 곳은 읽기 값만 받는다.

    센서          원인                          읽는 곳
    라이다·지도    x 위치 편향·노이즈              단계 계산의 몸통·발끝 x, 높이맵 위치 (map_mode="accumulated")
                  장착 각도, 거리 편향           높이맵 기울기, 높이맵 높이
                  프레임 주기, 구멍              높이맵 갱신, 직전 값이 남는 칸
    IMU           기울기                         중력 관측, 높이맵 기울기, 발끝 높이 계산
                  자이로 노이즈                  각속도 관측
    엔코더         편향, 노이즈                   joint_pos 관측, 발끝 위치·앞다리 접힘 계산
    발끝 스위치     오판                          feet_contact 관측, 단계 계산의 접촉
    발끝 높이 계산   단 높이 가정 오차, 기준 발 오차   발끝 높이 계산 (지지 발 디딤면 기준)

env마다 하나 (get_sensor_model). 에피소드 오차는 리셋 때 (StairTracker.reset이 부른다), step 오차는 step마다
처음 읽을 때 한 번 뽑는다 (보상 계산에서 먼저 읽든 관측에서 먼저 읽든 같은 값).
오차 설정(env.cfg.sensor_error)이 없거나 모두 0이면 모든 읽기 값이 참값과 같다.
"""

import torch
from isaaclab.envs import mdp as base_mdp
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply


@configclass
class SensorErrorCfg:
    """randomize=False(평가): 편향은 정확히 이 크기이고 부호만 무작위, 확률·노이즈는 이 값 그대로.
    randomize=True(학습): 이 값은 최댓값이다. 에피소드마다 env별로 0부터 이 값 사이에서 고르게 뽑는다."""

    # 라이다 (Point-LIO + 지도에서 찾은 수직면)
    lidar_x_bias: float = 0.0      # 수직면 기준 몸통 x 편향 [m]. 에피소드 고정
    lidar_x_noise: float = 0.0     # 몸통 x 노이즈 std [m]. step마다
    lidar_mount_tilt: float = 0.0  # 장착 각도 오차 [rad]. pitch, roll 각각. 높이맵만 기울어진다
    lidar_range_bias: float = 0.0  # 높이맵 전체 높이 편향 [m]
    lidar_period: float = 0.0      # 높이맵 갱신 주기 [s]. 그 사이는 마지막 값 유지. 0 = 매 step
    lidar_dropout: float = 0.0     # 칸마다 step마다 새 값 대신 직전 값이 남을 확률 (가려짐, 점 부족)
    # accumulated: 지도에 쌓아 잘라 쓴다 → x 위치 오차만큼 높이맵도 밀린다
    # per_frame: 매 프레임 스캔을 몸통 기준으로 쓴다 → 위치 오차가 높이맵에 들어가지 않는다
    map_mode: str = "accumulated"
    # IMU
    imu_tilt: float = 0.0          # 중력 방향 오차 [rad]. pitch, roll 각각
    gyro_noise: float = 0.0        # 각속도 노이즈 std [rad/s]
    # 엔코더
    encoder_bias: float = 0.0      # 관절마다 편향 [rad]. 에피소드 고정
    encoder_noise: float = 0.0     # 관절마다 노이즈 std [rad]. step마다
    # 관절각을 재지 못하는 실물: 관절각 자리에 마지막으로 보낸 목표각을 쓴다 (엔코더 오차 = 목표각 - 실제 각).
    # 관절 관측, 발끝 위치, 앞다리 접힘이 모두 목표각 기준이 된다. joint_vel 관측은 목표각의 변화율 (student_joint_vel)
    commanded_joints: bool = False
    # 발끝 스위치
    contact_flip: float = 0.0      # 발마다 step마다 뒤집힐 확률
    # 발끝 높이 계산 (IMU + 관절각 + 지지 발의 디딤면 기준). IMU·엔코더 몫은 위 원인에서 따로 들어간다
    step_height_error: float = 0.0  # 가정한 단 높이(16cm)와 실제의 차이 [m]. 기준 단과 단 수가 다를수록 커진다
    foot_height_bias: float = 0.0   # 나머지 편향 (기준 발이 모서리에 걸침 등) [m]. 모든 발 공통
    foot_height_noise: float = 0.0  # 나머지 노이즈 std [m]. step마다
    randomize: bool = False
    # 에피소드마다 이 확률로 편향(에피소드 고정 오차)을 모두 0으로 한다. 노이즈·오판·구멍·주기는 그대로 둔다.
    # 단계 추정 신경망 학습용: 편향이 늘 있으면 신경망이 "규칙이 멈추면 넘어간다"를 배워 맞는 규칙도 앞지른다
    bias_free_ratio: float = 0.0


class SensorModel:
    """env별 센서 오차와 읽기 값."""

    def __init__(self, env, cfg: SensorErrorCfg | None):
        from .stair_reward import FOOT_BODIES  # stair_reward가 이 모듈을 불러오므로 여기서

        self.env = env
        self.asset = env.scene["robot"]
        n, device = env.num_envs, env.device
        self.n, self.device = n, device
        self.num_feet = len(FOOT_BODIES)
        self.num_joints = self.asset.num_joints
        # 에피소드 고정 오차
        self.x_bias = torch.zeros(n, device=device)
        self.x_noise = torch.zeros(n, device=device)
        self.mount_tilt = torch.zeros(n, 2, device=device)  # (roll, pitch)
        self.imu_tilt = torch.zeros(n, 2, device=device)    # (roll, pitch)
        self.range_bias = torch.zeros(n, device=device)
        self.encoder_bias = torch.zeros(n, self.num_joints, device=device)
        self.flip_prob = torch.zeros(n, device=device)
        self.step_height_error = torch.zeros(n, device=device)
        self.foot_height_bias = torch.zeros(n, device=device)
        self.foot_height_noise = torch.zeros(n, device=device)
        self.period_offset = torch.zeros(n, dtype=torch.long, device=device)  # env마다 높이맵 갱신 시점을 어긋나게
        self.bias_free = torch.zeros(n, dtype=torch.bool, device=device)  # 이번 에피소드는 편향 없음 (bias_free_ratio)
        # step 오차 (begin_step에서 뽑는다)
        self.x_error = torch.zeros(n, device=device)
        self.flips = torch.zeros(n, self.num_feet, dtype=torch.bool, device=device)
        self.encoder_error = torch.zeros(n, self.num_joints, device=device)
        self.foot_height_error = torch.zeros(n, device=device)
        # commanded_joints: 이번 step 목표각과 그 변화율 (asset 관절 순서)
        self.commanded = torch.zeros(n, self.num_joints, device=device)
        self.commanded_vel = torch.zeros(n, self.num_joints, device=device)
        self.reset_step = torch.full((n,), -1, dtype=torch.long, device=device)  # 마지막 리셋의 step 번호
        self._step = None
        self.configure(cfg or SensorErrorCfg())
        self.reset(torch.arange(n, device=device))

    def configure(self, cfg: SensorErrorCfg):
        """설정을 바꾼다. 다음 리셋부터 새 오차가 적용된다 (평가 스크립트의 오차별 반복에서 쓴다)."""
        assert cfg.map_mode in ("accumulated", "per_frame"), cfg.map_mode
        self.cfg = cfg
        self.period_steps = max(1, round(cfg.lidar_period / self.env.step_dt))
        self._step = None

    # ----- 오차 뽑기 -----
    def _signed(self, shape, size: float) -> torch.Tensor:
        if self.cfg.randomize:
            return (torch.rand(shape, device=self.device) * 2.0 - 1.0) * size
        return (torch.randint(2, shape, device=self.device).float() * 2.0 - 1.0) * size

    def _scale(self, count: int, size: float) -> torch.Tensor:
        if self.cfg.randomize:
            return torch.rand(count, device=self.device) * size
        return torch.full((count,), size, device=self.device)

    def reset(self, env_ids):
        cfg, count = self.cfg, len(env_ids)
        self.x_bias[env_ids] = self._signed((count,), cfg.lidar_x_bias)
        self.x_noise[env_ids] = self._scale(count, cfg.lidar_x_noise)
        self.mount_tilt[env_ids] = self._signed((count, 2), cfg.lidar_mount_tilt)
        self.imu_tilt[env_ids] = self._signed((count, 2), cfg.imu_tilt)
        self.range_bias[env_ids] = self._signed((count,), cfg.lidar_range_bias)
        self.encoder_bias[env_ids] = self._signed((count, self.num_joints), cfg.encoder_bias)
        self.flip_prob[env_ids] = self._scale(count, cfg.contact_flip)
        self.step_height_error[env_ids] = self._signed((count,), cfg.step_height_error)
        self.foot_height_bias[env_ids] = self._signed((count,), cfg.foot_height_bias)
        self.foot_height_noise[env_ids] = self._scale(count, cfg.foot_height_noise)
        self.period_offset[env_ids] = torch.randint(self.period_steps, (count,), device=self.device)
        self.bias_free[env_ids] = False
        if cfg.bias_free_ratio > 0.0:
            free = torch.rand(count, device=self.device) < cfg.bias_free_ratio
            self.bias_free[env_ids] = free
            keep = (~free).float()
            for bias in (self.x_bias, self.range_bias, self.step_height_error, self.foot_height_bias):
                bias[env_ids] = bias[env_ids] * keep
            for bias in (self.mount_tilt, self.imu_tilt, self.encoder_bias):
                bias[env_ids] = bias[env_ids] * keep.unsqueeze(1)
        # 리셋 뒤 같은 step의 첫 관측도 새 편향을 쓴다
        self.x_error[env_ids] = self.x_bias[env_ids]
        self.flips[env_ids] = False
        self.encoder_error[env_ids] = self.encoder_bias[env_ids]
        self.foot_height_error[env_ids] = self.foot_height_bias[env_ids]
        # 리셋 직후는 기본 자세에 서 있고, 목표각도 기본 자세로 본다 (실물: 서기 자세를 보낸 뒤 시작)
        self.commanded[env_ids] = self._default_joint_pos()[env_ids]
        self.commanded_vel[env_ids] = 0.0
        self.reset_step[env_ids] = self.env.common_step_counter

    def begin_step(self):
        """step 오차를 이번 step에 처음 읽을 때 한 번 뽑는다."""
        step = self.env.common_step_counter
        if self._step == step:
            return
        self._step = step
        cfg = self.cfg
        self.x_error = self.x_bias + torch.randn(self.n, device=self.device) * self.x_noise
        self.flips = torch.rand(self.n, self.num_feet, device=self.device) < self.flip_prob.unsqueeze(1)
        self.encoder_error = self.encoder_bias
        if cfg.encoder_noise > 0.0:
            self.encoder_error = self.encoder_bias + torch.randn_like(self.encoder_bias) * cfg.encoder_noise
        if cfg.commanded_joints:
            # 이번 step에 리셋된 env는 행동 항에 지난 에피소드 목표각이 남아 있으므로 리셋 때 값(기본 자세)을 둔다
            fresh = (self.reset_step == step).unsqueeze(1)
            target = torch.where(fresh, self.commanded, self._joint_target())
            self.commanded_vel = torch.where(fresh, 0.0, (target - self.commanded) / self.env.step_dt)
            self.commanded = target
            self.encoder_error = self.encoder_error + target - _as_torch(self.asset.data.joint_pos)
        self.foot_height_error = self.foot_height_bias + torch.randn(self.n, device=self.device) * self.foot_height_noise

    # ----- 켜져 있는지 -----
    @property
    def has_encoder_error(self) -> bool:
        return self.cfg.encoder_bias > 0.0 or self.cfg.encoder_noise > 0.0 or self.cfg.commanded_joints

    def _default_joint_pos(self) -> torch.Tensor:
        return _as_torch(self.asset.data.default_joint_pos)

    def _joint_target(self) -> torch.Tensor:
        """이번 step에 보낸 관절 목표각 (N, 관절 수), asset 관절 순서. 행동 항 하나가 모든 관절을 맡는다."""
        term = self.env.action_manager.get_term("joint_pos")
        target = self._default_joint_pos().clone()
        target[:, term._joint_ids] = term.processed_actions
        return target

    @property
    def holds_map(self) -> bool:
        return self.period_steps > 1 or self.cfg.lidar_dropout > 0.0

    @property
    def distorts_map(self) -> bool:
        c = self.cfg
        return c.lidar_mount_tilt > 0.0 or c.imu_tilt > 0.0 or c.lidar_range_bias > 0.0 or self.holds_map

    # ----- 읽기 값 -----
    def read_x_error(self) -> torch.Tensor:
        """라이다로 잰 수직면 기준 x의 오차 (N,). 몸통과 발끝 x에 같이 들어간다."""
        self.begin_step()
        return self.x_error

    def map_x_shift(self) -> torch.Tensor | None:
        """높이맵 격자를 x로 미는 양 (N, 1). 지도에서 잘라 쓸 때만."""
        if self.cfg.map_mode != "accumulated" or (self.cfg.lidar_x_bias <= 0.0 and self.cfg.lidar_x_noise <= 0.0):
            return None
        return self.read_x_error().unsqueeze(1)

    def read_encoder_error(self) -> torch.Tensor:
        """관절각 오차 (N, 관절 수), asset 관절 순서."""
        self.begin_step()
        return self.encoder_error

    def read_flips(self) -> torch.Tensor:
        """발끝 스위치 오판 (N, 6), FOOT_BODIES 순서."""
        self.begin_step()
        return self.flips

    def map_tilt(self) -> torch.Tensor:
        """높이맵이 기울어지는 각도 (N, 2) (roll, pitch) = 장착 각도 + 중력 방향 오차."""
        return self.mount_tilt + self.imu_tilt

    def tip_height_error(self, tips: torch.Tensor, level_error: torch.Tensor) -> torch.Tensor:
        """발끝 높이 계산의 오차 (N, 6).
        - IMU 기울기: 지지 발(몸통 중심으로 근사)에서 멀수록 커진다. pitch × 앞 거리 - roll × 왼쪽 거리
        - 단 높이 가정 오차: (발이 있는 단 - 기준 단) 수만큼
        - 나머지: 모든 발 공통 편향 + 노이즈
        level_error: 발마다 (발 밑 단 - 기준 단) 단 수 (N, 6)."""
        self.begin_step()
        out = self.foot_height_error.unsqueeze(1) + self.step_height_error.unsqueeze(1) * level_error
        return out + self.imu_tilt_height_error(tips)

    def imu_tilt_height_error(self, tips: torch.Tensor) -> torch.Tensor:
        """IMU 기울기 오차 때문에 몸통 기준 발끝 높이가 틀어지는 양 (N, 6). pitch × 앞 거리 - roll × 왼쪽 거리."""
        if self.cfg.imu_tilt <= 0.0:
            return torch.zeros(tips.shape[:2], device=tips.device)
        pose = self.asset.data.root_pose_w.torch
        base = pose[:, :3] - self.env.scene.env_origins
        forward = torch.zeros_like(base)
        forward[:, 0] = 1.0
        forward = quat_apply(pose[:, 3:7], forward)
        heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
        dx, dy = tips[..., 0] - base[:, 0:1], tips[..., 1] - base[:, 1:2]
        ahead = torch.cos(heading) * dx + torch.sin(heading) * dy
        left = -torch.sin(heading) * dx + torch.cos(heading) * dy
        roll, pitch = self.imu_tilt.unbind(dim=1)
        return pitch.unsqueeze(1) * ahead - roll.unsqueeze(1) * left


def _as_torch(x) -> torch.Tensor:
    return x.torch if hasattr(x, "torch") else x


def get_sensor_model(env) -> SensorModel:
    if not hasattr(env, "sensor_model"):
        env.sensor_model = SensorModel(env, getattr(env.cfg, "sensor_error", None))
    return env.sensor_model


# ===== student 관측 (오차가 없으면 원래 관측과 같다) =====
def student_joint_pos(env) -> torch.Tensor:
    """joint_pos_rel + 엔코더 오차."""
    out = base_mdp.joint_pos_rel(env)
    sensors = get_sensor_model(env)
    if sensors.has_encoder_error:
        out = out + sensors.read_encoder_error()
    return out


def student_joint_vel(env) -> torch.Tensor:
    """joint_vel_rel. commanded_joints면 목표각의 변화율 (실물에 관절 속도 측정이 없을 때)."""
    sensors = get_sensor_model(env)
    if sensors.cfg.commanded_joints:
        sensors.begin_step()
        return sensors.commanded_vel.clone()
    return base_mdp.joint_vel_rel(env)


def student_base_ang_vel(env) -> torch.Tensor:
    out = base_mdp.base_ang_vel(env)
    noise = get_sensor_model(env).cfg.gyro_noise
    if noise > 0.0:
        out = out + torch.randn_like(out) * noise
    return out


def student_projected_gravity(env) -> torch.Tensor:
    """중력 방향 추정이 (roll, pitch)만큼 틀린 값. 작은 각이라 몸통 좌표에서 g + e × g로 돌린다.
    높이맵 기울기와 발끝 높이 계산이 같은 각도로 틀린다 (같은 부호: 앞쪽이 pitch × 거리만큼 높게 보인다)."""
    out = base_mdp.projected_gravity(env)
    sensors = get_sensor_model(env)
    if sensors.cfg.imu_tilt <= 0.0:
        return out
    tilt = torch.zeros_like(out)
    tilt[:, :2] = sensors.imu_tilt
    out = out + torch.cross(tilt, out, dim=-1)
    return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def student_feet_contact(env, sensor_cfg, threshold: float = 1.0) -> torch.Tensor:
    """발끝 스위치 + 오판. 오판은 단계 계산이 이번 step에 쓰는 것과 같다."""
    from .observations import feet_contact

    out = feet_contact(env, sensor_cfg, threshold)
    sensors = get_sensor_model(env)
    if sensors.cfg.contact_flip > 0.0:
        out = (out.bool() ^ sensors.read_flips()).float()
    return out
