"""계단 teacher용 특권 관측: 계단 형상을 알고 있다는 가정으로 해석적으로 계산."""

import torch
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply

from .sensor_model import get_sensor_model


@configclass
class StairsGeometryCfg:
    """stairs_*.usda의 형상 (env 원점 기준). USD를 바꾸면 여기도 같이 바꿀 것."""

    x0: float = 0.5            # 첫 단 시작 x (= StairsSceneCfg의 stairs 위치)
    step_height: float = 0.16  # 단 높이
    tread: float = 0.3         # 단 깊이 (마지막 단 제외)
    num_steps: int = 3
    top_length: float = 0.8    # 마지막 단(꼭대기) 깊이
    width: float = 1.0


def stairs_terrain_height(stairs: StairsGeometryCfg, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """env 원점 기준 (x, y) 지점의 지형 높이. 계단 밖은 바닥(0)."""
    xr = x - stairs.x0
    length = (stairs.num_steps - 1) * stairs.tread + stairs.top_length
    on_stairs = (xr >= 0.0) & (xr < length) & (torch.abs(y) <= stairs.width / 2)
    # i번째 단(0부터) 윗면 높이 = (i + 1) × 단 높이, 마지막 단은 top_length 끝까지
    k = torch.clamp(torch.floor(xr / stairs.tread) + 1, max=stairs.num_steps)
    return torch.where(on_stairs, k * stairs.step_height, torch.zeros_like(xr))


def stairs_next_riser_distance(stairs: StairsGeometryCfg, x: torch.Tensor, max_dist: float) -> torch.Tensor:
    """x 앞쪽의 가장 가까운 단 수직면(riser)까지 거리. 계단 앞에서는 = 계단까지 거리, 꼭대기 위는 max_dist."""
    risers = stairs.x0 + stairs.tread * torch.arange(stairs.num_steps, device=x.device)
    d = risers - x.unsqueeze(-1)
    d = torch.where(d >= 0.0, d, torch.full_like(d, max_dist))
    return torch.clamp(d.min(dim=-1).values, max=max_dist)


# tibia 원점(무릎) → 발끝 offset (tibia 프레임, URDF tibiatip 위치). L1("tibia")만 y 방향이 반대
TIP_OFFSET_L1 = (-0.004, -0.147, -0.021)
TIP_OFFSET = (0.004, 0.147, -0.021)


def tip_offsets(body_names: list[str], device) -> torch.Tensor:
    return torch.tensor([TIP_OFFSET_L1 if n == "tibia" else TIP_OFFSET for n in body_names], device=device)


class _FeetTipTerm(ManagerTermBase):
    """발끝 위치가 필요한 관측의 공통 부분: 바디별 발끝 offset을 한 번만 만들어 둠."""

    def __init__(self, cfg: ObservationTermCfg, env):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]
        self.tip_offset = tip_offsets([self.asset.body_names[i] for i in asset_cfg.body_ids], env.device)

    def tips_local(self, env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        """발끝 위치 (env 원점 기준) (N, B, 3)."""
        quat = self.asset.data.body_quat_w.torch[:, asset_cfg.body_ids]
        offset_w = quat_apply(quat, self.tip_offset.expand(quat.shape[0], -1, -1))
        tips_w = self.asset.data.body_pos_w.torch[:, asset_cfg.body_ids] + offset_w
        return tips_w - env.scene.env_origins.unsqueeze(1)


class front_feet_to_stairs(_FeetTipTerm):
    """1. 앞발 끝에서 앞쪽 다음 단 수직면까지 x 거리 (N, 앞발 수)."""

    def __call__(self, env, asset_cfg: SceneEntityCfg, stairs: StairsGeometryCfg, max_dist: float = 1.0):
        tips = self.tips_local(env, asset_cfg)
        return stairs_next_riser_distance(stairs, tips[..., 0], max_dist)


class feet_height_above_terrain(_FeetTipTerm):
    """2-a. 각 발끝이 바로 아래 지형에서 떠 있는 높이 (N, 6). 딛고 있으면 ≈ 0."""

    def __call__(self, env, asset_cfg: SceneEntityCfg, stairs: StairsGeometryCfg):
        tips = self.tips_local(env, asset_cfg)
        return tips[..., 2] - stairs_terrain_height(stairs, tips[..., 0], tips[..., 1])


class terrain_height_under_feet(_FeetTipTerm):
    """2-b. 각 발끝 아래 지형이 몸통보다 얼마나 낮은지 (N, 6). 발 밑에 단이 있으면 값이 작아짐."""

    def __call__(self, env, asset_cfg: SceneEntityCfg, stairs: StairsGeometryCfg):
        tips = self.tips_local(env, asset_cfg)
        base_z = self.asset.data.root_pos_w.torch[:, 2:3] - env.scene.env_origins[:, 2:3]
        return base_z - stairs_terrain_height(stairs, tips[..., 0], tips[..., 1])


def stairs_step_height(env, stairs: StairsGeometryCfg) -> torch.Tensor:
    """3. 단 높이 (N, 1). 지금은 모든 env가 같은 계단 → 계단 높이 커리큘럼 때 env별 값으로 교체."""
    return torch.full((env.num_envs, 1), stairs.step_height, device=env.device)


# ===== student 관측 =====
class stairs_height_map(ManagerTermBase):
    """가짜 라이다: 몸통 위치와 yaw를 따라가는 격자의 칸마다 지형이 몸통보다 얼마나 낮은지 (N, 칸 수)."""

    def __init__(self, cfg: ObservationTermCfg, env):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        (x_min, x_max), (y_min, y_max) = cfg.params["x_range"], cfg.params["y_range"]
        x_spacing, y_spacing = cfg.params["spacing"]
        # 간격이 범위를 정확히 나누지 않으면 끝을 넘지 않는 칸까지만 만든다
        x = x_min + x_spacing * torch.arange(int((x_max - x_min) / x_spacing + 1e-6) + 1, device=env.device)
        y = y_min + y_spacing * torch.arange(int((y_max - y_min) / y_spacing + 1e-6) + 1, device=env.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        self.grid_x, self.grid_y = grid_x.flatten(), grid_y.flatten()
        # 라이다 갱신 주기·구멍(sensor_model.py)에 쓰는 마지막 출력
        self.last = None
        self.refresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        # 새 에피소드의 첫 관측은 항상 새로 계산한다
        if env_ids is None:
            self.refresh[:] = True
        else:
            self.refresh[env_ids] = True

    def __call__(self, env, asset_cfg: SceneEntityCfg, stairs: StairsGeometryCfg, x_range, y_range, spacing):
        sensors = get_sensor_model(env)
        pose = self.asset.data.root_pose_w.torch
        base = pose[:, :3] - env.scene.env_origins
        forward = torch.zeros_like(base)
        forward[:, 0] = 1.0
        forward = quat_apply(pose[:, 3:7], forward)
        heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
        cos, sin = torch.cos(heading), torch.sin(heading)
        x = base[:, 0:1] + cos * self.grid_x - sin * self.grid_y
        y = base[:, 1:2] + sin * self.grid_x + cos * self.grid_y
        shift = sensors.map_x_shift()
        if shift is not None:
            # 몸통이 x + e에 있다고 믿으면 지도에서 x + e 주변을 잘라 온다 → 모서리가 단계 계산과 같은 만큼 밀려 보인다
            x = x + shift
        height = base[:, 2:3] - stairs_terrain_height(stairs, x, y)
        if not sensors.distorts_map:
            return height
        return self._distort(env, height, sensors)

    def _distort(self, env, height, sensors):
        """기울기, 높이 편향, 갱신 주기, 구멍. 기울기는 점군을 틀린 중력 방향으로 세운 결과라 앞뒤·좌우 거리에 비례한다."""
        roll, pitch = sensors.map_tilt().unbind(dim=1)
        # 점의 높이가 pitch × 앞 거리 - roll × 왼쪽 거리만큼 높게 보인다 (발끝 높이 계산, 중력 관측과 같은 부호)
        rise = pitch.unsqueeze(1) * self.grid_x - roll.unsqueeze(1) * self.grid_y
        height = height - rise + sensors.range_bias.unsqueeze(1)
        if not sensors.holds_map or self.last is None or self.last.shape != height.shape:
            self.last = height
            self.refresh[:] = False
            return height
        cfg = sensors.cfg
        step = env.common_step_counter
        due = ((step + sensors.period_offset) % sensors.period_steps == 0) | self.refresh
        new = height
        if cfg.lidar_dropout > 0.0:
            stale = torch.rand_like(height) < cfg.lidar_dropout
            new = torch.where(stale & ~self.refresh.unsqueeze(1), self.last, height)
        out = torch.where(due.unsqueeze(1), new, self.last)
        self.last = out
        self.refresh[:] = False
        return out


def stair_env_mask(env) -> torch.Tensor:
    """계단 env 표시 (N,). 앞쪽 stair_ratio 비율의 env가 계단, 나머지가 걷기다. stair_ratio가 없는 task는 전부 계단."""
    if not hasattr(env, "_stair_env_mask"):
        num_stair = round(getattr(env.cfg, "stair_ratio", 1.0) * env.num_envs)
        env._stair_env_mask = torch.arange(env.num_envs, device=env.device) < num_stair
    return env._stair_env_mask


def stair_env_flag(env) -> torch.Tensor:
    """teacher 선택용 표시 (N, 1). 계단 env = 1. student는 보지 않는다."""
    return stair_env_mask(env).float().unsqueeze(1)


def student_command(env, stair_velocity: tuple[float, float, float], stair_height: float) -> torch.Tensor:
    """student가 보는 속도(3)와 몸통 높이(1) 명령 (N, 4). 계단 teacher는 명령을 따르지 않으므로 계단 env에는 고정값을 보여 준다."""
    command = torch.cat(
        [env.command_manager.get_command("base_velocity"), env.command_manager.get_command("base_height")], dim=1
    )
    fixed = torch.tensor([*stair_velocity, stair_height], device=env.device)
    return torch.where(stair_env_mask(env).unsqueeze(1), fixed, command)


def feet_contact(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """발 접촉 (N, 발 수). 발끝 스위치처럼 수직 힘만 본다: 수직면에 부딪힌 tibia는 접촉이 아니다."""
    sensor = env.scene.sensors[sensor_cfg.name]
    forces_z = sensor.data.net_normal_forces_w_history.torch[:, :, sensor_cfg.body_ids, 2]
    return (forces_z.max(dim=1).values > threshold).float()
