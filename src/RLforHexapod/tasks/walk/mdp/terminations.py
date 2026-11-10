import torch
from isaaclab.envs.mdp import bad_orientation
from isaaclab.managers import SceneEntityCfg

from .observations import stair_env_mask


def out_of_stairs_width(env, half_width: float, center_y: float = 0.0,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """몸통 중심이 계단 폭(env 원점 기준 center_y ± half_width) 밖으로 나가면 종료."""
    asset = env.scene[asset_cfg.name]
    # env 원점 기준 옆(y) 위치
    y = asset.data.root_pos_w.torch[:, 1] - env.scene.env_origins[:, 1]
    return torch.abs(y - center_y) > half_width


def sim_nan(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """물리가 터져 상태에 NaN/Inf가 생기면 종료 (속도 한계 비교는 NaN을 못 잡음)."""
    d = env.scene[asset_cfg.name].data
    return ~(torch.isfinite(d.joint_pos.torch).all(1) & torch.isfinite(d.root_pose_w.torch).all(1))


# ===== 걷기·계단 통합 student =====
def mixed_bad_orientation(env, stair_limit: float, walk_limit: float) -> torch.Tensor:
    """각 teacher를 학습할 때의 기울기 한계로 종료."""
    return torch.where(stair_env_mask(env), bad_orientation(env, stair_limit), bad_orientation(env, walk_limit))


def stair_out_of_width(env, half_width: float) -> torch.Tensor:
    """계단 env만 계단 폭 이탈로 종료. 걷기 env는 계단 뒤쪽 평지를 자유롭게 다닌다."""
    return stair_env_mask(env) & out_of_stairs_width(env, half_width)


def walk_time_out(env, time_s: float) -> torch.Tensor:
    """걷기 env의 에피소드 시간 초과. 계단 env는 env 전체 길이(45초)를 쓴다."""
    return ~stair_env_mask(env) & (env.episode_length_buf * env.step_dt >= time_s)
