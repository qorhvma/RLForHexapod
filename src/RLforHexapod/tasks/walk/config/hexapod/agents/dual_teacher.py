"""걷기·계단 teacher 묶음: env 그룹에 맞는 teacher의 행동을 student의 정답으로 돌려준다.

rsl_rl의 증류는 teacher 하나만 받으므로, 두 MLP를 teacher 하나처럼 보이게 묶는다.
체크포인트는 scripts/merge_teachers.py로 합친 파일을 쓴다.
"""

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from tensordict import TensorDict


class DualTeacherModel(nn.Module):
    """obs_groups의 teacher 항목 순서: [걷기 teacher 관측, 계단 teacher 관측, 계단 env 표시 (N, 1)]."""

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        walk_clip: float = 1.0,
        stair_clip: float = 2.0,
    ) -> None:
        super().__init__()
        walk_group, stair_group, self.selector_group = obs_groups[obs_set]
        model_args = (output_dim, hidden_dims, activation, obs_normalization, distribution_cfg)
        self.walk = MLPModel(obs, {obs_set: [walk_group]}, obs_set, *model_args)
        self.stair = MLPModel(obs, {obs_set: [stair_group]}, obs_set, *model_args)
        self.walk_clip = walk_clip
        self.stair_clip = stair_clip

    def forward(self, obs: TensorDict, masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
        # teacher가 학습 때 실제로 낼 수 있던 범위로 자른다
        walk = self.walk(obs).clamp(-self.walk_clip, self.walk_clip)
        stair = self.stair(obs).clamp(-self.stair_clip, self.stair_clip)
        return torch.where(obs[self.selector_group] > 0.5, stair, walk)

    # 증류 알고리즘이 teacher에 호출하는 함수들. MLP라 기억이 없다
    def reset(self, dones: torch.Tensor | None = None, hidden_state=None) -> None:
        pass

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass
