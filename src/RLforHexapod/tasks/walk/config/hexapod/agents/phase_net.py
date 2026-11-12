"""단계 추정 신경망: student 센서 관측만 보고 지금 단계(7)를 맞힌다. 그 출력을 A-Phase student의 단계 입력에 넣는다.

규칙 추정기(mdp/phase_estimator.py)는 단계를 하나씩 진행하는 상태 기계라서, 편향 때문에 한 번 뒤처지면 돌아오지 못한다.
이 신경망은 매 step 센서 이력에서 다시 판단하므로 틀려도 다시 맞는 쪽(순간 오차)이 되는 것을 노린다.
A-Phase는 순간 오차(지연, 접촉 뒤집힘)에는 fine-tune 없이도 강하다.

    입력: A-Phase student 관측에서 단계 칸(phase)을 뺀 것 = Student-A 관측 (높이맵, 관절, IMU, 접촉, 직전 행동)
          + extra_groups의 관측 그룹. 기본:
            phase_net  = 라이다로 잰 몸통 x, 몸통 기준 발끝 높이
            phase_rule = 규칙 추정기의 단계와 판정 값 (발마다 수직면을 넘은 거리, 지면 위 높이 등).
                         오차가 없으면 규칙 단계를 따라 하면 되므로, 신경망은 규칙이 굳는 경우만 고치면 된다
            phase_timer = 규칙이 지금 단계에 머문 시간. 신경망 기억(1.28초)보다 긴 정지를 셀 수 있게 한다
    구조: 입력 정규화 → GRU → Linear → 단계 logits (7, stair_phase_obs와 같은 순서)
    학습: scripts/train_phase_net.py (참값 단계에 대한 cross-entropy)
    평가: scripts/eval_stairs.py --phase_net <model.pt>
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PHASE_TERM = "phase"  # A-Phase student 관측에서 단계 칸의 이름 (env_cfg.StudentAPhaseObservationsCfg)
EXTRA_GROUP = "phase_net"  # 신경망만 보는 추가 관측 그룹 (env_cfg.StudentAPhaseObservationsCfg.PhaseNetCfg)
RULE_GROUP = "phase_rule"  # 규칙 추정기의 판정과 판정 값 (env_cfg.StudentAPhaseObservationsCfg.PhaseRuleCfg)
TIMER_GROUP = "phase_timer"  # 규칙 추정기가 지금 단계에 머문 시간 (env_cfg.StudentAPhaseObservationsCfg.PhaseTimerCfg)


class RunningNorm(nn.Module):
    """관측 평균·분산을 누적해 정규화한다. 저장 파일에 같이 들어간다."""

    def __init__(self, dim: int, eps: float = 1e-2):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1])
        n = x.shape[0]
        batch_mean, batch_var = x.mean(dim=0), x.var(dim=0, unbiased=False)
        total = self.count + n
        delta = batch_mean - self.mean
        self.mean += delta * n / total
        self.var = (self.var * self.count + batch_var * n + delta**2 * self.count * n / total) / total
        self.count = total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps)


class PhaseNet(nn.Module):
    """한 step씩 돌리는 GRU 단계 분류기."""

    def __init__(
        self, input_dim: int, hidden_dim: int = 128, num_classes: int = 7, input_mode: str = "onehot",
        extra_groups: tuple[str, ...] | list[str] = (),
    ):
        super().__init__()
        assert input_mode in ("onehot", "probs"), input_mode
        self.extra_groups = list(extra_groups)
        self.config = {
            "input_dim": input_dim, "hidden_dim": hidden_dim, "num_classes": num_classes, "input_mode": input_mode,
            "extra_groups": self.extra_groups,
        }
        self.input_mode = input_mode
        self.norm = RunningNorm(input_dim)
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    @property
    def uses_rule(self) -> bool:
        """규칙 추정기의 판정을 입력으로 받는다 (발끝 높이 계산 오차와 판정 옵션이 신경망 입력에 영향을 준다)."""
        return RULE_GROUP in self.extra_groups

    def initial_hidden(self, num_envs: int, device) -> torch.Tensor:
        return torch.zeros(num_envs, self.gru.hidden_size, device=device)

    def step(self, x: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(logits, 다음 hidden). x는 정규화 전 관측."""
        hidden = self.gru(self.norm(x), hidden)
        return self.head(hidden), hidden

    def to_phase_input(self, logits: torch.Tensor) -> torch.Tensor:
        """A-Phase의 단계 입력으로 바꾼다. onehot = 가장 그럴듯한 단계 (A-Phase가 학습한 형태), probs = softmax."""
        if self.input_mode == "probs":
            return torch.softmax(logits, dim=-1)
        return F.one_hot(logits.argmax(dim=-1), logits.shape[-1]).float()

    def save(self, path: str, extra: dict | None = None):
        torch.save({"model": self.state_dict(), "config": self.config, **(extra or {})}, path)

    @classmethod
    def load(cls, path: str, device) -> tuple[PhaseNet, dict]:
        data = torch.load(path, map_location=device, weights_only=False)
        net = cls(**data["config"]).to(device)
        net.load_state_dict(data["model"])
        return net, data


def term_slice(observation_manager, group: str = "student", term: str = PHASE_TERM) -> slice:
    """관측 그룹 안에서 한 항의 위치 (이어 붙인 관측 기준)."""
    names = observation_manager.active_terms[group]
    dims = observation_manager.group_obs_term_dim[group]
    if term not in names:
        raise ValueError(f"'{group}' 관측에 '{term}' 항이 없습니다: {names}")
    start = 0
    for name, dim in zip(names, dims, strict=True):
        size = int(torch.tensor(dim).prod().item())
        if name == term:
            return slice(start, start + size)
        start += size
    raise AssertionError


def without(x: torch.Tensor, cut: slice) -> torch.Tensor:
    """관측에서 단계 칸을 뺀다."""
    return torch.cat([x[:, : cut.start], x[:, cut.stop :]], dim=1)


def net_input(obs, cut: slice, extra_groups, group: str = "student") -> torch.Tensor:
    """신경망 입력 = student 관측에서 단계 칸을 뺀 것 + 추가 관측 그룹들."""
    return torch.cat([without(obs[group], cut)] + [obs[g] for g in extra_groups], dim=1)


class PhaseNetPolicy:
    """A-Phase 정책을 감싼다. 매 step 신경망이 관측에서 단계를 추정해 관측의 단계 칸을 바꾼 뒤 정책에 넣는다.
    forced(학습 때만): True인 env는 신경망 대신 forced_input(참값 단계)을 넣는다."""

    def __init__(self, policy, net: PhaseNet, cut: slice, num_envs: int, device, group: str = "student"):
        self.policy, self.net, self.cut, self.group = policy, net, cut, group
        self.hidden = net.initial_hidden(num_envs, device)
        self.last_input: torch.Tensor | None = None   # 신경망이 본 관측 (단계 칸 뺀 것)
        self.last_logits: torch.Tensor | None = None
        self.forced: torch.Tensor | None = None
        self.forced_input: torch.Tensor | None = None

    def __call__(self, obs):
        observed = obs[self.group]
        self.last_input = net_input(obs, self.cut, self.net.extra_groups, self.group)
        self.last_logits, self.hidden = self.net.step(self.last_input, self.hidden)
        phase = self.net.to_phase_input(self.last_logits)
        if self.forced is not None:
            phase = torch.where(self.forced.unsqueeze(1), self.forced_input, phase)
        replaced = observed.clone()
        replaced[:, self.cut] = phase
        obs[self.group] = replaced
        return self.policy(obs)

    def reset(self, dones=None):
        self.policy.reset(dones)
        if dones is None:
            self.hidden.zero_()
        else:
            self.hidden[dones.bool()] = 0.0
