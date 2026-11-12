"""단계 보조 출력 student: GRU가 지금 단계를 알도록 참값 단계를 함께 가르친다.

student 관측은 Student-A와 같다 (단계 입력 없음). GRU 출력에 단계 분류 출력을 하나 더 붙이고,
학습 때만 행동 MSE에 단계 cross-entropy를 더한다. 행동 MLP는 단계 예측을 받지 않으므로
내보내는 모델(JIT/ONNX)은 GRU + 행동 MLP만 들어가 Student-A와 구조가 같다.

정답 단계는 env의 phase_label 관측 그룹이다 (student와 teacher 어느 쪽에도 주지 않는다).
rsl_rl 저장소는 관측 TensorDict를 통째로 저장하므로 학습 batch에서 꺼낼 수 있다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms import Distillation
from rsl_rl.models import RNNModel


class PhaseAuxRNNModel(RNNModel):
    """RNNModel + 단계 분류 출력. forward 때마다 마지막 단계 예측(logits)을 last_phase_logits에 남긴다."""

    def __init__(self, obs, obs_groups, obs_set, output_dim, *args, num_phases: int = 7, **kwargs) -> None:
        super().__init__(obs, obs_groups, obs_set, output_dim, *args, **kwargs)
        self.phase_head = nn.Linear(self.latent_dim, num_phases)
        self.last_phase_logits: torch.Tensor | None = None

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
        actions, _ = self.forward_with_phase(obs, masks, hidden_state, stochastic_output)
        return actions

    def forward_with_phase(self, obs, masks=None, hidden_state=None, stochastic_output: bool = False):
        """(행동, 단계 logits). GRU는 한 번만 돌린다 (기억이 두 번 진행되지 않게)."""
        latent = self.get_latent(obs, masks, hidden_state)
        mlp_output = self.mlp(latent)
        phase_logits = self.phase_head(latent)
        self.last_phase_logits = phase_logits.detach()
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample(), phase_logits
            return self.distribution.deterministic_output(mlp_output), phase_logits
        return mlp_output, phase_logits


class PhaseAuxDistillation(Distillation):
    """Distillation + 단계 분류 손실. 손실 = 행동 MSE + phase_loss_coef × cross-entropy."""

    def __init__(self, *args, phase_loss_coef: float = 0.1, phase_label_group: str = "phase_label", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.phase_loss_coef = phase_loss_coef
        self.phase_label_group = phase_label_group

    def update(self) -> dict[str, float]:
        """rsl_rl 5.5.1 Distillation.update와 같은 순서에 단계 손실만 더했다."""
        self.num_updates += 1
        mean_behavior_loss = 0.0
        mean_phase_loss = 0.0
        correct = 0.0
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.student.reset(hidden_state=self.last_hidden_states[0])
            self.teacher.reset(hidden_state=self.last_hidden_states[1])
            self.student.detach_hidden_state()
            for batch in self.storage.generator():
                with torch.amp.autocast(  # type: ignore
                    device_type=torch.device(self.device).type, enabled=self.use_mixed_precision, dtype=torch.bfloat16
                ):
                    actions, phase_logits = self.student.forward_with_phase(batch.observations)
                    behavior_loss = self.loss_fn(actions, batch.privileged_actions)
                    target = batch.observations[self.phase_label_group].argmax(dim=-1)
                    phase_loss = nn.functional.cross_entropy(phase_logits.float(), target)

                loss = loss + behavior_loss + self.phase_loss_coef * phase_loss
                mean_behavior_loss += behavior_loss.item()
                mean_phase_loss += phase_loss.item()
                correct += (phase_logits.argmax(dim=-1) == target).float().mean().item()
                cnt += 1

                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.student.detach_hidden_state()
                    loss = 0

                self.student.reset(batch.dones.view(-1))
                self.teacher.reset(batch.dones.view(-1))
                self.student.detach_hidden_state(batch.dones.view(-1))

        self.last_hidden_states = (self.student.get_hidden_state(), self.teacher.get_hidden_state())
        self.student.detach_hidden_state()
        self.student.update_normalization(self.storage.observations.flatten(0, 1))  # type: ignore
        self.storage.clear()

        # TensorBoard에는 Loss/behavior, Loss/phase, Loss/phase_accuracy로 나온다
        return {
            "behavior": mean_behavior_loss / cnt,
            "phase": mean_phase_loss / cnt,
            "phase_accuracy": correct / cnt,
        }
