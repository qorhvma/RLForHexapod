import torch
from isaaclab.envs import ManagerBasedRLEnv


class HexapodEnv(ManagerBasedRLEnv):
    """물리가 터진 step의 NaN 보상을 0으로 바꿈 (그 env는 sim_nan 종료로 리셋됨).

    phase_net_hook (agents/phase_net.PhaseNetObsHook): 설정하면 관측을 만든 뒤 student 관측의 단계 칸을
    단계 추정 신경망 출력으로 바꾼다 (scripts/finetune_phase_policy.py). 기본은 None (아무것도 안 함)."""

    phase_net_hook = None

    def step(self, action):
        obs, rew, term, trunc, extras = super().step(action)
        hook = self.phase_net_hook
        if hook is not None:
            # 이번 step에 리셋된 env는 새 에피소드의 첫 관측이므로 기억을 비우고 본다
            hook.clear(self.reset_buf.bool())
            hook.apply(obs)
        return obs, torch.nan_to_num(rew, 0.0, 0.0, 0.0), term, trunc, extras

    def reset(self, env_ids=slice(None), **kwargs):
        obs, extras = super().reset(env_ids, **kwargs)
        hook = self.phase_net_hook
        if hook is not None:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            if env_ids is None:
                env_ids = slice(None)
            mask[env_ids.long() if isinstance(env_ids, torch.Tensor) else env_ids] = True
            hook.clear(mask)
            hook.apply(obs, only=mask)
        return obs, extras
