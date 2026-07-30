"""Non-AMP parkour environment for BigReddog."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv, VecEnvStepReturn

from robolab.tasks.manager_based.parkour.managers import (
    DummyRewardCfg,
    MultiRewardCfg,
    MultiRewardManager,
)


def _reward_dict_to_vector(reward: dict[str, torch.Tensor]) -> torch.Tensor:
    """Convert grouped rewards to the tensor format expected by RSL-RL."""
    tensors = tuple(reward.values())
    if not tensors:
        raise ValueError("MultiRewardManager produced an empty reward dictionary.")
    reward_tensor = torch.stack(tensors, dim=-1)
    return reward_tensor.squeeze(-1) if reward_tensor.shape[-1] == 1 else reward_tensor


class BigReddogParkourEnv(ManagerBasedRLEnv):
    """Parkour environment without motion data, animation, or AMP managers."""

    def load_managers(self):
        """Load standard managers, then install the grouped reward manager."""
        reward_group_cfg = None
        if isinstance(self.cfg.rewards, MultiRewardCfg):
            reward_group_cfg = self.cfg.rewards
            self.cfg.rewards = DummyRewardCfg()

        super().load_managers()

        if reward_group_cfg is not None:
            self.cfg.rewards = reward_group_cfg
            self.reward_manager = MultiRewardManager(self.cfg.rewards, self)
            print("[INFO] Multi Reward Manager: ", self.reward_manager)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Run one environment step and flatten grouped rewards for RSL-RL."""
        obs, reward, terminated, truncated, extras = super().step(action)
        if isinstance(reward, dict):
            reward = _reward_dict_to_vector(reward)
            self.reward_buf = reward
        return obs, reward, terminated, truncated, extras
