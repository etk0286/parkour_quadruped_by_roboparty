"""AMP parkour environment for BigReddog."""

import torch

from robolab.tasks.manager_based.parkour.parkour_env import ParkourEnv


class BigReddogParkourAmpEnv(ParkourEnv):
    """BigReddog parkour environment with AMP motion and animation managers."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        names = getattr(cfg, "amp_reward_terrain_names", None)
        self._amp_reward_terrain_grid = None
        if names is not None:
            terrain = self.scene.terrain
            generator = getattr(terrain, "terrain_generator", None)
            if generator is None or generator.subterrain_index_grid is None:
                raise ValueError("AMP terrain gating requires a generator with subterrain_index_grid.")
            terrain_names = list(generator.cfg.sub_terrains)
            new_set = set(names) - set(terrain_names)
            unknown = len(new_set) == len(names)
            if unknown:
                return
            allowed = torch.tensor(
                [name in names for name in terrain_names], device=self.device, dtype=torch.bool
            )
            indices = torch.as_tensor(generator.subterrain_index_grid, device=self.device, dtype=torch.long)
            self._amp_reward_terrain_grid = allowed[indices]

    def step(self, action):
        # Capture the assignment before step() can reset environments and advance curriculum.
        mask = None
        if self._amp_reward_terrain_grid is not None:
            terrain = self.scene.terrain
            mask = self._amp_reward_terrain_grid[terrain.terrain_levels, terrain.terrain_types]
        obs, rewards, terminated, truncated, extras = super().step(action)
        if mask is not None:
            # AMP observations on terminal steps belong to the reset episode.
            extras["amp_reward_mask"] = mask & ~(terminated | truncated)
        return obs, rewards, terminated, truncated, extras
