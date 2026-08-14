"""Export all enabled BigReddog parkour training terrains to one MuJoCo heightfield."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
from PIL import Image

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--difficulty", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resolution", type=float, default=0.05)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from robolab.tasks.manager_based.parkour.terrain_generator_cfg import ROUGH_TERRAINS_CFG
from robolab.terrains.terrain_generator import FiledTerrainGenerator


def main() -> None:
    cfg = copy.deepcopy(ROUGH_TERRAINS_CFG)
    terrain_names = list(cfg.sub_terrains)
    if not 0.0 <= args.difficulty <= 1.0:
        raise ValueError("--difficulty must be between 0 and 1")

    # Equal proportions plus one column per entry guarantees that every enabled
    # training terrain appears exactly once. The four Perlin entries intentionally
    # remain separate because training assigns different commands to their names.
    for sub_cfg in cfg.sub_terrains.values():
        sub_cfg.proportion = 1.0
        if hasattr(sub_cfg, "wall_prob"):
            sub_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]
    cfg.seed = args.seed
    cfg.curriculum = True
    cfg.num_rows = 1
    cfg.num_cols = len(terrain_names)
    cfg.difficulty_range = (args.difficulty, args.difficulty)
    cfg.use_cache = False

    generator = FiledTerrainGenerator(cfg)
    mesh = generator.terrain_mesh
    bounds = mesh.bounds
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]

    resolution = float(args.resolution)
    x_values = np.arange(min_x, max_x + 0.5 * resolution, resolution)
    y_values = np.arange(min_y, max_y + 0.5 * resolution, resolution)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    ray_origins = np.column_stack(
        (grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, max_z + 1.0))
    )
    ray_directions = np.tile((0.0, 0.0, -1.0), (grid_x.size, 1))
    locations, ray_indices, _ = mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=True
    )
    heights = np.full(grid_x.size, min_z, dtype=np.float64)
    if len(locations):
        np.maximum.at(heights, ray_indices, locations[:, 2])
    heights = np.flipud(heights.reshape(grid_y.shape))

    height_min = float(heights.min())
    height_max = float(heights.max())
    height_range = height_max - height_min
    normalized = (heights - height_min) / max(height_range, 1.0e-8)

    output_dir = Path(__file__).resolve().parents[2] / "data/robots/roboparty/bigreddog_plus/terrain_assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "training_terrains_difficulty_050.png"
    Image.fromarray(np.round(normalized * 65535.0).astype(np.uint16)).save(image_path)

    metadata_path = output_dir / "training_terrains_difficulty_050.txt"
    cell_origins = np.asarray(generator.terrain_origins)[0]
    lines = [
        "BigReddog training terrain export",
        f"difficulty={args.difficulty}",
        f"seed={args.seed}",
        f"bounds_min={bounds[0].tolist()}",
        f"bounds_max={bounds[1].tolist()}",
        f"height_min={height_min}",
        f"height_range={height_range}",
        "",
        "Terrain centers (Isaac coordinates before centering in MuJoCo):",
    ]
    lines.extend(f"{name}: {origin.tolist()}" for name, origin in zip(terrain_names, cell_origins))
    metadata_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {image_path}")
    print(f"Wrote {metadata_path}")
    print(f"MuJoCo hfield size: {(max_x-min_x)/2:.6f} {(max_y-min_y)/2:.6f} {height_range:.6f} {abs(height_min):.6f}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
