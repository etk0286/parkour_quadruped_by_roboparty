# BigReddog AMP motion data

The AMP task currently reads retargeted BigReddog `.pkl` files from:

`walk/`

Every pickle in that directory is discovered automatically and sampled with
equal weight. Git LFS pointer files are rejected; the actual LFS objects must
be downloaded first.

Each file must contain a dictionary with:

- `fps`: positive frames per second
- `loop_mode`: `0` (clamp) or `1` (wrap)
- `root_pos`: float array shaped `(frames, 3)`
- `root_rot`: quaternion array shaped `(frames, 4)` in `(w, x, y, z)` order
- `dof_pos`: float array shaped `(frames, 12)` in Isaac BigReddog joint order
- `key_body_pos`: float array shaped `(frames, 4, 3)` in this order:
  `FL_foot`, `FR_foot`, `RL_foot`, `RR_foot`

Use at least two frames. Joint and key-body ordering must exactly match the
Isaac Lab articulation and AMP discriminator configuration.
