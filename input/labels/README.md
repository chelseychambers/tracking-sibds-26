# Label Format

Each `.json` file is a list of labeled frames:

```json
[
  {
    "frame_idx": 6,
    "labels": {
      "nose": [1, 984, 405],
      "tail_tip": [0, null, null]
    }
  }
]
```

`frame_idx`
- Zero-based frame index in the source video.

`labels`
- Maps each keypoint name to `[visible, x, y]`.
- `visible = 1`: keypoint is labeled at pixel `(x, y)`.
- `visible = 0`: keypoint is not visible / not labeled, stored as `[0, null, null]`.

Keypoints used here:
`nose`, `head`, `spine1`, `spine2`, `spine3`, `tailbase`, `tail1`, `tail2`, `tail_tip`, `L_shoulder`, `L_frontpaw`, `R_shoulder`, `R_frontpaw`, `L_hip`, `L_knee`, `L_backpaw`, `R_hip`, `R_knee`, `R_backpaw`.
