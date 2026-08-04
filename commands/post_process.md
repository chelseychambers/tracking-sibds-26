# Post-process Commands

# Benchmark a single model prediction
```bash
python scripts/post_process/benchmark.py \
  output/RTMPose/<run>/predictions.json \
  --labels input/labels/merged.json \
  --name "finetune"
```

# Merge labels and predictions before benchmarking
```bash
python scripts/post_process/merge_data.py \
  --labels-root input/labels \
  --labels-out input/labels/merged.json

python scripts/post_process/merge_data.py \
  --pred-root output/predicted_frames/finetune \
  --pred-out output/predicted_frames/finetune/merged.json
```

# Merge a subset of label files
```bash
python scripts/post_process/merge_data.py \
  --labels-root input/labels \
  --labels-out input/labels/merged_subset.json \
  --labels-include "*Camera4_stitched.json" "*RAT 11 FR1 10-02-25.json"
```

# Merge a subset of prediction files
```bash
python scripts/post_process/merge_data.py \
  --pred-root output/predicted_frames/finetune \
  --pred-out output/predicted_frames/finetune/merged_subset.json \
  --pred-include "*Camera4_stitched.json" "*RAT 11 FR1 10-02-25.json"
```

# Compare two model predictions
```bash
python scripts/post_process/compare_models.py \
  output/RTMPose/baseline/predictions.json \
  output/RTMPose/finetune/predictions.json \
  --labels input/labels/merged.json \
  --names "Baseline" "Finetune"
```

# Run the notebook wrapper interactively
```bash
# Open the notebook and update the prediction paths as needed.
# Then run the benchmark and comparison cells.
code scripts/post_process/benchmark_analysis.ipynb
```

# Notes
- `benchmark.py` is for evaluating a single prediction JSON against labels.
- `compare_models.py` is for comparing two prediction JSONs.
- `benchmark_analysis.ipynb` is an optional interactive notebook wrapper that runs both scripts.
