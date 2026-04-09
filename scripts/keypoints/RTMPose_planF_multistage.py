#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def latest_run_dir(output_root: Path, prefix: str) -> Path | None:
    if not output_root.exists():
        return None
    candidates = [p for p in output_root.iterdir() if p.is_dir() and p.name.startswith(prefix + "_")]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name.split("_")[-1])
    return candidates[-1]


def run_with_args(args: list[str]) -> None:
    target = Path(__file__).resolve().parent / "RTMPose.py"
    sys.argv = [str(target), *args]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    output_root = Path("output") / "RTMPose"
    stage1_prefix = "planF_stage1"
    stage2_prefix = "planF_stage2"
    stage3_prefix = "planF_stage3"

    run_with_args([
        "--config-overwrite",
        "input/RTMPose/config_no_weak.yaml",
        "--model-config",
        "input/RTMPose/model_rtmpose_x.yaml",
        "--init-checkpoint",
        "output/RTMPose/no_weak_20260328_174401/checkpoint_best.pt",
        "--eval-every-n-epoch",
        "1",
        "train",
        "--prefix",
        stage1_prefix,
        "--epochs",
        "10",
    ])

    stage1_dir = latest_run_dir(output_root, stage1_prefix)
    if stage1_dir is None:
        raise SystemExit(1)
    stage1_ckpt = stage1_dir / "checkpoint_best.pt"

    run_with_args([
        "--config-overwrite",
        "input/RTMPose/config_with_weak.yaml",
        "--model-config",
        "input/RTMPose/model_rtmpose_x.yaml",
        "--init-checkpoint",
        str(stage1_ckpt),
        "--no-mask-on-labeled",
        "--weak-sample-weight",
        "0.1",
        "--lambda-mask",
        "0.1",
        "--train-weak-samples-per-epoch",
        "300",
        "--mask-loss-mode",
        "variance",
        "--mask-variance-weight",
        "0.2",
        "--eval-every-n-epoch",
        "1",
        "train",
        "--prefix",
        stage2_prefix,
        "--epochs",
        "20",
    ])

    stage2_dir = latest_run_dir(output_root, stage2_prefix)
    if stage2_dir is None:
        raise SystemExit(1)
    stage2_ckpt = stage2_dir / "checkpoint_best.pt"

    run_with_args([
        "--config-overwrite",
        "input/RTMPose/config_no_weak.yaml",
        "--model-config",
        "input/RTMPose/model_rtmpose_x.yaml",
        "--init-checkpoint",
        str(stage2_ckpt),
        "--eval-every-n-epoch",
        "1",
        "train",
        "--prefix",
        stage3_prefix,
        "--epochs",
        "10",
    ])
