#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    extra_global = [
        "--no-mask-on-labeled",
        "--weak-sample-weight",
        "0.1",
    ]
    extra_train = [
        "--lambda-mask",
        "0.1",
        "--train-weak-samples-per-epoch",
        "300",
    ]
    target = Path(__file__).resolve().parent / "RTMPose.py"
    user_args = list(sys.argv[1:])
    if "train" in user_args:
        idx = user_args.index("train")
        user_args = user_args[: idx + 1] + extra_train + user_args[idx + 1 :]
    sys.argv = [str(target), *extra_global, *user_args]
    runpy.run_path(str(target), run_name="__main__")
