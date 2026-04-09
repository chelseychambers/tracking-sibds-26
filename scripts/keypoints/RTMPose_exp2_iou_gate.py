#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    extra = ["--weak-mask-iou-thresh", "0.5"]
    target = Path(__file__).resolve().parent / "RTMPose.py"
    sys.argv = [str(target), *extra, *sys.argv[1:]]
    runpy.run_path(str(target), run_name="__main__")
