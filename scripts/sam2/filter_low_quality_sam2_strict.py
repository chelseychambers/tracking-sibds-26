#!/usr/bin/env python3
from __future__ import annotations

import os

from scripts.sam2.filter_low_quality_sam2 import main


if __name__ == "__main__":
    os.environ.setdefault("SAM2_MASK_AREA_LOWER_QUANTILE", "0.20")
    os.environ.setdefault("SAM2_MASK_AREA_UPPER_QUANTILE", "0.80")
    os.environ.setdefault("SAM2_MIN_MASK_AREA_PIXELS", "1500")
    raise SystemExit(main())
