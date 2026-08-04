#!/usr/bin/env python3
"""Convert JSON label files to DLC CSV format for RTMPose training.

The label interface produces JSON files with [visible, x, y] per keypoint.
RTMPose training expects DeepLabCut (DLC) CSV format, which uses a 3-row
header (scorer, bodypart, coordinate) and one row per frame. This script
bridges the two formats.

Usage:
    python scripts/json_to_dlc.py
    python scripts/json_to_dlc.py --labels-dir path/to/jsons --output-dir path/to/csvs
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_LABELS_DIR = PROJECT_ROOT / "input" / "labels"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "input" / "labeled-data"
DEFAULT_BODYPARTS = [
    "head", "nose", "spine1", "spine2", "spine3", "tailbase",
    "tail1", "tail2", "tail_tip",
    "L_hip", "L_backpaw", "R_backpaw",
    "L_shoulder", "R_frontpaw", "R_shoulder",
    "R_hip", "R_knee", "L_knee", "L_frontpaw",
]
SCORER = "rats"
FIRST_COL_VALUE = "labeled-data"


def load_label_json(label_json_path: Path) -> pd.DataFrame:
    payload = json.loads(label_json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {label_json_path}, got {type(payload).__name__}")

    rows = []
    for item in payload:
        frame_idx = int(item["frame_idx"])
        row = {"frame": frame_idx}
        labels = item.get("labels") or {}
        for keypoint, value in labels.items():
            visible = int(value[0]) if value else 0
            x = value[1] if len(value) > 1 else None
            y = value[2] if len(value) > 2 else None
            row[f"{keypoint}_x"] = float(x) if visible and x is not None else float("nan")
            row[f"{keypoint}_y"] = float(y) if visible and y is not None else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_dlc_csv(df: pd.DataFrame, video_name: str, bodyparts: list[str], csv_path: Path) -> None:
    frame_col = "frame"
    keypoint_columns = []
    for bp in bodyparts:
        keypoint_columns.extend([f"{bp}_x", f"{bp}_y"])

    n_data_cols = 3 + len(keypoint_columns)

    header_row0 = [SCORER] + [""] * (n_data_cols - 1)
    header_row1 = ["", "", ""]
    for bp in bodyparts:
        header_row1.extend([bp, bp])
    header_row2 = ["", "", ""]
    for _ in bodyparts:
        header_row2.extend(["x", "y"])

    max_frame = int(pd.to_numeric(df[frame_col], errors="coerce").max())
    frame_width = max(1, len(str(max_frame)))

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [header_row0, header_row1, header_row2]

    for _, row in df.sort_values(frame_col).iterrows():
        frame_idx = int(float(row[frame_col]))
        data_row = [
            FIRST_COL_VALUE,
            video_name,
            f"img{str(frame_idx).zfill(frame_width)}.png",
        ]
        for col_name in keypoint_columns:
            val = row.get(col_name)
            if pd.isna(val):
                data_row.append("")
            else:
                data_row.append(str(int(round(float(val)))))
        rows.append(data_row)

    with open(csv_path, "w", newline="\n") as f:
        for row in rows:
            f.write(",".join(row) + "\n")


def convert_all(labels_dir: Path, output_dir: Path, bodyparts: list[str]) -> int:
    json_files = sorted(labels_dir.glob("*.json"))
    converted = 0
    for json_path in json_files:
        video_name = json_path.stem
        df = load_label_json(json_path)
        csv_path = output_dir / video_name / "CollectedData_rats.csv"
        build_dlc_csv(df, video_name, bodyparts, csv_path)
        converted += 1
        print(f"Converted {json_path.name} -> {csv_path} ({len(df)} frames)")
    return converted


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert JSON labels to DLC CSV format")
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bodyparts", nargs="+", default=DEFAULT_BODYPARTS)
    args = parser.parse_args()

    count = convert_all(args.labels_dir, args.output_dir, args.bodyparts)
    print(f"Done. Converted {count} label files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
