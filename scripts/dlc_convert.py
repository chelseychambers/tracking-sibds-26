#!/usr/bin/env python3
# Convert DLC labeled-data CSV files into per-video JSON label files
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.label_csv_utils import load_keypoints


def _discover_csv_files(labeled_data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in labeled_data_dir.rglob("CollectedData*.csv")
        if path.is_file()
    )


def _extract_keypoint_names(df) -> list[str]:
    names: set[str] = set()
    for col in df.columns:
        if col == "frame":
            continue
        if col.endswith("_x") or col.endswith("_y"):
            names.add(col.rsplit("_", 1)[0])
    return sorted(names)


def _to_float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _to_int_or_none(value: Any) -> int | None:
    numeric = _to_float_or_none(value)
    if numeric is None:
        return None
    return int(round(numeric))


def _convert_dataframe(df) -> list[dict[str, object]]:
    keypoints = _extract_keypoint_names(df)
    frame_records: list[dict[str, object]] = []

    for _, row in df.sort_values("frame").iterrows():
        frame_idx = int(float(row["frame"]))
        labels: dict[str, list[int | None]] = {}

        for keypoint in keypoints:
            x_val = _to_int_or_none(row.get(f"{keypoint}_x"))
            y_val = _to_int_or_none(row.get(f"{keypoint}_y"))
            visible = 1 if x_val is not None and y_val is not None else 0
            labels[keypoint] = [visible, x_val, y_val]

        frame_records.append({"frame_idx": frame_idx, "labels": labels})

    return frame_records


def convert_labeled_data(input_dir: Path, output_dir: Path) -> int:
    csv_files = _discover_csv_files(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for csv_path in csv_files:
        video_name = csv_path.parent.name
        keypoint_df = load_keypoints(csv_path)
        payload = _convert_dataframe(keypoint_df)

        out_path = output_dir / f"{video_name}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        converted += 1
        print(f"Converted {csv_path} -> {out_path}")

    return converted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert DLC labels in input/labeled-data into per-video JSON files "
            "under input/labels."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input/labeled-data"),
        help="Folder containing DLC labeled-data directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("input/labels"),
        help="Destination folder for per-video JSON labels.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input labeled-data directory not found: {args.input_dir}")

    converted = convert_labeled_data(args.input_dir, args.output_dir)
    print(f"Done. Converted {converted} video label files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
