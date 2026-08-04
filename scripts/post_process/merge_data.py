#!/usr/bin/env python3
"""Merge per-video label and prediction JSON files into single benchmark-ready JSONs."""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


def load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON file: {path}") from exc


def collect_json_files(
    root: Path,
    exclude_names: set[str],
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"JSON root directory not found: {root}")

    paths = sorted(root.glob("*.json"))
    if include_patterns:
        paths = [
            path
            for path in paths
            if any(fnmatch.fnmatch(path.name, pattern) for pattern in include_patterns)
        ]
    if exclude_patterns:
        paths = [
            path
            for path in paths
            if not any(fnmatch.fnmatch(path.name, pattern) for pattern in exclude_patterns)
        ]

    paths = [path for path in paths if path.name not in exclude_names]
    if not paths:
        raise FileNotFoundError(
            f"No JSON files found under {root} after exclusions: {sorted(exclude_names)}"
        )
    return paths


def merge_json_files(paths: Iterable[Path]) -> list[dict]:
    merged: list[dict] = []
    for path in paths:
        payload = load_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON list in {path}, got {type(payload).__name__}")
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError(f"Expected list of objects in {path}")
            if "frame_idx" not in item:
                raise ValueError(f"Missing 'frame_idx' in item from {path}")
            merged.append(item)
    merged.sort(key=lambda item: int(item["frame_idx"]))
    return merged


def validate_unique_frame_idx(payload: list[dict]) -> None:
    counts = Counter(int(item["frame_idx"]) for item in payload)
    duplicates = [frame for frame, count in counts.items() if count > 1]
    if duplicates:
        sample = sorted(duplicates)[:20]
        raise ValueError(
            f"Duplicate frame_idx values found in merged payload: {len(duplicates)} duplicates. "
            f"Sample duplicates: {sample}"
        )


def write_json(payload: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Written {len(payload)} entries to {out_path}")


def merge_directory(
    root: Path,
    out_path: Path,
    default_excludes: set[str],
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> None:
    exclude_names = set(default_excludes)
    if out_path.name:
        exclude_names.add(out_path.name)
    paths = collect_json_files(root, exclude_names, include_patterns, exclude_patterns)
    print(f"Merging {len(paths)} files from {root}")
    payload = merge_json_files(paths)
    validate_unique_frame_idx(payload)
    write_json(payload, out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-video label/prediction JSON files into a single JSON file."
    )
    parser.add_argument("--labels-root", type=Path,
                        help="Directory containing per-video label JSON files.")
    parser.add_argument("--labels-out", type=Path,
                        help="Output path for merged labels JSON.")
    parser.add_argument("--pred-root", type=Path,
                        help="Directory containing per-video prediction JSON files.")
    parser.add_argument("--pred-out", type=Path,
                        help="Output path for merged predictions JSON.")
    parser.add_argument("--labels-include", nargs="*",
                        help="Glob patterns for label filenames to include.")
    parser.add_argument("--labels-exclude", nargs="*",
                        help="Glob patterns for label filenames to exclude.")
    parser.add_argument("--pred-include", nargs="*",
                        help="Glob patterns for prediction filenames to include.")
    parser.add_argument("--pred-exclude", nargs="*",
                        help="Glob patterns for prediction filenames to exclude.")
    parser.add_argument("--json-files", type=Path, nargs="*",
                        help="Explicit JSON files to merge into a single output.")
    parser.add_argument("--out", type=Path,
                        help="Output path for --json-files mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.json_files and args.out:
        payload = merge_json_files(args.json_files)
        validate_unique_frame_idx(payload)
        write_json(payload, args.out)
        return

    if not args.labels_root and not args.pred_root:
        raise SystemExit("Please provide at least --labels-root or --pred-root, or use --json-files with --out.")

    if args.labels_root:
        if not args.labels_out:
            raise SystemExit("--labels-out is required when --labels-root is provided.")
        merge_directory(
            args.labels_root,
            args.labels_out,
            {"merged.json"},
            args.labels_include,
            args.labels_exclude,
        )

    if args.pred_root:
        if not args.pred_out:
            raise SystemExit("--pred-out is required when --pred-root is provided.")
        merge_directory(
            args.pred_root,
            args.pred_out,
            {args.pred_out.name},
            args.pred_include,
            args.pred_exclude,
        )


if __name__ == "__main__":
    main()
