#!/usr/bin/env python3
"""Evaluate a single model's predictions against ground-truth labels.

Computes pixel-error metrics, PCK curves, and per-keypoint breakdowns.
For comparing two models, see compare_models.py.

Usage:
    python scripts/post_process/benchmark.py predictions.json --labels labels.json
    python scripts/post_process/benchmark.py merged_test.json --name "finetune_100ep"
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from modules.manual_correction_utils import (
    get_prediction_point,
    load_label_json,
    load_prediction_map,
)


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

PCK_NORM_THRESHOLDS = [0.05, 0.10, 0.20]
OUTPUT_FOLDER = Path("output/benchmark")

PCK_CURVE_MAX = 0.30
PCK_HEADLINE_THRESHOLD = 0.10
PCK_TARGET = 0.90


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _get_keypoint_names(df: pd.DataFrame) -> list[str]:
    return sorted(
        col[:-2]
        for col in df.columns
        if col.endswith("_x") and f"{col[:-2]}_y" in df.columns
    )


def _parse(v: object) -> Optional[float]:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _cast_pred_map_keys(raw: dict) -> dict:
    return {int(k): v for k, v in raw.items()}


def _clean_prediction_map(raw: dict) -> tuple[dict, int]:
    cleaned: dict = {}
    dropped = 0
    for frame, rec in raw.items():
        kps: dict = {}
        for name, kp in (rec.get("keypoints") or {}).items():
            x = _parse(kp.get("x"))
            y = _parse(kp.get("y"))
            if x is None or y is None:
                dropped += 1
                continue
            kps[name] = {"x": x, "y": y, "score": float(kp.get("score") or 0.0)}
        cleaned[int(frame)] = {**rec, "keypoints": kps}
    return cleaned, dropped


def compute_bbox_scales(df: pd.DataFrame) -> dict[int, float]:
    scales: dict[int, float] = {}
    keys = _get_keypoint_names(df)
    for row in df.itertuples(index=False):
        frame = int(getattr(row, "frame"))
        xs, ys = [], []
        for kp in keys:
            x = _parse(getattr(row, f"{kp}_x", None))
            y = _parse(getattr(row, f"{kp}_y", None))
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        if len(xs) >= 2:
            diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            if diag > 1e-6:
                scales[frame] = diag
    return scales


def compute_metrics(
    df: pd.DataFrame,
    pred_map: dict,
    scales: dict[int, float],
) -> dict:
    keys = _get_keypoint_names(df)

    per_kp_errors: dict[str, list[float]] = {k: [] for k in keys}
    raw_errors:    list[float] = []
    norm_errors:   list[float] = []

    total_visible       = 0
    total_missing_kp    = 0
    total_missing_frame = 0

    for row in df.itertuples(index=False):
        frame = int(getattr(row, "frame"))
        pred  = pred_map.get(frame)

        if pred is None:
            for kp in keys:
                if _parse(getattr(row, f"{kp}_x", None)) is not None:
                    total_missing_frame += 1
            continue

        scale = scales.get(frame)

        for kp in keys:
            lx = _parse(getattr(row, f"{kp}_x", None))
            ly = _parse(getattr(row, f"{kp}_y", None))
            if lx is None or ly is None:
                continue

            total_visible += 1

            pp = get_prediction_point(pred_map, frame, kp)
            if pp is None:
                total_missing_kp += 1
                continue

            dist = math.hypot(float(pp[0]) - lx, float(pp[1]) - ly)
            raw_errors.append(dist)
            per_kp_errors[kp].append(dist)
            if scale:
                norm_errors.append(dist / scale)

    matched = len(raw_errors)
    arr     = np.array(raw_errors,  dtype=float) if raw_errors  else np.array([])
    narr    = np.array(norm_errors, dtype=float) if norm_errors else np.array([])

    return {
        "total_visible_labels":      total_visible,
        "matched_predictions":       matched,
        "missing_kp_predictions":    total_missing_kp,
        "missing_frame_predictions": total_missing_frame,

        "mean_pixel_error": float(arr.mean())                   if matched else 0.0,
        "rmse":             float(np.sqrt(np.mean(arr ** 2)))   if matched else 0.0,
        "p50":              float(np.median(arr))               if matched else 0.0,
        "p90":              float(np.quantile(arr, 0.90))       if matched else 0.0,

        "pck_pixel_auc": float(
            np.mean([(arr <= t).mean() for t in range(0, 51)])
        ) if matched else 0.0,

        "pck_norm": {
            a: float((narr <= a).mean()) if len(narr) else 0.0
            for a in PCK_NORM_THRESHOLDS
        },

        "raw_errors":  raw_errors,
        "norm_errors": norm_errors,
        "per_kp":      per_kp_errors,
    }


def _save(fig: plt.Figure, name: str) -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_FOLDER / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


def load_all(pred_path: str, label_df: pd.DataFrame
             ) -> tuple[pd.DataFrame, dict, dict, dict]:
    raw = load_prediction_map(Path(pred_path))
    pred_map, _ = _clean_prediction_map(_cast_pred_map_keys(raw))
    scales = compute_bbox_scales(label_df)
    metrics = compute_metrics(label_df, pred_map, scales)
    return label_df, pred_map, scales, metrics


def print_summary(name: str, metrics: dict, n_frames: int, n_missing: int,
                  n_scales: int) -> None:
    print()
    print("=" * 65)
    print(f"  {name}")
    print("=" * 65)
    print(f"  visible label keypoints      : {metrics['total_visible_labels']}")
    print(f"  matched (label + prediction) : {metrics['matched_predictions']}")
    print(f"  missing keypoint predictions : {metrics['missing_kp_predictions']}")
    print(f"  frames with no prediction    : {n_missing}")
    print(f"  bbox-scale frames available  : {n_scales}")
    print()
    print(f"  mean pixel error             : {metrics['mean_pixel_error']:.2f} px")
    narr = np.array(metrics['norm_errors']) if metrics['norm_errors'] else np.array([0.0])
    print(f"  mean bbox-normalised error   : {float(np.mean(narr)):.2%}  (error / body diagonal)")
    print(f"  RMSE                         : {metrics['rmse']:.2f} px")
    print(f"  median (p50)                 : {metrics['p50']:.2f} px")
    print(f"  p90                          : {metrics['p90']:.2f} px")
    print(f"  PCK-AUC@50px (pixel, diag.)  : {metrics['pck_pixel_auc']:.4f}")
    print()
    print("  PCK (bbox-normalised) -- PRIMARY metric")
    print(f"  Target: PCK@{PCK_HEADLINE_THRESHOLD:.2f} >= {PCK_TARGET:.2f}")
    for a in PCK_NORM_THRESHOLDS:
        bar = "\u2588" * int(metrics["pck_norm"][a] * 20)
        print(f"    PCK@{a:.2f}  : {metrics['pck_norm'][a]:.3f}  {bar}")
    print("=" * 65)


def plot_pck_curve_single(metrics: dict, name: str) -> None:
    ts = np.linspace(0, PCK_CURVE_MAX, 300)
    narr = np.array(metrics["norm_errors"])
    if not len(narr):
        return

    acc = np.array([(narr <= t).mean() * 100 for t in ts])
    auc = float(acc.mean() / 100)
    t_idx = np.searchsorted(ts, PCK_HEADLINE_THRESHOLD)
    pck_h = float(acc[min(t_idx, len(acc) - 1)]) / 100

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ts, acc, color="#2a78d6", linewidth=2,
            label=f"AUC={auc:.3f}  |  PCK@{PCK_HEADLINE_THRESHOLD:.2f}={pck_h:.1%}")
    ax.fill_between(ts, acc, alpha=0.12, color="#2a78d6")
    ax.scatter([PCK_HEADLINE_THRESHOLD], [pck_h * 100], color="#2a78d6", zorder=5, s=45,
               edgecolor="white", linewidth=0.8)

    ax.axvline(PCK_HEADLINE_THRESHOLD, color="#999999", linestyle="--", linewidth=1.1,
               label=f"\u03c4={PCK_HEADLINE_THRESHOLD:.2f}")
    ax.set_title(f"PCK curve (bbox-normalised) -- {name}", fontsize=12)
    ax.set_xlabel("Threshold \u03c4 (fraction of body-diagonal)")
    ax.set_ylabel("% predictions within \u03c4")
    ax.set_xlim(0, PCK_CURVE_MAX)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    fig.tight_layout()
    safe = name.replace(" ", "_")
    _save(fig, f"pck_curve_{safe}.png")


def plot_error_histogram(errors: list[float], name: str) -> None:
    if not errors:
        print("  [SKIP] error_histogram -- no errors")
        return

    arr    = np.array(errors)
    n_bins = min(60, max(10, int(np.ceil(np.log2(len(arr)) + 1))))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(arr, bins=n_bins, color="#2a78d6", alpha=0.75,
            edgecolor="white", linewidth=0.4)

    med = float(np.median(arr))
    p90 = float(np.quantile(arr, 0.90))
    ax.axvline(med, color="#1baf7a", linestyle="--", linewidth=1.4,
               label=f"median  {med:.1f} px")
    ax.axvline(p90, color="#e34948", linestyle="--", linewidth=1.4,
               label=f"p90  {p90:.1f} px")

    ax.set_title(f"Pixel error distribution -- {name}", fontsize=12)
    ax.set_xlabel("Pixel error")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    safe = name.replace(" ", "_")
    _save(fig, f"error_histogram_{safe}.png")


def plot_sample_count_vs_error(per_kp: dict[str, list[float]], name: str) -> None:
    names = [k for k, v in per_kp.items() if v]
    if not names:
        print("  [SKIP] sample_count_vs_error -- no data")
        return

    ns    = np.array([len(per_kp[k]) for k in names])
    means = np.array([float(np.mean(per_kp[k])) for k in names])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ns, means, color="#2a78d6", alpha=0.8, s=60, zorder=3)

    for label, n, m in zip(names, ns, means):
        ax.annotate(label, (n, m), textcoords="offset points", xytext=(5, 3),
                    fontsize=7, color="#333333")

    med_n = float(np.median(ns))
    med_m = float(np.median(means))
    ax.axvline(med_n, color="#aaaaaa", linestyle=":", linewidth=1)
    ax.axhline(med_m, color="#aaaaaa", linestyle=":", linewidth=1)

    ax.set_title(f"Sample count vs mean pixel error -- {name}", fontsize=12)
    ax.set_xlabel("Number of matched instances (n)")
    ax.set_ylabel("Mean pixel error (px)")
    ax.grid(True, linestyle=":", alpha=0.4)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(xlim[0] + (med_n - xlim[0]) * 0.05, ylim[1] * 0.97,
            "rare & bad\n(add labels)", fontsize=7, color="#e34948",
            va="top", ha="left")
    ax.text(med_n + (xlim[1] - med_n) * 0.05, ylim[1] * 0.97,
            "common & bad\n(model problem)", fontsize=7, color="#e34948",
            va="top", ha="left")
    ax.text(xlim[0] + (med_n - xlim[0]) * 0.05, ylim[0] + (med_m - ylim[0]) * 0.1,
            "rare & ok", fontsize=7, color="#1baf7a", va="bottom", ha="left")
    ax.text(med_n + (xlim[1] - med_n) * 0.05, ylim[0] + (med_m - ylim[0]) * 0.1,
            "common & ok", fontsize=7, color="#1baf7a", va="bottom", ha="left")

    safe = name.replace(" ", "_")
    _save(fig, f"sample_count_vs_error_{safe}.png")


def print_per_keypoint_table(name: str, df: pd.DataFrame, pred_map: dict,
                             scales: dict[int, float]) -> None:
    keys = _get_keypoint_names(df)
    per_kp_norm: dict[str, list[float]] = {k: [] for k in keys}

    for row in df.itertuples(index=False):
        frame = int(getattr(row, "frame"))
        pred  = pred_map.get(frame)
        if pred is None:
            continue
        scale = scales.get(frame)
        if not scale:
            continue
        for kp in keys:
            lx = _parse(getattr(row, f"{kp}_x", None))
            ly = _parse(getattr(row, f"{kp}_y", None))
            if lx is None or ly is None:
                continue
            pp = get_prediction_point(pred_map, frame, kp)
            if pp is None:
                continue
            dist = math.hypot(float(pp[0]) - lx, float(pp[1]) - ly)
            per_kp_norm[kp].append(dist / scale)

    raw_per_kp: dict[str, list[float]] = {k: [] for k in keys}
    for row in df.itertuples(index=False):
        frame = int(getattr(row, "frame"))
        pred  = pred_map.get(frame)
        if pred is None:
            continue
        for kp in keys:
            lx = _parse(getattr(row, f"{kp}_x", None))
            ly = _parse(getattr(row, f"{kp}_y", None))
            if lx is None or ly is None:
                continue
            pp = get_prediction_point(pred_map, frame, kp)
            if pp is None:
                continue
            dist = math.hypot(float(pp[0]) - lx, float(pp[1]) - ly)
            raw_per_kp[kp].append(dist)

    sorted_kps = sorted(
        ((k, float(np.mean(raw_per_kp[k])), len(raw_per_kp[k]))
         for k, v in raw_per_kp.items() if v),
        key=lambda x: x[1],
        reverse=True,
    )

    print(f"\n  Per-keypoint breakdown -- {name}")
    print(f"  {'keypoint':<30}  {'mean px':>7}  {'n':>5}  {'PCK@0.10 (bbox-norm)':>20}")
    print(f"  {'-'*30}  {'-'*7}  {'-'*5}  {'-'*20}")

    for kp_name, mean_err, count in sorted_kps:
        norm_vals = per_kp_norm.get(kp_name, [])
        pck10_norm = float((np.array(norm_vals) <= 0.10).mean()) if norm_vals else float("nan")
        print(f"  {kp_name:<30}  {mean_err:7.2f}  {count:5d}  {pck10_norm:20.3f}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Evaluate a single model's predictions")
    parser.add_argument("model", help="Path to prediction JSON file")
    parser.add_argument("--labels", default="input/labels/merged.json", help="Path to label JSON")
    parser.add_argument("--name", default="Model", help="Model name for display")
    args = parser.parse_args()

    df = load_label_json(Path(args.labels))

    print(f"\nLoading [{args.name}]")
    print(f"  predictions : {args.model}")

    _, pred_map, scales, metrics = load_all(args.model, df)

    all_label_frames = sorted(int(float(f)) for f in df["frame"].tolist() if pd.notna(f))
    predicted_frames = set(pred_map.keys())
    missing_frames = [f for f in all_label_frames if f not in predicted_frames]

    print_summary(args.name, metrics, len(all_label_frames), len(missing_frames), len(scales))
    print_per_keypoint_table(args.name, df, pred_map, scales)

    print("\nGenerating plots:")
    plot_pck_curve_single(metrics, args.name)
    plot_error_histogram(metrics["raw_errors"], args.name)
    plot_sample_count_vs_error(metrics["per_kp"], args.name)

    print(f"\nPlots saved to: {OUTPUT_FOLDER}")
    print("Done.\n")


if __name__ == "__main__":
    main()
