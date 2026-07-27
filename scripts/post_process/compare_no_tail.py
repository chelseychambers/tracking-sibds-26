#!/usr/bin/env python3
"""Compare old vs new model on the combined Camera4 + Rat4 + Rat9 test set.

Produces:
  - Per-model metric summaries
  - Combined PCK curve with both models

Tail keypoints (tailbase, tail1, tail2, tail_tip) are excluded from all
benchmarking metrics.
"""
from __future__ import annotations

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
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

LABEL_DIR = Path("input/labels")
PRED_DIR  = Path("output/predicted_frames")
OUTPUT_FOLDER = Path("output/benchmark_plots_compare")

CONFIGS = [
    {
        "name": "Old model",
        "label_path": LABEL_DIR / "merged.json",
        "pred_path":  PRED_DIR / "mikey_pred/camera4_Rat4_Rat9_merged.json",
    },
    {
        "name": "New model",
        "label_path": LABEL_DIR / "merged.json",
        "pred_path":  PRED_DIR / "finetune_100ep/merged_test.json",
    },
]

PCK_CURVE_MAX = 0.30
PCK_NORM_THRESHOLDS = [0.05, 0.10, 0.20]
PCK_HEADLINE_THRESHOLD = 0.10   # the "primary metric" threshold, per the target below
PCK_TARGET = 0.90               # target: PCK@0.10 >= 0.90

# Keypoints excluded from every benchmark computation (tail is unreliable / not needed)
EXCLUDED_KEYPOINTS = {"tailbase", "tail1", "tail2", "tail_tip"}

COLORS = ["#2a78d6", "#e34948"]

COMPARISON_PLOT_NAME = "pck_curve_all.png"

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _get_keypoint_names(df: pd.DataFrame) -> list[str]:
    return sorted(
        col[:-2]
        for col in df.columns
        if col.endswith("_x")
        and f"{col[:-2]}_y" in df.columns
        and col[:-2] not in EXCLUDED_KEYPOINTS
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
            if name in EXCLUDED_KEYPOINTS:
                continue
            x = _parse(kp.get("x"))
            y = _parse(kp.get("y"))
            if x is None or y is None:
                dropped += 1
                continue
            kps[name] = {"x": x, "y": y, "score": float(kp.get("score") or 0.0)}
        cleaned[int(frame)] = {**rec, "keypoints": kps}
    return cleaned, dropped


def _save(fig: plt.Figure, name: str) -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_FOLDER / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


# ──────────────────────────────────────────────────────────────────────────────
# BBOX SCALE
# ──────────────────────────────────────────────────────────────────────────────

def compute_bbox_scales(df: pd.DataFrame) -> dict[int, float]:
    """Bbox diagonal is computed from ALL available keypoints (including tail,
    if present in the label file) so that the normalising scale reflects the
    animal's true extent — only the *benchmark metrics* exclude tail points."""
    scales: dict[int, float] = {}
    keys = sorted(
        col[:-2]
        for col in df.columns
        if col.endswith("_x") and f"{col[:-2]}_y" in df.columns
    )
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


# ──────────────────────────────────────────────────────────────────────────────
# CORE METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    df: pd.DataFrame,
    pred_map: dict,
    scales: dict[int, float],
) -> dict:
    keys = _get_keypoint_names(df)  # already excludes tail keypoints
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


# ──────────────────────────────────────────────────────────────────────────────
# LOAD + CLEAN helper
# ──────────────────────────────────────────────────────────────────────────────

def load_config(label_path: Path, pred_path: Path) -> tuple[pd.DataFrame, dict, dict]:
    df = load_label_json(label_path)
    raw_pred = load_prediction_map(pred_path)
    raw_pred_int = _cast_pred_map_keys(raw_pred)
    pred_map, _ = _clean_prediction_map(raw_pred_int)
    return df, pred_map


# ──────────────────────────────────────────────────────────────────────────────
# COMBINED COMPARISON FIGURE (PCK curve + PCK@0.10 summary bar)
# ──────────────────────────────────────────────────────────────────────────────

def plot_combined_pck_curve(all_results: list[dict]) -> None:
    """
    Single-panel PCK curve. AUC and PCK@0.10 both live in the legend text
    (one line per config, plus one line for the tau=0.10 reference), so
    there's exactly one place to look and nothing floating on top of the
    plot area.
    """
    ts = np.linspace(0, PCK_CURVE_MAX, 300)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for res, color in zip(all_results, COLORS):
        name = res["name"]
        narr = np.array(res["norm_errors"])
        if not len(narr):
            continue

        acc = np.array([(narr <= t).mean() * 100 for t in ts])
        auc = float(acc.mean() / 100)

        t_idx = np.searchsorted(ts, PCK_HEADLINE_THRESHOLD)
        pck_headline = float(acc[min(t_idx, len(acc) - 1)]) / 100

        ax.plot(ts, acc, color=color, linewidth=2,
                label=f"{name}  AUC={auc:.3f}  |  PCK@{PCK_HEADLINE_THRESHOLD:.2f}={pck_headline:.1%}")
        ax.scatter([PCK_HEADLINE_THRESHOLD], [pck_headline * 100],
                   color=color, zorder=5, s=45,
                   edgecolor="white", linewidth=0.8)

    ax.axvline(PCK_HEADLINE_THRESHOLD, color="#999999", linestyle="--",
              linewidth=1.1, label=f"\u03c4={PCK_HEADLINE_THRESHOLD:.2f}")

    ax.set_title("PCK curve comparison (bbox-normalised, tail excluded)", fontsize=13)
    ax.set_xlabel("Threshold \u03c4 (fraction of body-diagonal)")
    ax.set_ylabel("% predictions within \u03c4")
    ax.set_xlim(0, PCK_CURVE_MAX)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)

    fig.tight_layout()
    _save(fig, COMPARISON_PLOT_NAME)


# ──────────────────────────────────────────────────────────────────────────────
# PRINT SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(name: str, metrics: dict, n_frames: int, n_missing: int,
                  n_scales: int) -> None:
    print()
    print("=" * 65)
    print(f"  {name}")
    print("=" * 65)
    print(f"  excluded keypoints            : {sorted(EXCLUDED_KEYPOINTS)}")
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


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    all_results = []

    for cfg in CONFIGS:
        name = cfg["name"]
        label_path = cfg["label_path"]
        pred_path  = cfg["pred_path"]

        print(f"\nLoading [{name}]")
        print(f"  labels      : {label_path}")
        print(f"  predictions : {pred_path}")

        df, pred_map = load_config(label_path, pred_path)

        all_label_frames = sorted(int(float(f)) for f in df["frame"].tolist() if pd.notna(f))
        predicted_frames = set(pred_map.keys())
        missing_frames   = [f for f in all_label_frames if f not in predicted_frames]

        scales  = compute_bbox_scales(df)
        metrics = compute_metrics(df, pred_map, scales)

        print_summary(name, metrics, len(all_label_frames), len(missing_frames),
                      len(scales))

        # Store for comparison plot + summary table (n_frames/n_missing_frames
        # added so the comparison table below can report real frame counts
        # instead of double-counting keypoints).
        all_results.append({
            **metrics,
            "name": name,
            "n_frames": len(all_label_frames),
            "n_missing_frames": len(missing_frames),
        })

    # ── Comparison PCK curve + bar summary ────────────────────────────────
    print("\nGenerating comparison plot:")
    plot_combined_pck_curve(all_results)

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print(f"(tail keypoints excluded: {sorted(EXCLUDED_KEYPOINTS)})")
    print("=" * 80)
    hdr = f"  {'Metric':<35}"
    for r in all_results:
        hdr += f"  {r['name']:>18}"
    print(hdr)
    print("  " + "-" * (35 + 20 * len(all_results)))

    rows = [
        ("Frames labeled",              lambda r: f"{r['n_frames']}"),
        ("Frames missing prediction",   lambda r: f"{r['n_missing_frames']}"),
        ("Matched predictions",          lambda r: f"{r['matched_predictions']}"),
        ("Mean pixel error (px)",        lambda r: f"{r['mean_pixel_error']:.2f}"),
        ("RMSE (px)",                    lambda r: f"{r['rmse']:.2f}"),
        ("Median p50 (px)",              lambda r: f"{r['p50']:.2f}"),
        ("p90 (px)",                     lambda r: f"{r['p90']:.2f}"),
        ("PCK-AUC@50px",                lambda r: f"{r['pck_pixel_auc']:.4f}"),
    ]
    for label, fn in rows:
        line = f"  {label:<35}"
        for r in all_results:
            line += f"  {fn(r):>18}"
        print(line)

    # PCK thresholds
    print()
    for a in PCK_NORM_THRESHOLDS:
        line = f"  PCK@{a:.2f}".ljust(37)
        for r in all_results:
            v = r["pck_norm"][a]
            bar = "\u2588" * int(v * 10)
            line += f"  {v:>8.3f} {bar:<8}"
        print(line)

    print("=" * 80)
    print(f"\nPlots saved to: {OUTPUT_FOLDER}")
    print(f"Comparison figure: {OUTPUT_FOLDER / COMPARISON_PLOT_NAME}")
    print("Done.\n")


if __name__ == "__main__":
    main()