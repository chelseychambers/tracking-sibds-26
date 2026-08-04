#!/usr/bin/env python3
"""Compare two models' predictions with t-tests and difference plots.

Requires exactly two models. For single-model evaluation, see benchmark.py.

Usage:
    python scripts/post_process/compare_models.py old.json new.json \\
        --labels labels.json \\
        --names "Old" "New"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from modules.manual_correction_utils import load_label_json

from scripts.post_process.benchmark import (
    PCK_NORM_THRESHOLDS,
    OUTPUT_FOLDER,
    _save,
    load_all,
    print_summary,
    print_per_keypoint_table,
)

COLORS = ["#2a78d6", "#e34948", "#1baf7a", "#f5a623", "#9b59b6"]

PCK_CURVE_MAX = 0.30
PCK_HEADLINE_THRESHOLD = 0.10


def plot_pck_curve_comparison(results: list[dict]) -> None:
    ts = np.linspace(0, PCK_CURVE_MAX, 300)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for res, color in zip(results, COLORS[:len(results)]):
        name = res["name"]
        narr = np.array(res["norm_errors"])
        if not len(narr):
            continue
        acc = np.array([(narr <= t).mean() * 100 for t in ts])
        auc = float(acc.mean() / 100)
        t_idx = np.searchsorted(ts, PCK_HEADLINE_THRESHOLD)
        pck_h = float(acc[min(t_idx, len(acc) - 1)]) / 100
        ax.plot(ts, acc, color=color, linewidth=2,
                label=f"{name}  AUC={auc:.3f}  |  PCK@{PCK_HEADLINE_THRESHOLD:.2f}={pck_h:.1%}")
        ax.scatter([PCK_HEADLINE_THRESHOLD], [pck_h * 100], color=color, zorder=5, s=45,
                   edgecolor="white", linewidth=0.8)

    ax.axvline(PCK_HEADLINE_THRESHOLD, color="#999999", linestyle="--", linewidth=1.1,
               label=f"\u03c4={PCK_HEADLINE_THRESHOLD:.2f}")
    ax.set_title("PCK curve comparison (bbox-normalised)", fontsize=13)
    ax.set_xlabel("Threshold \u03c4 (fraction of body-diagonal)")
    ax.set_ylabel("% predictions within \u03c4")
    ax.set_xlim(0, PCK_CURVE_MAX)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    fig.tight_layout()
    _save(fig, "pck_curve_comparison.png")


def plot_keypoint_diff(old_kp: dict[str, list[float]], new_kp: dict[str, list[float]],
                       all_kps: list[str]) -> None:
    names, diffs, old_means, new_means = [], [], [], []
    for kp in all_kps:
        o = old_kp.get(kp, [])
        n = new_kp.get(kp, [])
        if not o or not n:
            continue
        om = float(np.mean(o))
        nm = float(np.mean(n))
        names.append(kp)
        diffs.append(om - nm)
        old_means.append(om)
        new_means.append(nm)

    order = np.argsort(diffs)[::-1]
    names = [names[i] for i in order]
    diffs = [diffs[i] for i in order]

    colors = ["#2a78d6" if d >= 0 else "#e34948" for d in diffs]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.45 + 1)))
    bars = ax.barh(names, diffs, color=colors, alpha=0.8, height=0.65)

    for bar, d in zip(bars, diffs):
        w = bar.get_width()
        offset = max(abs(d) for d in diffs) * 0.01
        label_x = w + offset if w >= 0 else w - offset
        ha = "left" if w >= 0 else "right"
        ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                f"{d:+.1f} px", va="center", ha=ha, fontsize=7, color="#52514e")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Mean pixel error difference (old - new)\nBlue = new is better, Red = old is better",
                 fontsize=11)
    ax.set_xlabel("Improvement in mean pixel error (px)")
    ax.invert_yaxis()
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    xlim = ax.get_xlim()
    pad = max(abs(xlim[0]), abs(xlim[1])) * 0.15
    ax.set_xlim(xlim[0] - pad, xlim[1] + pad)
    fig.tight_layout()
    _save(fig, "keypoint_error_diff.png")


def print_comparison_table(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    hdr = f"  {'Metric':<35}"
    for r in results:
        hdr += f"  {r['name']:>18}"
    print(hdr)
    print("  " + "-" * (35 + 20 * len(results)))

    table_rows = [
        ("Frames labeled",            lambda r: f"{r['n_frames']}"),
        ("Frames missing prediction", lambda r: f"{r['n_missing_frames']}"),
        ("Matched predictions",       lambda r: f"{r['matched_predictions']}"),
        ("Mean pixel error (px)",     lambda r: f"{r['mean_pixel_error']:.2f}"),
        ("RMSE (px)",                 lambda r: f"{r['rmse']:.2f}"),
        ("Median p50 (px)",           lambda r: f"{r['p50']:.2f}"),
        ("p90 (px)",                  lambda r: f"{r['p90']:.2f}"),
        ("PCK-AUC@50px",             lambda r: f"{r['pck_pixel_auc']:.4f}"),
    ]
    for label, fn in table_rows:
        line = f"  {label:<35}"
        for r in results:
            line += f"  {fn(r):>18}"
        print(line)

    print()
    for a in PCK_NORM_THRESHOLDS:
        line = f"  PCK@{a:.2f}".ljust(37)
        for r in results:
            v = r["pck_norm"][a]
            bar = "\u2588" * int(v * 10)
            line += f"  {v:>8.3f} {bar:<8}"
        print(line)
    print("=" * 80)


def run_t_tests(old_kp: dict[str, list[float]], new_kp: dict[str, list[float]],
                old_name: str, new_name: str) -> None:
    all_kps = sorted(set(old_kp) | set(new_kp))

    print()
    header = "{:<20}  {:>5} {:>5}  {:>9} {:>9}  {:>8}  {:>10}  {}".format(
        "keypoint", "old_n", "new_n", "old_mean", "new_mean", "t-stat", "p-value", "sig"
    )
    print(header)
    print("-" * 100)

    sig_counts = {"better": 0, "worse": 0, "ns": 0}

    for kp in all_kps:
        old_errs = old_kp.get(kp, [])
        new_errs = new_kp.get(kp, [])
        if len(old_errs) < 2 or len(new_errs) < 2:
            print("{:<20}  {:5d} {:5d}  insufficient data".format(
                kp, len(old_errs), len(new_errs)))
            continue
        t_stat, p_val = stats.ttest_rel(old_errs, new_errs)
        old_mean = float(np.mean(old_errs))
        new_mean = float(np.mean(new_errs))
        if p_val < 0.05:
            sig = "Better" if new_mean < old_mean else "Worse"
            sig_counts["better" if sig == "Better" else "worse"] += 1
        else:
            sig = "ns"
            sig_counts["ns"] += 1
        print("{:<20}  {:5d} {:5d}  {:9.2f} {:9.2f}  {:8.3f}  {:10.6f}  {}".format(
            kp, len(old_errs), len(new_errs), old_mean, new_mean, t_stat, p_val, sig))

    print()
    print("{} better, {} worse, {} ns (out of {} keypoints)".format(
        sig_counts["better"], sig_counts["worse"], sig_counts["ns"], len(all_kps)))

    paired_kps = [k for k in all_kps
                  if len(old_kp.get(k, [])) >= 2 and len(new_kp.get(k, [])) >= 2]
    if len(paired_kps) >= 2:
        old_pooled = np.concatenate([np.asarray(old_kp[k]) for k in paired_kps])
        new_pooled = np.concatenate([np.asarray(new_kp[k]) for k in paired_kps])
        t_stat, p_val = stats.ttest_rel(old_pooled, new_pooled)
        print()
        print("Paired t-test, pooled across all matched observations "
              "(weighted by keypoint frequency):")
        print("  n pairs = {}".format(len(old_pooled)))
        print("  t={:.3f}, p={:.6f}".format(t_stat, p_val))
        print("  {} overall mean: {:.2f} px".format(old_name, float(np.mean(old_pooled))))
        print("  {} overall mean: {:.2f} px".format(new_name, float(np.mean(new_pooled))))

        old_means = [float(np.mean(old_kp[k])) for k in paired_kps]
        new_means = [float(np.mean(new_kp[k])) for k in paired_kps]
        t_stat_kp, p_val_kp = stats.ttest_rel(old_means, new_means)
        print()
        print("  (unweighted, one unit per keypoint: t={:.3f}, p={:.6f})".format(
            t_stat_kp, p_val_kp))

    plot_keypoint_diff(old_kp, new_kp, all_kps)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Compare two models' predictions")
    parser.add_argument("models", nargs=2, help="Paths to two prediction JSON files")
    parser.add_argument("--labels", default="input/labels/merged.json", help="Path to label JSON")
    parser.add_argument("--names", nargs=2, default=["Old", "New"],
                        help="Model names (default: Old, New)")
    args = parser.parse_args()

    df = load_label_json(Path(args.labels))

    all_results = []
    for pred_path, name in zip(args.models, args.names):
        print(f"\nLoading [{name}]")
        print(f"  predictions : {pred_path}")

        _, pred_map, scales, metrics = load_all(pred_path, df)

        all_label_frames = sorted(int(float(f)) for f in df["frame"].tolist() if pd.notna(f))
        predicted_frames = set(pred_map.keys())
        missing_frames = [f for f in all_label_frames if f not in predicted_frames]

        print_summary(name, metrics, len(all_label_frames), len(missing_frames), len(scales))
        print_per_keypoint_table(name, df, pred_map, scales)

        all_results.append({
            **metrics,
            "name": name,
            "n_frames": len(all_label_frames),
            "n_missing_frames": len(missing_frames),
        })

    print("\nGenerating plots:")
    plot_pck_curve_comparison(all_results)
    print_comparison_table(all_results)

    print("\nRunning t-tests:")
    run_t_tests(
        all_results[0]["per_kp"],
        all_results[1]["per_kp"],
        all_results[0]["name"],
        all_results[1]["name"],
    )

    print(f"\nPlots saved to: {OUTPUT_FOLDER}")
    print("Done.\n")


if __name__ == "__main__":
    main()
