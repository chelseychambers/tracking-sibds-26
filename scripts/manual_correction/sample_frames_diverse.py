#!/usr/bin/env python3
"""
Sample visually diverse frames from videos using brightness filtering
and embedding-based diversity selection.

For each video:
  1. Sample a large pool of evenly-spaced frames.
  2. Filter by mean grayscale brightness (discard dark frames).
  3. Downsample each bright frame to 32x32 grayscale -> 1024-d vector.
  4. Select diverse frames via agglomerative clustering.
  5. Split into train/test (75/25 temporal split)- remove split

Usage:
    python scripts/sample_frames_diverse.py \
        --video-names "Rat 4 2025-11-12 12-25-01" "Rat 9 2025-11-11 11-09-35" \
        --target-frames 150 \
        --min-brightness 50 \
        --train-split 0.75 \
        --pool-factor 3
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = PROJECT_ROOT / "videos"
EXTRACTED_DIR = PROJECT_ROOT / "output" / "extracted_frames"
LABELS_DIR = PROJECT_ROOT / "input" / "labels"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
EMBED_SIZE = 32


def find_video(video_name: str) -> Path | None:
    for ext in VIDEO_EXTENSIONS:
        p = VIDEOS_DIR / f"{video_name}{ext}"
        if p.is_file():
            return p
    return None


def get_video_info(video_path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        return total, fps
    finally:
        cap.release()


def even_indices(total_frames: int, count: int) -> list[int]:
    if count >= total_frames:
        return list(range(total_frames))
    stride = total_frames / count
    return [min(int(round(i * stride)), total_frames - 1) for i in range(count)]


def extract_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def compute_brightness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def embed_frame(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (EMBED_SIZE, EMBED_SIZE), interpolation=cv2.INTER_AREA)
    vec = small.astype(np.float32).flatten()
    vec /= 255.0
    return vec


def select_diverse_frames(
    vectors: np.ndarray,
    indices: list[int],
    target: int,
) -> list[int]:
    if len(indices) <= target:
        return indices

    dist_matrix = np.zeros((len(indices), len(indices)), dtype=np.float32)
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            d = np.linalg.norm(vectors[i] - vectors[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="complete")

    labels = fcluster(Z, t=target, criterion="maxclust")

    selected = []
    for cluster_id in range(1, target + 1):
        members = [i for i, c in enumerate(labels) if c == cluster_id]
        if not members:
            continue
        cluster_vecs = vectors[members]
        centroid = cluster_vecs.mean(axis=0)
        dists = np.linalg.norm(cluster_vecs - centroid, axis=1)
        best = members[int(np.argmin(dists))]
        selected.append(indices[best])

    return selected


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def save_config(config: dict):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def save_labels(video_name: str, data: list[dict]):
    label_path = LABELS_DIR / f"{video_name}.json"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        json.dump(data, f, indent=2)


def process_video(
    video_name: str,
    target_frames: int,
    min_brightness: float,
    train_split: float,
    pool_factor: int,
    dry_run: bool,
):
    video_path = find_video(video_name)
    if video_path is None:
        print(f"  ERROR: No video found for '{video_name}' in {VIDEOS_DIR}", file=sys.stderr)
        return

    total_frames, fps = get_video_info(video_path)
    print(f"  Video: {video_path.name}")
    print(f"  Total frames: {total_frames}  FPS: {fps:.1f}")

    pool_size = target_frames * pool_factor
    pool_indices = even_indices(total_frames, pool_size)
    print(f"  Pool: {len(pool_indices)} evenly-spaced frames")

    bright_indices = []
    bright_vectors = []

    for i, idx in enumerate(pool_indices):
        frame = extract_frame(video_path, idx)
        if frame is None:
            continue
        brightness = compute_brightness(frame)
        if brightness < min_brightness:
            continue
        vec = embed_frame(frame)
        bright_indices.append(idx)
        bright_vectors.append(vec)

    print(f"  Bright frames: {len(bright_indices)} (filtered from {len(pool_indices)})")

    if len(bright_indices) == 0:
        print(f"  WARNING: No bright frames found for {video_name}")
        return

    vectors = np.array(bright_vectors, dtype=np.float32)
    selected = select_diverse_frames(vectors, bright_indices, target_frames)
    selected.sort()

    print(f"  Selected {len(selected)} diverse frames")

    train_name = video_name
    test_name = f"{video_name}_test"
    train_dir = EXTRACTED_DIR / train_name
    test_dir = EXTRACTED_DIR / test_name

    if dry_run:
        print(f"  Would extract to:")
        print(f"    train: {train_dir}")
        print(f"    test:  {test_dir}")
        return

    for d in [train_dir, test_dir]:
        if d.exists():
            for f in d.glob("*.jpg"):
                f.unlink()
            d.rmdir()

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    split_point = max(1, int(len(selected) * train_split))
    train_frames = selected[:split_point]
    test_frames = selected[split_point:]

    for idx in train_frames:
        frame = extract_frame(video_path, idx)
        out_path = train_dir / f"{idx:08d}.jpg"
        cv2.imwrite(str(out_path), frame)

    for idx in test_frames:
        frame = extract_frame(video_path, idx)
        out_path = test_dir / f"{idx:08d}.jpg"
        cv2.imwrite(str(out_path), frame)

    print(f"  Saved: {len(train_frames)} train, {len(test_frames)} test")

    train_labels = [{"frame_idx": idx} for idx in train_frames]
    save_labels(train_name, train_labels)

    test_labels = [{"frame_idx": idx} for idx in test_frames]
    save_labels(test_name, test_labels)

    print(f"  Labels: {LABELS_DIR / f'{train_name}.json'}")
    print(f"  Labels: {LABELS_DIR / f'{test_name}.json'}")

    config = load_config()
    if train_name not in config.get("train_videos", []):
        config.setdefault("train_videos", []).append(train_name)
    if test_name not in config.get("test_videos", []):
        config.setdefault("test_videos", []).append(test_name)
    save_config(config)
    print(f"  Updated {CONFIG_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-names", nargs="+", required=True,
                        help="Video stems to process")
    parser.add_argument("--target-frames", type=int, default=150,
                        help="Target number of diverse frames (default: 150)")
    parser.add_argument("--min-brightness", type=float, default=50.0,
                        help="Minimum mean grayscale brightness (default: 50)")
    parser.add_argument("--train-split", type=float, default=0.75,
                        help="Fraction for training (default: 0.75)")
    parser.add_argument("--pool-factor", type=int, default=3,
                        help="Pool size = target * pool_factor (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without writing files")
    args = parser.parse_args()

    for name in args.video_names:
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"{'='*60}")
        process_video(name, args.target_frames, args.min_brightness,
                      args.train_split, args.pool_factor, args.dry_run)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
