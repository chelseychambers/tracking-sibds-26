from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_LABEL_DIR = PROJECT_ROOT / "input" / "labels"
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "videos"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "labeled_frames"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract labeled frames from videos listed in input/labels.",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=DEFAULT_LABEL_DIR,
        help=f"Directory containing label JSON files. Default: {DEFAULT_LABEL_DIR}",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help=f"Directory containing source videos. Default: {DEFAULT_VIDEO_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for extracted frames. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional list of label file stems or filenames to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted frames.",
    )
    return parser.parse_args()


def normalize_input_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_frame_indices(label_path: Path) -> list[int]:
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected {label_path} to contain a list, got {type(payload).__name__}")
    frame_indices = sorted({int(item["frame_idx"]) for item in payload})
    if not frame_indices:
        raise ValueError(f"No frame indices found in {label_path}")
    return frame_indices


def find_video_path(video_dir: Path, video_name: str) -> Path:
    for extension in VIDEO_EXTENSIONS:
        candidate = video_dir / f"{video_name}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find a video for '{video_name}' under {video_dir}")


def extract_frame(capture: cv2.VideoCapture, frame_idx: int) -> cv2.typing.MatLike | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = capture.read()
    if ok and frame is not None:
        return frame

    if frame_idx > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        ok, frame = capture.read()
        if ok and frame is not None:
            return frame
    return None


def process_label_file(label_path: Path, video_dir: Path, output_dir: Path, overwrite: bool) -> tuple[int, int]:
    video_name = label_path.stem
    video_path = find_video_path(video_dir, video_name)
    frame_indices = read_frame_indices(label_path)
    target_dir = output_dir / video_name
    target_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    saved = 0
    skipped = 0
    try:
        for frame_idx in frame_indices:
            output_path = target_dir / f"{frame_idx:08d}.jpg"
            if output_path.exists() and not overwrite:
                skipped += 1
                continue

            frame = extract_frame(capture, frame_idx)
            if frame is None:
                print(f"[warn] {video_name}: failed to read frame {frame_idx}")
                continue

            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"Failed to write frame to {output_path}")
            saved += 1
    finally:
        capture.release()

    return saved, skipped


def main() -> None:
    args = parse_args()
    label_dir = normalize_input_path(args.label_dir)
    video_dir = normalize_input_path(args.video_dir)
    output_dir = normalize_input_path(args.output_dir)

    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    label_paths = sorted(path for path in label_dir.glob("*.json") if path.is_file())
    if args.labels:
        requested = {name if name.endswith(".json") else f"{name}.json" for name in args.labels}
        label_paths = [path for path in label_paths if path.name in requested]
    if not label_paths:
        raise FileNotFoundError(f"No label JSON files found in {label_dir}")

    total_saved = 0
    total_skipped = 0
    failures = []
    for label_path in label_paths:
        try:
            saved, skipped = process_label_file(label_path, video_dir, output_dir, args.overwrite)
            total_saved += saved
            total_skipped += skipped
            print(f"[ok] {label_path.stem}: saved {saved}, skipped {skipped}")
        except Exception as exc:  # noqa: BLE001
            failures.append((label_path.name, str(exc)))
            print(f"[error] {label_path.name}: {exc}")

    print(
        f"Finished. Processed {len(label_paths)} label files, saved {total_saved} frames, "
        f"skipped {total_skipped} existing frames."
    )
    if failures:
        raise SystemExit(
            "Some files failed:\n" + "\n".join(f"- {name}: {message}" for name, message in failures)
        )


if __name__ == "__main__":
    main()
