#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _load_video_map(mapping_path: Path | None) -> dict[str, str]:
    if mapping_path is None:
        return {}
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Mapping file must be JSON object: {mapping_path}")
    return {str(k): str(v) for k, v in payload.items()}


def _collect_video_files(videos_root: Path) -> list[Path]:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
    return [p for p in videos_root.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def _resolve_video_path(video_name: str, candidates: list[Path], explicit_map: Mapping[str, str]) -> Path:
    if video_name in explicit_map:
        path = Path(explicit_map[video_name])
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Mapped video path not found for {video_name}: {path}")
        return path

    norm_target = _normalize_name(video_name)
    ranked: list[tuple[int, int, Path]] = []
    for cand in candidates:
        stem_norm = _normalize_name(cand.stem)
        full_norm = _normalize_name(cand.name)
        score = 0
        if stem_norm == norm_target:
            score = 100
        elif norm_target in stem_norm or stem_norm in norm_target:
            score = 70
        elif norm_target in full_norm or full_norm in norm_target:
            score = 60
        if score > 0:
            ranked.append((score, len(stem_norm), cand))
    if not ranked:
        raise FileNotFoundError(
            f"No video file match for '{video_name}'. Provide --video-map for explicit mapping."
        )
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked[0][2]


@dataclass
class VideoFramePlan:
    video_name: str
    source_path: Path
    target_frames: list[int]
    context_frames: list[int]
    all_frames: list[int]


def _build_frame_plan(
    frame_indices_by_video: Mapping[str, list[int]],
    video_files: list[Path],
    explicit_map: Mapping[str, str],
    context_radius: int,
    context_step: int,
) -> list[VideoFramePlan]:
    plans: list[VideoFramePlan] = []
    offsets = list(range(-context_radius, context_radius + 1, max(1, context_step)))
    for video_name, frames in frame_indices_by_video.items():
        src = _resolve_video_path(video_name, video_files, explicit_map)
        targets = sorted({int(f) for f in frames if int(f) >= 0})
        ctx: set[int] = set()
        for f in targets:
            for off in offsets:
                idx = f + off
                if idx >= 0:
                    ctx.add(idx)
        all_frames = sorted(ctx)
        context_only = sorted(set(all_frames) - set(targets))
        plans.append(
            VideoFramePlan(
                video_name=video_name,
                source_path=src,
                target_frames=targets,
                context_frames=context_only,
                all_frames=all_frames,
            )
        )
    return plans


def _extract_from_video(plan: VideoFramePlan, output_dir: Path, jpeg_quality: int) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(plan.source_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {plan.source_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_out = output_dir / plan.video_name.replace(" ", "_")
    video_out.mkdir(parents=True, exist_ok=True)

    saved = 0
    misses = 0
    details: list[dict[str, Any]] = []
    for frame_idx in plan.all_frames:
        if total > 0 and frame_idx >= total:
            misses += 1
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame_bgr = cap.read()
        if not ok:
            misses += 1
            continue
        dst = video_out / f"{frame_idx:06d}.jpg"
        cv2.imwrite(str(dst), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        saved += 1
        details.append(
            {
                "frame_idx": int(frame_idx),
                "is_target": int(frame_idx) in set(plan.target_frames),
                "path": str(dst),
            }
        )
    cap.release()

    return {
        "video_name": plan.video_name,
        "source_path": str(plan.source_path),
        "total_video_frames": total,
        "target_frame_count": len(plan.target_frames),
        "context_frame_count": len(plan.context_frames),
        "requested_frame_count": len(plan.all_frames),
        "saved_frame_count": saved,
        "missing_frame_count": misses,
        "frames": details,
    }


def _load_frame_indices_from_predictions(path: Path) -> dict[str, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in predictions file: {path}")
    by_video: dict[str, list[int]] = {}
    for rec in payload:
        if not isinstance(rec, Mapping):
            continue
        video_name = str(rec.get("video_name", ""))
        frame_idx = int(rec.get("frame_idx", -1))
        if not video_name or frame_idx < 0:
            continue
        by_video.setdefault(video_name, []).append(frame_idx)
    return by_video


def _load_frame_indices_from_manifest(path: Path) -> dict[str, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_video: dict[str, list[int]] = {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected object JSON in manifest file: {path}")
    frames = payload.get("frames_by_video", {})
    if not isinstance(frames, Mapping):
        raise ValueError(f"manifest missing frames_by_video: {path}")
    for name, arr in frames.items():
        if not isinstance(arr, list):
            continue
        by_video[str(name)] = [int(v) for v in arr]
    return by_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract sparse context frames for temporal post-processing.")
    parser.add_argument("--predictions-json", type=Path, default=None, help="Predictions JSON containing video_name/frame_idx.")
    parser.add_argument("--frame-manifest-json", type=Path, default=None, help="Optional frame manifest with frames_by_video object.")
    parser.add_argument("--videos-root", type=Path, default=PROJECT_ROOT / "videos")
    parser.add_argument("--video-map", type=Path, default=None, help="Optional JSON map: video_name -> absolute/relative video path.")
    parser.add_argument("--context-radius", type=int, default=30, help="Context radius around each target frame.")
    parser.add_argument("--context-step", type=int, default=5, help="Sample every N frames in context window.")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output" / "post_process" / "context_frames")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.predictions_json is None and args.frame_manifest_json is None:
        raise ValueError("Provide --predictions-json or --frame-manifest-json")

    if args.predictions_json is not None:
        by_video = _load_frame_indices_from_predictions(args.predictions_json)
    else:
        by_video = _load_frame_indices_from_manifest(args.frame_manifest_json)

    if not by_video:
        raise ValueError("No frame indices found in input.")

    video_files = _collect_video_files(args.videos_root)
    if not video_files:
        raise FileNotFoundError(f"No video files found under {args.videos_root}")
    explicit_map = _load_video_map(args.video_map)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / f"context_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    plans = _build_frame_plan(
        frame_indices_by_video=by_video,
        video_files=video_files,
        explicit_map=explicit_map,
        context_radius=int(args.context_radius),
        context_step=int(args.context_step),
    )

    records: list[dict[str, Any]] = []
    for plan in plans:
        records.append(_extract_from_video(plan, run_dir / "frames", int(args.jpeg_quality)))

    summary = {
        "videos_root": str(args.videos_root),
        "video_file_count": len(video_files),
        "context_radius": int(args.context_radius),
        "context_step": int(args.context_step),
        "run_dir": str(run_dir),
        "video_count": len(records),
        "total_saved_frames": int(sum(r["saved_frame_count"] for r in records)),
        "total_target_frames": int(sum(r["target_frame_count"] for r in records)),
        "records": records,
    }
    _write_json(run_dir / "extraction_manifest.json", summary)
    print(f"Context frame extraction complete: {run_dir}")
    print(f"Saved frames: {summary['total_saved_frames']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
