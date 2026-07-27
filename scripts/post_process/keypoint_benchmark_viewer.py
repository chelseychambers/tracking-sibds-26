"""
Keypoint Benchmark Viewer — Named video selector
=================================================
Run:  streamlit run scripts/post_process/keypoint_benchmark_viewer.py

Lets you pick a video name and auto-maps to the correct label, prediction, and image files.
"""

from __future__ import annotations

import base64
import math
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from modules.manual_correction_utils import (
    get_prediction_point,
    load_label_json,
    load_prediction_map,
)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
try:
    st.set_page_config(layout="wide", page_title="Keypoint Benchmark Viewer")
except Exception:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# NAMED VIDEOS — add/remove entries here
# ──────────────────────────────────────────────────────────────────────────────
VIDEOS = [
    {
        "name": "Merged Testing",
        "label":      "input/labels/merged.json",
        "prediction": "output/predicted_frames/mikey_pred/camera4_Rat4_Rat9_merged.json",
        "old_prediction": "output/predicted_frames/mikey_pred/camera4_Rat4_Rat9_merged.json",
        "new_prediction": "output/predicted_frames/finetune_100ep/merged_test.json",
        "frames":     [
            "output/extracted_frames/Camera4_stitched",
            "output/extracted_frames/Rat 4 2025-11-12 12-25-01_test",
            "output/extracted_frames/Rat 9 2025-11-11 11-09-35_test",
        ],
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# CANVAS COMPONENT
# ──────────────────────────────────────────────────────────────────────────────
_COMP_DIR = Path(__file__).resolve().parent / "_cam4_canvas"
_COMP_DIR.mkdir(exist_ok=True)

# Canvas uses fixed colors: green circles for manual, red crosshairs for
# predictions, blue dashed lines for errors.
_CANVAS_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0e1117; overflow: hidden; }
  canvas { display: block; margin: 0 auto; border-radius: 8px; }
</style>
</head>
<body>
<canvas id="c"></canvas>
<script>
var canvas = document.getElementById("c");
var ctx    = canvas.getContext("2d");

var GT   = [];
var PRED = [];
var W = 0, H = 0;
var POINT_R    = 7;
var SHOW_GT    = true;
var SHOW_PRED  = true;
var SHOW_LINES = true;
var ISOLATE    = "None (Show All)";

var img = new Image();
img.onload = function() { draw(); };

window.parent.postMessage({
  isStreamlitMessage: true,
  type: "streamlit:componentReady",
  apiVersion: 1
}, "*");

window.addEventListener("message", function(e) {
  if (!e.data || e.data.type !== "streamlit:render") return;
  var a = e.data.args;

  W          = a.display_w;
  H          = a.display_h;
  POINT_R    = a.point_r    !== undefined ? a.point_r    : 7;
  SHOW_GT    = a.show_gt    !== undefined ? a.show_gt    : true;
  SHOW_PRED  = a.show_pred  !== undefined ? a.show_pred  : true;
  SHOW_LINES = a.show_lines !== undefined ? a.show_lines : true;
  ISOLATE    = a.isolate    !== undefined ? a.isolate    : "None (Show All)";

  canvas.width  = W;
  canvas.height = H;

  GT   = (a.gt_points   || []).map(function(p) { return p; });
  PRED = (a.pred_points || []).map(function(p) { return p; });

  var newSrc = a.img_src;
  if (img.src !== newSrc) {
    img.src = newSrc;
    if (img.complete && img.naturalWidth) draw();
  } else {
    draw();
  }
});

function draw() {
  if (!W || !H) return;
  ctx.clearRect(0, 0, W, H);
  if (img.complete && img.naturalWidth) {
    ctx.drawImage(img, 0, 0, W, H);
  }

  var gtMap = {};
  GT.forEach(function(p) { gtMap[p.label] = p; });

  if (SHOW_GT && SHOW_PRED) {
    PRED.forEach(function(pred) {
      var gt = gtMap[pred.label];
      if (!gt) return;

      var isIsolated = (ISOLATE !== "None (Show All)" && pred.label === ISOLATE);
      var showAll = (SHOW_LINES && ISOLATE === "None (Show All)");

      if (isIsolated || showAll) {
        ctx.save();
        ctx.setLineDash([3, 2]);
        ctx.strokeStyle = "#4da6ff";
        ctx.lineWidth   = 2.5;
        ctx.globalAlpha = 0.95;
        ctx.beginPath();
        ctx.moveTo(gt.x, gt.y);
        ctx.lineTo(pred.x, pred.y);
        ctx.stroke();
        ctx.restore();
      }
    });
  }

  if (SHOW_GT) {
    GT.forEach(function(p) {
      ctx.save();
      ctx.shadowColor = "rgba(0,0,0,0.8)";
      ctx.shadowBlur  = 8;
      ctx.beginPath();
      ctx.arc(p.x, p.y, POINT_R, 0, Math.PI * 2);
      ctx.fillStyle   = "#00cc66";
      ctx.fill();
      ctx.restore();
    });
  }

  if (SHOW_PRED) {
    PRED.forEach(function(p) {
      var len = POINT_R + 1;
      ctx.save();
      ctx.shadowColor = "rgba(0,0,0,0.8)";
      ctx.shadowBlur  = 6;

      ctx.beginPath();
      ctx.moveTo(p.x - len, p.y);
      ctx.lineTo(p.x + len, p.y);
      ctx.moveTo(p.x, p.y - len);
      ctx.lineTo(p.x, p.y + len);

      ctx.strokeStyle = "#ff2a55";
      ctx.lineWidth   = 2.5;
      ctx.stroke();

      ctx.restore();
    });
  }
}
</script>
</body>
</html>"""

(_COMP_DIR / "index.html").write_text(_CANVAS_HTML, encoding="utf-8")
_canvas = components.declare_component("cam4_canvas", path=str(_COMP_DIR))


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

FRAME_SOURCE_RANGES = [
    (1506, 159340, "Camera 4"),
    (188196, 253582, "Rat 4"),
    (260236, 513594, "Rat 9"),
]


def get_frame_source(frame_idx: int) -> str:
    for lo, hi, source in FRAME_SOURCE_RANGES:
        if lo <= frame_idx <= hi:
            return source
    return "Unknown"


def _parse(v: object) -> Optional[float]:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def pil_to_b64(img_pil: Image.Image) -> str:
    buf = BytesIO()
    img_pil.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


@st.cache_data
def build_image_index(folders_tuple: tuple) -> Dict[int, str]:
    index: Dict[int, str] = {}
    for folder_str in folders_tuple:
        folder = Path(folder_str)
        if not folder.exists():
            continue
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            nums = re.findall(r"\d+", f.stem)
            if nums:
                index.setdefault(int(nums[-1]), str(f))
    return index

def find_image(folders: List[str], frame_idx: int) -> Optional[Path]:
    match = build_image_index(tuple(folders)).get(frame_idx)
    return Path(match) if match else None


def _frame_mean_errors_vs_gt(df, pred_map: Dict, kp_names: List[str], valid_frames: List[int]) -> Dict[int, float]:
    """Per-frame mean pixel error between a prediction set and the ground-truth labels."""
    frame_mean_errors = {}
    for f in valid_frames:
        f_df = df[df["frame"] == f]
        errs = []
        if not f_df.empty:
            row = f_df.iloc[0]
            for kp in kp_names:
                gx = _parse(getattr(row, f"{kp}_x", None))
                gy = _parse(getattr(row, f"{kp}_y", None))
                pp = get_prediction_point(pred_map, f, kp)
                if gx is not None and gy is not None and pp is not None:
                    px, py = _parse(pp[0]), _parse(pp[1])
                    if px is not None and py is not None:
                        errs.append(math.hypot(px - gx, py - gy))
        frame_mean_errors[f] = sum(errs) / len(errs) if errs else 0.0
    return frame_mean_errors


@st.cache_data
def load_data(label_path: str, pred_path: Optional[str]):
    """Load labels and (optionally) predictions.

    pred_path may be None when no prediction file exists for the selected
    video — in that case we skip trying to parse the labels file as if it
    were a predictions file (which used to either throw an exception or
    silently produce meaningless zero-error stats).
    """
    df = load_label_json(Path(label_path))

    if pred_path:
        raw_pred = load_prediction_map(Path(pred_path))
        pred_map = {int(k): v for k, v in raw_pred.items()}
    else:
        pred_map = {}

    kp_names = sorted(
        col[:-2] for col in df.columns
        if col.endswith("_x") and f"{col[:-2]}_y" in df.columns
    )
    valid_frames = sorted(int(f) for f in df["frame"].dropna().unique())

    frame_mean_errors = _frame_mean_errors_vs_gt(df, pred_map, kp_names, valid_frames)

    return df, pred_map, kp_names, valid_frames, frame_mean_errors


@st.cache_data
def compute_improvement(
    label_path: str,
    old_pred_path: str,
    new_pred_path: str,
) -> Dict[int, Dict[str, float]]:
    """Per-frame comparison of Old vs New prediction accuracy against ground truth.

    Returns, for every labeled frame, the mean pixel error of the Old
    predictions vs. the manual labels, the mean pixel error of the New
    predictions vs. the manual labels, and the improvement
    (old_error - new_error — positive means the New predictions are closer
    to the ground truth than the Old ones).
    """
    df = load_label_json(Path(label_path))

    kp_names = sorted(
        col[:-2] for col in df.columns
        if col.endswith("_x") and f"{col[:-2]}_y" in df.columns
    )
    valid_frames = sorted(int(f) for f in df["frame"].dropna().unique())

    old_map = {int(k): v for k, v in load_prediction_map(Path(old_pred_path)).items()}
    new_map = {int(k): v for k, v in load_prediction_map(Path(new_pred_path)).items()}

    old_errors = _frame_mean_errors_vs_gt(df, old_map, kp_names, valid_frames)
    new_errors = _frame_mean_errors_vs_gt(df, new_map, kp_names, valid_frames)

    result: Dict[int, Dict[str, float]] = {}
    for f in valid_frames:
        oe = old_errors.get(f, 0.0)
        ne = new_errors.get(f, 0.0)
        result[f] = {"old": oe, "new": ne, "improvement": oe - ne}

    return result


def build_gt_points(
    df, frame_idx: int, kp_names: List[str],
    sx: float, sy: float,
) -> List[Dict]:
    rows = df[df["frame"] == frame_idx]
    if rows.empty:
        return []
    row = rows.iloc[0]
    pts = []
    for kp in kp_names:
        x = _parse(getattr(row, f"{kp}_x", None))
        y = _parse(getattr(row, f"{kp}_y", None))
        if x is not None and y is not None:
            pts.append({
                "label": kp,
                "x": x * sx,
                "y": y * sy,
            })
    return pts


def build_pred_points(pred_map: Dict, frame_idx: int, gt_pts: List[Dict], sx: float, sy: float) -> List[Dict]:
    pts = []
    for gp in gt_pts:
        pp = get_prediction_point(pred_map, frame_idx, gp["label"])
        if pp is None:
            continue
        x, y = _parse(pp[0]), _parse(pp[1])
        if x is not None and y is not None:
            pts.append({
                "label": gp["label"],
                "x": x * sx,
                "y": y * sy,
            })
    return pts


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Video Selection")

    video_names = [v["name"] for v in VIDEOS]
    selected_name = st.selectbox("Select video", options=video_names, index=0)

    video_cfg = next(v for v in VIDEOS if v["name"] == selected_name)
    label_path      = video_cfg["label"]
    images_dirs     = video_cfg["frames"]

    has_comparison = "old_prediction" in video_cfg and "new_prediction" in video_cfg
    if has_comparison:
        pred_choice = st.radio(
            "Prediction Set",
            ["Old Predictions", "New Predictions"],
            horizontal=True,
        )
        pred_path = video_cfg["old_prediction"] if pred_choice == "Old Predictions" else video_cfg["new_prediction"]
    else:
        pred_path = video_cfg["prediction"]

    st.markdown(f"**Labels:** `{label_path}`")
    st.markdown(f"**Predictions:** `{pred_path}`")
    for d in images_dirs:
        st.markdown(f"**Frames:** `{d}`")

    st.markdown("---")
    st.subheader("Frame Navigation")

    sort_options = ["Default order", "Lowest Mean Pixel Error"]
    if has_comparison:
        sort_options.append("Most Improved (Old→New Mean Error)")
    sort_mode = st.selectbox("Sort / filter frames by", sort_options)

    st.markdown("---")
    st.subheader("Appearance")
    point_r = st.slider("Point size", 3, 20, 7)

    st.markdown("---")
    st.subheader("Layers")
    show_gt = st.checkbox("Manual keypoints (Circles)", value=True)
    show_pred = st.checkbox("Predicted keypoints (Crosshairs)", value=True)
    show_lines = st.checkbox("Show ALL error lines", value=False)

    st.markdown("---")
    with st.expander("Debug: image folders"):
        for d in images_dirs:
            img_index_dbg = build_image_index(tuple([d]))
            if img_index_dbg:
                st.caption(f"`{d}` — {len(img_index_dbg)} images (range {min(img_index_dbg)}–{max(img_index_dbg)})")
            else:
                st.caption(f"`{d}` — folder empty or not found.")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

header_container = st.container()
canvas_container = st.container()
metrics_container = st.container()
table_container = st.container()

with header_container:
    title = "Keypoint Benchmark Viewer"
    if has_comparison:
        title += f"  ({pred_choice})"
    st.title(title)

    label_ok = Path(label_path).exists()
    pred_ok = Path(pred_path).exists()

    if not label_ok:
        st.error(f"Labels file not found: `{label_path}`")
        st.stop()
    if not pred_ok:
        st.warning(f"Predictions file not found: `{pred_path}` — only Manual points will be shown.")

    try:
        df, pred_map, kp_names, valid_frames, frame_mean_errors = load_data(
            label_path, pred_path if pred_ok else label_path
        )
    except Exception as ex:
        st.error(f"Failed to load data: {ex}")
        st.stop()

    if not kp_names:
        st.warning("The label file contains no keypoint annotations (entries have frame indices but no labels). Select a different video.")
        st.stop()

    if not valid_frames:
        st.warning("No valid frames found in label data.")
        st.stop()

    # ── Resolve frame ordering / filtering based on the sidebar sort_mode ────
    frame_improvement: Dict[int, Dict[str, float]] = {}

    if sort_mode == "Lowest Mean Pixel Error":
        display_frames = sorted(valid_frames, key=lambda x: frame_mean_errors.get(x, 0))

    elif sort_mode == "Most Improved (Old→New Mean Error)":
        old_pred_ok = Path(video_cfg["old_prediction"]).exists()
        new_pred_ok = Path(video_cfg["new_prediction"]).exists()

        if not (old_pred_ok and new_pred_ok):
            st.warning(
                "Can't compute Old vs New improvement — one or both "
                "prediction files are missing. Showing default order instead."
            )
            display_frames = valid_frames
        else:
            frame_improvement = compute_improvement(
                label_path, video_cfg["old_prediction"], video_cfg["new_prediction"]
            )

            improvements = [v["improvement"] for v in frame_improvement.values()]
            min_imp, max_imp = (min(improvements), max(improvements)) if improvements else (0.0, 0.0)
            min_improvement_threshold = st.sidebar.slider(
                "Min. improvement to include (px)",
                min_value=float(min(min_imp, 0.0)),
                max_value=float(max(max_imp, 1.0)),
                value=0.0,
                help=(
                    "Improvement = Old mean error minus New mean error (both vs. the "
                    "manual labels). Positive means the New predictions are closer to "
                    "ground truth than the Old ones for that frame."
                ),
            )

            display_frames = sorted(
                (
                    f for f in valid_frames
                    if frame_improvement.get(f, {}).get("improvement", 0.0) >= min_improvement_threshold
                ),
                key=lambda x: frame_improvement.get(x, {}).get("improvement", 0.0),
                reverse=True,
            )

            if not display_frames:
                st.warning("No frames meet that improvement threshold — try lowering it.")
                st.stop()
    else:
        display_frames = valid_frames

    st.caption(f"Total frames: **{len(valid_frames)}** (showing **{len(display_frames)}**)")

    def _format_frame(x: int) -> str:
        base = f"Frame {x} — {get_frame_source(x)}"
        if sort_mode == "Most Improved (Old→New Mean Error)":
            info = frame_improvement.get(x, {})
            return (
                f"{base} (Improved {info.get('improvement', 0):.1f} px: "
                f"{info.get('old', 0):.1f}→{info.get('new', 0):.1f})"
            )
        return f"{base} (Mean Error: {frame_mean_errors.get(x, 0):.1f} px)"

    selected_frame = st.selectbox(
        "Select Frame",
        options=display_frames,
        format_func=_format_frame,
    )

    img_path = find_image(images_dirs, selected_frame)
    if img_path is None:
        img_index = build_image_index(tuple(images_dirs))
        if not img_index:
            st.error(f"No images found in any of the configured frame folders.")
        else:
            available = sorted(img_index.keys())
            st.error(f"No image found for frame **{selected_frame}**.\n\nAvailable frames: **{len(available)}** (range {min(available)}–{max(available)})")
        st.stop()

# ── Scale image & Extract Baseline Points ─────────────────────────────────────
pil_img = Image.open(img_path).convert("RGB")
orig_w, orig_h = pil_img.width, pil_img.height
MAX_W = 1060
scale = min(1.0, MAX_W / orig_w)
display_w = int(orig_w * scale)
display_h = int(orig_h * scale)
sx, sy = display_w / orig_w, display_h / orig_h

display_img = pil_img.resize((display_w, display_h), Image.LANCZOS) if scale < 1.0 else pil_img
img_src = f"data:image/jpeg;base64,{pil_to_b64(display_img)}"

gt_pts = build_gt_points(df, selected_frame, kp_names, sx, sy)
pred_pts = build_pred_points(pred_map, selected_frame, gt_pts, sx, sy) if pred_map else []


# ── Build the Interactive Bottom Panel ────────────────────────────────────────
isolate_choice = "None (Show All)"

with table_container:
    if gt_pts:
        st.markdown("---")
        st.subheader("Keypoint Errors")

        all_labels = sorted(p["label"] for p in gt_pts)
        isolate_choice = st.selectbox(
            "Isolate Keypoint (Select to view its error line):",
            options=["None (Show All)"] + all_labels
        )

        gt_lut2 = {p["label"]: p for p in gt_pts}
        pred_lut2 = {p["label"]: p for p in pred_pts}

        rows = []
        for lbl in sorted(gt_lut2.keys()):
            gp = gt_lut2[lbl]
            pp = pred_lut2.get(lbl)
            err = (math.hypot(pp["x"] - gp["x"], pp["y"] - gp["y"]) / scale) if pp else None

            rows.append({
                "Keypoint": lbl,
                "Manual (x, y)": f"{int(gp['x'] / sx)}, {int(gp['y'] / sy)}",
                "Pred (x, y)": f"{int(pp['x'] / sx)}, {int(pp['y'] / sy)}" if pp else "—",
                "Error (px)": float(err) if err is not None else None,
            })

        tdf = pd.DataFrame(rows)

        def _row_style(row):
            lbl = row["Keypoint"]
            styles = ["text-align: center;"] * len(row)

            if lbl == isolate_choice:
                styles = ["text-align: center; background-color: #1c3d5a; color: #ffffff; font-weight: bold"] * len(row)
            elif row["Pred (x, y)"] == "—":
                styles = ["text-align: center; background-color: #331a1a; color: #ff8888"] * len(row)

            return styles

        st.dataframe(
            tdf.style
               .set_properties(**{'text-align': 'center'})
               .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
               .apply(_row_style, axis=1),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Error (px)": st.column_config.NumberColumn(
                    "Error (px)",
                    format="%.1f",
                )
            }
        )

# ── Render the Canvas ─────────────────────────────────────────────────────────
with canvas_container:
    _canvas(
        img_src=img_src,
        display_w=display_w,
        display_h=display_h,
        gt_points=gt_pts,
        pred_points=pred_pts,
        point_r=point_r,
        show_gt=show_gt,
        show_pred=show_pred,
        show_lines=show_lines,
        isolate=isolate_choice,
        key=f"cam4_{selected_name}_{selected_frame}_{point_r}_{show_gt}_{show_pred}_{show_lines}_{isolate_choice}",
        default=None,
        height=display_h + 4,
    )
    st.caption(f"Frame {selected_frame} ({get_frame_source(selected_frame)})  |  {img_path.name}  |  {orig_w} x {orig_h} px  |  circle = Manual    crosshair = Prediction")

# ── Render Metrics Row ────────────────────────────────────────────────────────
with metrics_container:
    if pred_pts:
        gt_lut = {p["label"]: p for p in gt_pts}
        pred_lut = {p["label"]: p for p in pred_pts}
        matched = sorted(set(gt_lut) & set(pred_lut))

        if matched:
            errors = [
                math.hypot(pred_lut[l]["x"] - gt_lut[l]["x"],
                           pred_lut[l]["y"] - gt_lut[l]["y"]) / scale
                for l in matched
            ]
            mean_e = sum(errors) / len(errors)
            max_e = max(errors)

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Matched", f"{len(matched)} / {len(gt_pts)}")
            c2.metric("Mean error", f"{mean_e:.1f} px")
            c3.metric("Max error", f"{max_e:.1f} px")
            c4.metric("Missing pred", str(len(gt_pts) - len(matched)))