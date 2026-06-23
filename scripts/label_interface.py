"""
Simple Streamlit label editor.

Features:
- Choose an images folder (defaults to output/extracted_frames).
- Choose a labels JSON (searches input/labels and output/predicted_frames).
- View selected image and its keypoint labels (per-frame JSON format used in this repo).
- Edit existing keypoints, add new keypoints, delete, and save back to JSON.

replaces the previous placeholder UI.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import streamlit as st
from PIL import Image, ImageDraw

# Compatibility wrapper for Streamlit image URL helper (cross-version)
try:
    from streamlit.elements.lib.image_utils import image_to_url as _streamlit_image_to_url
    import streamlit.elements.image as st_image
    from types import SimpleNamespace

    def _compat_image_to_url(image_data, *args, **kwargs):
        # Handle the older positional signature used by some callers:
        # (image, width, clamp, channels, output_format, image_id)
        if args and isinstance(args[0], int):
            width = args[0]
            clamp = args[1] if len(args) > 1 else kwargs.get("clamp", True)
            channels = args[2] if len(args) > 2 else kwargs.get("channels", "RGB")
            output_format = args[3] if len(args) > 3 else kwargs.get("output_format", "PNG")
            image_id = args[4] if len(args) > 4 else kwargs.get("image_id")
            layout = SimpleNamespace(width=width, height=kwargs.get("height"))
            return _streamlit_image_to_url(image_data, layout, clamp, channels, output_format, image_id)

        # Newer signature: (image, layout_config, clamp, channels, output_format, image_id)
        if args and hasattr(args[0], "width"):
            layout = args[0]
            clamp = args[1] if len(args) > 1 else kwargs.get("clamp", True)
            channels = args[2] if len(args) > 2 else kwargs.get("channels", "RGB")
            output_format = args[3] if len(args) > 3 else kwargs.get("output_format", "PNG")
            image_id = args[4] if len(args) > 4 else kwargs.get("image_id")
            return _streamlit_image_to_url(image_data, layout, clamp, channels, output_format, image_id)

        # Fallback using keyword args
        layout = kwargs.get("layout") or kwargs.get("layout_config")
        return _streamlit_image_to_url(
            image_data,
            layout,
            kwargs.get("clamp", True),
            kwargs.get("channels", "RGB"),
            kwargs.get("output_format", "PNG"),
            kwargs.get("image_id"),
        )

    if not hasattr(st_image, "image_to_url"):
        st_image.image_to_url = _compat_image_to_url
except Exception:
    # If we cannot access internals, continue and hope streamlit_drawable_canvas
    # can function with the installed Streamlit version.
    pass

try:
    from streamlit_drawable_canvas import st_canvas
except Exception:
    st_canvas = None


# Prefer a wide layout so the canvas can use more horizontal space
try:
    st.set_page_config(layout="wide")
except Exception:
    pass


@st.cache_data
def list_json_label_files() -> List[str]:
    candidates = []
    for base in ("input/labels", "output/predicted_frames"):
        p = Path(base)
        if p.exists() and p.is_dir():
            for f in sorted(p.glob("*.json")):
                candidates.append(str(f))
    return candidates


@st.cache_data
def list_image_files(images_dir: str) -> List[str]:
    p = Path(images_dir)
    if not p.exists() or not p.is_dir():
        return []
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for e in exts:
        files.extend(sorted(p.glob(e)))
    return [str(x) for x in files]


def load_labels(path: str) -> List[Dict[str, Any]]:
    if not Path(path).exists():
        return []
    with open(path, "r") as fh:
        data = json.load(fh)
    return data


def save_labels(path: str, data: List[Dict[str, Any]]):
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def get_output_label_path(original_path: str, output_dir: str, suffix: str) -> str:
    original_name = Path(original_path).stem
    ext = Path(original_path).suffix or ".json"
    return str(Path(output_dir) / f"{original_name}{suffix}{ext}")


def annotate_label_indices(image: Image.Image, labels: Dict[str, Any], scale: float) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for idx, (name, v) in enumerate(labels.items()):
        visible = bool(v[0]) if isinstance(v, list) and len(v) >= 1 else False
        x = v[1] if isinstance(v, list) and len(v) >= 2 else None
        y = v[2] if isinstance(v, list) and len(v) >= 3 else None
        # tolerate a nested list (some files may store [x, y] together)
        if isinstance(x, (list, tuple)) and len(x) > 0:
            x = x[0]
        if isinstance(y, (list, tuple)) and len(y) > 0:
            y = y[0]
        if visible and x is not None and y is not None:
            xx = int(round(float(x) * scale))
            yy = int(round(float(y) * scale))
            label_text = str(idx)
            text_pos = (xx + 8, yy - 8)
            outline_color = "black"
            fill_color = "yellow"
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                draw.text((text_pos[0] + dx, text_pos[1] + dy), label_text, fill=outline_color)
            draw.text(text_pos, label_text, fill=fill_color)
    return annotated


def annotate_canvas_objects(image: Image.Image, objects: List[Dict[str, Any]]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for idx, obj in enumerate(objects):
        left = int(obj.get("left", 0))
        top = int(obj.get("top", 0))
        label_text = str(idx)
        text_pos = (left + 8, top - 8)
        outline_color = "black"
        fill_color = "yellow"
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((text_pos[0] + dx, text_pos[1] + dy), label_text, fill=outline_color)
        draw.text(text_pos, label_text, fill=fill_color)
    return annotated


def frame_list_to_map(frames: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    m = {}
    for entry in frames:
        idx = int(entry.get("frame_idx", -1))
        if idx >= 0:
            m[idx] = entry.get("labels", {})
    return m


def map_to_frame_list(m: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for idx in sorted(m.keys()):
        out.append({"frame_idx": idx, "labels": m[idx]})
    return out


def digits_from_filename(name: str) -> Optional[int]:
    m = re.findall(r"(\d+)", name)
    if not m:
        return None
    return int(m[-1])


def guess_images_dir_from_label_path(label_path: str) -> Optional[str]:
    if not label_path:
        return None
    stem = Path(label_path).stem
    candidate = Path("output") / "extracted_frames" / stem
    if candidate.is_dir():
        return str(candidate)
    return None


st.title("Label Editor")

# Give the main canvas more room by biasing the right column
col1, col2 = st.columns([1, 3])

with col1:
    label_files = list_json_label_files()
    label_choice = st.selectbox("Labels JSON", label_files)

    default_images_dir = guess_images_dir_from_label_path(label_choice) or "output/extracted_frames"
    images_dir = st.text_input("Images folder", default_images_dir)
    image_files = list_image_files(images_dir)
    st.write(f"{len(image_files)} images found")
    st.markdown("---")
    st.write("Controls")

    # Make the display-width control clearly visible in the left column
    max_display_width = st.number_input("Max display width (px)", value=1200, step=100)
    output_labels_dir = st.text_input("Save labels to folder", "output/labels_edited")
    output_name_suffix = st.text_input("Filename suffix", "_new")
    selected_image = st.selectbox("Choose image", image_files)
    filename = Path(selected_image).name if selected_image else ""
    parsed_idx = digits_from_filename(filename) if filename else None
    frame_idx = st.number_input("Frame index (override)", value=parsed_idx or 0, step=1)

    if st.button("Reload labels"):
        st.experimental_rerun()

with col2:
    if not selected_image:
        st.info("No image selected or images folder empty.")
        st.stop()

    if st_canvas is None:
        st.error("Please install streamlit-drawable-canvas: pip install streamlit-drawable-canvas")
        st.stop()
    img = Image.open(selected_image).convert("RGB")
    if not hasattr(img, "width") or not hasattr(img, "height"):
        st.error(f"Background image missing width/height attributes: {type(img)}")
        st.stop()
    # Choose a final display width for the image/canvas based on the left
    # sidebar control and the right-column width approximation.
    col_ratio = 3.0 / 4.0
    canvas_width = min(img.width, int(max_display_width * col_ratio), 1100)
    if canvas_width < img.width:
        scale = float(canvas_width) / float(img.width)
        display_img = img.resize((canvas_width, int(img.height * scale)), Image.LANCZOS)
    else:
        scale = 1.0
        display_img = img

    labels_data = load_labels(label_choice)
    frame_map = frame_list_to_map(labels_data)
    labels_for_frame = frame_map.get(int(frame_idx), {})

    st.subheader(f"Labels for frame {frame_idx}")

    # prepare initial objects for the canvas from existing labels
    initial_objects = []
    for name, v in labels_for_frame.items():
        visible = bool(v[0]) if isinstance(v, list) and len(v) >= 1 else False
        x = v[1] if isinstance(v, list) and len(v) >= 2 else None
        y = v[2] if isinstance(v, list) and len(v) >= 3 else None
        # tolerate nested [x,y] lists
        if isinstance(x, (list, tuple)) and len(x) > 0:
            x = x[0]
        if isinstance(y, (list, tuple)) and len(y) > 0:
            y = y[0]
        if visible and x is not None and y is not None:
            obj = {
                "type": "circle",
                # scale coordinates for display
                "left": float(x) * scale,
                "top": float(y) * scale,
                "radius": 6,
                "fill": "rgba(255,0,0,0.6)",
                "stroke": "black",
                "label": name,
                "originX": "center",
                "originY": "center",
            }
            initial_objects.append(obj)

    tool = st.radio("Tool", ["Move points", "Add points"], index=0, horizontal=True)
    drawing_mode = "transform" if tool == "Move points" else "point"

    # Annotate the display image with initial point indices before showing the canvas
    # (avoid redrawing the background on every canvas update to prevent flicker)
    annotated_display = annotate_canvas_objects(display_img, initial_objects)

    # Shrink the canvas to the right column width approximation.
    # We use the 1:3 column split above, so the right column is ~75% of the
    # available content width. Clamp to avoid overflow.
    canvas_result = st_canvas(
        background_image=annotated_display,
        height=annotated_display.height,
        width=canvas_width,
        drawing_mode=drawing_mode,
        stroke_width=2,
        stroke_color="#00ffa294",
        background_color="#5b0e5b",
        update_streamlit=True,
        initial_drawing={"objects": initial_objects},
        key=f"canvas_{frame_idx}",
    )

    objects = []
    if canvas_result and canvas_result.json_data and "objects" in canvas_result.json_data:
        objects = canvas_result.json_data["objects"]

    # Use session state to preserve label names mapped to object indices
    session_key = f"labels_{frame_idx}"
    if session_key not in st.session_state:
        st.session_state[session_key] = {}

    label_map = st.session_state[session_key]
    
    # Initialize label map from initial_objects if not already set
    if not label_map:
        for idx, init_obj in enumerate(initial_objects):
            label_map[idx] = init_obj.get("label", "")
    
    # Restore or update labels for current objects
    for idx, obj in enumerate(objects):
        if idx in label_map:
            obj["label"] = label_map[idx]
        else:
            obj["label"] = obj.get("label", "")

    # show list of objects and allow labeling
    st.markdown("---")
    st.write("Edit object labels (drag points on the image above). Use the delete checkbox to remove points.")
    obj_rows = []
    for i, obj in enumerate(objects):
        left = int(obj.get("left", 0))
        top = int(obj.get("top", 0))
        label_val = obj.get("label", "")
        cols = st.columns([1, 1, 2, 1])
        with cols[0]:
            st.write(f"#{i}")
        with cols[1]:
            new_label = st.text_input(f"label_{i}", value=label_val, key=f"label_{frame_idx}_{i}")
            label_map[i] = new_label  # Update session state with new label
        with cols[2]:
            st.write(f"x: {left}  y: {top}")
        with cols[3]:
            remove = st.checkbox("Delete", key=f"del_{frame_idx}_{i}")
        obj_rows.append({"index": i, "left": left, "top": top, "label": new_label, "remove": remove})

    # Build updated labels dict from objects
    updated = {}
    for r in obj_rows:
        if r["remove"]:
            continue
        name = r["label"] if r["label"] else f"pt_{r['index']}"
        # un-scale coordinates back to original image space before saving
        orig_x = int(round(r["left"] / scale)) if scale != 0 else int(r["left"])
        orig_y = int(round(r["top"] / scale)) if scale != 0 else int(r["top"])
        updated[name] = [1, orig_x, orig_y]

    st.markdown("---")
    if st.button("Save labels"):
        frame_map[int(frame_idx)] = updated
        out_list = map_to_frame_list(frame_map)
        output_path = get_output_label_path(label_choice, output_labels_dir, output_name_suffix)

        existing_data = []
        if Path(output_path).exists():
            with open(output_path, "r") as fh:
                existing_data = json.load(fh)

        if existing_data == out_list:
            st.info(f"No changes detected. {output_path} was not modified.")
        else:
            save_labels(output_path, out_list)
            st.success(f"Saved {len(updated)} labels to {output_path}")
