from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from modules.manual_correction_utils import build_label_json_payload, load_label_json


REPO_ROOT = Path(__file__).resolve().parents[2]
LABEL_JSON_ROOT = REPO_ROOT / "input" / "labels"
FRAMES_ROOT = REPO_ROOT / "output" / "extracted_frames"

# Keypoints from input/labels/README.md. Add or remove names here as needed.
DEFAULT_KEYPOINTS = [
    "nose",
    "head",
    "spine1",
    "spine2",
    "spine3",
    "tailbase",
    "tail1",
    "tail2",
    "tail_tip",
    "L_shoulder",
    "L_frontpaw",
    "R_shoulder",
    "R_frontpaw",
    "L_hip",
    "L_knee",
    "L_backpaw",
    "R_hip",
    "R_knee",
    "R_backpaw",
]


def get_video_names() -> list[str]:
    video_names = set()
    if FRAMES_ROOT.is_dir():
        for path in sorted(FRAMES_ROOT.iterdir()):
            if path.is_dir():
                video_names.add(path.name)
    if LABEL_JSON_ROOT.is_dir():
        for path in sorted(LABEL_JSON_ROOT.glob("*.json")):
            video_names.add(path.stem)
    return sorted(video_names)


def get_frame_image_path(video_name: str, frame_idx: int) -> Path:
    return FRAMES_ROOT / video_name / f"{frame_idx:08d}.jpg"


def load_labels(video_name: str) -> pd.DataFrame:
    label_path = LABEL_JSON_ROOT / f"{video_name}.json"
    if label_path.is_file():
        df = load_label_json(label_path)
    else:
        df = pd.DataFrame(columns=["frame"])
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype("Int64")
    return df


def ensure_keypoint_columns(df: pd.DataFrame, keypoints: list[str]) -> pd.DataFrame:
    for keypoint in keypoints:
        xcol = f"{keypoint}_x"
        ycol = f"{keypoint}_y"
        if xcol not in df.columns:
            df[xcol] = pd.NA
        if ycol not in df.columns:
            df[ycol] = pd.NA
    return df


def ensure_frame_row(df: pd.DataFrame, frame_idx: int) -> pd.DataFrame:
    if not (df["frame"] == frame_idx).any():
        row = {col: pd.NA for col in df.columns}
        row["frame"] = int(frame_idx)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df.sort_values("frame").reset_index(drop=True)


def get_existing_keypoints(df: pd.DataFrame) -> list[str]:
    keypoints = sorted({col[:-2] for col in df.columns if col.endswith("_x") and f"{col[:-2]}_y" in df.columns})
    return keypoints


def update_label(df: pd.DataFrame, frame_idx: int, keypoint: str, visible: bool, x: int | None, y: int | None) -> pd.DataFrame:
    xcol = f"{keypoint}_x"
    ycol = f"{keypoint}_y"
    df = ensure_frame_row(df, frame_idx)
    row_mask = df["frame"] == frame_idx
    if visible and x is not None and y is not None:
        df.loc[row_mask, xcol] = int(x)
        df.loc[row_mask, ycol] = int(y)
    else:
        df.loc[row_mask, xcol] = pd.NA
        df.loc[row_mask, ycol] = pd.NA
    return df


def frame_to_image_bytes(video_name: str, frame_idx: int) -> bytes | None:
    image_path = get_frame_image_path(video_name, frame_idx)
    if not image_path.is_file():
        return None
    return image_path.read_bytes()


def save_labels(video_name: str, df: pd.DataFrame, keypoints: list[str]) -> None:
    payload = build_label_json_payload(df, keypoints)
    LABEL_JSON_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = LABEL_JSON_ROOT / f"{video_name}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


st.set_page_config(page_title="Tracking Labeler", layout="wide")
st.title("Streamlit Tracking Labeler")

video_names = get_video_names()
if not video_names:
    st.error("No videos found in output/extracted_frames or input/labels.")
    st.stop()

video_name = st.sidebar.selectbox("Video", video_names)
label_df = load_labels(video_name)
existing_keypoints = get_existing_keypoints(label_df)
all_keypoints = sorted(set(DEFAULT_KEYPOINTS) | set(existing_keypoints))
label_df = ensure_keypoint_columns(label_df, all_keypoints)

frame_indices = sorted(label_df["frame"].dropna().astype(int).unique().tolist())
if not frame_indices:
    frame_dir = FRAMES_ROOT / video_name
    if frame_dir.is_dir():
        frame_indices = sorted(
            int(p.stem) for p in frame_dir.glob("*.jpg") if p.stem.isdigit()
        )

if not frame_indices:
    st.warning("No frame images or existing labels found for this video.")
    st.stop()

frame_idx = st.sidebar.selectbox("Frame index", frame_indices)
selected_keypoint = st.sidebar.selectbox("Keypoint", all_keypoints)
show_json = st.sidebar.checkbox("Show JSON payload", value=False)

image_bytes = frame_to_image_bytes(video_name, frame_idx)
if image_bytes is None:
    st.error(f"Frame image not found: {video_name}/{frame_idx:08d}.jpg")
    st.stop()

col_left, col_right = st.columns([3, 1])
with col_left:
    st.image(image_bytes, caption=f"{video_name} frame {frame_idx}", use_column_width=True)

with col_right:
    with st.form("label_form"):
        visible = st.checkbox("Visible", value=True)
        x = st.number_input("X", min_value=0, value=0, step=1)
        y = st.number_input("Y", min_value=0, value=0, step=1)
        add_new_keypoint = st.text_input("Add new keypoint", value="")
        submitted = st.form_submit_button("Update label")

    if add_new_keypoint.strip():
        new_keypoint = add_new_keypoint.strip()
        if new_keypoint not in all_keypoints:
            all_keypoints.append(new_keypoint)
            label_df = ensure_keypoint_columns(label_df, [new_keypoint])
            st.experimental_rerun()

    if submitted:
        label_df = update_label(label_df, frame_idx, selected_keypoint, visible, x if visible else None, y if visible else None)
        st.success(f"Updated {selected_keypoint} on frame {frame_idx}")
        st.experimental_rerun()

    st.markdown("### Existing labels")
    row_mask = label_df["frame"] == frame_idx
    if row_mask.any():
        display_row = label_df.loc[row_mask, ["frame"] + [f"{kp}_{c}" for kp in all_keypoints for c in ("x", "y")]]
        st.dataframe(display_row.fillna(""), use_container_width=True)
    else:
        st.info("No label row exists for this frame yet.")

    if st.button("Save JSON"):
        save_labels(video_name, label_df, all_keypoints)
        st.success(f"Saved labels to input/labels/{video_name}.json")

    if show_json:
        payload = build_label_json_payload(label_df, all_keypoints)
        st.json(payload)
