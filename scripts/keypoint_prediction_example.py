from modules.videos import extract_frames, save_frames_with_predictions, get_frame_numbers 
from modules.keypoint_rtmpose_predict_common import keypoint_extraction_rtmpose, load_model_from_checkpoint_for_inference
from modules.detector_ssdlite_model import detector_extraction_ssdlite, load_detector
from pathlib import Path

device_obj = "cuda"

detector_path = "output/ssdlite/weak_20260326_090250"
keypoint_path = "output/RTMPose/no_weak_20260328_174401"

video_path = "videos/RAT 2 FR1 10-02-25.mp4"

# get the frame numbers in the video
frame_numbers = get_frame_numbers(video_path)
print(frame_numbers)

# extract frames from the video
frames = extract_frames(video_path, 
                        frame_indices = [0, 100, 200])

len(frames)
frames[0].shape


# bounding boxes
detector = load_detector(
    detector_path=detector_path,
    device=device_obj,
)

detection_boxes = detector_extraction_ssdlite(
    detector,
    frames,
    score_threshold = 0.1
)

# keypoints models
model, _ = load_model_from_checkpoint_for_inference(
    model_path=keypoint_path,
    device=device_obj,
)

keypoints = keypoint_extraction_rtmpose(
    model, 
    frames,
    detection_boxes,
)


# save the frame along with the detection and keypoint results for visualization
save_frames_with_predictions(
    frames=frames,
    detection_boxes=detection_boxes,
    keypoints=keypoints,
    output_folder="output/debug_predictions",
    file_names=None,
)