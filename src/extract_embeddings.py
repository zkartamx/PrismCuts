import cv2
import numpy as np
import torch

def get_video_clips(video_path, seconds_per_clip=1.0):
    """
    Samples clips from a video. Each clip is a list of frames.
    This implementation samples a fixed number of frames from each time interval
    to match the model's expected input size.
    """
    NUM_FRAMES_PER_CLIP = 16  # The model expects 16 frames per clip

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print("Warning: Could not determine video FPS. Assuming 30.")
            fps = 30
        clip_duration_in_frames = int(fps * seconds_per_clip)

        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()

    if not all_frames:
        return []

    clips = []
    for i in range(0, len(all_frames), clip_duration_in_frames):
        chunk_of_frames = all_frames[i:i + clip_duration_in_frames]
        if len(chunk_of_frames) == 0:
            continue

        # Uniformly sample frames from the chunk
        indices = np.linspace(0, len(chunk_of_frames) - 1, num=NUM_FRAMES_PER_CLIP, dtype=int)
        sampled_frames = [chunk_of_frames[idx] for idx in indices]
        clips.append(sampled_frames)
        
    return clips


def extract_embeddings(clips, model, processor):
    embeddings = []
    for clip in clips:
        inputs = processor(images=clip, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(emb)
    # Return as a 2D numpy array for consistency with scene detection logic
    return np.array(embeddings)
