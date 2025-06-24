import cv2
import numpy as np
from transformers import AutoProcessor, AutoModel
import torch

def get_video_clips(video_path, seconds_per_clip=1.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_per_clip = int(fps * seconds_per_clip)
    clips = []

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

        if len(frames) == frames_per_clip:
            clips.append(frames.copy())
            frames.clear()

    cap.release()
    return clips

def extract_embeddings(clips, model, processor):
    embeddings = []
    for clip in clips:
        inputs = processor(videos=[clip], return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(emb)
    return embeddings
