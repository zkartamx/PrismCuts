import ffmpeg
import os

def split_video(input_path, scene_changes, seconds_per_clip, output_dir):
    for i in range(len(scene_changes) - 1):
        start = scene_changes[i] * seconds_per_clip
        duration = (scene_changes[i + 1] - scene_changes[i]) * seconds_per_clip
        output_path = os.path.join(output_dir, f"scene_{i + 1}.mp4")

        (
            ffmpeg
            .input(input_path, ss=start, t=duration)
            .output(output_path, codec='copy')
            .run(overwrite_output=True)
        )
