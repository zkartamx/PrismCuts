import ffmpeg
import os

def split_video(input_path, scene_changes, seconds_per_clip, output_dir):
    """Cuts a video into scenes based on a list of start times."""
    if not scene_changes:
        print("Warning: No scene changes provided.")
        return

    try:
        probe = ffmpeg.probe(input_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        total_duration = float(video_info['duration'])
    except (ffmpeg.Error, StopIteration) as e:
        print(f"Error probing video file: {e}")
        return

    # Create a list of start times for each scene
    start_times = [sc * seconds_per_clip for sc in scene_changes]

    for i, start_time in enumerate(start_times):
        # The end time is the start of the next scene, or the total duration for the last scene
        end_time = start_times[i + 1] if i + 1 < len(start_times) else total_duration
        duration = end_time - start_time

        if duration <= 0:
            continue

        output_path = os.path.join(output_dir, f"scene_{i + 1}.mp4")
        print(f"Creating scene {i+1}: from {start_time:.2f}s to {end_time:.2f}s")

        try:
            (
                ffmpeg
                .input(input_path, ss=start_time, t=duration)
                .output(output_path, c='copy', avoid_negative_ts=1)
                .run(overwrite_output=True, quiet=True)
            )
        except ffmpeg.Error as e:
            print(f"Error during ffmpeg processing for scene {i+1}: {e.stderr}")
