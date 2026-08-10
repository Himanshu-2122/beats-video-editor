# import os
# import shutil
# import gc

# try:
#     from PIL import Image
#     if not hasattr(Image, "ANTIALIAS"):
#         Image.ANTIALIAS = Image.Resampling.LANCZOS
# except Exception:
#     pass

# import moviepy.config as cfg
# from moviepy.editor import AudioFileClip, concatenate_videoclips, VideoFileClip

# from app.beat import get_beats
# from app.sync import sync_clips_with_beats
# from app.video import load_video_paths

# cfg.FFMPEG_BINARY = "ffmpeg"

# music_path = "data/music/song.mp3"
# video_folder = "data/videos"

# print("[1/5] Loading video paths...")
# video_paths = load_video_paths(video_folder)
# print(f"       Found {len(video_paths)} videos")

# print("[2/5] Loading beats...")
# beats = get_beats(music_path)
# print(f"       Found {len(beats)} beats")

# print("[3/5] Processing clips...")
# temp_clip_paths, temp_dir = sync_clips_with_beats(video_paths, beats)
# print(f"       Created {len(temp_clip_paths)} temp clips")

# print("[4/5] Concatenating clips...")
# # Load clips one at a time to avoid memory issues
# clips = []
# for p in temp_clip_paths:
#     clip = VideoFileClip(p)
#     clips.append(clip)

# final = concatenate_videoclips(clips, method="compose")
# print("       Concatenation done")

# # Close individual clips to free memory
# for c in clips:
#     c.close()
# gc.collect()

# print("[5/5] Adding audio and rendering...")
# audio = AudioFileClip(music_path)
# final = final.set_audio(audio)

# output_path = "output/final.mp4"
# os.makedirs("output", exist_ok=True)

# print("       Writing video file...")
# final.write_videofile(
#     output_path,
#     fps=30,
#     codec="libx264",
#     audio_codec="aac",
#     bitrate="10000k",
#     preset="fast",
#     threads=2,
# )

# print("       Cleaning up...")
# for c in clips:
#     c.close()
# audio.close()
# final.close()

# # Clean up temp files
# shutil.rmtree(temp_dir, ignore_errors=True)

# print(f"Done! Video saved at {output_path}")

import os
import shutil

from app.beat import get_beats
from app.sync import sync_clips_with_beats
from app.video import concatenate_videos, load_video_paths, add_audio


MUSIC_PATH = "data/music/song.mp3"
VIDEO_FOLDER = "data/videos"
OUTPUT_PATH = "output/final.mp4"


def main():
    print("=" * 60)
    print("        BEATS VIDEO EDITOR")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load videos
    # ---------------------------------------------------------
    print("\n[1/5] Loading video paths...")
    video_paths = load_video_paths(VIDEO_FOLDER)

    if not video_paths:
        raise RuntimeError(f"No video files found in: {VIDEO_FOLDER}")

    print(f"       Found {len(video_paths)} videos")

    # ---------------------------------------------------------
    # 2. Detect beats
    # ---------------------------------------------------------
    print("\n[2/5] Loading beats...")
    beats = get_beats(MUSIC_PATH)

    if len(beats) < 2:
        raise RuntimeError("Not enough beats detected from the music.")

    print(f"       Found {len(beats)} beats")

    # ---------------------------------------------------------
    # 3. Generate temporary clips
    # ---------------------------------------------------------
    print("\n[3/5] Processing clips...")
    temp_clip_paths, temp_dir = sync_clips_with_beats(video_paths, beats)

    if not temp_clip_paths:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError("No temporary clips were generated.")

    print(f"       Created {len(temp_clip_paths)} temp clips")

    # ---------------------------------------------------------
    # 4. Concatenate clips
    # ---------------------------------------------------------
    print("\n[4/5] Concatenating clips...")

    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    video_without_audio = os.path.join(temp_dir, "combined_video.mp4")

    concatenate_videos(temp_clip_paths, video_without_audio)
    print("       Concatenation done")

    # ---------------------------------------------------------
    # 5. Add music + final render
    # ---------------------------------------------------------
    print("\n[5/5] Adding audio and rendering...")
    add_audio(video_path=video_without_audio, audio_path=MUSIC_PATH, output_path=OUTPUT_PATH)

    print("\nCleaning up temporary files...")
    shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Video saved at: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()