import os
import shutil

from app.beat import get_beats

from app.sync import (
    sync_clips_with_beats,
)

from app.video import (
    load_video_paths,
    concatenate_videos,
    add_audio,
)


MUSIC_PATH = "data/music/song.mp3"

VIDEO_FOLDER = "data/videos"

OUTPUT_FOLDER = "output"


def main():

    print("=" * 60)
    print("        BEAT VIDEO EDITOR")
    print("=" * 60)

    # ========================================================
    # Videos
    # ========================================================

    print(
        "\n[1/5] Loading videos..."
    )

    video_paths = load_video_paths(
        VIDEO_FOLDER
    )

    print(
        f"Found {len(video_paths)} videos."
    )

    if not video_paths:

        raise RuntimeError(
    "No videos found."
)

# ========================================================
# Beats
# ========================================================

print(
    "\n[2/5] Detecting beats..."
)

beats = get_beats(
    MUSIC_PATH
)

if len(beats) < 2:

    raise RuntimeError(
"Not enough beats."
)

# ========================================================
# Work directory
# ========================================================

work_dir = os.path.join(
    OUTPUT_FOLDER,
    "_work",
)

os.makedirs(
    work_dir,
    exist_ok=True,
)

# ========================================================
# Clips
# ========================================================

print(
    "\n[3/5] Generating clips..."
)

clip_paths, beat_groups, _ = (
    sync_clips_with_beats(
        video_paths,
        beats,
        work_dir,
        min_beats=4,
        max_beats=8,
    )
)

if not clip_paths:

    raise RuntimeError(
"No clips generated."
)

# ========================================================
# Transitions
# ========================================================

print(
    "\n[4/5] Creating final video..."
)

no_audio = os.path.join(
    work_dir,
    "video_no_audio.mp4",
)

concatenate_videos(
    clip_paths,
    no_audio,
)

# ========================================================
# Audio
# ========================================================

print(
    "\n[5/5] Adding music..."
)

final_path = os.path.join(
    OUTPUT_FOLDER,
    "final.mp4",
)

add_audio(
    no_audio,
    MUSIC_PATH,
    final_path,
)

# ========================================================
# Cleanup
# ========================================================

shutil.rmtree(
    work_dir,
    ignore_errors=True,
)

print(
    "\nDONE!"
)

print(
    f"Final video: {final_path}"
)


if __name__ == "__main__":
    main()