import os
import shutil

from app.beat import get_beats, analyze_music_full

from app.sync import (
    sync_clips_with_beats,
)

from app.video import (
    load_video_paths,
    concatenate_videos,
    add_audio,
    encode_in_chunks,
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

    # Full music analysis for AI matching
    music_analysis = analyze_music_full(MUSIC_PATH)
    print(f"    BPM: {music_analysis.get('bpm', 'N/A'):.1f}")
    print(f"    Beats: {len(music_analysis.get('beats', []))}")
    print(f"    Drops: {len(music_analysis.get('drops', []))}")
    print(f"    Sections: {len(music_analysis.get('sections', []))}")

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
            ai_mode=True,
            music_analysis=music_analysis,
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
        beat_groups=beat_groups,
    )

    # ========================================================
    # Low-RAM Re-encode
    # ========================================================

    encoded = os.path.join(work_dir, "video_encoded.mp4")
    encode_in_chunks(no_audio, encoded, segment_time=60)

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
        encoded,
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