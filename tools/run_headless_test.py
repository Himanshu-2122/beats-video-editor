import os
import subprocess
import sys
import argparse

# Ensure project root is on sys.path for local imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.beat import get_beats
from app.sync import sync_clips_with_beats
from app.video import concatenate_videos, encode_in_chunks, add_audio


def parse_args():
    parser = argparse.ArgumentParser(description="Headless test runner for beats-video-editor")
    parser.add_argument("--ai", action="store_true", help="Enable AI-assisted clip matching")
    parser.add_argument("--debug", action="store_true", help="Write debug output (sync_debug_ai.json) and print duration diagnostics")
    parser.add_argument("--sample-interval", type=float, default=0.5, help="Uniform sampling interval in seconds (AI mode)")
    parser.add_argument("--scene-threshold", type=float, default=0.3, help="Scene detection threshold (AI mode)")
    parser.add_argument("--reuse-limit", type=int, default=2, help="Max clips per source video (AI mode)")
    parser.add_argument("--compute-motion", action="store_true", default=True, help="Enable motion analysis (AI mode)")
    parser.add_argument("--no-motion", dest="compute_motion", action="store_false", help="Disable motion analysis (AI mode)")
    parser.add_argument("--min-beats", type=int, default=2, help="Minimum beats per clip")
    parser.add_argument("--max-beats", type=int, default=4, help="Maximum beats per clip")
    return parser.parse_args()


def main():
    args = parse_args()

    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_VIDEOS = os.path.join(ROOT, "data", "videos")
    DATA_MUSIC = os.path.join(ROOT, "data", "music")
    OUTPUT = os.path.join(ROOT, "output")

    os.makedirs(OUTPUT, exist_ok=True)

    # Ensure there is at least one test video
    video_path = os.path.join(DATA_VIDEOS, "test_video.mp4")
    if not os.path.exists(video_path):
        print("No test video found — creating a synthetic test video with ffmpeg (10s)")
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30", "-t", "10",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", video_path
        ]
        subprocess.run(cmd, check=True)

    # Music file
    music_file = os.path.join(DATA_MUSIC, "song.mp3")
    if not os.path.exists(music_file):
        print("No music file found at data/music/song.mp3 — aborting test")
        sys.exit(1)

    print("Running beat detection...")
    beats = get_beats(music_file)
    print(f"Detected {len(beats)} beats")
    if len(beats) < 2:
        print("Not enough beats to run full pipeline; exiting.")
        sys.exit(1)

    # Run sync to create clips
    print("Syncing clips with beats...")
    video_paths = [os.path.join(DATA_VIDEOS, f) for f in os.listdir(DATA_VIDEOS) if f.lower().endswith('.mp4')]

    clip_paths, beat_groups, temp_dir = sync_clips_with_beats(video_paths, beats, **sync_kwargs)

    # Report target vs actual durations
    music_duration = None
    if music_analysis and isinstance(music_analysis, dict):
        md = music_analysis.get("duration", None)
        if isinstance(md, (int, float)):
            music_duration = float(md)
    if music_duration is None or music_duration <= 0:
        music_duration = float(beats[-1]["time"]) if beats else 0.0

    total_clip_duration = sum(bg.get("clip_duration", 0.0) for bg in beat_groups)
    print(f"TARGET music_duration={music_duration:.3f}s  total_clip_duration={total_clip_duration:.3f}s  clip_count={len(clip_paths)}")
    print(f"DEBUG: beat_groups={len(beat_groups)}")
    for i, bg in enumerate(beat_groups):
        print(f"DEBUG beat_group#{i+1}: start={bg.get('start',0):.3f} end={bg.get('end',0):.3f} dur={bg.get('clip_duration', bg.get('duration',0)):.3f} src_start={bg.get('source_start',0):.3f}")

    print(f"Created {len(clip_paths)} clips in {temp_dir}")
    if not clip_paths:
        print("No clips created; exiting.")
        sys.exit(1)

    # Concatenate
    out_no_audio = os.path.join(OUTPUT, "test_video_no_audio.mp4")
    print("Concatenating clips into", out_no_audio)
    concatenate_videos(clip_paths, out_no_audio, beat_groups=beat_groups)

    # Re-encode low-RAM
    encoded = os.path.join(OUTPUT, "test_video_encoded.mp4")
    print("Encoding in chunks...")
    encode_in_chunks(out_no_audio, encoded, segment_time=5, logger=print)

    # Add audio
    final = os.path.join(OUTPUT, "test_final.mp4")
    print("Adding audio to final video...", final)
    add_audio(encoded, music_file, final)

    print("Test pipeline complete. Output:", final)

    # Verify final output
    try:
        import shutil
        total_video_duration = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", final],
            stdout=subprocess.PIPE, text=True).stdout.strip())
        print(f"FINAL output duration={total_video_duration:.3f}s  music_duration={music_duration:.3f}s  delta={total_video_duration - music_duration:+.3f}s")
    except Exception as exc:
        print(f"Could not probe final output: {exc}")

    # Cleanup temp dir
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()