import os
import subprocess
import sys
import argparse
import shutil
import numpy as np
import soundfile as sf

# Ensure project root is on sys.path for local imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.beat import get_beats, analyze_music_full
from app.sync import sync_clips_with_beats
from app.video import concatenate_videos, encode_in_chunks, add_audio


def build_synthetic_music(path: str, duration_seconds: float = 12.0, sample_rate: int = 44100) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = np.linspace(0.0, duration_seconds, int(duration_seconds * sample_rate), endpoint=False)
    beat_gate = (np.sin(2 * np.pi * 2.0 * t) > 0).astype(np.float32)
    carrier = np.sin(2 * np.pi * 220.0 * t)
    signal = carrier * (0.35 + 0.65 * beat_gate)
    signal = np.clip(signal, -1.0, 1.0)
    sf.write(path, signal.astype(np.float32), sample_rate)


def ensure_test_assets():
    data_videos = os.path.join(ROOT, "data", "videos")
    data_music = os.path.join(ROOT, "data", "music")
    os.makedirs(data_videos, exist_ok=True)
    os.makedirs(data_music, exist_ok=True)

    video_path = os.path.join(data_videos, "test_video.mp4")
    if not os.path.exists(video_path):
        print("No test video found — creating a synthetic test video with ffmpeg (8s)")
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30", "-t", "8",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", video_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    music_file = os.path.join(data_music, "song.mp3")
    if not os.path.exists(music_file):
        print("No music file found — creating a synthetic music file")
        build_synthetic_music(music_file, duration_seconds=12.0)

    return video_path, music_file, data_videos, data_music


def parse_args():
    parser = argparse.ArgumentParser(description="Headless test runner for beats-video-editor")
    parser.add_argument("--ai", action="store_true", help="Enable heuristic clip matching")
    parser.add_argument("--debug", action="store_true", help="Write debug output and print duration diagnostics")
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
    output_dir = os.path.join(ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)

    video_path, music_file, data_videos, _ = ensure_test_assets()
    music_analysis = analyze_music_full(music_file)
    beats = music_analysis.get("beats", get_beats(music_file))

    print("Running beat detection...")
    print(f"Detected {len(beats)} beats")
    if len(beats) < 2:
        print("Not enough beats to run full pipeline; exiting.")
        sys.exit(1)

    video_paths = [os.path.join(data_videos, f) for f in sorted(os.listdir(data_videos)) if f.lower().endswith(".mp4")]
    if not video_paths:
        print("No mp4 test videos found; exiting.")
        sys.exit(1)

    sync_kwargs = {
        "output_dir": os.path.join(output_dir, "headless_test_run"),
        "min_beats": args.min_beats,
        "max_beats": args.max_beats,
        "ai_mode": args.ai,
        "sample_interval": args.sample_interval,
        "scene_threshold": args.scene_threshold,
        "reuse_limit": args.reuse_limit,
        "compute_motion": args.compute_motion,
        "music_analysis": music_analysis,
    }

    print("Syncing clips with beats...")
    clip_paths, beat_groups, temp_dir = sync_clips_with_beats(video_paths, beats, **sync_kwargs)

    music_duration = float(music_analysis.get("duration", beats[-1]["time"])) if beats else 0.0
    total_clip_duration = sum(float(bg.get("clip_duration", bg.get("duration", 0.0))) for bg in beat_groups)
    print(f"TARGET music_duration={music_duration:.3f}s  total_clip_duration={total_clip_duration:.3f}s  clip_count={len(clip_paths)}")
    print(f"DEBUG: beat_groups={len(beat_groups)}")
    for i, bg in enumerate(beat_groups):
        print(f"DEBUG beat_group#{i + 1}: start={bg.get('start', 0):.3f} end={bg.get('end', 0):.3f} dur={bg.get('clip_duration', bg.get('duration', 0)):.3f} src_start={bg.get('source_start', 0):.3f}")

    print(f"Created {len(clip_paths)} clips in {temp_dir}")
    if not clip_paths:
        print("No clips created; exiting.")
        sys.exit(1)

    out_no_audio = os.path.join(output_dir, "test_video_no_audio.mp4")
    print("Concatenating clips into", out_no_audio)
    concatenate_videos(clip_paths, out_no_audio, beat_groups=beat_groups)

    encoded = os.path.join(output_dir, "test_video_encoded.mp4")
    print("Encoding in chunks...")
    encode_in_chunks(out_no_audio, encoded, segment_time=5, logger=print)

    final = os.path.join(output_dir, "test_final.mp4")
    print("Adding audio to final video...", final)
    add_audio(encoded, music_file, final)

    print("Test pipeline complete. Output:", final)

    try:
        total_video_duration = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", final],
            stdout=subprocess.PIPE, text=True,
            check=True,
        ).stdout.strip())
        print(f"FINAL output duration={total_video_duration:.3f}s  music_duration={music_duration:.3f}s  delta={total_video_duration - music_duration:+.3f}s")
    except Exception as exc:
        print(f"Could not probe final output: {exc}")

    if args.debug:
        debug_path = os.path.join(output_dir, "sync_debug_ai.json")
        with open(debug_path, "w", encoding="utf-8") as fh:
            fh.write("{\n  \"status\": \"ok\",\n  \"video\": \"%s\",\n  \"clip_count\": %d\n}\n" % (video_path, len(clip_paths)))
        print(f"Wrote debug metadata to {debug_path}")

    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()