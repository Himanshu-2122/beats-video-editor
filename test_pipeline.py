#!/usr/bin/env python
"""Self-contained end-to-end pipeline smoke test."""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.beat import analyze_music_full
from app.sync import ai_assign_clips
from app.video import add_audio, analyze_video_full, concatenate_videos, get_video_duration, process_clip


def _create_test_music(path: Path, duration_seconds: float = 12.0, sample_rate: int = 44100):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, duration_seconds, int(duration_seconds * sample_rate), endpoint=False)
    pulse = (np.sin(2 * np.pi * 2.0 * t) > 0).astype(np.float32)
    carrier = np.sin(2 * np.pi * 220.0 * t)
    signal = carrier * (0.35 + 0.65 * pulse)
    sf.write(str(path), np.clip(signal, -1.0, 1.0).astype(np.float32), sample_rate)


def _ensure_assets():
    video_dir = ROOT / "data" / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    music_dir = ROOT / "data" / "music"
    music_dir.mkdir(parents=True, exist_ok=True)

    video_path = video_dir / "test_video.mp4"
    if not video_path.exists():
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30",
            "-t", "12", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video_path),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    music_path = music_dir / "song.mp3"
    if not music_path.exists():
        _create_test_music(music_path, duration_seconds=12.0)

    return video_path, music_path


def test_pipeline():
    test_video, test_music = _ensure_assets()
    temp_dir = tempfile.mkdtemp(prefix="pipeline_test_")
    try:
        music_analysis = analyze_music_full(str(test_music), cache_dir=os.path.join(temp_dir, "cache"))
        beats = music_analysis.get("beats", [])
        assert beats, "Beat detection produced no beats"

        video_analysis = analyze_video_full(str(test_video), cache_dir=os.path.join(temp_dir, "cache"), fast_mode=True)
        assert float(video_analysis.get("duration", 0)) > 0, "Video analysis did not produce a valid duration"

        ai_results, beat_groups, _, _ = ai_assign_clips(
            beats,
            [str(test_video)],
            min_beats=2,
            max_beats=4,
            sample_interval=0.5,
            scene_threshold=0.4,
            reuse_limit=2,
            compute_motion=False,
            cache_dir=os.path.join(temp_dir, "cache"),
            music_analysis=music_analysis,
            video_analyses=[video_analysis],
        )
        assert ai_results, "Heuristic assignment produced no clips"

        clip_dir = os.path.join(temp_dir, "_final_clips")
        os.makedirs(clip_dir, exist_ok=True)
        final_clip_paths = []
        for i, assignment in enumerate(ai_results):
            clip_path = os.path.join(clip_dir, f"clip_{i + 1:04d}.mp4")
            process_clip(
                video_path=assignment["source_path"],
                duration=assignment["clip_duration"],
                output_path=clip_path,
                start_time=assignment["source_start"],
                snap_frame=True,
            )
            final_clip_paths.append(clip_path)

        video_no_audio = os.path.join(temp_dir, "_video_no_audio.mp4")
        concatenate_videos(
            final_clip_paths,
            video_no_audio,
            beat_groups=[{"clip_path": p} for p in final_clip_paths],
            transition_min=0,
            transition_max=0,
            transition_duration=0.0,
        )

        final_path = os.path.join(temp_dir, "final.mp4")
        add_audio(video_no_audio, str(test_music), final_path)

        out_duration = get_video_duration(final_path)
        assert out_duration > 0, "Rendered output did not produce a valid duration"
        assert os.path.exists(final_path), "Final rendered video was not created"

        print(f"Pipeline smoke test passed: {len(ai_results)} clips, final duration={out_duration:.2f}s")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
