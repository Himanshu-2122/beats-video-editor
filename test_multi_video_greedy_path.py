#!/usr/bin/env python
"""Regression test for the multi-video greedy path."""
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
from app.video import analyze_video_full


def _write_music(path: Path, duration_seconds: float = 30.0, sample_rate: int = 44100):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, duration_seconds, int(duration_seconds * sample_rate), endpoint=False)
    pulse = (np.sin(2 * np.pi * 2.0 * t) > 0).astype(np.float32)
    carrier = np.sin(2 * np.pi * 220.0 * t)
    signal = carrier * (0.45 + 0.55 * pulse)
    sf.write(str(path), np.clip(signal, -1.0, 1.0).astype(np.float32), sample_rate)


def _create_video(path: Path, duration_seconds: float = 18.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30",
        "-t", str(duration_seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_multi_video_greedy_path():
    temp_dir = tempfile.mkdtemp(prefix="regression_test_")
    try:
        video_dir = Path(temp_dir) / "videos"
        video_dir.mkdir(exist_ok=True)
        video_paths = [video_dir / f"video_{i}.mp4" for i in range(3)]
        for path in video_paths:
            _create_video(path, duration_seconds=18.0)

        music_path = Path(temp_dir) / "music.mp3"
        _write_music(music_path, duration_seconds=30.0)

        music_analysis = analyze_music_full(str(music_path), cache_dir=os.path.join(temp_dir, "cache"))
        beats = music_analysis.get("beats", [])
        assert beats, "Regression audio produced no detected beats"

        video_analyses = [
            analyze_video_full(str(vp), cache_dir=os.path.join(temp_dir, "cache"), fast_mode=True)
            for vp in video_paths
        ]

        ai_results, beat_groups, _, _ = ai_assign_clips(
            beats,
            [str(vp) for vp in video_paths],
            min_beats=4,
            max_beats=8,
            sample_interval=2.0,
            scene_threshold=0.4,
            reuse_limit=None,
            compute_motion=False,
            cache_dir=os.path.join(temp_dir, "cache"),
            music_analysis=music_analysis,
            video_analyses=video_analyses,
        )

        num_assignments = len(ai_results)
        num_beat_groups = len(beat_groups)
        assert num_assignments > 0, "REGRESSION: No clips assigned in the multi-video path"
        coverage_ratio = num_assignments / num_beat_groups if num_beat_groups else 0.0
        assert coverage_ratio >= 0.5, f"Coverage too low: {coverage_ratio:.1%}"

        videos_used = {a["source_path"] for a in ai_results}
        assert len(videos_used) >= 2, f"Expected at least 2 videos used, got {len(videos_used)}"

        print(f"Multi-video regression passed: {num_assignments} assignments across {len(videos_used)} videos")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
