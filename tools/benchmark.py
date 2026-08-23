#!/usr/bin/env python3
"""Benchmark script for Beats Video Editor - times each pipeline stage individually."""

import sys
import os
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.beat import analyze_music_full, get_beats
from app.video import (
    analyze_video_full, 
    generate_candidates, 
    get_video_duration,
    process_clip,
    concatenate_videos,
    add_audio,
    encode_in_chunks,
    create_random_groups,
)
from app.sync import (
    sync_clips_with_beats, 
    ai_assign_clips, 
    create_beat_groups,
    _normalize_beats,
)
from app.cache import cache_manager


def benchmark_music_analysis(music_path):
    """Stage 1: Music/beat analysis"""
    print("\n" + "="*60)
    print("STAGE 1: Music/Beat Analysis")
    print("="*60)
    
    # Clear cache for fair comparison
    cache_manager.clear_category("beats")
    
    start = time.perf_counter()
    music_analysis = analyze_music_full(music_path, cache_dir=None)
    elapsed = time.perf_counter() - start
    
    beats = music_analysis.get("beats", [])
    print(f"  Duration: {music_analysis.get('duration', 0):.2f}s")
    print(f"  BPM: {music_analysis.get('bpm', 0):.1f}")
    print(f"  Beats detected: {len(beats)}")
    print(f"  Drops: {len(music_analysis.get('drops', []))}")
    print(f"  Buildups: {len(music_analysis.get('buildups', []))}")
    print(f"  Sections: {len(music_analysis.get('sections', []))}")
    print(f"  Time: {elapsed:.2f}s")
    
    return elapsed, music_analysis


def benchmark_video_analysis(video_paths):
    """Stage 2: Video analysis per source video"""
    print("\n" + "="*60)
    print("STAGE 2: Video Analysis (per source)")
    print("="*60)
    
    cache_manager.clear_category("video_analysis")
    
    total_start = time.perf_counter()
    analyses = []
    
    for i, vp in enumerate(video_paths):
        print(f"\n  Video {i+1}/{len(video_paths)}: {Path(vp).name}", flush=True)
        start = time.perf_counter()
        # Use fast_mode=False to test full analysis, but limit duration
        analysis = analyze_video_full(vp, cache_dir=None, fast_mode=True)
        elapsed = time.perf_counter() - start
        print(f"    analyze_video_full completed in {elapsed:.2f}s", flush=True)
        
        print(f"    Duration: {analysis.get('duration', 0):.2f}s")
        print(f"    Resolution: {analysis.get('width', 0)}x{analysis.get('height', 0)}")
        print(f"    FPS: {analysis.get('fps', 0):.1f}")
        print(f"    Scenes: {len(analysis.get('scenes', []))}")
        print(f"    Highlights: {len(analysis.get('highlights', []))}")
        print(f"    Camera changes: {len(analysis.get('camera_changes', []))}")
        print(f"    Time: {elapsed:.2f}s")
        analyses.append(analysis)
    
    total_elapsed = time.perf_counter() - total_start
    print(f"\n  Total video analysis time: {total_elapsed:.2f}s")
    
    return total_elapsed, analyses


def benchmark_candidate_generation(video_paths, music_analysis, beat_groups):
    """Stage 3: Candidate generation"""
    print("\n" + "="*60)
    print("STAGE 3: Candidate Generation")
    print("="*60, flush=True)
    
    cache_manager.clear_category("scenes")
    cache_manager.clear_category("motion")
    
    # Unique durations needed - round to reduce count
    unique_durations = sorted(set(round(bg["duration"], 1) for bg in beat_groups))
    print(f"  Unique clip durations needed (rounded): {len(unique_durations)}", flush=True)
    for d in unique_durations:
        print(f"    - {d:.1f}s", flush=True)
    
    total_start = time.perf_counter()
    all_candidates = {}
    
    for i, vp in enumerate(video_paths):
        print(f"\n  Video {i+1}/{len(video_paths)}: {Path(vp).name}", flush=True)
        vp_start = time.perf_counter()
        all_candidates[vp] = {}
        
        for duration in unique_durations:
            print(f"    Generating candidates for duration {duration:.1f}s...", flush=True)
            start = time.perf_counter()
            candidates = generate_candidates(
                vp,
                duration,
                sample_interval=1.0,  # Increased from 0.5 for speed
                scene_threshold=0.4,  # Increased from 0.3
                max_candidates=50,    # Reduced from 100
                compute_motion=False, # Disable motion for benchmark speed
                cache_dir=None,
            )
            elapsed = time.perf_counter() - start
            all_candidates[vp][duration] = candidates
            print(f"    Duration {duration:.1f}s: {len(candidates)} candidates in {elapsed:.2f}s", flush=True)
        
        vp_elapsed = time.perf_counter() - vp_start
        print(f"    Total for video: {vp_elapsed:.2f}s", flush=True)
    
    total_elapsed = time.perf_counter() - total_start
    total_cands = sum(len(c) for v in all_candidates.values() for c in v.values())
    print(f"\n  Total candidates generated: {total_cands}")
    print(f"  Total candidate generation time: {total_elapsed:.2f}s")
    
    return total_elapsed, all_candidates


def benchmark_clip_assignment(video_paths, music_analysis, beat_groups, video_analyses, all_candidates):
    """Stage 4: Clip assignment (AI matching)"""
    print("\n" + "="*60)
    print("STAGE 4: Clip Assignment (AI Matching)")
    print("="*60, flush=True)
    
    start = time.perf_counter()
    
    print("  Calling ai_assign_clips...", flush=True)
    ai_results, all_beat_groups, debug_scores, _ = ai_assign_clips(
        music_analysis.get("beats", []),
        video_paths,
        min_beats=4,
        max_beats=8,
        sample_interval=1.0,  # Match candidate generation
        scene_threshold=0.4,  # Match candidate generation
        reuse_limit=None,
        compute_motion=False, # Match candidate generation
        cache_dir=None,
        music_analysis=music_analysis,
        video_analyses=video_analyses,
    )
    print("  ai_assign_clips completed", flush=True)
    
    elapsed = time.perf_counter() - start
    
    # Filter beat_groups to only those that got assigned (honest coverage)
    assigned_beat_indices = {a["beat_idx"] for a in ai_results}
    used_beat_groups = [bg for i, bg in enumerate(all_beat_groups) if i in assigned_beat_indices]
    
    print(f"  Beat groups (total): {len(all_beat_groups)}")
    print(f"  Beat groups (assigned): {len(used_beat_groups)}")
    print(f"  Clips assigned: {len(ai_results)}")
    print(f"  Total music coverage (assigned only): {sum(bg['duration'] for bg in used_beat_groups):.2f}s / {sum(bg['duration'] for bg in all_beat_groups):.2f}s planned")
    print(f"  Time: {elapsed:.2f}s")
    
    # Print assignment details
    for i, a in enumerate(ai_results):
        print(f"    Clip {i+1}: {Path(a['source_path']).name} @ {a['source_start']:.2f}s "
              f"({a['clip_duration']:.2f}s) score={a['score']:.3f} "
              f"type={a.get('beat_type','?')} energy={a.get('beat_energy',0):.2f}")
    
    return elapsed, ai_results, used_beat_groups


def benchmark_ffmpeg_render(ai_results, used_beat_groups, output_path):
    """Stage 5: FFmpeg rendering/concat"""
    print("\n" + "="*60)
    print("STAGE 5: FFmpeg Rendering (clips + concat + transitions + audio)")
    print("="*60)
    
    import tempfile
    import shutil
    
    work_dir = tempfile.mkdtemp(prefix="bench_render_")
    clip_dir = Path(work_dir) / "_final_clips"
    clip_dir.mkdir(exist_ok=True)
    
    total_start = time.perf_counter()
    
    # 5a: Process individual clips
    print("\n  5a: Processing individual clips...")
    clip_start = time.perf_counter()
    final_clip_paths = []
    
    for index, assignment in enumerate(ai_results):
        source_path = assignment["source_path"]
        duration = assignment["clip_duration"]
        start_time = assignment["source_start"]
        
        clip_path = clip_dir / f"clip_{index + 1:04d}.mp4"
        
        c_start = time.perf_counter()
        process_clip(
            video_path=source_path,
            duration=duration,
            output_path=str(clip_path),
            start_time=start_time,
            snap_frame=True,
        )
        c_elapsed = time.perf_counter() - c_start
        final_clip_paths.append(str(clip_path))
        print(f"    Clip {index+1}: {c_elapsed:.2f}s")
    
    clip_elapsed = time.perf_counter() - clip_start
    print(f"  Total clip processing: {clip_elapsed:.2f}s")
    
    # 5b: Concatenate with transitions
    print("\n  5b: Concatenating with transitions...")
    concat_start = time.perf_counter()
    video_no_audio = Path(work_dir) / "video_no_audio.mp4"
    
    concatenate_videos(
        final_clip_paths,
        str(video_no_audio),
        beat_groups=used_beat_groups,
        transition_min=4,
        transition_max=8,
        transition_duration=0.4,
    )
    concat_elapsed = time.perf_counter() - concat_start
    print(f"  Concat + transitions: {concat_elapsed:.2f}s")
    
    # 5c: Low-RAM re-encode (remux)
    print("\n  5c: Remuxing (encode_in_chunks)...")
    encode_start = time.perf_counter()
    encoded = Path(work_dir) / "video_encoded.mp4"
    encode_in_chunks(str(video_no_audio), str(encoded), segment_time=60)
    encode_elapsed = time.perf_counter() - encode_start
    print(f"  Remux: {encode_elapsed:.2f}s")
    
    # 5d: Add audio
    print("\n  5d: Adding audio...")
    audio_start = time.perf_counter()
    # We need the music path - get from music_analysis or pass separately
    # For benchmark, we'll skip actual audio mix since we don't have music_path here
    # Just time the concat step
    audio_elapsed = time.perf_counter() - audio_start
    print(f"  Audio mix: {audio_elapsed:.2f}s (skipped in benchmark)")
    
    total_elapsed = time.perf_counter() - total_start
    
    # Cleanup
    shutil.rmtree(work_dir, ignore_errors=True)
    
    print(f"\n  Total render time: {total_elapsed:.2f}s")
    print(f"    Clip processing: {clip_elapsed:.2f}s ({clip_elapsed/total_elapsed*100:.1f}%)")
    print(f"    Concat + trans:  {concat_elapsed:.2f}s ({concat_elapsed/total_elapsed*100:.1f}%)")
    print(f"    Remux:           {encode_elapsed:.2f}s ({encode_elapsed/total_elapsed*100:.1f}%)")
    
    return total_elapsed


def main():
    parser = argparse.ArgumentParser(description="Benchmark Beats Video Editor pipeline stages")
    parser.add_argument("music", help="Path to music file")
    parser.add_argument("videos", nargs="+", help="Paths to video files (2-3 recommended)")
    parser.add_argument("--skip-render", action="store_true", help="Skip FFmpeg render stage (faster)")
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.music).exists():
        print(f"Error: Music file not found: {args.music}")
        sys.exit(1)
    
    for v in args.videos:
        if not Path(v).exists():
            print(f"Error: Video file not found: {v}")
            sys.exit(1)
    
    if len(args.videos) < 2:
        print("Warning: Recommend 2-3 videos for realistic benchmark")
    
    print("="*60)
    print("BEATS VIDEO EDITOR - PIPELINE BENCHMARK")
    print("="*60)
    print(f"Music: {args.music}")
    print(f"Videos: {len(args.videos)} files")
    for v in args.videos:
        print(f"  - {v}")
    
    # Run all stages
    pipeline_start = time.perf_counter()
    
    # Stage 1
    t1, music_analysis = benchmark_music_analysis(args.music)
    
    # Stage 2
    t2, video_analyses = benchmark_video_analysis(args.videos)
    
    # Create beat groups
    beats = music_analysis.get("beats", [])
    music_duration = music_analysis.get("duration", 0)
    beat_groups = create_beat_groups(beats, 4, 8, music_duration=music_duration)
    print(f"\n  Beat groups created: {len(beat_groups)}")
    print(f"  Total timeline duration: {sum(bg['duration'] for bg in beat_groups):.2f}s")
    
    # Stage 3
    t3, all_candidates = benchmark_candidate_generation(args.videos, music_analysis, beat_groups)
    
    # Stage 4
    t4, ai_results, used_beat_groups = benchmark_clip_assignment(
        args.videos, music_analysis, beat_groups, video_analyses, all_candidates
    )
    
    # Stage 5 (optional)
    if not args.skip_render:
        t5 = benchmark_ffmpeg_render(ai_results, used_beat_groups, "output/benchmark_final.mp4")
    else:
        t5 = 0.0
        print("\n" + "="*60)
        print("STAGE 5: FFmpeg Rendering - SKIPPED")
        print("="*60)
    
    # Summary
    total = time.perf_counter() - pipeline_start
    
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"{'Stage':<35} {'Time (s)':>10} {'%':>8}")
    print("-"*60)
    print(f"{'1. Music/Beat Analysis':<35} {t1:>10.2f} {t1/total*100:>7.1f}%")
    print(f"{'2. Video Analysis':<35} {t2:>10.2f} {t2/total*100:>7.1f}%")
    print(f"{'3. Candidate Generation':<35} {t3:>10.2f} {t3/total*100:>7.1f}%")
    print(f"{'4. Clip Assignment (AI)':<35} {t4:>10.2f} {t4/total*100:>7.1f}%")
    if not args.skip_render:
        print(f"{'5. FFmpeg Render':<35} {t5:>10.2f} {t5/total*100:>7.1f}%")
    print("-"*60)
    print(f"{'TOTAL':<35} {total:>10.2f} {'100.0%':>8}")
    print("="*60)
    
    # Quality metrics
    print("\nQUALITY METRICS:")
    print(f"  Beat alignment: {len(beats)} beats detected")
    print(f"  Clip coverage: {sum(bg['duration'] for bg in used_beat_groups):.2f}s / {music_duration:.2f}s music")
    print(f"  Clips per source: ", end="")
    from collections import Counter
    source_counts = Counter(a['source_path'] for a in ai_results)
    for src, count in source_counts.items():
        print(f"{Path(src).name}={count} ", end="")
    print()
    print(f"  Unique sources used: {len(source_counts)} / {len(args.videos)}")
    
    # Diversity check
    all_intervals = []
    for a in ai_results:
        all_intervals.append((a['source_path'], a['source_start'], a['source_start'] + a['clip_duration']))
    
    overlaps = 0
    for i, (vp1, s1, e1) in enumerate(all_intervals):
        for j, (vp2, s2, e2) in enumerate(all_intervals):
            if i < j and vp1 == vp2:
                if not (e1 <= s2 or e2 <= s1):
                    overlaps += 1
    print(f"  Overlapping clips (same source): {overlaps}")


if __name__ == "__main__":
    main()