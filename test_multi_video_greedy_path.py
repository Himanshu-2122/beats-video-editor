#!/usr/bin/env python
"""
Regression test for the multi-video greedy path bug.

This test specifically covers the scenario that broke:
- 3+ videos
- Long enough total duration that use_hungarian gets estimated False
- Forces the greedy path with use_hungarian=False
- Asserts that assigned clips > 0

This is different from test_long_video.py (single video) - this tests the
exact bug where all_candidates={} was passed to Hungarian when use_hungarian=False.
"""
import os
import sys
import tempfile
import shutil

sys.path.insert(0, 'C:/Users/himan/Projects/beats-video-editor')

from app.beat import analyze_music_full
from app.video import analyze_video_full
from app.sync import ai_assign_clips

def test_multi_video_greedy_path():
    """
    Test that multi-video greedy path (use_hungarian=False) generates clips.
    
    This is a regression test for the bug where:
    - use_hungarian=False caused all_candidates={} to be passed to _hungarian_assign
    - _hungarian_assign didn't fallback to greedy because total_cands=0 made problem size 0
    - _greedy_assign received empty candidates and returned 0 assignments
    """
    # Use 3 output videos (~200s each) + music (173s)
    video_paths = [
        'output/beat_edit_20260816_214752.mp4',  # 209s
        'output/beat_edit_20260817_000114.mp4',  # 206s
        'output/beat_edit_20260817_003752.mp4',  # 209s
    ]
    music_path = 'temp_uploads/beats_upload_lnbmq50e/music.mp3'  # 173s

    # Verify all files exist
    for vp in video_paths:
        assert os.path.exists(vp), f"Video not found: {vp}"
    assert os.path.exists(music_path), f"Music not found: {music_path}"

    temp_dir = tempfile.mkdtemp(prefix='regression_test_')
    
    try:
        music_analysis = analyze_music_full(music_path, cache_dir=os.path.join(temp_dir, 'cache'))
        beats = music_analysis.get('beats', [])
        
        video_analyses = []
        for vp in video_paths:
            va = analyze_video_full(vp, cache_dir=os.path.join(temp_dir, 'cache'), fast_mode=True)
            video_analyses.append(va)

        # This should trigger use_hungarian=False due to:
        # 40 beat_groups * 3 videos * 7 durations * 10 = 8400 > 5000
        ai_results, beat_groups, debug_scores, all_candidates = ai_assign_clips(
            beats,
            video_paths,
            min_beats=4,
            max_beats=8,
            sample_interval=2.0,
            scene_threshold=0.4,
            reuse_limit=None,
            compute_motion=False,
            cache_dir=os.path.join(temp_dir, 'cache'),
            music_analysis=music_analysis,
            video_analyses=video_analyses,
        )

        # THE ASSERTION THAT WOULD HAVE FAILED BEFORE THE FIX:
        # Before fix: ai_results would be [] (empty list) -> len == 0
        # After fix: ai_results should have assignments > 0
        num_assignments = len(ai_results)
        num_beat_groups = len(beat_groups)
        
        print(f"Beat groups: {num_beat_groups}")
        print(f"Assignments made: {num_assignments}")
        print(f"Coverage: {num_assignments}/{num_beat_groups} = {num_assignments/num_beat_groups*100:.1f}%")
        
        # Assert we got assignments (the core fix)
        assert num_assignments > 0, "REGRESSION: No clips assigned! The greedy path bug has returned."
        
        # Assert reasonable coverage (at least 50% of beat groups covered)
        # This is a sanity check - with 3 videos of ~200s each and 173s music, 
        # we should cover most beat groups
        coverage_ratio = num_assignments / num_beat_groups
        assert coverage_ratio >= 0.5, f"Coverage too low: {coverage_ratio:.1%} (expected >= 50%)"
        
        # Verify assignments are distributed across videos (not all from one)
        videos_used = set(a['source_path'] for a in ai_results)
        assert len(videos_used) >= 2, f"Only {len(videos_used)} video(s) used, expected at least 2"
        
        print("[PASS] REGRESSION TEST PASSED")
        print(f"  - Assignments > 0: {num_assignments} > 0 [OK]")
        print(f"  - Coverage >= 50%: {coverage_ratio:.1%} >= 50% [OK]")
        print(f"  - Videos used: {len(videos_used)}/3 [OK]")
        
        return True
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    print("=" * 60)
    print("REGRESSION TEST: Multi-video greedy path (use_hungarian=False)")
    print("=" * 60)
    try:
        success = test_multi_video_greedy_path()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FAIL] REGRESSION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)