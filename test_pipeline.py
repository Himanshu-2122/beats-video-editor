#!/usr/bin/env python
"""
Test script to verify the full pipeline works.
Run this after starting the Streamlit app to test with sample files.
"""
import os
import sys
import tempfile

# Add project root to path
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from app.beat import analyze_music_full
from app.video import analyze_video_full, process_clip, get_video_duration
from app.sync import ai_assign_clips, sync_clips_with_beats
from app.video import concatenate_videos, add_audio

def test_pipeline():
    print("=" * 60)
    print("TESTING FULL PIPELINE")
    print("=" * 60)
    
    # Check if test files exist
    test_video = os.path.join(ROOT, "data", "videos", "test_video.mp4")
    test_music = os.path.join(ROOT, "data", "music", "song.mp3")
    
    if not os.path.exists(test_video):
        print(f"Test video not found: {test_video}")
        print("Create test_assets/ folder with test_video.mp4 and test_music.mp3")
        return False
    
    if not os.path.exists(test_music):
        print(f"Test music not found: {test_music}")
        return False
    
    temp_dir = tempfile.mkdtemp(prefix="pipeline_test_")
    print(f"Temp dir: {temp_dir}")
    
    try:
        # 1. Music analysis
        print("\n[1/7] Analyzing music...")
        music_analysis = analyze_music_full(test_music, cache_dir=os.path.join(temp_dir, "cache"))
        beats = music_analysis.get("beats", [])
        print(f"  BPM: {music_analysis.get('bpm', 'N/A'):.1f}, Beats: {len(beats)}, Drops: {len(music_analysis.get('drops', []))}")
        
        # 2. Video analysis
        print("\n[2/7] Analyzing video...")
        video_analysis = analyze_video_full(test_video, cache_dir=os.path.join(temp_dir, "cache"), fast_mode=True)
        print(f"  Duration: {video_analysis['duration']:.1f}s, Scenes: {len(video_analysis['scenes'])}")
        
        # 3. AI clip assignment
        print("\n[3/7] AI clip assignment...")
        ai_results, beat_groups, debug_scores, all_candidates = ai_assign_clips(
            beats,
            [test_video],
            min_beats=4,
            max_beats=8,
            sample_interval=2.0,
            scene_threshold=0.4,
            reuse_limit=None,
            compute_motion=False,
            cache_dir=os.path.join(temp_dir, "cache"),
            music_analysis=music_analysis,
            video_analyses=[video_analysis],
        )
        print(f"  Generated {len(ai_results)} clip assignments")
        
        # 4. Process clips
        print("\n[4/7] Processing clips...")
        clip_dir = os.path.join(temp_dir, "_final_clips")
        os.makedirs(clip_dir, exist_ok=True)
        
        final_clip_paths = []
        for i, assignment in enumerate(ai_results):
            clip_path = os.path.join(clip_dir, f"clip_{i+1:04d}.mp4")
            process_clip(
                video_path=assignment["source_path"],
                duration=assignment["clip_duration"],
                output_path=clip_path,
                start_time=assignment["source_start"],
                snap_frame=True,
            )
            final_clip_paths.append(clip_path)
            print(f"  Clip {i+1}/{len(ai_results)}: {assignment['clip_duration']:.2f}s @ {assignment['source_start']:.2f}s")
        
        # 5. Concatenate with transitions
        print("\n[5/7] Concatenating with transitions...")
        video_no_audio = os.path.join(temp_dir, "_video_no_audio.mp4")
        concatenate_videos(
            final_clip_paths,
            video_no_audio,
            beat_groups=[{"clip_path": p} for p in final_clip_paths],
            transition_min=4,
            transition_max=8,
            transition_duration=0.4,
        )
        
        # 6. Add audio
        print("\n[6/7] Adding audio...")
        final_path = os.path.join(temp_dir, "final.mp4")
        add_audio(video_no_audio, test_music, final_path)
        
        # 7. Verify output
        print("\n[7/7] Verifying output...")
        out_duration = get_video_duration(final_path)
        print(f"  Final video: {final_path}")
        print(f"  Duration: {out_duration:.1f}s")
        print(f"  File size: {os.path.getsize(final_path) / 1024 / 1024:.1f} MB")
        
        print("\n" + "=" * 60)
        print("PIPELINE TEST PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\nPIPELINE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)