import os
import sys
import tempfile
import threading
import shutil
import streamlit as st
from datetime import datetime
import psutil
import gc
import warnings

# Suppress Streamlit thread warnings
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Temp directory for uploaded files (persists across reruns)
UPLOAD_TEMP_DIR = os.path.join(ROOT, "temp_uploads")
os.makedirs(UPLOAD_TEMP_DIR, exist_ok=True)

# Chunk size for streaming uploads (50MB)
UPLOAD_CHUNK_SIZE = 50 * 1024 * 1024  # 50MB

from app.beat import analyze_music_full
from app.video import analyze_video_full, process_clip, concatenate_videos, add_audio, encode_in_chunks, concat_group
from app.sync import sync_clips_with_beats, ai_assign_clips
from app.progress_tracker import ProgressTracker, create_default_stages

st.set_page_config(page_title="Beat Video Editor", layout="wide", initial_sidebar_state="collapsed")


def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_free_disk_space_gb(path):
    """Get free disk space in GB for the given path."""
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024 ** 3)
    except Exception:
        return 0.0


def stream_uploaded_file_to_disk(uploaded_file, dest_path):
    """
    Stream an uploaded file to disk in chunks without loading entire file into memory.
    Uses uploaded_file.read(chunk_size) which streams from the underlying buffer.
    """
    try:
        with open(dest_path, "wb") as f:
            while True:
                chunk = uploaded_file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
        return True, None
    except Exception as e:
        # Clean up partial file on error
        try:
            if os.path.exists(dest_path):
                os.remove(dest_path)
        except Exception:
            pass
        return False, str(e)


def check_disk_space_required(total_size_bytes):
    """
    Check if there's enough disk space for uploads + pipeline working space.
    
    At scale (3GB+ files), need to account for:
    - Upload temp files: total_size
    - Pipeline cache/ (extracted frames, analysis): ~0.5-1x total_size
    - Temp clips during rendering: ~0.5-1x total_size  
    - Final render output: ~1-2 GB
    - OS / other: 2-5 GB
    
    Use 3x multiplier as safety margin for 9GB+ total uploads.
    """
    free_gb = get_free_disk_space_gb(UPLOAD_TEMP_DIR)
    # 3x multiplier: 1x uploads + 1x pipeline cache/temp + 1x render output + headroom
    required_gb = total_size_bytes / (1024 ** 3) * 3.0
    if free_gb < required_gb:
        return False, (
            f"❌ Insufficient disk space.\n"
            f"   Upload size: {total_size_bytes / (1024**3):.1f} GB\n"
            f"   Required (3× for uploads + pipeline cache + render): {required_gb:.1f} GB\n"
            f"   Available: {free_gb:.1f} GB\n"
            f"   Free up at least {required_gb - free_gb:.1f} GB more space."
        )
    return True, None


def process_uploaded_files(video_files, music_file):
    """Process uploaded files immediately - stream to disk, return file paths."""
    if not video_files or not music_file:
        return None, None, "Please upload at least one video file and a music track."
    
    # Estimate total size (file_uploader gives us file size)
    total_size = sum(getattr(vf, 'size', 0) for vf in video_files) + getattr(music_file, 'size', 0)
    total_size_gb = total_size / (1024 ** 3)
    
    # Disk space check
    has_space, err = check_disk_space_required(total_size)
    if not has_space:
        return None, None, f"❌ {err}"
    
    # Warn if large
    if total_size_gb > 2:  # > 2GB
        st.warning(
            f"⚠️ Total upload size: {total_size_gb:.1f} GB. "
            f"Ensure you have enough disk space and RAM."
        )
    
    # Check individual file sizes and warn for very large files
    LARGE_FILE_THRESHOLD_GB = 4  # Browser uploads > 4GB are slow/unreliable
    for vf in video_files:
        vf_size_gb = getattr(vf, 'size', 0) / (1024 ** 3)
        if vf_size_gb > LARGE_FILE_THRESHOLD_GB:
            st.warning(
                f"⚠️ **Large file detected:** '{vf.name}' is {vf_size_gb:.1f} GB.\n"
                f"   Browser uploads over {LARGE_FILE_THRESHOLD_GB} GB can be slow and may fail due to "
                f"network timeouts or browser memory limits.\n"
                f"   **Recommended:** Place the file in `{UPLOAD_TEMP_DIR}/local_input/` "
                f"and use the 'Local file path' input below instead of browser upload."
            )
    
    # Create unique temp directory for this session
    session_temp_dir = tempfile.mkdtemp(prefix="beats_upload_", dir=UPLOAD_TEMP_DIR)
    
    # Save music file - stream to disk
    music_ext = os.path.splitext(music_file.name)[1]
    music_path = os.path.join(session_temp_dir, f"music{music_ext}")
    
    # Show progress for large files
    music_size = getattr(music_file, 'size', 0)
    if music_size > 100 * 1024 * 1024:  # > 100MB
        with st.spinner(f"Saving music file ({music_size / (1024**2):.1f} MB)..."):
            success, err = stream_uploaded_file_to_disk(music_file, music_path)
    else:
        success, err = stream_uploaded_file_to_disk(music_file, music_path)
    
    if not success:
        shutil.rmtree(session_temp_dir, ignore_errors=True)
        return None, None, f"Failed to save music file: {err}"
    
    # Save video files one at a time - stream to disk
    video_paths = []
    for idx, vf in enumerate(video_files):
        ext = os.path.splitext(vf.name)[1]
        vp = os.path.join(session_temp_dir, f"video_{idx}{ext}")
        
        # Show progress for large files
        vf_size = getattr(vf, 'size', 0)
        if vf_size > 100 * 1024 * 1024:  # > 100MB
            with st.spinner(f"Saving video {idx+1}/{len(video_files)} ({vf_size / (1024**3):.2f} GB)..."):
                success, err = stream_uploaded_file_to_disk(vf, vp)
        else:
            success, err = stream_uploaded_file_to_disk(vf, vp)
        
        if not success:
            # Cleanup on failure
            for saved_vp in video_paths:
                try:
                    os.remove(saved_vp)
                except Exception:
                    pass
            try:
                os.remove(music_path)
            except Exception:
                pass
            shutil.rmtree(session_temp_dir, ignore_errors=True)
            return None, None, f"Failed to save video '{vf.name}': {err}"
        
        video_paths.append(vp)
    
    # Force garbage collection to free UploadedFile objects
    del video_files
    del music_file
    gc.collect()
    
    return video_paths, music_path, None


def run_generation(tracker, temp_dir, output_dir, video_paths, music_path, params):
    """Background thread for video generation."""
    print(f"\n{'='*60}")
    print(f"🚀 STARTING VIDEO GENERATION")
    print(f"{'='*60}")
    print(f"📹 Videos: {len(video_paths)}")
    for vp in video_paths:
        print(f"   - {os.path.basename(vp)}")
    print(f"🎵 Music: {os.path.basename(music_path)}")
    print(f"📁 Temp dir: {temp_dir}")
    print(f"📁 Output dir: {output_dir}")
    print(f"⚙️  Params: {params}")
    print(f"{'='*60}\n")
    
    try:
        # STAGE 1: Music Analysis
        print(f"\n🎵 STAGE 1: Music Analysis")
        tracker.start_stage(0, "Analyzing music...")
        music_analysis = analyze_music_full(music_path, cache_dir=os.path.join(temp_dir, "cache"))
        beats = music_analysis.get("beats", [])
        print(f"   ✅ Music analyzed: {len(beats)} beats, BPM={music_analysis.get('bpm', 'N/A'):.1f}, Duration={music_analysis.get('duration', 0):.1f}s")
        if len(beats) < 2:
            raise RuntimeError("Not enough beats detected in music.")
        tracker.complete_stage(0)

        # STAGE 2: Video Analysis
        print(f"\n🎬 STAGE 2: Video Analysis ({len(video_paths)} videos)")
        tracker.start_stage(1, "Analyzing videos...")
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def analyze_one(vp):
            print(f"   🔍 Analyzing: {os.path.basename(vp)}")
            analysis = analyze_video_full(vp, cache_dir=os.path.join(temp_dir, "cache"), fast_mode=True)
            print(f"   ✅ Done: {os.path.basename(vp)} - {len(analysis.get('scenes', []))} scenes")
            return analysis

        video_analyses = []
        max_workers = min(len(video_paths), 4)
        print(f"   🔧 Using {max_workers} worker threads")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(analyze_one, vp): vp for vp in video_paths}
            for i, future in enumerate(as_completed(futures)):
                analysis = future.result()
                video_analyses.append(analysis)
                tracker.update_stats(
                    videos_analyzed=i + 1,
                    total_videos=len(video_paths),
                    scenes_detected=sum(len(a['scenes']) for a in video_analyses)
                )
                tracker.update_stage_progress(1, (i + 1) / len(video_paths))
        tracker.complete_stage(1)
        print(f"   ✅ All {len(video_analyses)} videos analyzed")

        # STAGE 3: Beat Groups & Clip Discovery
        print(f"\n🎯 STAGE 3: Beat Groups & Clip Discovery")
        tracker.start_stage(2, "Creating beat groups...")
        tracker.update_stage_progress(2, 0.3, "Discovering candidate clips...")
        cache_dir = os.path.join(temp_dir, "cache")
        reuse_limit = int(params['max_clips_per_source']) if params['max_clips_per_source'] > 0 else None
        print(f"   🎯 Calling ai_assign_clips...")
        print(f"   📊 Params: min_beats={params['min_beats']}, max_beats={params['max_beats']}, compute_motion={params['compute_motion']}")
        
        ai_results, beat_groups, debug_scores, all_candidates = ai_assign_clips(
            beats,
            video_paths,
            min_beats=int(params['min_beats']),
            max_beats=int(params['max_beats']),
            sample_interval=float(params['sample_interval']),
            scene_threshold=float(params['scene_threshold']),
            reuse_limit=reuse_limit,
            compute_motion=bool(params['compute_motion']),
            cache_dir=cache_dir,
            music_analysis=music_analysis,
            video_analyses=video_analyses,
        )

        if not ai_results:
            raise RuntimeError("No clips could be generated.")
        print(f"   ✅ Assigned {len(ai_results)} clips across {len(beat_groups)} beat groups")
        tracker.update_stage_progress(2, 0.7, "Scoring & matching clips...")
        tracker.update_stats(
            clips_generated=len(ai_results),
            target_duration=sum(bg.get('duration', 0) for bg in beat_groups)
        )
        tracker.complete_stage(2)

        # STAGE 4: Process Clips
        print(f"\n✂️ STAGE 4: Processing Clips ({len(ai_results)} clips)")
        tracker.start_stage(3, "Building timeline...")
        tracker.update_stage_progress(3, 0.5, "Processing clips...")
        clip_dir = os.path.join(temp_dir, "_final_clips")
        os.makedirs(clip_dir, exist_ok=True)

        final_clip_paths = []
        used_beat_groups = []
        total = len(ai_results)

        for index, assignment in enumerate(ai_results):
            source_path = assignment["source_path"]
            duration = assignment["clip_duration"]
            start_time = assignment["source_start"]
            beat_group = {
                "start": assignment["music_start"],
                "end": assignment["music_end"],
                "duration": duration,
            }

            clip_path = os.path.join(clip_dir, f"clip_{index + 1:04d}.mp4")

            print(f"   🎬 Clip {index+1}/{total}: {os.path.basename(source_path)} [{start_time:.1f}s-{start_time+duration:.1f}s]")
            
            process_clip(
                video_path=source_path,
                duration=duration,
                output_path=clip_path,
                start_time=start_time,
                snap_frame=True,
                target_width=int(params['target_width']),
                target_height=int(params['target_height']),
            )
            print(f"      ✅ Clip saved: {clip_path}")

            final_clip_paths.append(clip_path)
            used_beat_groups.append({
                "clip_path": clip_path,
                "source_path": source_path,
                "music_start": beat_group["start"],
                "music_end": beat_group["end"],
                "clip_duration": duration,
                "source_start": start_time,
                "score": assignment["score"],
                "scene_flag": assignment["scene_flag"],
                "motion_score": assignment.get("motion_score", 0.0),
                "beat_strength": assignment.get("beat_strength", 0.5),
                "beat_energy": assignment.get("beat_energy", 0.5),
                "beat_type": assignment.get("beat_type", "regular"),
            })

            tracker.update_stats(
                clips_generated=index + 1,
                generated_duration=sum(bg["clip_duration"] for bg in used_beat_groups)
            )
            tracker.update_stage_progress(3, (index + 1) / total)

        tracker.complete_stage(3)
        print(f"\n🔗 STAGE 5: Concatenate & Transitions")
        tracker.start_stage(4, "Planning transitions...")
        tracker.update_stage_progress(4, 0.3, "Planning transitions...")
        tracker.update_stage_progress(4, 0.7, "Concatenating clips...")
        video_no_audio = os.path.join(output_dir, "_video_no_audio.mp4")

        print(f"   🔗 Concatenating {len(final_clip_paths)} clips...")
        if params['use_transitions']:
            print(f"   🎞️  With transitions (duration={params['transition_duration']}s)")
            concatenate_videos(
                final_clip_paths,
                video_no_audio,
                beat_groups=used_beat_groups,
                transition_min=4,
                transition_max=8,
                transition_duration=float(params['transition_duration']),
                target_width=int(params['target_width']),
                target_height=int(params['target_height']),
            )
        else:
            print(f"   🔗 Concatenating without transitions")
            concat_group(final_clip_paths, video_no_audio)

        print(f"   ✅ Concatenated: {video_no_audio}")
        tracker.complete_stage(4)

        # STAGE 6: Low-RAM Re-encode (Optional)
        print(f"\n🎬 STAGE 6: Rendering Video")
        tracker.start_stage(5, "Rendering video...")
        final_input = video_no_audio
        if params['use_low_ram']:
            encoded_tmp = os.path.join(output_dir, "_video_encoded.mp4")
            try:
                def enc_progress(cur, total):
                    tracker.update_stage_progress(5, cur / max(total, 1))
                print(f"   🔄 Encoding in chunks (segment={params['chunk_size']}s)...")
                encode_in_chunks(
                    video_no_audio,
                    encoded_tmp,
                    segment_time=int(params['chunk_size']),
                    progress_callback=enc_progress,
                )
                final_input = encoded_tmp
                print(f"   ✅ Encoded: {encoded_tmp}")
            except Exception as e:
                print(f"   ⚠️  Chunked encoding failed, using original: {e}")
                pass

        tracker.update_stage_progress(5, 0.8, "Finalizing render...")
        tracker.complete_stage(5)

        # STAGE 7: Add Audio
        print(f"\n🎵 STAGE 7: Adding Music")
        tracker.start_stage(6, "Adding music...")
        final_filename = f"beat_edit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        final_path = os.path.join(OUTPUT_DIR, final_filename)
        print(f"   🎵 Adding audio: {os.path.basename(music_path)}")
        add_audio(final_input, music_path, final_path)
        print(f"   ✅ Final video: {final_filename}")
        tracker.complete_stage(6)

        # STAGE 8: Finalizing
        print(f"\n✅ STAGE 8: Finalizing")
        tracker.start_stage(7, "Finalizing...")
        tracker.update_stage_progress(7, 1.0)
        tracker.complete_stage(7)

        print(f"\n{'='*60}")
        print(f"🎉 GENERATION COMPLETE!")
        print(f"📁 Output: {final_path}")
        print(f"{'='*60}\n")

        st.session_state.generation_result = {"success": True, "final_path": final_path, "final_filename": final_filename}
    except Exception as exc:
        print(f"\n❌ GENERATION FAILED: {exc}")
        import traceback
        traceback.print_exc()
        st.session_state.generation_result = {"success": False, "error": str(exc)}
    finally:
        st.session_state.generation_running = False
        if music_path and os.path.exists(music_path):
            try:
                os.remove(music_path)
            except Exception:
                pass
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


@st.fragment(run_every=1)
def progress_dashboard(tracker: ProgressTracker):
    """Progress dashboard fragment that updates every second."""
    data = tracker.get_dashboard_data()

    st.markdown(f"### {data['current_stage']}")
    st.progress(data['overall_progress'] / 100)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Progress", f"{data['overall_progress']}%")
    with col2:
        st.metric("Elapsed", data['elapsed'])
    with col3:
        st.metric("Remaining", data['eta'])
    with col4:
        st.metric("Timeline", data['stats']['timeline_progress'])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Videos Processed", data['stats']['videos_analyzed'])
    with col2:
        st.metric("Scenes Detected", data['stats']['scenes_detected'])
    with col3:
        st.metric("Clips Generated", data['stats']['clips_generated'])
    with col4:
        st.metric("Generated Duration", data['stats']['generated_duration'])


def get_generated_videos():
    """Get list of previously generated videos from output directory."""
    videos = []
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith(".mp4") and f.startswith("beat_edit_"):
                path = os.path.join(OUTPUT_DIR, f)
                stat = os.stat(path)
                videos.append({
                    "filename": f,
                    "path": path,
                    "size_mb": round(stat.st_size / (1024 * 1024), 1),
                    "created": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                })
    return sorted(videos, key=lambda x: x["created"], reverse=True)


def main():
    st.title("Beat Video Editor")
    st.caption("Upload videos + music → AI creates beat-synced edit automatically")

    # Initialize session state for file paths (not UploadedFile objects)
    if "uploaded_video_paths" not in st.session_state:
        st.session_state.uploaded_video_paths = []
    if "uploaded_music_path" not in st.session_state:
        st.session_state.uploaded_music_path = None
    if "upload_temp_dir" not in st.session_state:
        st.session_state.upload_temp_dir = None
    if "generation_running" not in st.session_state:
        st.session_state.generation_running = False
    if "generation_result" not in st.session_state:
        st.session_state.generation_result = None

    with st.sidebar:
        st.header("Upload")
        
        # Disk space & memory indicator
        free_gb = get_free_disk_space_gb(UPLOAD_TEMP_DIR)
        mem_mb = get_memory_usage_mb()
        st.caption(f"💾 Disk: {free_gb:.1f} GB free | 🧠 RAM: {mem_mb:.1f} MB")
        
        # Local file paths for videos (primary method - avoids Streamlit memory issues with large files)
        st.info("📁 **Video input: Local file paths** (bypasses browser upload memory limits)")
        st.caption(f"Tip: Drag files from Windows Explorer to get full paths, or copy path from address bar")
        
        # Create local input directory if it doesn't exist
        local_input_dir = os.path.join(UPLOAD_TEMP_DIR, "local_input")
        os.makedirs(local_input_dir, exist_ok=True)
        
        # Video file paths input
        video_paths_input = st.text_area(
            "Source video file paths (one per line)",
            placeholder="C:/Videos/video1.mp4\nC:/Videos/video2.mp4\n...",
            help="Enter full paths to video files, one per line. Supports: mp4, mov, webm, mkv",
            height=120
        )
        
        # Parse and validate video paths
        video_files = []
        if video_paths_input.strip():
            for line in video_paths_input.strip().split('\n'):
                # Strip whitespace AND surrounding quotes (both " and ')
                path = line.strip().strip('"').strip("'")
                if not path:
                    continue
                # Normalize to absolute path before checking
                path = os.path.abspath(path)
                # DEBUG: show exact string being checked
                st.caption(f"🔍 Checking: {repr(path)}")
                if os.path.exists(path):
                    # Validate video format
                    ext = os.path.splitext(path)[1].lower()
                    if ext in ['.mp4', '.mov', '.webm', '.mkv']:
                        video_files.append(path)
                    else:
                        st.warning(f"⚠️ Unsupported format '{ext}' in '{path}' — skipping")
                else:
                    st.warning(f"⚠️ File not found: {path}")
        
        if not video_files:
            st.warning("⬆️ Enter at least one valid video file path above to continue")
        
        st.markdown("---")
        
        # Music file - keep browser upload since it's small (a few MB)
        music_file = st.file_uploader(
            "Music track",
            type=["mp3", "wav", "m4a", "aac"],
            help="Upload the music/audio track (kept as browser upload since small)"
        )

        st.markdown("---")
        st.header("Advanced (Optional)")
        with st.expander("Processing Options"):
            min_beats = st.number_input("Min beats per clip", 1, 20, 4)
            max_beats = st.number_input("Max beats per clip", 1, 40, 8)
            max_clips_per_source = st.number_input("Max clips per source video (0=auto)", 0, 50, 0)
            sample_interval = st.number_input("Candidate sample interval (s)", 0.1, 5.0, 2.0, 0.1)
            scene_threshold = st.number_input("Scene threshold", 0.05, 0.8, 0.4, 0.05)
            compute_motion = st.checkbox("Compute motion (optical flow)", False)
            use_transitions = st.checkbox("Add transitions", True)
            transition_duration = st.selectbox("Transition duration (s)", [0.2, 0.25, 0.3, 0.35, 0.4, 0.5], index=3)
            use_low_ram = st.checkbox("Low-RAM chunked encoding", True)
            chunk_size = st.number_input("Chunk size (s)", 10, 300, 60)
            
            st.markdown("---")
            st.caption("Output Settings")
            target_resolution = st.selectbox(
                "Target Resolution",
                ["720p (1280x720)", "1080p (1920x1080)", "4K (3840x2160)"],
                index=1,
                help="All clips will be scaled/padded to this resolution. 1080p is recommended for best compatibility."
            )
            # Parse resolution string to width/height
            res_map = {
                "720p (1280x720)": (1280, 720),
                "1080p (1920x1080)": (1920, 1080),
                "4K (3840x2160)": (3840, 2160),
            }
            target_width, target_height = res_map[target_resolution]

    # Process music file upload immediately when it appears
    if music_file:
        current_music_name = music_file.name
        stored_music_name = os.path.basename(st.session_state.uploaded_music_path) if st.session_state.uploaded_music_path else None
        
        if current_music_name != stored_music_name:
            # Show memory before upload
            mem_before = get_memory_usage_mb()
            st.sidebar.info(f"📥 Uploading music... Memory: {mem_before:.1f} MB")
            
            # Save music file to temp dir
            session_temp_dir = st.session_state.upload_temp_dir or tempfile.mkdtemp(prefix="beats_upload_", dir=UPLOAD_TEMP_DIR)
            music_ext = os.path.splitext(music_file.name)[1]
            music_path = os.path.join(session_temp_dir, f"music{music_ext}")
            
            success, err = stream_uploaded_file_to_disk(music_file, music_path)
            if not success:
                st.error(f"Failed to save music file: {err}")
                st.stop()
            
            st.session_state.uploaded_music_path = music_path
            st.session_state.upload_temp_dir = session_temp_dir
            
            mem_after = get_memory_usage_mb()
            st.sidebar.success(f"✅ Music uploaded. Memory: {mem_after:.1f} MB (Δ{mem_after - mem_before:+.1f} MB)")
            
            del music_file
            gc.collect()

    # Process video file paths (they are already local paths, no upload needed)
    if video_files:
        current_video_names = {os.path.basename(p) for p in video_files}
        stored_video_names = {os.path.basename(p) for p in st.session_state.uploaded_video_paths}
        
        if current_video_names != stored_video_names:
            # Validate all paths exist and are readable
            valid_paths = []
            for path in video_files:
                if os.path.exists(path):
                    valid_paths.append(path)
                else:
                    st.error(f"❌ File no longer accessible: {path}")
            
            if valid_paths:
                st.session_state.uploaded_video_paths = valid_paths
                # Ensure we have a temp dir for output
                if not st.session_state.upload_temp_dir:
                    session_temp_dir = tempfile.mkdtemp(prefix="beats_upload_", dir=UPLOAD_TEMP_DIR)
                    st.session_state.upload_temp_dir = session_temp_dir
                
                mem_mb = get_memory_usage_mb()
                st.sidebar.success(f"✅ {len(valid_paths)} video(s) ready. Memory: {mem_mb:.1f} MB")
            else:
                st.error("❌ No valid video files found")

    # Show previously generated videos
    generated_videos = get_generated_videos()
    if generated_videos:
        with st.expander(f"📁 Previous Videos ({len(generated_videos)})", expanded=False):
            for vid in generated_videos:
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.text(vid["filename"])
                with col2:
                    st.text(f"{vid['size_mb']} MB")
                with col3:
                    st.text(vid["created"])
                with col4:
                    with open(vid["path"], "rb") as f:
                        st.download_button(
                            "Download",
                            data=f.read(),
                            file_name=vid["filename"],
                            mime="video/mp4",
                            key=f"dl_{vid['filename']}",
                            use_container_width=True
                        )
                st.video(vid["path"])

    st.markdown("---")

    # Show upload status
    if st.session_state.uploaded_video_paths:
        st.sidebar.info(
            f"📁 Ready: {len(st.session_state.uploaded_video_paths)} video(s), "
            f"{'music ✓' if st.session_state.uploaded_music_path else 'music ✗'}"
        )
    else:
        st.sidebar.warning("⬆️ Upload videos and music to begin")

    if not st.session_state.generation_running:
        if st.button("Generate AI Video", type="primary", use_container_width=True, 
                       disabled=not (st.session_state.uploaded_video_paths and st.session_state.uploaded_music_path)):
            if not st.session_state.uploaded_video_paths:
                st.error("Please upload at least one video file.")
                st.stop()
            if not st.session_state.uploaded_music_path:
                st.error("Please upload a music file.")
                st.stop()
            if min_beats > max_beats:
                st.error("Minimum beats cannot exceed maximum beats.")
                st.stop()

            video_paths = st.session_state.uploaded_video_paths
            music_path = st.session_state.uploaded_music_path
            temp_dir = st.session_state.upload_temp_dir
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            # Estimate total duration from first video
            total_duration = 0.0
            try:
                import cv2
                cap = cv2.VideoCapture(video_paths[0])
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                total_duration = frame_count / fps if fps > 0 else 0
                cap.release()
            except Exception:
                total_duration = 60.0

            stages = create_default_stages(total_duration)
            tracker = ProgressTracker(stages, total_duration)

            st.session_state.tracker = tracker
            st.session_state.temp_dir = temp_dir
            st.session_state.output_dir = output_dir
            st.session_state.video_paths = video_paths
            st.session_state.music_path = music_path
            st.session_state.params = {
                'min_beats': min_beats,
                'max_beats': max_beats,
                'max_clips_per_source': max_clips_per_source,
                'sample_interval': sample_interval,
                'scene_threshold': scene_threshold,
                'compute_motion': compute_motion,
                'use_transitions': use_transitions,
                'transition_duration': transition_duration,
                'use_low_ram': use_low_ram,
                'chunk_size': chunk_size,
                'target_width': target_width,
                'target_height': target_height,
            }
            st.session_state.generation_running = True
            st.session_state.generation_result = None

            thread = threading.Thread(
                target=run_generation,
                args=(tracker, temp_dir, output_dir, video_paths, music_path, st.session_state.params),
                daemon=True
            )
            thread.start()
            st.rerun()

    if st.session_state.generation_running:
        progress_dashboard(st.session_state.tracker)
    elif st.session_state.generation_result:
        result = st.session_state.generation_result
        if result["success"]:
            st.success("Video generated successfully!")
            with open(result["final_path"], "rb") as f:
                st.download_button(
                    "Download Final Video",
                    data=f.read(),
                    file_name=result["final_filename"],
                    mime="video/mp4",
                    use_container_width=True,
                )
            st.video(result["final_path"])
        else:
            st.error(f"Generation failed: {result['error']}")


if __name__ == "__main__":
    main()