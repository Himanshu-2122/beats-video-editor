import os
import sys
import tempfile
import threading
import shutil
import streamlit as st
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

from app.beat import analyze_music_full
from app.video import analyze_video_full, process_clip, concatenate_videos, add_audio, encode_in_chunks, concat_group
from app.sync import sync_clips_with_beats, ai_assign_clips
from app.progress_tracker import ProgressTracker, create_default_stages

st.set_page_config(page_title="Beat Video Editor", layout="wide", initial_sidebar_state="collapsed")


def run_generation(tracker, temp_dir, output_dir, video_paths, music_path, params):
    """Background thread for video generation."""
    try:
        # STAGE 1: Music Analysis
        tracker.start_stage(0, "Analyzing music...")
        music_analysis = analyze_music_full(music_path, cache_dir=os.path.join(temp_dir, "cache"))
        beats = music_analysis.get("beats", [])
        if len(beats) < 2:
            raise RuntimeError("Not enough beats detected in music.")
        tracker.complete_stage(0)

        # STAGE 2: Video Analysis
        tracker.start_stage(1, "Analyzing videos...")
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def analyze_one(vp):
            analysis = analyze_video_full(vp, cache_dir=os.path.join(temp_dir, "cache"), fast_mode=True)
            return analysis

        video_analyses = []
        max_workers = min(len(video_paths), 4)
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

        # STAGE 3: Beat Groups & Clip Discovery
        tracker.start_stage(2, "Creating beat groups...")
        tracker.update_stage_progress(2, 0.3, "Discovering candidate clips...")
        cache_dir = os.path.join(temp_dir, "cache")
        reuse_limit = int(params['max_clips_per_source']) if params['max_clips_per_source'] > 0 else None
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
        tracker.update_stage_progress(2, 0.7, "Scoring & matching clips...")
        tracker.update_stats(
            clips_generated=len(ai_results),
            target_duration=sum(bg.get('duration', 0) for bg in beat_groups)
        )
        tracker.complete_stage(2)

        # STAGE 4: Process Clips
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

            process_clip(
                video_path=source_path,
                duration=duration,
                output_path=clip_path,
                start_time=start_time,
                snap_frame=True,
            )

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

        # STAGE 5: Concatenate & Transitions
        tracker.start_stage(4, "Planning transitions...")
        tracker.update_stage_progress(4, 0.3, "Planning transitions...")
        tracker.update_stage_progress(4, 0.7, "Concatenating clips...")
        video_no_audio = os.path.join(output_dir, "_video_no_audio.mp4")

        if params['use_transitions']:
            concatenate_videos(
                final_clip_paths,
                video_no_audio,
                beat_groups=used_beat_groups,
                transition_min=4,
                transition_max=8,
                transition_duration=float(params['transition_duration']),
            )
        else:
            concat_group(final_clip_paths, video_no_audio)

        tracker.complete_stage(4)

        # STAGE 6: Low-RAM Re-encode (Optional)
        tracker.start_stage(5, "Rendering video...")
        final_input = video_no_audio
        if params['use_low_ram']:
            encoded_tmp = os.path.join(output_dir, "_video_encoded.mp4")
            try:
                def enc_progress(cur, total):
                    tracker.update_stage_progress(5, cur / max(total, 1))
                encode_in_chunks(
                    video_no_audio,
                    encoded_tmp,
                    segment_time=int(params['chunk_size']),
                    progress_callback=enc_progress,
                )
                final_input = encoded_tmp
            except Exception:
                pass

        tracker.update_stage_progress(5, 0.8, "Finalizing render...")
        tracker.complete_stage(5)

        # STAGE 7: Add Audio
        tracker.start_stage(6, "Adding music...")
        final_filename = f"beat_edit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        final_path = os.path.join(OUTPUT_DIR, final_filename)
        add_audio(final_input, music_path, final_path)
        tracker.complete_stage(6)

        # STAGE 8: Finalizing
        tracker.start_stage(7, "Finalizing...")
        tracker.update_stage_progress(7, 1.0)
        tracker.complete_stage(7)

        st.session_state.generation_result = {"success": True, "final_path": final_path, "final_filename": final_filename}
    except Exception as exc:
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

    with st.sidebar:
        st.header("Upload")
        video_files = st.file_uploader(
            "Source videos",
            type=["mp4", "mov", "webm", "mkv"],
            accept_multiple_files=True,
            help="Upload one or more full-length videos"
        )
        music_file = st.file_uploader(
            "Music track",
            type=["mp3", "wav", "m4a", "aac"],
            help="Upload the music/audio track"
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

    if "generation_running" not in st.session_state:
        st.session_state.generation_running = False
    if "generation_result" not in st.session_state:
        st.session_state.generation_result = None

    if not st.session_state.generation_running:
        if st.button("Generate AI Video", type="primary", use_container_width=True):
            if not video_files:
                st.error("Please upload at least one video file.")
                st.stop()
            if not music_file:
                st.error("Please upload a music file.")
                st.stop()
            if min_beats > max_beats:
                st.error("Minimum beats cannot exceed maximum beats.")
                st.stop()

            temp_dir = tempfile.mkdtemp(prefix="beats_editor_")
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            music_ext = os.path.splitext(music_file.name)[1]
            music_path = os.path.join(temp_dir, f"music{music_ext}")
            with open(music_path, "wb") as f:
                f.write(music_file.getbuffer())

            video_paths = []
            for idx, vf in enumerate(video_files):
                ext = os.path.splitext(vf.name)[1]
                vp = os.path.join(temp_dir, f"video_{idx}{ext}")
                with open(vp, "wb") as f:
                    f.write(vf.getbuffer())
                video_paths.append(vp)

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