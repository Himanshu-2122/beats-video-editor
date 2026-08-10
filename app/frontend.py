import os
import shutil
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.beat import get_beats
from app.sync import sync_clips_with_beats
from app.video import add_audio, concat_group, concatenate_videos


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Beat Video Editor",
    page_icon="🎬",
    layout="wide",
)


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SESSION STATE
# ============================================================

if "generated_video" not in st.session_state:
    st.session_state.generated_video = None


def save_uploaded_file(uploaded_file, destination_path):
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    with open(destination_path, "wb") as file:
        file.write(uploaded_file.getbuffer())


def render_generated_video():
    final_path = st.session_state.generated_video
    if final_path and os.path.exists(final_path):
        st.divider()
        st.header("🎉 Generated Video")
        st.video(final_path)
        with open(final_path, "rb") as video_file:
            st.download_button(
                label="⬇️ Download Final Video",
                data=video_file,
                file_name=os.path.basename(final_path),
                mime="video/mp4",
                use_container_width=True,
            )


def main():
    st.title("🎬 Beat Video Editor")
    st.markdown(
        """
        Create beat-synchronized videos automatically.

        Upload your music and videos, choose your settings,
        and generate the final video with random transitions.
        """
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        st.subheader("Clip Settings")

        min_beats = st.slider(
            "Minimum beats per clip",
            min_value=2,
            max_value=10,
            value=4,
        )

        max_beats = st.slider(
            "Maximum beats per clip",
            min_value=3,
            max_value=12,
            value=8,
        )

        if max_beats < min_beats:
            max_beats = min_beats

        st.divider()
        st.subheader("🎞️ Transitions")

        enable_transitions = st.checkbox(
            "Enable random transitions",
            value=True,
        )

        transition_min = st.slider(
            "Transition every minimum clips",
            min_value=2,
            max_value=10,
            value=4,
            disabled=not enable_transitions,
        )

        transition_max = st.slider(
            "Transition every maximum clips",
            min_value=3,
            max_value=15,
            value=8,
            disabled=not enable_transitions,
        )

        transition_duration = st.slider(
            "Transition duration",
            min_value=0.10,
            max_value=1.00,
            value=0.35,
            step=0.05,
            disabled=not enable_transitions,
        )

        st.divider()
        st.subheader("🎥 Quality")

        quality = st.selectbox(
            "Video quality",
            ["Very High", "High", "Balanced"],
            index=1,
        )

        if quality == "Very High":
            crf = 18
            preset = "fast"
        elif quality == "High":
            crf = 20
            preset = "veryfast"
        else:
            crf = 22
            preset = "veryfast"

        st.caption(f"CRF: {crf} | Encoder: {preset}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎵 Music")
        music_file = st.file_uploader(
            "Upload your music",
            type=["mp3", "wav", "m4a", "aac"],
            accept_multiple_files=False,
        )

    with col2:
        st.subheader("🎬 Videos")
        video_files = st.file_uploader(
            "Upload source videos",
            type=["mp4", "mov", "webm", "mkv"],
            accept_multiple_files=True,
        )

    if music_file:
        st.success(f"Music: {music_file.name}")

    if video_files:
        st.success(f"{len(video_files)} source video(s) uploaded")

    if not music_file:
        st.info("👆 Upload a music file to continue.")

    if music_file and not video_files:
        st.info("👆 Upload at least one source video.")

    can_generate = music_file is not None and video_files and len(video_files) > 0

    if can_generate:
        if st.button("🚀 Generate Video", type="primary", use_container_width=True):
            generate_video(
                music_file=music_file,
                video_files=video_files,
                min_beats=min_beats,
                max_beats=max_beats,
                enable_transitions=enable_transitions,
                transition_min=transition_min,
                transition_max=transition_max,
                transition_duration=transition_duration,
                crf=crf,
                preset=preset,
            )

    render_generated_video()


def generate_video(
    music_file,
    video_files,
    min_beats,
    max_beats,
    enable_transitions,
    transition_min,
    transition_max,
    transition_duration,
    crf,
    preset,
):
    project_dir = tempfile.mkdtemp(prefix="beats_editor_ui_")
    music_dir = os.path.join(project_dir, "music")
    videos_dir = os.path.join(project_dir, "videos")
    temp_output_dir = os.path.join(project_dir, "output")

    os.makedirs(music_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(temp_output_dir, exist_ok=True)

    temp_dir = None
    final_path = os.path.join(OUTPUT_DIR, "final.mp4")

    try:
        music_path = os.path.join(music_dir, music_file.name)
        save_uploaded_file(music_file, music_path)

        saved_video_paths = []
        saving_progress = st.progress(0, text="Saving uploaded videos...")

        for index, uploaded_video in enumerate(video_files, start=1):
            video_path = os.path.join(videos_dir, uploaded_video.name)
            save_uploaded_file(uploaded_video, video_path)
            saved_video_paths.append(video_path)
            saving_progress.progress(index / len(video_files), text=f"Saving videos {index}/{len(video_files)}")

        st.subheader("🎵 Beat Detection")
        beat_status = st.empty()
        beat_status.info("Analyzing music...")

        beats = get_beats(music_path)
        if len(beats) < 2:
            raise RuntimeError("Could not detect enough beats.")

        beat_status.success(f"Detected {len(beats)} beats")

        st.subheader("🎬 Generating Clips")
        clip_status = st.empty()
        clip_progress = st.progress(0, text="Preparing clips...")

        def progress_callback(current, total):
            clip_progress.progress(min(current / max(total, 1), 1.0), text=f"Generating clip {current}/{total}")

        temp_clip_paths, temp_dir = sync_clips_with_beats(
            saved_video_paths,
            beats,
            min_beats=min_beats,
            max_beats=max_beats,
            progress_callback=progress_callback,
        )

        if not temp_clip_paths:
            raise RuntimeError("No clips were generated.")

        clip_status.success(f"Created {len(temp_clip_paths)} clips")

        st.subheader("🎞️ Creating Final Video")
        transition_status = st.empty()
        transition_status.info("Combining clips and adding transitions...")

        video_without_audio = os.path.join(temp_output_dir, "video.mp4")

        if enable_transitions:
            concatenate_videos(temp_clip_paths, video_without_audio)
        else:
            concat_group(temp_clip_paths, video_without_audio)

        transition_status.success(
            "Transitions added successfully" if enable_transitions else "Clips combined without transitions"
        )

        st.subheader("🎵 Adding Music")
        audio_status = st.empty()
        audio_status.info("Adding music to video...")

        add_audio(
            video_path=video_without_audio,
            audio_path=music_path,
            output_path=final_path,
        )

        audio_status.success("Music added successfully")
        st.session_state.generated_video = final_path
        st.success("🎉 Video generated successfully!")

    except Exception as exc:
        st.error(f"❌ Video generation failed: {exc}")
        st.exception(exc)

    finally:
        shutil.rmtree(project_dir, ignore_errors=True)
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
