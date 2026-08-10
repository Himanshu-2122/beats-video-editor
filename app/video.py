import os
import random
import subprocess

# ============================================================
# VIDEO SETTINGS
# ============================================================

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
FPS = 30
# Quality: lower = better quality / larger file
CRF = 20
# Encoding preset
PRESET = "veryfast"

VIDEO_EXTENSIONS = (
    ".mp4",
    ".webm",
    ".mov",
    ".mkv",
)


# ============================================================
# FFmpeg helper
# ============================================================

def run_ffmpeg(command):
    """
    Execute FFmpeg command and raise a useful error if FFmpeg fails.
    Returns the completed subprocess.Process instance.
    """
    print("       Running FFmpeg...")

    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if process.returncode != 0:
        print("\nFFmpeg ERROR:")
        print(process.stderr)
        raise RuntimeError("FFmpeg command failed.\n" + process.stderr)

    return process


# ============================================================
# Load video paths
# ============================================================

def load_video_paths(video_folder):
    """
    Find supported video files inside the folder and return their paths sorted.
    """
    if not os.path.isdir(video_folder):
        raise FileNotFoundError(f"Video folder does not exist: {video_folder}")

    video_files = []
    for filename in os.listdir(video_folder):
        path = os.path.join(video_folder, filename)
        if not os.path.isfile(path):
            continue
        if filename.lower().endswith(VIDEO_EXTENSIONS):
            video_files.append(path)

    video_files.sort()
    return video_files


# ============================================================
# Process individual clip
# ============================================================

def process_clip(video_path, duration, output_path):
    """
    Create one normalized video clip.

    Processing:
        - random starting position
        - exact duration
        - 1280x720
        - 30 FPS
        - remove original audio
        - H.264

    FFmpeg handles everything directly.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if duration <= 0:
        raise ValueError(f"Invalid duration: {duration}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get source duration
    source_duration = get_video_duration(video_path)
    if source_duration <= 0:
        raise RuntimeError(f"Could not determine duration: {video_path}")

    # Random start position
    if source_duration > duration:
        max_start = source_duration - duration
        start_time = random.uniform(0, max_start)
    else:
        start_time = 0

    duration = float(duration)

    # Video filter: scale while preserving aspect, then pad/crop to exact size
    video_filter = (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:"
        f"force_original_aspect_ratio=decrease," 
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        f"setsar=1,fps={FPS}"
    )

    command = [
        "ffmpeg",
        "-y",
        # Seek before decoding.
        "-ss",
        f"{start_time:.3f}",
        "-i",
        video_path,
        # Exact output duration.
        "-t",
        f"{duration:.3f}",
        # Video processing.
        "-vf",
        video_filter,
        # Remove source audio.
        "-an",
        # H.264 encoder.
        "-c:v",
        "libx264",
        # Quality.
        "-crf",
        str(CRF),
        # Faster encoding.
        "-preset",
        PRESET,
        # Pixel format compatible with most devices.
        "-pix_fmt",
        "yuv420p",
        # Better seeking/streaming.
        "-movflags",
        "+faststart",
        output_path,
    ]

    run_ffmpeg(command)

    if not os.path.exists(output_path):
        raise RuntimeError(f"FFmpeg did not create: {output_path}")

    return output_path


# ============================================================
# Get video duration
# ============================================================

def get_video_duration(video_path):
    """
    Get video duration using ffprobe.
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]

    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if process.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}\n{process.stderr}")

    try:
        return float(process.stdout.strip())
    except (ValueError, TypeError):
        raise RuntimeError(f"Invalid duration returned for: {video_path}")


# ============================================================
# Create concat file
# ============================================================

def create_concat_file(clip_paths, concat_file):
    """
    Create FFmpeg concat demuxer file.
    """
    with open(concat_file, "w", encoding="utf-8") as file:
        for path in clip_paths:
            escaped_path = os.path.abspath(path).replace("\\", "/").replace("'", "'\\''")
            file.write(f"file '{escaped_path}'\n")


# ============================================================
# Concatenate videos
# ============================================================

def concatenate_videos(clip_paths, output_path):
    """
    Concatenate already normalized clips using the concat demuxer.
    """
    if not clip_paths:
        raise ValueError("No clips to concatenate.")

    concat_file = os.path.join(os.path.dirname(output_path), "concat.txt")
    create_concat_file(clip_paths, concat_file)

    command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_file,
        # Since all clips have identical format, stream copy is fast.
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        output_path,
    ]

    try:
        run_ffmpeg(command)
    finally:
        if os.path.exists(concat_file):
            os.remove(concat_file)

    if not os.path.exists(output_path):
        raise RuntimeError("Combined video was not created.")

    return output_path


# ============================================================
# Add music
# ============================================================

def add_audio(video_path, audio_path, output_path):
    """
    Add music to final video. Video is copied without re-encoding; audio is encoded.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        # Loop audio indefinitely so it fills the video length.
        "-stream_loop",
        "-1",
        "-i",
        audio_path,
        # Take video from first input, audio from second.
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        # Do not re-encode video.
        "-c:v",
        "copy",
        # AAC audio.
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        # Stop when video ends.
        "-shortest",
        "-movflags",
        "+faststart",
        output_path,
    ]

    run_ffmpeg(command)

    if not os.path.exists(output_path):
        raise RuntimeError("Final video was not created.")

    return output_path
