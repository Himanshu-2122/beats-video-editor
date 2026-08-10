import os
import random
import subprocess
import tempfile


# ============================================================
# VIDEO SETTINGS
# ============================================================

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

FPS = 30

# Lower CRF = better quality / larger file
# 18 = very high quality
# 20 = excellent balance
CRF = 20

# Encoding speed
PRESET = "veryfast"

# Transition settings
TRANSITION_MIN_CLIPS = 4
TRANSITION_MAX_CLIPS = 8

TRANSITION_DURATION = 0.35

TRANSITIONS = [
    "fade",
    "fadeblack",
    "wipeleft",
    "wiperight",
    "slideleft",
    "slideright",
]


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
    Run FFmpeg and raise an error if it fails.
    """

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
        raise RuntimeError("FFmpeg command failed.")

    return process


# ============================================================
# Load video paths
# ============================================================

def load_video_paths(video_folder):
    """
    Find all supported videos in a folder.
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
        raise RuntimeError(f"ffprobe failed:\n{process.stderr}")

    try:
        return float(process.stdout.strip())
    except (ValueError, TypeError):
        raise RuntimeError(f"Invalid duration for: {video_path}")


# ============================================================
# Process individual clip
# ============================================================

def process_clip(video_path, duration, output_path):
    """
    Create one normalized clip.

    Output:
        1280x720
        30 FPS
        H.264
        no audio
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if duration <= 0:
        raise ValueError(f"Invalid duration: {duration}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    source_duration = get_video_duration(video_path)
    if source_duration <= 0:
        raise RuntimeError(f"Invalid source duration: {video_path}")

    if source_duration > duration:
        max_start = source_duration - duration
        start_time = random.uniform(0, max_start)
    else:
        start_time = 0

    video_filter = (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:"
        f"force_original_aspect_ratio=decrease,"
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:"
        f"(ow-iw)/2:(oh-ih)/2,"
        f"setsar=1,"
        f"fps={FPS}"
    )

    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_time:.3f}",
        "-i",
        video_path,
        "-t",
        f"{duration:.3f}",
        "-vf",
        video_filter,
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        str(CRF),
        "-preset",
        PRESET,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]

    run_ffmpeg(command)

    if not os.path.exists(output_path):
        raise RuntimeError(f"Clip was not created: {output_path}")

    return output_path


# ============================================================
# Create concat file
# ============================================================

def create_concat_file(clip_paths, concat_file):
    """
    Create FFmpeg concat demuxer file.
    """

    with open(concat_file, "w", encoding="utf-8") as file:
        for path in clip_paths:
            absolute_path = os.path.abspath(path).replace("\\", "/")
            absolute_path = absolute_path.replace("'", "'\\''")
            file.write(f"file '{absolute_path}'\n")


# ============================================================
# Concatenate normal clips
# ============================================================

def concat_group(clip_paths, output_path):
    """
    Concatenate clips without transitions.

    This is very fast because stream copy is used.
    """

    if not clip_paths:
        raise ValueError("No clip paths provided for concatenation.")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    concat_file = os.path.join(
        os.path.dirname(output_path),
        f"concat_{random.randint(100000, 999999)}.txt",
    )

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
        raise RuntimeError(f"Concatenated group was not created: {output_path}")

    return output_path


# ============================================================
# Split clips into random groups
# ============================================================

def create_random_groups(clip_paths):
    """
    Split clips into random groups.

    Each group contains 4-8 clips.

    A transition will be added BETWEEN groups.
    """

    groups = []
    index = 0
    total = len(clip_paths)

    while index < total:
        remaining = total - index
        if remaining <= TRANSITION_MAX_CLIPS:
            group_size = remaining
        else:
            group_size = random.randint(TRANSITION_MIN_CLIPS, TRANSITION_MAX_CLIPS)
            group_size = min(group_size, remaining)

        groups.append(clip_paths[index:index + group_size])
        index += group_size

    return groups


# ============================================================
# Add random transitions
# ============================================================

def create_transition_video(group_paths, output_path):
    """
    Combine groups using random xfade transitions.

    Only group boundaries receive transitions.
    """

    if not group_paths:
        raise ValueError("No group paths provided for transition video.")

    if len(group_paths) == 1:
        return concat_group(group_paths, output_path)

    print(f"       Adding {len(group_paths) - 1} random transitions...")

    durations = [get_video_duration(path) for path in group_paths]

    command = ["ffmpeg", "-y"]
    for path in group_paths:
        command.extend(["-i", path])

    filters = []
    cumulative_duration = durations[0]
    previous_label = "[0:v]"

    for i in range(1, len(group_paths)):
        current_duration = durations[i]
        transition = random.choice(TRANSITIONS)
        transition_duration = min(
            TRANSITION_DURATION,
            durations[i - 1] / 2,
            current_duration / 2,
        )
        transition_duration = max(0.10, transition_duration)
        offset = cumulative_duration - transition_duration
        output_label = f"[v{i}]"

        filters.append(
            f"{previous_label}[{i}:v]"
            f"xfade=transition={transition}:duration={transition_duration:.3f}:offset={offset:.3f}"
            f"{output_label}"
        )

        print(
            f"       Transition {i}: {transition} "
            f"({transition_duration:.2f}s)"
        )

        cumulative_duration = cumulative_duration + current_duration - transition_duration
        previous_label = output_label

    filter_complex = ";".join(filters)

    command.extend([
        "-filter_complex",
        filter_complex,
        "-map",
        previous_label,
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        str(CRF),
        "-preset",
        PRESET,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ])

    run_ffmpeg(command)

    if not os.path.exists(output_path):
        raise RuntimeError("Transition video was not created.")

    return output_path


# ============================================================
# Main concatenate function
# ============================================================

def concatenate_videos(clip_paths, output_path):
    """
    Concatenate clips with selective random transitions.

    Example:

        clips 1-6
        ↓ transition
        clips 7-12
        ↓ transition
        clips 13-17
        ↓ transition
        ...

        Normal cuts happen inside each group.
    """

    if not clip_paths:
        raise ValueError("No clips to concatenate.")

    temp_dir = os.path.dirname(output_path)
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)

    groups = create_random_groups(clip_paths)
    print(f"       Created {len(groups)} clip groups")

    group_paths = []
    for index, group in enumerate(groups, start=1):
        group_output = os.path.join(temp_dir, f"group_{index:04d}.mp4")
        print(
            f"       Building group {index}/{len(groups)} "
            f"({len(group)} clips)"
        )
        concat_group(group, group_output)
        group_paths.append(group_output)

    create_transition_video(group_paths, output_path)

    for path in group_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as exc:
            print(f"       Could not remove {path}: {exc}")

    return output_path


# ============================================================
# Add music
# ============================================================

def add_audio(video_path, audio_path, output_path):
    """
    Add music to the final video.

    Video is copied without re-encoding.
    Only audio is encoded.
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-stream_loop",
        "-1",
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        output_path,
    ]

    run_ffmpeg(command)

    if not os.path.exists(output_path):
        raise RuntimeError("Final video was not created.")

    return output_path

