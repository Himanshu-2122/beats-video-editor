import os
import random
import re
import subprocess

# ============================================================
# VIDEO SETTINGS
# ============================================================

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
FPS = 30

# ============================================================
# QSV QUALITY
# ============================================================
#
# Lower number = better quality / larger file
#
# 18 = Very High
# 20 = High / Recommended
# 22 = Balanced
#
QSV_QUALITY = 20
QSV_PRESET = "medium"

# CPU fallback settings.
CPU_CRF = 20
CPU_PRESET = "veryfast"

# ============================================================
# TRANSITIONS
# ============================================================

TRANSITION_MIN_CLIPS = 4
TRANSITION_MAX_CLIPS = 8
TRANSITION_DURATION = 0.25
TRANSITIONS = [
    "fade",
    "fadeblack",
]

VIDEO_EXTENSIONS = (
    ".mp4",
    ".webm",
    ".mov",
    ".mkv",
)

# ============================================================
# FFMPEG
# ============================================================

def run_ffmpeg(command):
    """
    Run FFmpeg.
    """

    print("\nRunning FFmpeg...")

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


def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def check_qsv():
    try:
        process = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = process.stdout + process.stderr
        return "h264_qsv" in output
    except Exception:
        return False


def load_video_paths(video_folder):
    if not os.path.isdir(video_folder):
        raise FileNotFoundError(f"Video folder does not exist: {video_folder}")

    videos = []
    for filename in os.listdir(video_folder):
        path = os.path.join(video_folder, filename)
        if not os.path.isfile(path):
            continue
        if filename.lower().endswith(VIDEO_EXTENSIONS):
            videos.append(path)
    return videos


def natural_sort_key(path):
    filename = os.path.basename(path)
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", filename)
    ]


def get_video_duration(video_path):
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
        raise RuntimeError(f"Could not determine video duration: {video_path}")


def get_video_filter():
    return (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:"
        f"force_original_aspect_ratio=increase,"
        f"crop={TARGET_WIDTH}:{TARGET_HEIGHT},"
        f"setsar=1,"
        f"fps={FPS},"
        f"eq=contrast=1.04:saturation=1.06:brightness=0.005,"
        f"unsharp=5:5:0.25:5:5:0"
    )


def get_qsv_video_args():
    return [
        "-c:v",
        "h264_qsv",
        "-global_quality",
        str(QSV_QUALITY),
        "-preset",
        QSV_PRESET,
        "-pix_fmt",
        "nv12",
        "-profile:v",
        "high",
    ]


def get_cpu_video_args():
    return [
        "-c:v",
        "libx264",
        "-crf",
        str(CPU_CRF),
        "-preset",
        CPU_PRESET,
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
    ]


def process_clip(video_path, duration, output_path, start_time=None):
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

    if start_time is None:
        if source_duration > duration:
            max_start = source_duration - duration
            start_time = random.uniform(0, max_start)
        else:
            start_time = 0.0
            duration = source_duration
    else:
        start_time = max(0.0, start_time)
        if start_time >= source_duration:
            start_time = 0.0
        available_duration = source_duration - start_time
        if duration > available_duration:
            duration = available_duration
            if duration <= 0:
                raise RuntimeError("Clip duration became zero.")

    video_filter = get_video_filter()

    qsv_command = [
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
    ]
    qsv_command.extend(get_qsv_video_args())
    qsv_command.extend([
        "-movflags",
        "+faststart",
        output_path,
    ])

    try:
        run_ffmpeg(qsv_command)
    except RuntimeError:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        cpu_command = [
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
        ]
        cpu_command.extend(get_cpu_video_args())
        cpu_command.extend([
            "-movflags",
            "+faststart",
            output_path,
        ])
        run_ffmpeg(cpu_command)

    if not os.path.exists(output_path):
        raise RuntimeError(f"Clip was not created: {output_path}")

    return output_path


def create_concat_file(clip_paths, concat_file):
    with open(concat_file, "w", encoding="utf-8") as file:
        for path in clip_paths:
            absolute_path = os.path.abspath(path).replace("\\", "/")
            absolute_path = absolute_path.replace("'", "'\\''")
            file.write(f"file '{absolute_path}'\n")


def concat_group(clip_paths, output_path):
    if not clip_paths:
        raise ValueError("No clips supplied.")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    concat_file = os.path.join(output_dir, f"_concat_{random.randint(100000, 999999)}.txt")
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
            try:
                os.remove(concat_file)
            except OSError:
                pass

    return output_path


def create_random_groups(clip_paths):
    groups = []
    index = 0
    total = len(clip_paths)

    while index < total:
        remaining = total - index
        if remaining <= TRANSITION_MAX_CLIPS:
            group_size = remaining
        else:
            group_size = random.randint(TRANSITION_MIN_CLIPS, TRANSITION_MAX_CLIPS)

        group = clip_paths[index:index + group_size]
        if group:
            groups.append(group)
        index += group_size

    return groups


def create_transition_video(group_paths, output_path):
    if not group_paths:
        raise ValueError("No groups supplied.")

    if len(group_paths) == 1:
        return concat_group(group_paths, output_path)

    print("\nAdding selective transitions...")

    durations = [get_video_duration(path) for path in group_paths]
    for path, duration in zip(group_paths, durations):
        if duration <= 0:
            raise RuntimeError(f"Invalid group duration: {path}")

    command = ["ffmpeg", "-y"]
    for path in group_paths:
        command.extend(["-i", path])

    filters = []
    cumulative_duration = durations[0]
    previous_label = "[0:v]"

    for i in range(1, len(group_paths)):
        current_duration = durations[i]
        transition = random.choice(TRANSITIONS)
        transition_duration = min(TRANSITION_DURATION, durations[i - 1] / 2, current_duration / 2)
        transition_duration = max(0.10, transition_duration)
        offset = cumulative_duration - transition_duration
        output_label = f"[v{i}]"

        filters.append(
            f"{previous_label}[{i}:v]xfade=transition={transition}:duration={transition_duration:.3f}:offset={offset:.3f}{output_label}"
        )

        print(f"       Transition {i}: {transition} {transition_duration:.2f}s")

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
        "-preset",
        CPU_PRESET,
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


def concatenate_videos(clip_paths, output_path):
    if not clip_paths:
        raise ValueError("No clips to concatenate.")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    groups = create_random_groups(clip_paths)
    print(f"       Created {len(groups)} clip groups.")

    group_paths = []
    try:
        for index, group in enumerate(groups, start=1):
            group_output = os.path.join(output_dir, f"_group_{index:04d}.mp4")
            print(f"       Building group {index}/{len(groups)} ({len(group)} clips)")
            concat_group(group, group_output)
            group_paths.append(group_output)

        create_transition_video(group_paths, output_path)
    finally:
        for path in group_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

    return output_path


def add_audio(video_path, audio_path, output_path):
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
        "256k",
        "-shortest",
        "-movflags",
        "+faststart",
        output_path,
    ]

    run_ffmpeg(command)

    if not os.path.exists(output_path):
        raise RuntimeError("Final video was not created.")

    return output_path
