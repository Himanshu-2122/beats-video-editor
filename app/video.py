import os
import random
import re
import subprocess
import tempfile
import io
import hashlib
import json
import numpy as np
from typing import Optional

# ============================================================
# VIDEO SETTINGS
# ============================================================

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
FPS = 30

# Scene detection threshold for candidate generation
SCENE_THRESH = 0.3

# Default sample interval for uniform candidate sampling (seconds)
SAMPLE_INTERVAL = 0.5

# Maximum candidates per source video
MAX_CANDIDATES_PER_SOURCE = 100

# Maximum clips per source video
MAX_CLIPS_PER_SOURCE = 2

# Clip duration bounds (in beats)
MIN_BEATS_PER_CLIP = 4
MAX_BEATS_PER_CLIP = 8

# Motion analysis settings - OPTIMIZED FOR SPEED
MOTION_SAMPLE_FPS = 1  # Sample frames at this rate for motion analysis (was 2)
MOTION_WINDOW_SECONDS = 1.0  # Window size for motion averaging
MOTION_DOWNSCALE = 0.125  # Downscale factor for faster optical flow (0.125 = 1/8 resolution, was 0.25)

# Transition settings
TRANSITION_DURATION = 0.4
TRANSITION_MIN_CLIPS = 4
TRANSITION_MAX_CLIPS = 8
TRANSITIONS = [
    "fade",
    "wipeleft",
    "wiperight",
    "wipeup",
    "wipedown",
    "slideleft",
    "slideright",
    "slideup",
    "slidedown",
    "circlecrop",
    "rectcrop",
    "distance",
    "fadeblack",
    "fadewhite",
    "radial",
    "smoothleft",
    "smoothright",
    "smoothup",
    "smoothdown",
    "circleopen",
    "circleclose",
    "vertopen",
    "vertclose",
    "horzopen",
    "horzclose",
    "dissolve",
    "pixelize",
    "diagtl",
    "diagtr",
    "diagbl",
    "diagbr",
    "hlslice",
    "hrslice",
    "vuslice",
    "vdslice",
    "hblur",
    "fadegrays",
    "wipetl",
    "wipetr",
    "wipebl",
    "wipebr",
    "squeezeh",
    "squeezev",
    "zoomin",
    "fadefast",
    "fadeslow",
    "hlwind",
    "hrwind",
    "vuwind",
    "vdwind",
    "coverleft",
    "coverright",
    "coverup",
    "coverdown",
    "revealleft",
    "revealright",
    "revealup",
    "revealdown",
]

# Minimum gap between clips from same source (seconds) - prevents rapid reuse feel
MIN_CLIP_GAP = 3.0

VIDEO_EXTENSIONS = (
    ".mp4",
    ".webm",
    ".mov",
    ".mkv",
)


def snap_to_frame(time_s: float, fps: int = FPS) -> float:
    """Snap a timestamp to the nearest frame boundary at given FPS."""
    return round(time_s * fps) / fps


# ============================================================
# COLOR NORMALIZATION (for visual consistency)
# ============================================================

def get_color_normalization_filter():
    """FFmpeg filter to normalize color/brightness/contrast across clips.
    Reduced intensity to preserve source quality while maintaining consistency.
    """
    return (
        "eq=contrast=1.01:saturation=1.02:brightness=0.005:gamma=1.0,"
        "hue=s=1.0"
    )


def get_video_filter():
    """Full filter chain for final output - applied ONCE during transition/encode stage."""
    return (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:"
        f"force_original_aspect_ratio=increase,"
        f"crop={TARGET_WIDTH}:{TARGET_HEIGHT},"
        f"setsar=1,"
        f"fps={FPS}:round=near,"
        f"{get_color_normalization_filter()},"
        f"unsharp=3:3:0.5:3:3:0.2"
    )


def extract_frame(video_path: str, time_s: float):
    """Extract a single frame from video at given timestamp using ffmpeg pipe to PIL."""
    import subprocess
    from PIL import Image

    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{time_s:.3f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]

    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    if process.returncode != 0 or not process.stdout:
        return None

    try:
        return Image.open(io.BytesIO(process.stdout))
    except Exception:
        return None


def extract_audio_segment(video_path: str, start_s: float, duration_s: float):
    """Extract audio segment using librosa."""
    import librosa
    import numpy as np

    try:
        y, sr = librosa.load(
            video_path,
            sr=None,
            mono=True,
            offset=start_s,
            duration=duration_s,
        )
        return y, sr
    except Exception:
        return None, None


def detect_scene_changes(video_path: str, threshold: float = SCENE_THRESH):
    """Detect scene change timestamps using ffmpeg scene detection filter."""
    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        f"select='gt(scene,{threshold})',showinfo",
        "-vsync",
        "vfr",
        "-f",
        "null",
        "-",
    ]

    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    scene_times = []
    output = process.stderr + process.stdout

    for line in output.split("\n"):
        if "pts_time:" in line:
            try:
                pts_part = line.split("pts_time:")[1].split()[0]
                scene_time = float(pts_part)
                scene_times.append(scene_time)
            except (IndexError, ValueError):
                continue

    return scene_times


def generate_candidates(
    video_path: str,
    clip_duration: float,
    sample_interval: float = SAMPLE_INTERVAL,
    scene_threshold: float = SCENE_THRESH,
    max_candidates: int = MAX_CANDIDATES_PER_SOURCE,
    compute_motion: bool = False,
    cache_dir: str = None,
    cached_scenes: list = None,  # Accept pre-computed scenes
):
    """Generate candidate clip start times from scene anchors and uniform sampling,
    distributed across the full video duration."""
    source_duration = get_video_duration(video_path)
    if source_duration <= clip_duration:
        return [{"t": 0.0, "scene_flag": True, "motion_score": 0.0}]

    # Auto-adjust sample interval for long videos to limit raw candidate count
    max_raw_candidates = 500
    if source_duration / sample_interval > max_raw_candidates:
        sample_interval = source_duration / max_raw_candidates
        print(f"       Auto-adjusted sample_interval to {sample_interval:.1f}s for {source_duration:.0f}s video")

    candidates = []

    # Scene anchors (high priority) - use cached scenes if available
    if cached_scenes is not None:
        scene_times = [s["start"] for s in cached_scenes if s.get("duration", 0) > 0.5]
    else:
        scene_times = detect_scene_changes(video_path, scene_threshold)
    for t in scene_times:
        if source_duration - t >= clip_duration:
            candidates.append({"t": t, "scene_flag": True})

    # Uniform sampling across full duration - vectorized for speed
    max_start = source_duration - clip_duration
    if max_start > 0:
        # Calculate number of samples directly (avoids loop)
        num_samples = int(max_start / sample_interval) + 1
        if num_samples > 0:
            # Generate timestamps directly using list comprehension (faster than while loop)
            uniform_times = [i * sample_interval for i in range(num_samples)]
            # Filter in comprehension (avoids second pass)
            candidates.extend({"t": t, "scene_flag": False} for t in uniform_times if t <= max_start)

    # Deduplicate (keep scene_flag=True if duplicate)
    seen = {}
    for c in candidates:
        key = round(c["t"], 3)
        if key not in seen or c["scene_flag"]:
            seen[key] = c

    candidates = list(seen.values())

    # Separate scene anchors and uniform candidates
    scene_candidates = [c for c in candidates if c["scene_flag"]]
    uniform_candidates = [c for c in candidates if not c["scene_flag"]]

    # Always keep all scene anchors (they're high priority)
    # Distribute remaining slots across uniform candidates spanning full duration
    remaining_slots = max_candidates - len(scene_candidates)
    if remaining_slots > 0 and uniform_candidates:
        # Sort uniform candidates by time
        uniform_candidates.sort(key=lambda x: x["t"])
        # Sample evenly across the full timeline
        if len(uniform_candidates) > remaining_slots:
            if remaining_slots == 1:
                uniform_candidates = [uniform_candidates[len(uniform_candidates) // 2]]
            else:
                indices = [int(i * (len(uniform_candidates) - 1) / (remaining_slots - 1)) for i in range(remaining_slots)]
                uniform_candidates = [uniform_candidates[i] for i in indices]
    elif remaining_slots <= 0:
        # Too many scene anchors, keep only the first max_candidates
        scene_candidates = scene_candidates[:max_candidates]
        uniform_candidates = []

    # Combine: scene anchors first (high priority), then distributed uniform candidates
    candidates = scene_candidates + uniform_candidates
    candidates.sort(key=lambda x: (not x["scene_flag"], x["t"]))

    # Compute motion scores if requested
    if compute_motion:
        candidates = analyze_motion_at_candidates(video_path, candidates, clip_duration, cache_dir)

    return candidates


def get_video_hash(video_path: str) -> str:
    """Generate a hash for video file to use as cache key."""
    stat = os.stat(video_path)
    content = f"{video_path}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


# ============================================================
# MOTION ANALYSIS (Stage 3)
# ============================================================

# Motion analysis settings - OPTIMIZED FOR SPEED
MOTION_SAMPLE_FPS = 1  # Sample frames at this rate for motion analysis (was 2)
MOTION_WINDOW_SECONDS = 1.0  # Window size for motion averaging
MOTION_DOWNSCALE = 0.125  # Downscale factor for faster optical flow (0.125 = 1/8 resolution, was 0.25)


def compute_optical_flow_magnitude(prev_frame, curr_frame):
    """Compute mean optical flow magnitude between two frames using Farneback."""
    import cv2
    import numpy as np

    # Convert PIL to numpy if needed
    if hasattr(prev_frame, 'convert'):
        prev_frame = np.array(prev_frame.convert('L'))
    if hasattr(curr_frame, 'convert'):
        curr_frame = np.array(curr_frame.convert('L'))

    # Downscale for speed
    h, w = prev_frame.shape[:2]
    new_w, new_h = int(w * MOTION_DOWNSCALE), int(h * MOTION_DOWNSCALE)
    if new_w < 16 or new_h < 16:
        new_w, new_h = max(16, new_w), max(16, new_h)

    prev_small = cv2.resize(prev_frame, (new_w, new_h))
    curr_small = cv2.resize(curr_frame, (new_w, new_h))

    # Compute Farneback optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_small, curr_small,
        None,  # flow output
        0.5,   # pyr_scale
        3,     # levels
        15,    # winsize
        3,     # iterations
        5,     # poly_n
        1.2,   # poly_sigma
        0      # flags
    )

    # Compute magnitude
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return float(np.mean(magnitude))


def analyze_motion_segment(video_path: str, start_s: float, duration: float, sample_fps: int = MOTION_SAMPLE_FPS) -> float:
    """
    Analyze motion in a video segment using optical flow.
    Returns average motion magnitude (0.0 = static, higher = more motion).
    """
    import cv2
    import numpy as np
    from PIL import Image

    if duration <= 0:
        return 0.0

    # Calculate frame interval
    frame_interval = 1.0 / sample_fps
    num_samples = max(1, int(duration * sample_fps))

    magnitudes = []
    prev_frame = None

    for i in range(num_samples + 1):
        t = start_s + i * frame_interval
        if t > start_s + duration:
            break

        frame = extract_frame(video_path, t)
        if frame is None:
            continue

        # Convert to grayscale numpy
        gray = np.array(frame.convert('L'))

        if prev_frame is not None:
            mag = compute_optical_flow_magnitude(prev_frame, gray)
            magnitudes.append(mag)

        prev_frame = gray

    if not magnitudes:
        return 0.0

    # Return mean motion magnitude
    return float(np.mean(magnitudes))


def analyze_motion_at_candidates(
    video_path: str,
    candidates: list[dict],
    clip_duration: float,
    cache_dir: str = None,
) -> list[dict]:
    """
    Compute motion scores for all candidates.
    Adds 'motion_score' to each candidate dict.
    """
    import os
    import json

    # Check cache first
    video_hash = get_video_hash(video_path)
    cache_key = f"motion_{video_hash}_{clip_duration}.json"
    cache_path = os.path.join(cache_dir, cache_key) if cache_dir else None

    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            # Merge cached scores
            cached_dict = {c['t']: c['motion_score'] for c in cached}
            for c in candidates:
                c['motion_score'] = cached_dict.get(round(c['t'], 3), 0.0)
            return candidates
        except Exception:
            pass

    # Compute motion for each candidate
    for c in candidates:
        score = analyze_motion_segment(video_path, c['t'], clip_duration)
        c['motion_score'] = score

    # Save to cache
    if cache_path:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            serializable = [{'t': c['t'], 'motion_score': c['motion_score']} for c in candidates]
            with open(cache_path, 'w') as f:
                json.dump(serializable, f)
        except Exception:
            pass

    return candidates


# ============================================================
# QSV QUALITY
# ============================================================
#
# Lower number = better quality / larger file
#
# 16 = Near Lossless / Maximum Quality
# 18 = Very High
# 20 = High / Recommended
# 22 = Balanced
#
QSV_QUALITY = 16
QSV_PRESET = "slow"

# CPU fallback settings.
# CRF 15-16 = Visually transparent / Near-lossless
# CRF 17-18 = High quality
CPU_CRF = 16
CPU_PRESET = "slow"

# ============================================================
# TRANSITIONS
# ============================================================

TRANSITION_MIN_CLIPS = 4
TRANSITION_MAX_CLIPS = 8
TRANSITION_DURATION = 0.4
TRANSITIONS = [
    "fade",
    "wipeleft",
    "wiperight",
    "wipeup",
    "wipedown",
    "slideleft",
    "slideright",
    "slideup",
    "slidedown",
    "circlecrop",
    "rectcrop",
    "distance",
    "fadeblack",
    "fadewhite",
    "radial",
    "smoothleft",
    "smoothright",
    "smoothup",
    "smoothdown",
    "circleopen",
    "circleclose",
    "vertopen",
    "vertclose",
    "horzopen",
    "horzclose",
    "dissolve",
    "pixelize",
    "diagtl",
    "diagtr",
    "diagbl",
    "diagbr",
    "hlslice",
    "hrslice",
    "vuslice",
    "vdslice",
    "hblur",
    "fadegrays",
    "wipetl",
    "wipetr",
    "wipebl",
    "wipebr",
    "squeezeh",
    "squeezev",
    "zoomin",
    "fadefast",
    "fadeslow",
    "hlwind",
    "hrwind",
    "vuwind",
    "vdwind",
    "coverleft",
    "coverright",
    "coverup",
    "coverdown",
    "revealleft",
    "revealright",
    "revealup",
    "revealdown",
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


def run_ffmpeg_stream(command, logger=None, progress_callback=None, total_duration=None):
    """
    Run FFmpeg and stream stderr output line-by-line to both stdout and an optional
    logger callable. FFmpeg prints progress to stderr, so we capture that.

    `logger` is a callable that accepts a single string argument.
    """

    print("\nRunning FFmpeg (streaming)...")

    # Use raw bytes read so we can handle progress lines that use carriage
    # returns (\r) rather than newlines. This keeps the UI/terminal updated
    # during long ffmpeg filter_complex/encode runs.
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    buffer = b""
    try:
        while True:
            chunk = process.stderr.read(1024)
            if chunk:
                buffer += chunk
                try:
                    text = buffer.decode("utf-8", errors="replace")
                except Exception:
                    text = buffer.decode("utf-8", errors="replace")

                # Find the last line break (either \n or \r). Process complete
                # lines and keep the remainder in the buffer.
                last_break = max(text.rfind("\n"), text.rfind("\r"))
                if last_break != -1:
                    to_emit = text[: last_break + 1]
                    remainder = text[last_break + 1 :]

                    for line in re.split(r"[\r\n]+", to_emit):
                        if not line:
                            continue
                        # print to terminal
                        print(line)
                        # forward to logger (e.g., Streamlit UI)
                        try:
                            if logger:
                                logger(line)
                        except Exception:
                            pass

                        # parse ffmpeg progress time=HH:MM:SS.xx and call progress callback
                        try:
                            m = re.search(r"time=(\d{2}:\d{2}:\d{2}\.\d{2})", line)
                            if m and progress_callback and total_duration:
                                hh, mm, ss = m.group(1).split(":")
                                elapsed = int(hh) * 3600 + int(mm) * 60 + float(ss)
                                try:
                                    progress_callback(elapsed, total_duration)
                                except Exception:
                                    pass
                        except Exception:
                            pass

                    buffer = remainder.encode("utf-8")
            else:
                if process.poll() is not None:
                    break

        # flush any remaining buffered output
        if buffer:
            try:
                text = buffer.decode("utf-8", errors="replace")
            except Exception:
                text = str(buffer)
            for line in re.split(r"[\r\n]+", text):
                if not line:
                    continue
                print(line)
                try:
                    if logger:
                        logger(line)
                except Exception:
                    pass
                try:
                    m = re.search(r"time=(\d{2}:\d{2}:\d{2}\.\d{2})", line)
                    if m and progress_callback and total_duration:
                        hh, mm, ss = m.group(1).split(":")
                        elapsed = int(hh) * 3600 + int(mm) * 60 + float(ss)
                        try:
                            progress_callback(elapsed, total_duration)
                        except Exception:
                            pass
                except Exception:
                    pass

        process.wait()

    except Exception:
        try:
            process.kill()
        except Exception:
            pass
        raise

    if process.returncode != 0:
        raise RuntimeError("FFmpeg command failed. See output for details.")

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


def get_clip_trim_filter():
    """Minimal filter for clip trimming - only fps/SAR normalization, no quality loss."""
    return (
        f"fps={FPS}:round=near,"
        f"setsar=1"
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
        "-level",
        "4.2",
        "-look_ahead",
        "1",
        "-b_strategy",
        "1",
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
        "-level",
        "4.2",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-color_range",
        "tv",
        "-x264-params",
        "colormatrix=bt709:colorprim=bt709:transfer=bt709:range=tv:deblock=0:0:psy-rd=1.0:0.15",
    ]


def process_clip(video_path, duration, output_path, start_time=None, snap_frame: bool = True):
    """
    Trim clip with stream copy (no re-encode) when possible.
    Only applies minimal fps/SAR normalization filter.
    Full quality processing happens once during transition stage.
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

    # Snap start_time to frame boundary for precise beat alignment
    if snap_frame:
        start_time = snap_to_frame(start_time)

    print(f"[diagnostic] process_clip: start={start_time:.3f}s duration={duration:.3f}s source_duration={source_duration:.3f}s")

    # Try stream copy first (fastest, no quality loss)
    # Use -avoid_negative_ts make_zero to fix timestamp issues
    copy_command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_time:.3f}",
        "-i",
        video_path,
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "copy",
        "-an",
        "-avoid_negative_ts",
        "make_zero",
        "-fflags",
        "+genpts",
        output_path,
    ]

    try:
        run_ffmpeg(copy_command)
        # Verify output duration matches expected
        out_dur = get_video_duration(output_path)
        if abs(out_dur - duration) > 0.1:  # Allow small tolerance
            raise RuntimeError(f"Stream copy duration mismatch: expected {duration:.3f}s, got {out_dur:.3f}s")
        return output_path
    except RuntimeError:
        # Fallback: minimal re-encode with fps/SAR normalization only
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass

        video_filter = get_clip_trim_filter()

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
        # Use streaming runner when available so callers can receive realtime
        # ffmpeg stderr lines (for UI/terminal feedback).
        run_ffmpeg_stream(command)
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


def create_transition_video(group_paths, output_path, logger=None, progress_callback=None):
    """
    Apply transitions AND full video processing (scale, crop, color norm, fps) in ONE pass.
    Uses hardware encoder (QSV) when available.
    """
    if not group_paths:
        raise ValueError("No groups supplied.")

    if len(group_paths) == 1:
        # Single group: apply full filter chain without transitions
        return apply_full_filter_chain(group_paths[0], output_path, logger=logger)

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

    # Normalize each input to consistent timebase/fps before xfade
    # Apply fps filter to each input to ensure matching timebases for xfade
    normalized_labels = []
    for i in range(len(group_paths)):
        norm_label = f"[norm{i}]"
        filters.append(f"[{i}:v]fps={FPS}:round=near,setpts=PTS-STARTPTS{norm_label}")
        normalized_labels.append(norm_label)

    previous_label = normalized_labels[0]

    for i in range(1, len(group_paths)):
        current_duration = durations[i]
        transition = random.choice(TRANSITIONS)
        transition_duration = min(TRANSITION_DURATION, durations[i - 1] / 2, current_duration / 2)
        transition_duration = max(0.10, transition_duration)
        offset = cumulative_duration - transition_duration
        output_label = f"[v{i}]"

        filters.append(
            f"{previous_label}{normalized_labels[i]}xfade=transition={transition}:duration={transition_duration:.3f}:offset={offset:.3f}{output_label}"
        )

        try:
            if logger:
                logger(f"       Transition {i}: {transition} {transition_duration:.2f}s")
            else:
                print(f"       Transition {i}: {transition} {transition_duration:.2f}s")
        except Exception:
            pass

        cumulative_duration = cumulative_duration + current_duration - transition_duration
        previous_label = output_label

    # Append full video filter chain to the final output
    final_filter = f"{previous_label}{get_video_filter()}[outv]"
    filters.append(final_filter)

    filter_complex = ";".join(filters)
    command.extend([
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
        "-an",
    ])

    # Use hardware encoder if available
    use_qsv = check_qsv()
    if use_qsv:
        command.extend(get_qsv_video_args())
    else:
        command.extend(get_cpu_video_args())

    command.extend([
        "-movflags",
        "+faststart",
        output_path,
    ])

    total_output_duration = cumulative_duration
    run_ffmpeg_stream(command, logger=logger, progress_callback=progress_callback, total_duration=total_output_duration)

    if not os.path.exists(output_path):
        raise RuntimeError("Transition video was not created.")

    return output_path


def apply_full_filter_chain(input_path, output_path, logger=None):
    """Apply full video filter chain (scale, crop, color, fps) to a single video."""
    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        get_video_filter(),
        "-an",
    ]

    use_qsv = check_qsv()
    if use_qsv:
        command.extend(get_qsv_video_args())
    else:
        command.extend(get_cpu_video_args())

    command.extend([
        "-movflags",
        "+faststart",
        output_path,
    ])

    run_ffmpeg_stream(command, logger=logger)

    if not os.path.exists(output_path):
        raise RuntimeError("Filter chain video was not created.")

    return output_path


def concatenate_videos(
    clip_paths,
    output_path,
    beat_groups=None,
    transition_min=None,
    transition_max=None,
    transition_duration=None,
    logger=None,
    progress_callback=None,
):
    """
    Concatenate clips into `output_path`.

    - If `beat_groups` is provided, it will be used as the groups of clips.
    - Otherwise groups are generated via `create_random_groups()`; optional
      `transition_min`/`transition_max` override the module defaults while
      grouping.
    - `transition_duration` temporarily overrides `TRANSITION_DURATION` when
      creating transitions.
    """

    if not clip_paths:
        raise ValueError("No clips to concatenate.")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Determine groups
    if beat_groups is not None:
        # `beat_groups` may be a list of dicts (from sync_clips_with_beats),
        # where each dict includes a `clip_path` key. Convert that into the
        # expected list-of-lists format used by the concatenation logic.
        if isinstance(beat_groups, list) and beat_groups and isinstance(beat_groups[0], dict):
            groups = [[g["clip_path"]] for g in beat_groups]
        else:
            groups = beat_groups
    else:
        old_min = globals().get("TRANSITION_MIN_CLIPS", TRANSITION_MIN_CLIPS)
        old_max = globals().get("TRANSITION_MAX_CLIPS", TRANSITION_MAX_CLIPS)
        try:
            if transition_min is not None:
                globals()["TRANSITION_MIN_CLIPS"] = int(transition_min)
            if transition_max is not None:
                globals()["TRANSITION_MAX_CLIPS"] = int(transition_max)

            groups = create_random_groups(clip_paths)
        finally:
            globals()["TRANSITION_MIN_CLIPS"] = old_min
            globals()["TRANSITION_MAX_CLIPS"] = old_max

    print(f"       Created {len(groups)} clip groups.")

    group_paths = []
    try:
        for index, group in enumerate(groups, start=1):
            group_output = os.path.join(output_dir, f"_group_{index:04d}.mp4")
            try:
                if logger:
                    logger(f"       Building group {index}/{len(groups)} ({len(group)} clips)")
                else:
                    print(f"       Building group {index}/{len(groups)} ({len(group)} clips)")
            except Exception:
                pass

            # pass logger to concat_group so it can stream ffmpeg output
            concat_group(group, group_output)
            group_paths.append(group_output)

        # Optionally override transition duration when building the final video
        if transition_duration is not None:
            old_duration = globals().get("TRANSITION_DURATION", TRANSITION_DURATION)
            try:
                globals()["TRANSITION_DURATION"] = float(transition_duration)
                create_transition_video(group_paths, output_path, logger=logger, progress_callback=progress_callback)
            finally:
                globals()["TRANSITION_DURATION"] = old_duration
        else:
            create_transition_video(group_paths, output_path, logger=logger, progress_callback=progress_callback)
    finally:
        for path in group_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

    return output_path


def encode_in_chunks(input_path, output_path, segment_time=60, crf=None, preset=None, use_qsv=True, logger=None, progress_callback=None):
    """
    Fast remux - the input is already fully processed (transitions + filter chain applied).
    This function now just remuxes to ensure clean container, no re-encoding.
    Kept for API compatibility with main.py.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    if not check_ffmpeg():
        raise RuntimeError("ffmpeg/ffprobe not available on PATH")

    # Simple remux - no re-encoding, preserves quality exactly
    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        output_path,
    ]

    if logger:
        logger(f"       Remuxing (no re-encode)...")
    else:
        print("       Remuxing (no re-encode)...")

    run_ffmpeg_stream(command, logger=logger)

    if not os.path.exists(output_path):
        raise RuntimeError("Remux failed.")

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
        "320k",
        "-profile:a",
        "aac_low",
        "-shortest",
        "-movflags",
        "+faststart",
        output_path,
    ]

    run_ffmpeg(command)

    if not os.path.exists(output_path):
        raise RuntimeError("Final video was not created.")

    return output_path


# ============================================================
# ADVANCED VIDEO ANALYSIS (Stage 2)
# ============================================================

def analyze_video_full(video_path: str, cache_dir: str = None, fast_mode: bool = True) -> dict:
    """
    Comprehensive video analysis: scenes, motion profile, highlights, 
    camera changes, visual quality.
    
    Args:
        video_path: Path to video file
        cache_dir: Optional cache directory
        fast_mode: If True, skip heavy computations (motion profile, camera changes, quality, colors)
    
    Returns:
        dict: Complete video analysis
    """
    if cache_dir:
        video_hash = get_video_hash(video_path)
        cache_path = os.path.join(cache_dir, f"video_{video_hash}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached = json.load(f)
                print(f"       Loaded video analysis from cache: {os.path.basename(video_path)}")
                return cached
            except Exception:
                pass
    
    print(f"       Analyzing video: {os.path.basename(video_path)}")
    
    try:
        duration = get_video_duration(video_path)
        
        # Get video info
        info = get_video_info(video_path)
        width = info.get('width', 1920)
        height = info.get('height', 1080)
        fps = info.get('fps', 30)
        
        # Scene detection (fast, uses ffmpeg)
        scenes = detect_scenes_detailed(video_path)
        
        if fast_mode:
            # FAST MODE: Skip heavy computations
            motion_profile = []
            highlights = []
            camera_changes = []
            quality_profile = []
            dominant_colors = []
        else:
            # Motion profile (sampled) - SLOW
            motion_profile = analyze_motion_profile(video_path)
            
            # Highlights detection
            highlights = detect_highlights(video_path, scenes, motion_profile)
            
            # Camera changes - SLOW
            camera_changes = detect_camera_changes(video_path)
            
            # Visual quality profile - SLOW
            quality_profile = analyze_visual_quality_profile(video_path)
            
            # Dominant colors - SLOW
            dominant_colors = analyze_dominant_colors(video_path)
        
        result = {
            "path": video_path,
            "duration": duration,
            "fps": fps,
            "width": width,
            "height": height,
            "scenes": scenes,
            "motion_profile": motion_profile,
            "highlights": highlights,
            "camera_changes": camera_changes,
            "visual_quality": quality_profile,
            "dominant_colors": dominant_colors,
        }
        
        # Cache result
        if cache_dir:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_path, 'w') as f:
                    json.dump(result, f)
            except Exception:
                pass
        
        print(f"       Video analysis complete: {len(scenes)} scenes, {len(highlights)} highlights, {len(camera_changes)} camera changes")
        return result
        
    except Exception as exc:
        print(f"Video analysis error for {video_path}: {exc}")
        return {
            "path": video_path,
            "duration": 0.0,
            "fps": 30,
            "width": 1920,
            "height": 1080,
            "scenes": [],
            "motion_profile": [],
            "highlights": [],
            "camera_changes": [],
            "visual_quality": [],
            "dominant_colors": [],
        }


def get_video_info(video_path: str) -> dict:
    """Get video metadata using ffprobe."""
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json",
        video_path,
    ]
    
    try:
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        
        if process.returncode != 0:
            return {"width": 1920, "height": 1080, "fps": 30}
        
        data = json.loads(process.stdout)
        if not data.get('streams'):
            return {"width": 1920, "height": 1080, "fps": 30}
        
        stream = data['streams'][0]
        width = stream.get('width', 1920)
        height = stream.get('height', 1080)
        
        # Parse frame rate
        fps_str = stream.get('r_frame_rate', '30/1')
        try:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den > 0 else 30
        except Exception:
            fps = 30
        
        return {"width": width, "height": height, "fps": fps}
    except Exception:
        return {"width": 1920, "height": 1080, "fps": 30}


def detect_scenes_detailed(video_path: str, threshold: float = SCENE_THRESH) -> list:
    """Detect scene changes with detailed info."""
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        "-f", "null", "-",
    ]
    
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    
    scene_times = []
    output = process.stderr + process.stdout
    
    for line in output.split("\n"):
        if "pts_time:" in line:
            try:
                pts_part = line.split("pts_time:")[1].split()[0]
                scene_time = float(pts_part)
                scene_times.append(scene_time)
            except (IndexError, ValueError):
                continue
    
    # Build scene segments
    scenes = []
    all_times = [0.0] + scene_times
    duration = get_video_duration(video_path)
    all_times.append(duration)
    
    for i in range(len(all_times) - 1):
        start = all_times[i]
        end = all_times[i + 1]
        if end - start > 0.5:  # Minimum scene duration
            scenes.append({
                "start": start,
                "end": end,
                "duration": end - start,
                "score": 1.0 if i > 0 and i < len(all_times) - 1 else 0.5,
                "type": "cut"
            })
    
    return scenes


def analyze_motion_profile(video_path: str, sample_fps: int = MOTION_SAMPLE_FPS) -> list:
    """Analyze motion throughout the video at regular intervals."""
    duration = get_video_duration(video_path)
    if duration <= 0:
        return []
    
    frame_interval = 1.0 / sample_fps
    num_samples = int(duration * sample_fps)
    
    motion_profile = []
    prev_frame = None
    
    for i in range(num_samples + 1):
        t = i * frame_interval
        if t > duration:
            break
        
        frame = extract_frame(video_path, t)
        if frame is None:
            motion_profile.append((t, 0.0))
            continue
        
        import cv2
        import numpy as np
        gray = np.array(frame.convert('L'))
        
        if prev_frame is not None:
            mag = compute_optical_flow_magnitude(prev_frame, gray)
            motion_profile.append((t, mag))
        else:
            motion_profile.append((t, 0.0))
        
        prev_frame = gray
    
    # Normalize motion to 0-1
    if motion_profile:
        max_motion = max(m[1] for m in motion_profile)
        if max_motion > 0:
            motion_profile = [(t, m / max_motion) for t, m in motion_profile]
    
    return motion_profile


def detect_highlights(video_path: str, scenes: list, motion_profile: list) -> list:
    """Detect visually interesting moments (highlights)."""
    highlights = []
    duration = get_video_duration(video_path)
    
    # High motion peaks
    if motion_profile:
        motions = [m[1] for m in motion_profile]
        if motions:
            mean_motion = np.mean(motions) if 'np' in globals() else sum(motions) / len(motions)
            std_motion = np.std(motions) if 'np' in globals() else (sum((m - mean_motion)**2 for m in motions) / len(motions))**0.5
            threshold = mean_motion + std_motion
            
            for t, m in motion_profile:
                if m > threshold:
                    highlights.append({
                        "start": max(0, t - 1.0),
                        "end": min(duration, t + 1.0),
                        "score": min(1.0, m / max(motions) if max(motions) > 0 else 0),
                        "type": "motion"
                    })
    
    # Scene boundaries as highlights
    for scene in scenes:
        if scene["duration"] > 1.0:
            highlights.append({
                "start": scene["start"],
                "end": min(duration, scene["start"] + 2.0),
                "score": 0.7,
                "type": "scene_change"
            })
    
    # Merge overlapping highlights
    highlights.sort(key=lambda x: x["start"])
    merged = []
    for h in highlights:
        if not merged or h["start"] > merged[-1]["end"]:
            merged.append(h)
        else:
            merged[-1]["end"] = max(merged[-1]["end"], h["end"])
            merged[-1]["score"] = max(merged[-1]["score"], h["score"])
    
    return merged[:50]  # Cap at 50 highlights


def detect_camera_changes(video_path: str, sample_fps: float = 1.0) -> list:
    """Detect camera angle changes using color histogram differences."""
    duration = get_video_duration(video_path)
    if duration <= 0:
        return []
    
    frame_interval = 1.0 / sample_fps
    num_samples = int(duration * sample_fps)
    
    camera_changes = []
    prev_hist = None
    
    for i in range(num_samples + 1):
        t = i * frame_interval
        if t > duration:
            break
        
        frame = extract_frame(video_path, t)
        if frame is None:
            continue
        
        import cv2
        import numpy as np
        # Resize for speed
        small = cv2.resize(np.array(frame), (64, 64))
        # Compute color histogram
        hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        if prev_hist is not None:
            # Correlation distance
            correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if correlation < 0.7:  # Significant change
                camera_changes.append(t)
        
        prev_hist = hist
    
    return camera_changes


def analyze_visual_quality_profile(video_path: str, sample_fps: float = 0.5) -> list:
    """Analyze visual quality (sharpness, brightness, contrast) over time."""
    duration = get_video_duration(video_path)
    if duration <= 0:
        return []
    
    frame_interval = 1.0 / sample_fps
    num_samples = int(duration * sample_fps)
    
    quality_profile = []
    
    for i in range(num_samples + 1):
        t = i * frame_interval
        if t > duration:
            break
        
        frame = extract_frame(video_path, t)
        if frame is None:
            quality_profile.append((t, 0.5, 0.5, 0.5))
            continue
        
        import cv2
        import numpy as np
        gray = np.array(frame.convert('L'))
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = min(1.0, laplacian.var() / 1000.0)
        
        # Brightness
        brightness = gray.mean() / 255.0
        
        # Contrast (RMS)
        contrast = gray.std() / 128.0
        contrast = min(1.0, contrast)
        
        quality_profile.append((t, sharpness, brightness, contrast))
    
    return quality_profile


def get_visual_quality_at_time(video_path: str, time_s: float) -> tuple:
    """Get visual quality metrics at a specific time."""
    frame = extract_frame(video_path, time_s)
    if frame is None:
        return (0.5, 0.5, 0.5)
    
    import cv2
    import numpy as np
    gray = np.array(frame.convert('L'))
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = min(1.0, laplacian.var() / 1000.0)
    brightness = gray.mean() / 255.0
    contrast = min(1.0, gray.std() / 128.0)
    
    return (sharpness, brightness, contrast)


def analyze_dominant_colors(video_path: str, sample_fps: float = 0.5) -> list:
    """Analyze dominant colors throughout the video at regular intervals."""
    duration = get_video_duration(video_path)
    if duration <= 0:
        return []
    
    frame_interval = 1.0 / sample_fps
    num_samples = int(duration * sample_fps)
    
    color_profile = []
    
    for i in range(num_samples + 1):
        t = i * frame_interval
        if t > duration:
            break
        
        frame = extract_frame(video_path, t)
        if frame is None:
            color_profile.append((t, [128, 128, 128]))
            continue
        
        import cv2
        import numpy as np
        # Resize for speed
        small = cv2.resize(np.array(frame), (64, 64))
        # Convert to LAB for better color clustering
        lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
        # Reshape for k-means
        pixels = lab.reshape((-1, 3))
        # Simple dominant color: mean of pixels
        dominant = pixels.mean(axis=0)
        # Convert back to RGB approximation
        dominant_rgb = cv2.cvtColor(
            np.uint8([[dominant]]), cv2.COLOR_LAB2RGB
        )[0][0]
        color_profile.append((float(t), [float(dominant_rgb[0]), float(dominant_rgb[1]), float(dominant_rgb[2])]))
    
    return color_profile
