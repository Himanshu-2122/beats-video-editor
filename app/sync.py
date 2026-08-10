# import random
# import subprocess
# import os
# import tempfile

# from moviepy.editor import VideoFileClip, vfx

# TARGET_HEIGHT = 720
# FPS = 30


# def process_clip(video_path, duration, output_path):
#     """Process clip with moviepy - trim, resize, set fps for consistent concatenation"""
#     clip = None
#     try:
#         clip = VideoFileClip(video_path)
#         max_start = max(0, clip.duration - duration)
#         start = random.uniform(0, max_start)
        
#         # Trim clip
#         trimmed = clip.subclip(start, start + duration)
        
#         # Resize to target height, maintain aspect ratio
#         if trimmed.h != TARGET_HEIGHT:
#             trimmed = trimmed.resize(height=TARGET_HEIGHT)
        
#         # Set consistent FPS
#         if trimmed.fps != FPS:
#             trimmed = trimmed.set_fps(FPS)
        
#         # Remove audio (we'll add music later)
#         trimmed = trimmed.without_audio()
        
#         # Write with fast preset
#         trimmed.write_videofile(
#             output_path,
#             fps=FPS,
#             codec="libx264",
#             preset="fast",
#             threads=2,
#             logger=None
#         )
        
#         return output_path
        
#     except Exception as e:
#         print(f"Clip error ({video_path}): {e}")
#         return None
#     finally:
#         if clip:
#             clip.close()


# def sync_clips_with_beats(video_paths, beat_times):
#     temp_dir = tempfile.mkdtemp(prefix="beats_editor_")
#     temp_files = []
#     i = 0
#     clip_num = 0

#     while i < len(beat_times) - 1:
#         group_size = random.randint(4, 8)

#         if i + group_size >= len(beat_times):
#             group_size = len(beat_times) - i - 1
#             if group_size <= 0:
#                 break

#         duration = beat_times[i + group_size] - beat_times[i]

#         if duration <= 0:
#             i += 1
#             continue

#         video_path = random.choice(video_paths)
#         clip_num += 1
#         output_path = os.path.join(temp_dir, f"clip_{clip_num:03d}.mp4")

#         result = process_clip(video_path, duration, output_path)

#         if result:
#             temp_files.append(result)
#             print(f"Processed clip {clip_num} (duration: {duration:.2f}s)")

#         i += group_size

#     return temp_files, temp_dir

import os
import random
import tempfile

from app.video import process_clip


def sync_clips_with_beats(video_paths, beat_times):
    """
    Create video clips whose durations are synchronized
    with groups of detected beats.

    Only one source video is processed at a time, so RAM
    usage stays low.
    """

    temp_dir = tempfile.mkdtemp(
        prefix="beats_editor_"
    )

    temp_files = []

    i = 0
    clip_num = 0

    total_beats = len(beat_times)

    while i < total_beats - 1:

        # Randomly group 4-8 beats together
        group_size = random.randint(4, 8)

        remaining_beats = (
            total_beats - i - 1
        )

        group_size = min(
            group_size,
            remaining_beats,
        )

        if group_size <= 0:
            break

        start_time = beat_times[i]

        end_time = beat_times[
            i + group_size
        ]

        duration = end_time - start_time

        if duration <= 0:
            i += group_size
            continue

        video_path = random.choice(
            video_paths
        )

        clip_num += 1

        output_path = os.path.join(
            temp_dir,
            f"clip_{clip_num:04d}.mp4",
        )

        try:

            result = process_clip(
                video_path=video_path,
                duration=duration,
                output_path=output_path,
            )

            if result:
                temp_files.append(result)

                print(
                    f"Processed clip {clip_num} "
                    f"(duration: {duration:.2f}s)"
                )

        except Exception as exc:
            print(f"Failed clip {clip_num}: {exc}")
            # Skip this group of beats and continue
            i += group_size
            continue

        # Move to the next group of beats
        i += group_size

    # Finished processing all beat groups
    return temp_files, temp_dir