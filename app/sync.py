import os
import random
import tempfile

from app.video import process_clip


def sync_clips_with_beats(
    video_paths,
    beat_times,
    min_beats=4,
    max_beats=8,
    progress_callback=None,
):
    """
    Randomly select source videos, then order the selected videos
    in ascending filename order before generating clips.

    Each selected source video is used only once. Beat timings
    determine the duration of each generated clip.
    """

    if min_beats < 1:
        min_beats = 1

    if max_beats < min_beats:
        max_beats = min_beats

    temp_dir = tempfile.mkdtemp(prefix="beats_editor_")
    temp_files = []

    if not video_paths:
        print("No video paths provided.")
        return temp_files, temp_dir

    if len(beat_times) < 2:
        print("Not enough beats.")
        return temp_files, temp_dir

    max_possible_clips = len(video_paths)
    beat_groups = []

    i = 0
    total_beats = len(beat_times)

    while i < total_beats - 1 and len(beat_groups) < max_possible_clips:
        group_size = random.randint(min_beats, max_beats)
        remaining_beats = total_beats - i - 1
        group_size = min(group_size, remaining_beats)

        if group_size <= 0:
            break

        start_time = beat_times[i]
        end_time = beat_times[i + group_size]
        duration = end_time - start_time

        if duration <= 0:
            i += group_size
            continue

        beat_groups.append({
            "start": start_time,
            "end": end_time,
            "duration": duration,
        })

        i += group_size

    number_of_clips = min(len(beat_groups), max_possible_clips)

    if number_of_clips == 0:
        print("No valid clip groups found.")
        return temp_files, temp_dir

    selected_videos = random.sample(video_paths, number_of_clips)

    print(f"\nRandomly selected {len(selected_videos)} videos.")
    print("\nRandom selection:")
    for path in selected_videos:
        print(f"  {os.path.basename(path)}")

    selected_videos.sort(key=lambda path: os.path.basename(path).lower())

    print("\nFinal ascending order:")
    for index, path in enumerate(selected_videos, start=1):
        print(f"  {index:03d}: {os.path.basename(path)}")

    for index in range(number_of_clips):
        video_path = selected_videos[index]
        beat_group = beat_groups[index]
        duration = beat_group["duration"]
        clip_number = index + 1
        output_path = os.path.join(temp_dir, f"clip_{clip_number:04d}.mp4")

        print(f"\nProcessing clip {clip_number}/{number_of_clips}")
        print(f"Source: {os.path.basename(video_path)}")
        print(f"Duration: {duration:.2f}s")

        try:
            result = process_clip(
                video_path=video_path,
                duration=duration,
                output_path=output_path,
            )

            if result:
                temp_files.append(result)
                print(f"Created: clip_{clip_number:04d}.mp4")
            else:
                print(f"Failed to create clip {clip_number}")

        except Exception as exc:
            print(f"Clip {clip_number} failed:")
            print(exc)

        finally:
            if progress_callback:
                progress_callback(clip_number, number_of_clips)

    print(f"\nCreated {len(temp_files)} temporary clips.")
    return temp_files, temp_dir
