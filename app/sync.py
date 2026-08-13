import os
import random
import tempfile
import json

from app.video import (
    get_video_duration,
    natural_sort_key,
    process_clip,
)


def create_beat_groups(
    beat_times,
    min_beats,
    max_beats,
):
    """
    Convert beat timestamps into clip durations.

    Each group corresponds to a contiguous range of beats.
    The clip boundaries align exactly with detected beat timestamps.
    """

    groups = []
    i = 0
    total = len(beat_times)

    while i < total - 1:
        group_size = random.randint(min_beats, max_beats)
        remaining = total - i - 1
        group_size = min(group_size, remaining)

        if group_size <= 0:
            break

        start = beat_times[i]
        end = beat_times[i + group_size]
        duration = end - start

        if duration > 0:
            groups.append({
                "start": start,
                "end": end,
                "duration": duration,
            })

        i += group_size

    return groups


def sync_clips_with_beats(
    video_paths,
    beat_times,
    output_dir=None,
    min_beats=4,
    max_beats=8,
    use_proxies=False,
    progress_callback=None,
    proxy_progress_callback=None,
):
    """
    Generate beat-synchronized clips from source videos.

    Returns:
        tuple[list[str], list[dict], str]: generated clip paths, beat groups used, and temporary output directory.
    """

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="beats_video_")
    else:
        os.makedirs(output_dir, exist_ok=True)

    clip_dir = os.path.join(output_dir, "_final_clips")
    os.makedirs(clip_dir, exist_ok=True)

    if not video_paths:
        return [], [], output_dir

    if len(beat_times) < 2:
        return [], [], output_dir

    beat_groups = create_beat_groups(beat_times, min_beats, max_beats)
    if not beat_groups:
        return [], [], output_dir

    number_of_clips = min(len(video_paths), len(beat_groups))
    selected_videos = random.sample(video_paths, number_of_clips)

    print("\nRandomly selected:")
    for path in selected_videos:
        print(f"  {os.path.basename(path)}")

    selected_videos.sort(key=natural_sort_key)
    print("\nFinal ascending order:")
    for index, path in enumerate(selected_videos, start=1):
        print(f"  {index:03d} -> {os.path.basename(path)}")

    final_clip_paths = []
    used_beat_groups = []
    total = len(selected_videos)

    for index, source_path in enumerate(selected_videos):
        beat_group = beat_groups[index]
        duration = beat_group["duration"]

        source_duration = get_video_duration(source_path)
        if source_duration <= 0:
            print(f"Skipping invalid video: {source_path}")
            continue

        if source_duration > duration:
            max_start = source_duration - duration
            start_time = random.uniform(0, max_start)
        else:
            start_time = 0.0
            duration = source_duration

        if duration <= 0:
            continue

        clip_path = os.path.join(clip_dir, f"clip_{index + 1:04d}.mp4")

        print(f"\nFinal clip {index + 1}/{total}")
        print(f"Source: {os.path.basename(source_path)}")
        print(f"Music start: {beat_group['start']:.3f}s")
        print(f"Music end:   {beat_group['end']:.3f}s")
        print(f"Duration:    {duration:.3f}s")
        print(f"Source start: {start_time:.3f}s")

        process_clip(
            video_path=source_path,
            duration=duration,
            output_path=clip_path,
            start_time=start_time,
        )

        final_clip_paths.append(clip_path)
        used_beat_groups.append({
            "clip_path": clip_path,
            "source_path": source_path,
            "music_start": beat_group["start"],
            "music_end": beat_group["end"],
            "clip_duration": duration,
            "source_start": start_time,
        })

        if progress_callback:
            progress_callback(index + 1, total)

    debug_path = os.path.join(output_dir, "sync_debug.json")
    try:
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump({
                "beat_groups": used_beat_groups,
                "total_music_duration": beat_times[-1] if beat_times else 0,
            }, f, indent=2)
        print(f"\nSync debug written to: {debug_path}")
    except Exception as exc:
        print(f"Warning: Could not write sync debug: {exc}")

    return final_clip_paths, used_beat_groups, output_dir
