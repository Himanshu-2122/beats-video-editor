# import random
# from moviepy.editor import vfx

# TARGET_HEIGHT = 360
# FPS = 24

# def sync_clips_with_beats(clips, beat_times):
#     final_clips = []
#     i = 0

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

#         clip = random.choice(clips)

#         try:
#             # 🔥 process ONLY required portion
#             max_start = max(0, clip.duration - duration)
#             start = random.uniform(0, max_start)

#             trimmed = (
#                 clip
#                 .subclip(start, start + duration)
#                 .resize(height=TARGET_HEIGHT)
#                 .without_audio()
#                 .set_fps(FPS)
#             )

#             # ================= EFFECTS ================= #

#             if random.random() > 0.5:
#                 trimmed = trimmed.fx(vfx.mirror_x)

#             if random.random() > 0.7:
#                 trimmed = trimmed.resize(lambda t: 1 + 0.02 * t)

#             fade_dur = min(0.2, duration / 4)
#             trimmed = trimmed.fadein(fade_dur).fadeout(fade_dur)

#             # ========================================== #

#             final_clips.append(trimmed)

#         except Exception as e:
#             print(f"Clip processing error: {e}")

#         i += group_size

#     return final_clips

# # import random
# # from moviepy.editor import vfx
# # from app.theme import apply_instagram_theme

# # TARGET_HEIGHT = 360
# # FPS = 24

# # def sync_clips_with_beats(clips, beat_times):
# #     final_clips = []
# #     i = 0

# #     while i < len(beat_times) - 1:

# #         group_size = random.randint(3, 6)

# #         if i + group_size >= len(beat_times):
# #             group_size = len(beat_times) - i - 1
# #             if group_size <= 0:
# #                 break

# #         duration = beat_times[i + group_size] - beat_times[i]

# #         if duration <= 0:
# #             i += 1
# #             continue

# #         clip = random.choice(clips)

# #         try:
# #             max_start = max(0, clip.duration - duration)
# #             start = random.uniform(0, max_start)

# #             trimmed = (
# #                 clip
# #                 .subclip(start, start + duration)
# #                 .resize(height=TARGET_HEIGHT)
# #                 .without_audio()
# #                 .set_fps(FPS)
# #             )

# #             # 🔥 mirror effect
# #             if random.random() > 0.5:
# #                 trimmed = trimmed.fx(vfx.mirror_x)

# #             # 🔥 Beat intensity logic
# #             if duration < 0.5:
# #                 intensity = "high"
# #             elif duration < 1.5:
# #                 intensity = "medium"
# #             else:
# #                 intensity = "low"

# #             # 🔥 Apply Instagram theme
# #             trimmed = apply_instagram_theme(trimmed, intensity)

# #             final_clips.append(trimmed)

# #         except Exception as e:
# #             print(f"Clip processing error: {e}")

# #         i += group_size

# #     return final_clips
#
#
import random

from moviepy.editor import VideoFileClip, vfx

TARGET_HEIGHT = 360
FPS = 24


def process_clip(video_path, duration):
    try:
        clip = VideoFileClip(video_path)

        max_start = max(0, clip.duration - duration)
        start = random.uniform(0, max_start)

        sub = (
            clip.subclip(start, start + duration)
            .resize(height=TARGET_HEIGHT)
            .without_audio()
            .set_fps(FPS)
        )

        # Effects
        if random.random() > 0.5:
            sub = sub.fx(vfx.mirror_x)

        if random.random() > 0.7:
            sub = sub.resize(lambda t: 1 + 0.02 * t)

        fade_dur = min(0.2, duration / 4)
        sub = sub.fadein(fade_dur).fadeout(fade_dur)

        # ❌ DON'T CLOSE HERE

        return sub

    except Exception as e:
        print(f"Clip error ({video_path}): {e}")
        return None


def sync_clips_with_beats(video_paths, beat_times):
    final_clips = []
    i = 0

    while i < len(beat_times) - 1:
        group_size = random.randint(4, 8)

        if i + group_size >= len(beat_times):
            group_size = len(beat_times) - i - 1
            if group_size <= 0:
                break

        duration = beat_times[i + group_size] - beat_times[i]

        if duration <= 0:
            i += 1
            continue

        video_path = random.choice(video_paths)

        clip = process_clip(video_path, duration)

        if clip:
            final_clips.append(clip)

        i += group_size

    return final_clips
