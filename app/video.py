# from moviepy.editor import VideoFileClip

# TARGET_HEIGHT = 360
# FPS = 24

# def load_clips(video_paths):
#     clips = []

#     for p in video_paths:
#         try:
#             clip = (
#                 VideoFileClip(p)
#                 .resize(height=TARGET_HEIGHT)   # 🔥 reduce RAM
#                 .without_audio()                # 🔥 remove audio
#                 .set_fps(FPS)
#             )
#             clips.append(clip)

#         except Exception as e:
#             print(f"Error loading {p}: {e}")

#     return clips
# # from moviepy.editor import VideoFileClip

# # def load_clips(video_paths):
# #     clips = []

# #     for p in video_paths:
# #         try:
# #             clip = VideoFileClip(p)
# #             clips.append(clip)
# #         except Exception as e:
# #             print(f"Error loading {p}: {e}")

# #     return clips
import os


def load_video_paths(video_folder):
    video_files = [
        os.path.join(video_folder, f)
        for f in os.listdir(video_folder)
        if f.endswith((".mp4", ".webm"))
    ]

    return video_files
