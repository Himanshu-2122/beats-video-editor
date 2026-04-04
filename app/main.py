# from moviepy.editor import AudioFileClip, concatenate_videoclips
# from app.beat import get_beats
# from app.video import load_clips
# from app.sync import sync_clips_with_beats
# import os

# music_path = "data/music/song.mp3"
# video_folder = "data/videos"

# video_files = [
#     os.path.join(video_folder, f)
#     for f in os.listdir(video_folder)
#     if f.endswith((".mp4", ".webm"))
# ]

# print("Loading beats...")
# beats = get_beats(music_path)

# print("Loading clips...")
# clips = load_clips(video_files)

# print("Syncing clips...")
# final_clips = sync_clips_with_beats(clips, beats)

# print("Rendering video...")

# final = concatenate_videoclips(
#     final_clips,
#     method="compose",
#     padding=-0.1
# )

# audio = AudioFileClip(music_path)
# final = final.set_audio(audio)

# output_path = "output/final.mp4"

# final.write_videofile(
#     output_path,
#     fps=30,
#     codec="libx264",
#     audio_codec="aac",
#     bitrate="5000k",
#     preset="ultrafast",   # 🔥 RAM SAFE (IMPORTANT CHANGE)
#     threads=2             # 🔥 reduce threads
# )

# # 🔥 VERY IMPORTANT (FREE RAM)
# for c in clips:
#     c.close()

# for c in final_clips:
#     c.close()

# audio.close()
# final.close()

# print(f"Done! Video saved at {output_path}")


# # from moviepy.editor import AudioFileClip, concatenate_videoclips
# # from app.beat import get_beats
# # from app.video import load_clips
# # from app.sync import sync_clips_with_beats
# # import os

# # music_path = "data/music/song.mp3"
# # video_folder = "data/videos"

# # video_files = [
# #     os.path.join(video_folder, f)
# #     for f in os.listdir(video_folder)
# #     if f.endswith((".mp4", ".webm"))
# # ]

# # print("Loading beats...")
# # beats = get_beats(music_path)

# # print("Loading clips...")
# # clips = load_clips(video_files)

# # print("Syncing clips...")
# # final_clips = sync_clips_with_beats(clips, beats)

# # print("Rendering video...")

# # final = concatenate_videoclips(
# #     final_clips,
# #     method="compose",
# #     padding=-0.1
# # )

# # audio = AudioFileClip(music_path)
# # final = final.set_audio(audio)

# # output_path = "output/final.mp4"

# # final.write_videofile(
# #     output_path,
# #     fps=30,
# #     codec="libx264",
# #     audio_codec="aac",
# #     bitrate="5000k",
# #     preset="ultrafast",   # 🔥 RAM safe
# #     threads=2
# # )

# # # 🔥 CLEAN MEMORY (VERY IMPORTANT)
# # for c in clips:
# #     c.close()

# # for c in final_clips:
# #     c.close()

# # audio.close()
# # final.close()

# # print(f"Done! Video saved at {output_path}")
#
import os

import moviepy.config as cfg
from moviepy.editor import AudioFileClip, concatenate_videoclips

from app.beat import get_beats
from app.sync import sync_clips_with_beats
from app.video import load_video_paths

cfg.change_settings({"FFMPEG_BINARY": "ffmpeg"})

music_path = "data/music/song.mp3"
video_folder = "data/videos"

print("Loading video paths...")
video_paths = load_video_paths(video_folder)

print("Loading beats...")
beats = get_beats(music_path)

print("Syncing clips...")
final_clips = sync_clips_with_beats(video_paths, beats)

print("Rendering video...")

final = concatenate_videoclips(final_clips, method="chain")

audio = AudioFileClip(music_path)
final = final.set_audio(audio)

output_path = "output/final.mp4"

final.write_videofile(
    output_path,
    fps=30,
    codec="libx264",
    audio_codec="aac",
    bitrate="3000k",  # 🔥 reduce memory
    preset="ultrafast",
    threads=2,
)

# Cleanup
for c in final_clips:
    c.close()

audio.close()
final.close()

print(f"Done! Video saved at {output_path}")
