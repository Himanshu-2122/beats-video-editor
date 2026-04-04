# import librosa

# def get_beats(audio_path):
#     try:
#         # 🔥 mono=True → less RAM
#         y, sr = librosa.load(audio_path, sr=None, mono=True)

#         # 🔥 better beat tracking
#         tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

#         times = librosa.frames_to_time(beats, sr=sr)

#         # 🔥 ensure list format
#         times = list(times)

#         # 🔥 edge case: empty beats
#         if not times:
#             return [0.0]

#         # 🔥 ensure first beat = 0
#         if times[0] > 0:
#             times.insert(0, 0.0)

#         return times

#     except Exception as e:
#         print(f"Beat detection error: {e}")
#         return [0.0]


# # import librosa
q
# # def get_beats(audio_path):
# #     try:
# #         y, sr = librosa.load(audio_path, sr=None, mono=True)

# #         tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

# #         times = librosa.frames_to_time(beats, sr=sr)
# #         times = list(times)

# #         if not times:
# #             return [0.0]

# #         if times[0] > 0:
# #             times.insert(0, 0.0)

# #         return times

# #     except Exception as e:
# #         print(f"Beat detection error: {e}")
# #         return [0.0]
import librosa


def get_beats(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)

        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

        times = librosa.frames_to_time(beats, sr=sr)
        times = list(times)

        if not times:
            return [0.0]

        if times[0] > 0:
            times.insert(0, 0.0)

        return times

    except Exception as e:
        print(f"Beat detection error: {e}")
        return [0.0]
