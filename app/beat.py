# # import librosa

# # def get_beats(audio_path):
# #     try:
# #         # 🔥 mono=True → less RAM
# #         y, sr = librosa.load(audio_path, sr=None, mono=True)

# #         # 🔥 better beat tracking
# #         tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

# #         times = librosa.frames_to_time(beats, sr=sr)

# #         # 🔥 ensure list format
# #         times = list(times)

# #         # 🔥 edge case: empty beats
# #         if not times:
# #             return [0.0]

# #         # 🔥 ensure first beat = 0
# #         if times[0] > 0:
# #             times.insert(0, 0.0)

# #         return times

# #     except Exception as e:
# #         print(f"Beat detection error: {e}")
# #         return [0.0]


# # # import librosa
# # # def get_beats(audio_path):
# # #     try:
# # #         y, sr = librosa.load(audio_path, sr=None, mono=True)

# # #         tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

# # #         times = librosa.frames_to_time(beats, sr=sr)
# # #         times = list(times)

# # #         if not times:
# # #             return [0.0]

# # #         if times[0] > 0:
# # #             times.insert(0, 0.0)

# # #         return times

# # #     except Exception as e:
# # #         print(f"Beat detection error: {e}")
# # #         return [0.0]
# import librosa


# def get_beats(audio_path):
#     try:
#         y, sr = librosa.load(audio_path, sr=None, mono=True)

#         tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

#         times = librosa.frames_to_time(beats, sr=sr)
#         times = list(times)

#         if not times:
#             return [0.0]

#         if times[0] > 0:
#             times.insert(0, 0.0)

#         return times

#     except Exception as e:
#         print(f"Beat detection error: {e}")
#         return [0.0]
import librosa


def get_beats(audio_path):
    """Detect musical beats from an audio file.

    Returns:
        list[float]: beat timestamps in seconds
    """
    try:
        print("Analyzing music...")

        # Mono uses considerably less memory.
        y, sr = librosa.load(audio_path, sr=None, mono=True)

        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        beat_times = list(beat_times)

        if not beat_times:
            print("No beats detected.")
            return [0.0]

        # Make sure video starts at 0.
        if beat_times[0] > 0:
            beat_times.insert(0, 0.0)

        print(f"Detected tempo: {tempo}")
        return beat_times

    except Exception as exc:
        print(f"Beat detection error: {exc}")
        return [0.0]
