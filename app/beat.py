import librosa
import numpy as np


def get_beats(audio_path):
    """
    Detect musical beats from an audio file.

    Returns:
        list[float]: beat timestamps in seconds, monotonically increasing,
                     duplicates removed, no artificial 0.0 unless music starts at 0.
    """

    try:
        print("       Analyzing music...")

        y, sr = librosa.load(
            audio_path,
            sr=None,
            mono=True,
        )

        tempo, beat_frames = librosa.beat.beat_track(
            y=y,
            sr=sr,
        )

        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        beat_times = np.asarray(beat_times, dtype=float)

        if beat_times.size == 0:
            print("       No beats detected.")
            return []

        beat_times = np.unique(beat_times)
        beat_times = beat_times[beat_times > 0.01]

        if beat_times.size == 0:
            print("       No valid beats after filtering.")
            return []

        if not np.all(np.diff(beat_times) > 0):
            beat_times = np.sort(beat_times)
            beat_times = np.unique(beat_times)

        min_gap = 0.05
        keep = [True]
        for i in range(1, len(beat_times)):
            if beat_times[i] - beat_times[i - 1] >= min_gap:
                keep.append(True)
            else:
                keep.append(False)
        beat_times = beat_times[keep]

        print(f"       Detected tempo: {tempo:.1f} BPM")
        print(f"       Valid beats: {len(beat_times)}")

        return beat_times.tolist()

    except Exception as exc:
        print(f"Beat detection error: {exc}")
        return []