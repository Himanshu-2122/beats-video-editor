"""
Music beat detection and analysis module.

Responsibility:
- Detect musical beats from audio files using librosa
- Provide comprehensive music analysis including tempo, beats, energy curve,
  drops, buildups, and section labels
- Support caching of analysis results for performance

Key Functions:
- get_beats(): Detect beats from audio file (simple beat tracking)
- analyze_music_full(): Comprehensive analysis with energy, drops, buildups, sections
- get_beat_times(): Legacy adapter returning just timestamps

Dependencies:
- librosa: Beat tracking, onset strength, RMS energy, chroma features
- numpy: Numerical operations
- app.cache: Caching infrastructure
"""

import librosa
import numpy as np
import json
import os
from app.cache import cache_manager, cached


def get_beats(audio_path):
    """
    Detect musical beats from an audio file.

    Uses librosa onset strength envelope and beat tracking to find musical beats.
    Results are cached per audio file path for performance.

    Returns:
        list[dict]: list of dicts with 'time' (float, seconds from start)
                    and 'strength' (float 0-1, normalized beat strength).
                    Returns empty list if no beats detected.

    Raises:
        None: Errors are caught and an empty list is returned.

    Side Effects:
        Prints detection progress to stdout (tempo, beat count, average strength).

    Example:
        >>> beats = get_beats("song.mp3")
        >>> len(beats)
        42
        >>> beats[0]
        {"time": 0.5, "strength": 0.87}
    """
    return _get_beats_cached(audio_path)


@cached("beats", "_beats")
def _get_beats_cached(audio_path):
    try:
        print("       Analyzing music...")

        y, sr = librosa.load(
            audio_path,
            sr=None,
            mono=False,
        )
        # beat_track(y=...) can return no beats on some librosa versions for clean mono
        # inputs. Use a robust path: compute onset envelope and track beats from it.
        mono = y.mean(axis=0) if y.ndim > 1 else y
        hop_length = 512
        onset_env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop_length)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)

        try:
            tempo_value = float(np.squeeze(tempo))
        except Exception:
            tempo_value = tempo

        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
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

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        hop_length = 512
        beat_strengths = []
        for bt in beat_times:
            frame_idx = int(bt * sr / hop_length)
            if frame_idx < len(onset_env):
                beat_strengths.append(float(onset_env[frame_idx]))
            else:
                beat_strengths.append(0.0)

        if beat_strengths:
            max_str = max(beat_strengths)
            if max_str > 0:
                beat_strengths = [s / max_str for s in beat_strengths]

        beats = [{"time": float(t), "strength": float(s)} for t, s in zip(beat_times, beat_strengths)]

        try:
            print(f"       Detected tempo: {tempo_value:.1f} BPM")
        except Exception:
            print(f"       Detected tempo: {tempo_value}")
        print(f"       Valid beats: {len(beats)}")
        if beats:
            avg_str = sum(b['strength'] for b in beats) / len(beats)
            print(f"       Avg beat strength: {avg_str:.3f}")

        return beats

    except Exception as exc:
        print(f"Beat detection error: {exc}")
        return []


def analyze_music_full(audio_path, cache_dir=None):
    """
    Comprehensive music analysis: beats, energy, drops, buildups, sections.
    
    Performs full audio analysis using librosa to extract:
    - Beat tracking (tempo and beat timestamps)
    - Normalized beat strength from onset envelope
    - RMS energy curve in sliding windows (normalized 0-1)
    - Beat energy at each beat position
    - Drop detection: sudden energy increases + spectral flux peaks
    - Buildup detection: sustained energy increases over 4+ beats
    - Section labeling using chroma features and energy profiling
    
    Uses CacheManager for consistent caching of results per audio file.
    The cache_dir parameter is accepted for interface compatibility but
    the global CacheManager instance is used.
    
    Args:
        audio_path (str): Path to audio file (MP3, WAV, M4A, AAC supported)
        cache_dir (str, optional): Deprecated; use global CacheManager instead.
            Not actually used for cache selection in current implementation.
    
    Returns:
        dict: Complete music analysis with the following keys:
            - bpm (float): Detected tempo in beats per minute
            - beats (list[dict]): List of beat dicts with 'time' and 'strength'
              and optional 'energy' key. Beats are timestamps in seconds from
              audio start, strength is normalized 0-1.
            - drops (list[dict]): Detected drop timestamps with 'time' (float)
              and 'intensity' (float 0-1). A drop is a sudden energy increase
              accompanied by a spectral flux peak.
            - buildups (list[dict]): Detected buildup segments with 'start',
              'end' (floats in seconds) and 'intensity' (float 0-1). A buildup
              is a sustained energy rise over at least 4 beats.
            - sections (list[dict]): Detected sections with 'label', 'start',
              'end' (floats) and 'energy' (float 0-1). Labels are from
              ['intro', 'verse', 'chorus', 'drop', 'bridge', 'outro'] and
              are assigned based on mean energy in the segment.
            - energy_curve (list[tuple]): List of (time, energy) tuples
              representing normalized RMS energy at regular intervals across
              the full duration, energy normalized to 0-1 range.
            - duration (float): Audio duration in seconds.
    
    Raises:
        None: All exceptions are caught internally. On error, a minimal fallback
            result is returned with bpm=120.0, empty beats list, and basic
            structure.
    
    Side Effects:
        Prints analysis progress to stdout including tempo, beat count,
        drop/buildup/section counts, and analysis completion status.
        Uses CacheManager which may read/write JSON cache files.
    
    Example:
        >>> result = analyze_music_full("song.mp3")
        >>> result["bpm"]
        124.5
        >>> len(result["beats"])
        42
        >>> len(result["drops"])
        3
        >>> len(result["energy_curve"])
        100
        >>> result["duration"]
        185.5
    """
    from app.cache import cache_manager
    
    # Use CacheManager for consistent caching
    cached = cache_manager.load("music_analysis", audio_path)
    if cached is not None:
        print(f"       Loaded music analysis from cache")
        return cached

    try:
        print("       Performing full music analysis...")
        
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = float(len(y) / sr)
        
        # Beat tracking
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        try:
            tempo_value = float(np.squeeze(tempo))
        except Exception:
            tempo_value = tempo
            
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        beat_times = np.asarray(beat_times, dtype=float)
        beat_times = np.unique(beat_times)
        beat_times = beat_times[beat_times > 0.01]
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
        
        # Onset strength for beat strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        hop_length = 512
        beat_strengths = []
        for bt in beat_times:
            frame_idx = int(bt * sr / hop_length)
            if frame_idx < len(onset_env):
                beat_strengths.append(float(onset_env[frame_idx]))
            else:
                beat_strengths.append(0.0)
        
        if beat_strengths:
            max_str = max(beat_strengths)
            if max_str > 0:
                beat_strengths = [s / max_str for s in beat_strengths]
        
        beats = [{"time": float(t), "strength": float(s)} for t, s in zip(beat_times, beat_strengths)]
        
        # Energy curve (RMS in sliding windows)
        frame_length = 2048
        hop_length_energy = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length_energy)[0]
        energy_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length_energy)
        
        # Normalize energy to 0-1
        if rms.max() > 0:
            energy_normalized = rms / rms.max()
        else:
            energy_normalized = rms
            
        energy_curve = [(float(t), float(e)) for t, e in zip(energy_times, energy_normalized)]
        
        # Beat energy (energy at each beat)
        beat_energies = []
        for bt in beat_times:
            idx = np.argmin(np.abs(energy_times - bt))
            beat_energies.append(float(energy_normalized[idx]))
        
        # Add energy to beats
        for i, b in enumerate(beats):
            b["energy"] = beat_energies[i] if i < len(beat_energies) else 0.5
        
        # Detect drops: sudden energy increase + spectral flux peak
        spectral_flux = np.diff(rms, prepend=rms[0])
        flux_normalized = spectral_flux / (rms.max() + 1e-6)
        
        drops = []
        for i in range(1, len(energy_times) - 1):
            if (energy_normalized[i] > 0.6 and 
                flux_normalized[i] > 0.3 and
                energy_normalized[i] > energy_normalized[i-1] * 1.5):
                drops.append({
                    "time": float(energy_times[i]),
                    "intensity": float(min(1.0, energy_normalized[i] * flux_normalized[i] * 2))
                })
        
        # Merge nearby drops
        merged_drops = []
        for d in drops:
            if not merged_drops or d["time"] - merged_drops[-1]["time"] > 1.0:
                merged_drops.append(d)
            elif d["intensity"] > merged_drops[-1]["intensity"]:
                merged_drops[-1] = d
        drops = merged_drops
        
        # Detect buildups: sustained energy increase over multiple beats
        buildups = []
        i = 0
        while i < len(beats) - 3:
            if (beats[i+3]["energy"] - beats[i]["energy"]) > 0.3:
                start_time = beats[i]["time"]
                # Find end of buildup
                j = i + 3
                while j < len(beats) - 1 and beats[j+1]["energy"] >= beats[j]["energy"]:
                    j += 1
                end_time = beats[j]["time"]
                intensity = (beats[j]["energy"] - beats[i]["energy"])
                buildups.append({
                    "start": float(start_time),
                    "end": float(end_time),
                    "intensity": float(min(1.0, intensity * 2))
                })
                i = j
            else:
                i += 1
        
        # Detect sections using self-similarity (simplified)
        # Use chroma features for harmonic analysis
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
        # Simple section detection: find boundaries where chroma changes significantly
        chroma_diff = np.mean(np.abs(np.diff(chroma, axis=1)), axis=0)
        chroma_diff = np.concatenate([[0], chroma_diff])
        section_boundaries = np.where(chroma_diff > np.percentile(chroma_diff, 85))[0]
        section_times = librosa.frames_to_time(section_boundaries, sr=sr, hop_length=512)
        
        sections = []
        section_labels = ["intro", "verse", "chorus", "drop", "bridge", "outro"]
        for i, st in enumerate(section_times):
            if i + 1 < len(section_times):
                end_t = section_times[i + 1]
            else:
                end_t = duration
            # Assign label based on energy
            seg_energy = np.mean(energy_normalized[
                (energy_times >= st) & (energy_times < end_t)
            ]) if len(energy_times[(energy_times >= st) & (energy_times < end_t)]) > 0 else 0.5
            
            label_idx = min(int(seg_energy * len(section_labels)), len(section_labels) - 1)
            sections.append({
                "label": section_labels[label_idx],
                "start": float(st),
                "end": float(end_t),
                "energy": float(seg_energy)
            })
        
        # If no sections found, create basic structure
        if not sections:
            third = duration / 3
            sections = [
                {"label": "intro", "start": 0.0, "end": third, "energy": 0.3},
                {"label": "verse", "start": third, "end": 2*third, "energy": 0.6},
                {"label": "chorus", "start": 2*third, "end": duration, "energy": 0.8}
            ]
        
        result = {
            "bpm": float(tempo_value),
            "beats": beats,
            "drops": drops,
            "buildups": buildups,
            "sections": sections,
            "energy_curve": energy_curve,
            "duration": float(duration)
        }
        
        # Cache result using CacheManager
        cache_manager.save("music_analysis", audio_path, result)
        
        print(f"       Music analysis complete: {len(beats)} beats, {len(drops)} drops, {len(buildups)} buildups, {len(sections)} sections")
        return result
        
    except Exception as exc:
        print(f"Music analysis error: {exc}")
        # Return basic beats as fallback
        basic_beats = get_beats(audio_path)
        return {
            "bpm": 120.0,
            "beats": basic_beats,
            "drops": [],
            "buildups": [],
            "sections": [{"label": "full", "start": 0.0, "end": 0.0, "energy": 0.5}],
            "energy_curve": [],
            "duration": 0.0
        }


def _get_file_hash(filepath):
    """Generate hash for caching.
    
    Creates a hash based on file path, size, and modification time.
    The hash is used as a cache key that invalidates when the file
    changes at the same path (size or mtime change).
    
    Args:
        filepath (str): Path to the file to hash
    
    Returns:
        str: 16-character MD5 hash hex digest
    
    Side Effects:
        Uses os.stat() to read file metadata. No file contents are read.
    """
    import hashlib
    stat = os.stat(filepath)
    content = f"{filepath}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


# Backward compatibility
def get_beat_times(audio_path):
    """Return just beat timestamps as list[float] for legacy code.
    
    Convenience adapter that calls get_beats() and extracts just the timestamps.
    
    Args:
        audio_path (str): Path to audio file
    
    Returns:
        list[float]: List of beat timestamps in seconds from audio start.
                     Returns empty list if no beats detected.
    
    Example:
        >>> times = get_beat_times("song.mp3")
        >>> times[0]
        0.5
        >>> len(times)
        42
    """
    beats = get_beats(audio_path)
    return [b["time"] for b in beats]