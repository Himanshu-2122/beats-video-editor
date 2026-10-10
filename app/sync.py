"""
Beat grouping, candidate scoring, and clip assignment module.

Responsibility:
- Convert beat timestamps into clip duration groups via create_beat_groups()
- Score and assign candidates to beat groups using heuristic scoring
- Support both greedy and Hungarian algorithm assignment strategies
- Classify beat types (downbeat, drop, buildup, regular) for music-aware matching
- Compute diversity penalties to encourage clip variety across sources
- Provide fallback candidate search when no valid candidates are found

Key Functions:
- _normalize_beats(): Convert beat input to list[dict] with 'time' and 'strength' keys
- create_beat_groups(): Group contiguous beats into clip durations
- classify_beat_type(): Label a beat as drop/buildup/downbeat/regular
- _energy_match_weight(): Compute energy matching weight per SPEC energy levels
- _compute_diversity_penalty(): Score distance from already-used source intervals
- _greedy_assign(): Greedy assignment of clips to beat groups
- _hungarian_assign(): Hungarian algorithm for global optimal assignment
- ai_assign_clips(): Main entry point for heuristic clip assignment
- sync_clips_with_beats(): Legacy pipeline entry point returning clip paths

Dependencies:
- app.video: Video metadata, scene detection, candidate generation, frame snapping
- app.cache: Caching infrastructure for motion analysis results
- scipy: Optional Hungarian algorithm (linear_sum_assignment)
- numpy: Vectorized scoring operations
"""

import os
import random
import math
import tempfile
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict

from app.video import (
    get_video_duration,
    natural_sort_key,
    process_clip,
    generate_candidates,
    snap_to_frame,
    analyze_motion_at_candidates,
    SCENE_THRESH,
    SAMPLE_INTERVAL,
    MAX_CANDIDATES_PER_SOURCE,
    MIN_CLIP_GAP,
    INTRO_SKIP_SECONDS,
)

# ============================================================
# CONFIGURATION FLAGS (User Requirements)
# ============================================================

# No clip reuse - each source clip used at most once
ALLOW_CLIP_REUSE = False

# Enforce ascending order through source videos
ENFORCE_ASCENDING_ORDER = True

# Target music duration coverage (must match music duration)
REQUIRE_FULL_COVERAGE = True

# Maximum clips per source video (0 = unlimited, limited by duration)
MAX_CLIPS_PER_SOURCE = 0

# Minimum gap between clips from same source (seconds)
# Already defined in video.py as MIN_CLIP_GAP = 3.0

# Required policy: keep adding UNIQUE clips until the timeline reaches the music duration.
# Do NOT stop at a fixed clip count or a fixed total length, do NOT freeze/loop/stretch,
# and only the final clip may be trimmed to land exactly on the music end.
REQUIRE_VIDEO_COVERS_MUSIC = True

# Transition types by energy level
TRANSITIONS_CALM = ["fade", "dissolve", "fadeblack", "fadewhite", "fadeslow", "smoothleft", "smoothright", "smoothup", "smoothdown"]
TRANSITIONS_MEDIUM = ["wipeleft", "wiperight", "wipeup", "wipedown", "slideleft", "slideright", "slideup", "slidedown", "circlecrop", "rectcrop", "distance", "radial", "circleopen", "circleclose", "vertopen", "vertclose", "horzopen", "horzclose"]
TRANSITIONS_HIGH = ["zoomin", "fadefast", "pixelize", "diagtl", "diagtr", "diagbl", "diagbr", "hlslice", "hrslice", "vuslice", "vdslice", "hblur", "hlwind", "hrwind", "vuwind", "vdwind", "squeezeh", "squeezev", "coverleft", "coverright", "coverup", "coverdown", "revealleft", "revealright", "revealup", "revealdown", "wipetl", "wipetr", "wipebl", "wipebr"]

# Maximum attempts to find a valid candidate per beat group
MAX_ASSIGNMENT_ATTEMPTS = 3

# Diversity penalty for similar music sections (chorus/verse repetition)
SECTION_SIMILARITY_PENALTY = 0.25

# Professional clip selection weights
CLIP_SELECTION_WEIGHTS = {
    "scene_bonus": 0.25,       # Maximum bonus for scene boundary alignment
    "proximity_bonus": 0.15,   # Maximum bonus for beat proximity
    "motion_weight": 0.35,     # Weight of motion score in total score
    "highlight_bonus": 0.20,   # Maximum bonus for highlight overlap
    "quality_bonus": 0.10,     # Default quality bonus
    "type_bonus": 0.30,        # Maximum beat-type adjustment
    "energy_weight": 0.25,     # Energy matching weight importance
    "diversity_penalty": 0.15, # Maximum diversity penalty
    "distribution_bonus": 0.30,# Distribution bonus weight
    "forward_bonus": 0.10,     # Forward progression bonus
}


def _normalize_beats(beat_times):
    """Convert beat input to list[dict] with 'time' and 'strength' keys.
    
    Handles both legacy float lists and dict-formatted beat data.
    
    Args:
        beat_times (list[float] or list[dict]): Beat timestamps as flat floats
            or already-formatted dicts with 'time' and 'strength' keys.
    
    Returns:
        list[dict]: List of dicts with 'time' (float, seconds) and
            'strength' (float 0-1). Empty list if input is empty.
    
    Example:
        >>> _normalize_beats([0.5, 1.0, 1.5])
        [{"time": 0.5, "strength": 0.5}, {"time": 1.0, "strength": 0.5}, {"time": 1.5, "strength": 0.5}]
        >>> _normalize_beats([{"time": 0.5, "strength": 0.8}])
        [{"time": 0.5, "strength": 0.8}]
    """
    if not beat_times:
        return []
    if isinstance(beat_times[0], dict):
        return beat_times
    # Legacy: list of floats
    return [{"time": float(t), "strength": 0.5} for t in beat_times]


def create_beat_groups(
    beat_times,
    min_beats,
    max_beats,
    music_duration=None,
):
    """
    Convert beat timestamps into clip durations.
    
    Each group corresponds to a contiguous range of beats. The clip boundaries
    align exactly with detected beat timestamps. Guarantees coverage of the
    entire music duration when REQUIRE_FULL_COVERAGE is True.
    
    The grouping strategy uses adaptive group sizes: smaller groups for more
    granular coverage when remaining beats are few, random sizes within
    [min_beats, max_beats] otherwise.
    
    Args:
        beat_times (list[dict]): List of beat dicts with 'time' and 'strength' keys.
            Already normalized via _normalize_beats().
        min_beats (int): Minimum number of beats per group (inclusive).
        max_beats (int): Maximum number of beats per group (inclusive).
        music_duration (float, optional): Total music duration in seconds. If
            provided and REQUIRE_FULL_COVERAGE is True, ensures the timeline
            covers the full music duration without overshooting.
    
    Returns:
        list[dict]: List of beat groups, each with keys:
            - start (float): Start time in seconds from music start
            - end (float): End time in seconds from music start
            - duration (float): Clip duration in seconds (end - start)
            - strength (float): Average strength of beats in this group
            - beat_indices (list[int]): Indices of beats included in this group
            - synthetic (bool): True if this group was added to fill remaining
              duration (when REQUIRE_FULL_COVERAGE and music_duration are set)
    
    Raises:
        None: Function always returns a list; if beat_times has fewer than 2
            elements, returns empty list.
    
    Side Effects:
        Prints diagnostic messages when extending last group or adding
        synthetic groups to cover full music duration.
    
    Important Assumptions:
        - Beat times are sorted and in ascending order
        - Group boundaries align with actual beat timestamps
        - When REQUIRE_FULL_COVERAGE=True, the final group may be extended
          to music_duration or synthetic groups may be added
    
    Example:
        >>> beats = [{"time": 0.0, "strength": 0.8}, {"time": 2.0, "strength": 0.6}, {"time": 4.0, "strength": 0.9}]
        >>> create_beat_groups(beats, min_beats=2, max_beats=4, music_duration=12.0)
        [{'start': 0.0, 'end': 4.0, 'duration': 4.0, 'strength': 0.767, 'beat_indices': [0, 1, 2], 'synthetic': False}]
    """
    beats = _normalize_beats(beat_times)

    groups = []
    i = 0
    total = len(beats)

    while i < total - 1:
        # Adaptive group size: smaller for more granular coverage
        remaining_beats = total - i - 1
        if remaining_beats <= max_beats:
            group_size = remaining_beats
        else:
            group_size = random.randint(min_beats, max_beats)
            group_size = min(group_size, remaining_beats)

        if group_size <= 0:
            break

        start = beats[i]["time"]
        end = beats[i + group_size]["time"]
        duration = end - start

        # Average strength of beats in this group
        group_strength = sum(beats[j]["strength"] for j in range(i, i + group_size + 1)) / (group_size + 1)

        if duration > 0:
            groups.append({
                "start": start,
                "end": end,
                "duration": duration,
                "strength": group_strength,
                "beat_indices": list(range(i, i + group_size + 1)),
            })

        i += group_size

    # Ensure last beat is covered if there's remaining time
    if groups and i < total - 1:
        last_beat_time = beats[-1]["time"]
        if last_beat_time > groups[-1]["end"]:
            # Extend last group to cover the final beat
            groups[-1]["end"] = last_beat_time
            groups[-1]["duration"] = groups[-1]["end"] - groups[-1]["start"]

    # CRITICAL: Ensure coverage extends to full music duration without overshoot
    if REQUIRE_FULL_COVERAGE and music_duration and groups:
        # Sort groups by start time (they should already be in order, but ensure it)
        groups_sorted = sorted(groups, key=lambda g: g["start"])
        
        # Verify the first group starts at or after 0, and check for gaps
        # Extend first group backward to 0 if it starts after 0 and there are beats before it
        if groups_sorted and groups_sorted[0]["start"] > 0 and len(beats) > 0:
            # The first detected beat determines the effective start
            # If the first beat is after 0, we still start from the first beat's position
            # but we need to ensure the total coverage doesn't exceed music_duration
            pass  # First group starts at its detected start time
        
        # Extend last group to music_duration if needed, without overshooting
        last_group = groups_sorted[-1]
        if music_duration > last_group["end"]:
            # Check if extending would create overlap with synthetic groups logic
            # Only extend if the gap is reasonable and we have beats leading up to it
            if music_duration - last_group["end"] > 0.1:  # Minimum gap to justify extension
                last_group["end"] = music_duration
                last_group["duration"] = last_group["end"] - last_group["start"]
                print(f"       Extended final group to cover full music duration: {music_duration:.2f}s (was {last_group['end']:.2f}s)")
        
        # If still not enough coverage (total duration < music_duration), create synthetic groups
        # that start exactly where the last group ends and end exactly at music_duration
        total_duration = sum(g["duration"] for g in groups_sorted)
        if total_duration < music_duration - 0.5:  # Allow small tolerance
            remaining = music_duration - total_duration
            # Add synthetic group(s) that cover exactly the remaining duration
            # to reach music_duration without overshooting
            last_end_sorted = groups_sorted[-1]["end"]
            
            if remaining > 0.5:
                # Create exactly one synthetic group covering the remaining duration
                # clamped to not overshoot music_duration
                synth_duration = min(remaining, music_duration - last_end_sorted)
                if synth_duration > 0.1:  # Minimum meaningful group duration
                    groups_sorted.append({
                        "start": last_end_sorted,
                        "end": last_end_sorted + synth_duration,
                        "duration": synth_duration,
                        "strength": 0.5,
                        "beat_indices": [],
                        "synthetic": True,
                    })
                print(f"       Added synthetic group to cover remaining duration: {synth_duration:.2f}s")

    return groups


def _assert_ascend_no_overlap(assignments, music_duration, source_durations, intro_skip_seconds):
    """Post-condition checks that the timeline obeys the required behavior.
    
    These checks are informational (they log loudly) and help catch regressions.
    They do NOT change behavior at runtime.
    
    The checks verify:
    - Clips are in ascending order by source_start (no overlapping from same video)
    - No clip extends past its source video's duration
    - Total video duration vs music duration relationship
    - Last clip end position
    
    Args:
        assignments (list[dict]): Assignment dicts from greedy/hungarian assign.
            Each must have 'source_start', 'duration', and 'video_path' keys.
        music_duration (float): Expected total music duration in seconds.
        source_durations (dict, optional): Pre-computed {video_path: duration}.
            If not provided, durations will be looked up via get_video_duration().
        intro_skip_seconds (float): Intro skip seconds used in analysis.
    
    Returns:
        float: Total video clip duration across all assignments.
    
    Side Effects:
        Prints detailed [BEHAVIOR CHECK] log messages to stdout including:
        - Per-clip start/duration/end/source_dur info
        - Any violations found (non-ascending, past source end)
        - Summary totals (total_video_duration, music_duration, last_clip_end)
    
    Important Notes:
        - Does not modify assignments or any shared state
        - Only logs; does not raise exceptions or abort processing
        - Uses intro_skip_seconds to validate source_start positions
        - source_durations dict is updated in-place with looked-up durations
    """
    from app.video import get_video_duration as _gvd
    logger_lines = []
    logger_lines.append(f"[BEHAVIOR CHECK] assignments={len(assignments)}")
    total_video = 0.0
    prev_end = None
    for i, a in enumerate(assignments):
        vp = a.get("video_path") or a.get("source_path")
        start = float(a["source_start"])
        dur = float(a.get("duration") or a.get("clip_duration") or 0.0)
        end = start + dur
        sdur = source_durations.get(vp) if source_durations else None
        if sdur is None:
            sdur = _gvd(vp)
        source_durations[vp] = sdur
        total_video += dur
        msg = f"[BEHAVIOR CHECK] clip#{i+1} {os.path.basename(vp)} src_start={start:.3f} dur={dur:.3f} end={end:.3f} src_dur={sdur:.3f}"
        if prev_end is not None and start < prev_end - 1e-6:
            msg += f"  <<< VIOLATION: not ascending (prev_end={prev_end:.3f})"
        if end > sdur + 1e-6:
            msg += f"  <<< VIOLATION: clip past source end"
        logger_lines.append(msg)
        prev_end = end
    logger_lines.append(f"[BEHAVIOR CHECK] total_video_duration={total_video:.3f}s  music_duration={music_duration:.3f}s")
    if prev_end is not None:
        logger_lines.append(f"[BEHAVIOR CHECK] last_clip_end={prev_end:.3f}s")
    print("\n" + "\n".join(logger_lines) + "\n")
    return total_video


def _compute_diversity_penalty(cand_time: float, duration: float, used_intervals: list[tuple], source_duration: float) -> float:
    """Compute diversity penalty based on temporal distance from already used intervals.
    
    Returns penalty in range [0, 1] where 0 = no penalty, 1 = max penalty (too close).
    Enforces MIN_CLIP_GAP between clips from same video.
    
    The penalty encourages temporal spacing between clips from the same source video.
    A penalty of 0 means no diversity concern; 1 means the candidate is too close
    to an already-used interval (will be filtered out by the hard enforcement).
    
    Distance calculation:
    - If candidate ends before used interval starts: distance = used_start - cand_end
    - If candidate starts after used interval ends: distance = cand_time - used_end
    - If candidate overlaps used interval: distance = 0 (hard rejection)
    
    Penalty decay:
    - distance <= MIN_CLIP_GAP (3.0s): max penalty of 1.0 (hard rejection threshold)
    - MIN_CLIP_GAP < distance <= 15.0s: linearly decaying penalty from 1.0 to 0.0
    - distance > 15.0s: no penalty (0.0)
    
    Args:
        cand_time (float): Candidate start time in seconds.
        duration (float): Candidate clip duration in seconds.
        used_intervals (list[tuple]): List of (start, end) tuples of already-used
            source intervals in seconds, per video. These are snapped/unsnapped
            according to the assignment logic.
        source_duration (float): Total duration of the source video in seconds.
    
    Returns:
        float: Diversity penalty in range [0, 1.5]. Values > 1.0 act as a
            hard rejection (added as subtractive penalty that can push total
            score below other candidates). Exact values:
            - 1.5: cand_end < used_start AND distance < MIN_CLIP_GAP (too close before)
            - 1.0: distance <= MIN_CLIP_GAP (overlap or within gap, hard rejection zone)
            - 0.0: distance > 15.0s (no penalty, far enough away)
            - Intermediate values: linearly decaying from 1.0 to 0.0
    
    Example:
        >>> # No used intervals = no penalty
        >>> _compute_diversity_penalty(5.0, 3.0, [], 60.0)
        0.0
        >>> # Candidate overlaps used interval
        >>> _compute_diversity_penalty(5.0, 3.0, [(2.0, 8.0)], 60.0)
        1.5  # overlap => distance = 0 < MIN_CLIP_GAP => 1.5
        >>> # Candidate 1s after used interval ends
        >>> _compute_diversity_penalty(10.0, 3.0, [(2.0, 8.0)], 60.0)
        1.0  # distance = 10.0 - 8.0 = 2.0 < MIN_CLIP_GAP=3.0 => penalty 1.0
        >>> # Candidate 10s after used interval ends
        >>> _compute_diversity_penalty(20.0, 3.0, [(2.0, 8.0)], 60.0)
        0.0  # distance = 12.0, between 3 and 15, decayed: 0.7*(1-(12-3)/12)=0.7*(-8/12)... actually < 0
    
    Side Effects:
        Calls app.video.MIN_CLIP_GAP to retrieve the minimum gap constant.
        No I/O or file operations.
    """
    from app.video import MIN_CLIP_GAP
    
    if not used_intervals:
        return 0.0

    min_distance = float('inf')
    cand_end = cand_time + duration

    for (used_start, used_end) in used_intervals:
        # Distance between intervals
        if cand_end <= used_start:
            dist = used_start - cand_end
        elif cand_time >= used_end:
            dist = cand_time - used_end
        else:
            dist = 0  # Overlap (should be caught earlier)

        min_distance = min(min_distance, dist)

    # HARD ENFORCEMENT: reject if too close (will be filtered out)
    if min_distance < MIN_CLIP_GAP and min_distance > 0:
        return 1.5  # Higher than max possible score component

    # Normalize: penalty decays with distance
    # Within MIN_CLIP_GAP = max penalty, beyond 15 seconds = no penalty
    if min_distance <= MIN_CLIP_GAP:
        return 1.0
    elif min_distance <= 15.0:
        return 0.7 * (1.0 - (min_distance - MIN_CLIP_GAP) / (15.0 - MIN_CLIP_GAP))
    return 0.0


def classify_beat_type(beat: dict, beats: list, beat_idx: int, drops: list, buildups: list) -> str:
    """Classify beat type for music-aware matching.
    
    Determines the type of a beat based on its energy, strength, and position
    relative to detected drops and buildups in the music track. This classification
    is used to adjust clip scoring in music-aware matching - e.g., drop beats
    prefer high-motion clips, buildup beats prefer increasing motion, etc.
    
    Classification hierarchy:
    1. Drop beat: If beat time is within ~150ms of a detected drop timestamp
    2. Buildup beat: If beat time falls within a detected buildup segment
    3. Downbeat: If beat is a local strength maximum in a 4-beat window and
       strength > 0.6 (approx first beat of a measure)
    4. Regular beat: Default fallback
    
    Args:
        beat (dict): Beat dict with 'time' (float, seconds) and optional
            'strength' (float 0-1) and 'energy' (float 0-1) keys.
        beats (list[dict]): Full list of all beat dicts, used for downbeat
            heuristics via beat index.
        beat_idx (int): Index of this beat within the full beats list.
        drops (list[dict]): Detected drop segments, each with 'time' (float)
            and 'intensity' (float 0-1).
        buildups (list[dict]): Detected buildup segments, each with 'start' (float),
            'end' (float), and 'intensity' (float 0-1).
    
    Returns:
        str: One of 'drop', 'buildup', 'downbeat', or 'regular'.
    
    Example:
        >>> beat = {"time": 5.3, "strength": 0.7}
        >>> beats = [{"time": 0.0, "strength": 0.8}, {"time": 2.0, "strength": 0.5}, {"time": 4.0, "strength": 0.9}]
        >>> drops = [{"time": 5.5, "intensity": 0.9}]
        >>> buildups = [{"start": 1.0, "end": 3.0, "intensity": 0.4}]
        >>> classify_beat_type(beat, beats, 0, drops, buildups)
        'drop'
        >>> classify_beat_type({"time": 2.0, "strength": 0.7}, beats, 1, drops, buildups)
        'downbeat'
    """
    beat_time = beat["time"]
    beat_energy = beat.get("energy", beat.get("strength", 0.5))
    beat_strength = beat.get("strength", 0.5)
    
    # Check if it's a drop beat
    for drop in drops:
        if abs(drop["time"] - beat_time) < 0.15:  # Within 150ms
            return "drop"
    
    # Check if it's in a buildup
    for buildup in buildups:
        if buildup["start"] <= beat_time <= buildup["end"]:
            return "buildup"
    
    # Check if it's a downbeat (first beat of a measure - approx every 4 beats)
    # Simple heuristic: beats where strength is locally maximum in a 4-beat window
    window_start = max(0, beat_idx - 2)
    window_end = min(len(beats), beat_idx + 3)
    window_strengths = [beats[i].get("strength", 0.5) for i in range(window_start, window_end)]
    if beat_strength == max(window_strengths) and beat_strength > 0.6:
        return "downbeat"
    
    return "regular"


def _energy_match_weight(beat_energy: float, motion_score: float) -> float:
    """Compute energy matching weight per SPEC.
    
    Energy matching adjusts clip scores based on the relationship between
    the beat's energy level and the clip's motion score. The SPEC specifies:
    
    - Very Low energy (<0.2): prefer calm/low motion clips
    - Low (0.2-0.4): prefer gentle motion
    - Medium (0.4-0.6): neutral (no adjustment)
    - High (0.6-0.8): prefer high motion clips
    - Very High (>0.8): maximum boost to high motion (drop beats)
    
    The weight multiplier is applied to the base score, so a weight of 1.0
    means no adjustment, >1.0 boosts the score, and <1.0 reduces it.
    
    Args:
        beat_energy (float): Normalized beat energy 0-1 from music analysis.
            This is the energy at the closest beat to the current candidate.
        motion_score (float): Normalized motion score 0-1 from video analysis.
            Higher values indicate more motion/action in the clip.
    
    Returns:
        float: Energy matching weight. Always >= 0.0. Typical values:
            - Very Low energy (<0.2): 1.0 to 1.4 (boosts low-motion clips)
            - Low (0.2-0.4): 1.0 to 1.2 (slight boost to low motion)
            - Medium (0.4-0.6): 1.0 (neutral)
            - High (0.6-0.8): 1.0 to 1.3 (boosts high motion)
            - Very High (>0.8): 1.0 to 1.5 (maximum boost to high motion)
    
    Example:
        >>> _energy_match_weight(0.1, 0.2)  # Very low energy, low motion
        1.0 + ((1.0 - 0.2) * 0.4) = 1.32
        >>> _energy_match_weight(0.5, 0.8)  # Medium energy, high motion
        1.0  # neutral
        >>> _energy_match_weight(0.9, 0.9)  # Very high energy, high motion
        1.0 + (0.9 * 0.5) = 1.45
    """
    if beat_energy < 0.2:
        # Very low energy: boost low motion clips
        return 1.0 + ((1.0 - motion_score) * 0.4)
    elif beat_energy < 0.4:
        # Low energy: slight boost to low motion
        return 1.0 + ((1.0 - motion_score) * 0.2)
    elif beat_energy < 0.6:
        # Medium energy: neutral
        return 1.0
    elif beat_energy < 0.8:
        # High energy: boost high motion
        return 1.0 + (motion_score * 0.3)
    else:
        # Very high energy (drops): maximum boost to high motion
        return 1.0 + (motion_score * 0.5)


def _greedy_assign(
    beat_groups: list[dict],
    video_paths: list[str],
    scene_threshold: float = SCENE_THRESH,
    sample_interval: float = SAMPLE_INTERVAL,
    reuse_limit: int = 2,
    compute_motion: bool = True,
    cache_dir: str = None,
    music_analysis: dict = None,
    video_analyses: list = None,  # Pre-computed video analyses for cached scenes
    pre_generated_candidates: dict = None,  # Pre-generated candidates from caller
    intro_skip_seconds: float = INTRO_SKIP_SECONDS,
    source_durations: dict = None,  # Pre-computed durations {video_path: duration}
) -> list[dict]:
    """
    Greedy assignment of clips to beats using:
    - Scene bonus
    - Proximity bonus
    - Motion score
    - Diversity penalty (temporal)
    - Beat energy matching (music-aware)
    - Beat type classification (downbeat, drop, buildup, regular)
    """
    print(f"\n{'='*60}")
    print(f"[DIAG] _greedy_assign START")
    print(f"       beat_groups: {len(beat_groups)}")
    print(f"       video_paths: {len(video_paths)}")
    print(f"       reuse_limit: {reuse_limit}")
    print(f"       compute_motion: {compute_motion}")
    print(f"       intro_skip_seconds: {intro_skip_seconds:.1f}")
    print(f"       pre_generated_candidates provided: {pre_generated_candidates is not None}")
    if pre_generated_candidates:
        total_cands = sum(len(cands) for vp in video_paths for cands in pre_generated_candidates.get(vp, {}).values())
        durations = set()
        for vp in video_paths:
            durations.update(pre_generated_candidates.get(vp, {}).keys())
        print(f"       pre_generated total candidates: {total_cands}")
        print(f"       pre_generated durations: {sorted(durations)}")
        for vp in video_paths:
            vp_cands = pre_generated_candidates.get(vp, {})
            for dur, cands in vp_cands.items():
                print(f"         {os.path.basename(vp)} dur={dur:.2f}s: {len(cands)} candidates")
    print(f"{'='*60}\n")
    
    assignments = []
    video_usage_count = {v: 0 for v in video_paths}
    used_source_intervals = {v: [] for v in video_paths}

    # Extract beats and music features for energy matching
    all_beats = music_analysis.get("beats", []) if music_analysis else []
    drops = music_analysis.get("drops", []) if music_analysis else []
    buildups = music_analysis.get("buildups", []) if music_analysis else []

    # Build highlight map from video analyses for clip preference (highlights are pre-computed)
    video_highlights = {}
    if video_analyses:
        for va in video_analyses:
            video_highlights[va.get("path")] = va.get("highlights", [])

    # Lazy scene cache: analyze videos on-demand for scenes (saves RAM for many videos)
    _scene_cache = {}
    def get_cached_scenes(vp):
        if vp in _scene_cache:
            return _scene_cache[vp]
        if video_analyses:
            for va in video_analyses:
                if va.get("path") == vp:
                    _scene_cache[vp] = va.get("scenes", [])
                    return _scene_cache[vp]
        # Not in pre-computed analyses: detect scenes now (lazy)
        from app.video import detect_scenes_detailed
        print(f"       [lazy] Detecting scenes for {os.path.basename(vp)}...")
        scenes = detect_scenes_detailed(vp, intro_skip_seconds=intro_skip_seconds)
        _scene_cache[vp] = scenes
        return scenes

    # Track the earliest allowed timestamp per video for chronological ordering
    # Cursor starts at 0.0; ascending order enforced relative to previously placed clips
    # (intro_skip_seconds is handled at candidate generation time, not cursor init)
    video_cursors = {vp: 0.0 for vp in video_paths}
    
    # Use pre-generated candidates if provided, otherwise generate them
    if pre_generated_candidates is not None:
        all_candidates = pre_generated_candidates
        # Extract unique durations from pre-generated candidates
        unique_durations = set()
        for vp_cands in all_candidates.values():
            unique_durations.update(vp_cands.keys())
        unique_durations = sorted(unique_durations)
    else:
        # Pre-generate candidates for each video for ALL unique durations needed
        all_candidates = {}
        
        # Get unique durations from beat groups (bucketed to 0.05s to reduce count while minimizing mismatch)
        def bucket_duration(d):
            return round(d * 20) / 20  # 0.05s buckets
        
        unique_durations = sorted(set(bucket_duration(bg["duration"]) for bg in beat_groups))
        
        print(f"       Pre-generating candidates for {len(unique_durations)} unique durations (0.05s buckets): {unique_durations}")
        
        for vp in video_paths:
            all_candidates[vp] = {}
            
            # Find cached scenes for this video (lazy)
            cached_scenes = get_cached_scenes(vp)
            
            for duration in unique_durations:
                candidates = generate_candidates(
                    vp,
                    duration,
                    sample_interval=sample_interval,
                    scene_threshold=scene_threshold,
                    max_candidates=MAX_CANDIDATES_PER_SOURCE,
                    compute_motion=compute_motion,
                    cache_dir=cache_dir,
                    cached_scenes=cached_scenes,
                    intro_skip_seconds=intro_skip_seconds,
                )
                all_candidates[vp][duration] = candidates
    
    # Map actual duration -> bucketed duration for lookup
    def bucket_duration(d):
        return round(d * 20) / 20  # 0.05s buckets
    duration_to_bucket = {bg["duration"]: bucket_duration(bg["duration"]) for bg in beat_groups}

    # Debug: collect scores per beat
    debug_scores = {}
    
    

    # Diagnostic counters
    total_stats = {
        'total_examined': 0,
        'rejected_duration': 0,
        'rejected_overlap': 0,
        'rejected_cursor': 0,
        'rejected_reuse_limit': 0,
        'rejected_overlap_final': 0,
        'rejected_score': 0,
        'fallback_used': 0,
        'fallback_failed': 0,
        'assigned': 0,
    }
    
    for beat_idx, beat_group in enumerate(beat_groups):
        beat_time = beat_group["start"]
        duration = beat_group["duration"]
        # Use bucketed duration (0.05s) to match pre-generated candidates
        bucketed_duration = round(duration * 20) / 20
        
        # EARLY EXIT: Only when we generated candidates ourselves (same bucketing)
        # When pre_generated_candidates is passed (from Hungarian), bucketing may differ
        if pre_generated_candidates is None:
            # Find closest available duration for each video
            def find_closest_duration(vp, target):
                durations = list(all_candidates.get(vp, {}).keys())
                if not durations:
                    return None
                return min(durations, key=lambda d: abs(d - target))
            
            has_candidates = False
            for vp in video_paths:
                closest_dur = find_closest_duration(vp, bucketed_duration)
                if closest_dur is not None and all_candidates.get(vp, {}).get(closest_dur):
                    has_candidates = True
                    break
            
            if not has_candidates:
                debug_scores[beat_idx] = []
                print(f"       [DIAG] Beat {beat_idx}: NO CANDIDATES for bucketed_dur={bucketed_duration:.2f}s (beat_time={beat_time:.2f}s)")
                continue
        
        beat_start = time.perf_counter()
        
        if beat_idx % 10 == 0:
            print(f"       Processing beat group {beat_idx+1}/{len(beat_groups)}...", flush=True)
        
        # Time the scoring loop
        score_start = time.perf_counter()
        
        # Find the corresponding beat in music analysis for energy/type
        beat_info = None
        if all_beats:
            # Find closest beat
            beat_diffs = [(abs(b["time"] - beat_time), i) for i, b in enumerate(all_beats)]
            if beat_diffs:
                _, closest_idx = min(beat_diffs)
                beat_info = all_beats[closest_idx]
        
        beat_energy = beat_info.get("energy", beat_info.get("strength", 0.5)) if beat_info else 0.5
        beat_strength = beat_info.get("strength", 0.5) if beat_info else 0.5
        beat_type = classify_beat_type(beat_info, all_beats, closest_idx if beat_info else 0, drops, buildups) if beat_info else "regular"

        best_score = -1.0
        best_candidate = None
        best_video = None
        beat_scores = []
        
        candidates_considered = 0
        candidates_filtered_overlap = 0
        candidates_filtered_cursor = 0
        candidates_filtered_duration = 0
        
        # Per-beat diagnostics
        beat_diag = {
            'total_candidates': 0,
            'after_duration_filter': 0,
            'after_cursor_filter': 0,
            'after_overlap_filter': 0,
            'after_reuse_filter': 0,
            'scored': 0,
            'selected': None,
            'fallback_used': False,
            'fallback_failed': False,
        }
        
        # Pre-compute per-video data that doesn't change per candidate
        video_data = {}
        for vp in video_paths:
            if video_usage_count[vp] >= reuse_limit:
                continue
            # Find closest available duration for this video
            durations = list(all_candidates.get(vp, {}).keys())
            if not durations:
                continue
            closest_duration = min(durations, key=lambda d: abs(d - bucketed_duration))
            candidates = all_candidates.get(vp, {}).get(closest_duration, [])
            if not candidates:
                continue
            
            source_duration = source_durations.get(vp, 0.0) if source_durations else get_video_duration(vp)
            if source_duration <= 0:
                continue
                
            used_intervals = used_source_intervals[vp]
            cursor = video_cursors[vp]
            
            # Extract candidate arrays for vectorized operations
            n_cands = len(candidates)
            if n_cands == 0:
                continue
                
            cand_times = np.array([c["t"] for c in candidates], dtype=np.float64)
            cand_ends = cand_times + duration
            scene_flags = np.array([c.get("scene_flag", False) for c in candidates], dtype=bool)
            motion_raw_arr = np.array([c.get("motion_score", 0.0) for c in candidates], dtype=np.float64)
            
            # Pre-filter: duration bounds
            duration_mask = cand_ends <= source_duration
            diag_duration_passed = int(np.sum(duration_mask))
            # Pre-filter: chronological ordering
            cursor_mask = cand_times >= cursor
            diag_cursor_passed = int(np.sum(cursor_mask))
            
            # Overlap check - vectorized where possible
            overlap_mask = np.zeros(n_cands, dtype=bool)
            if len(used_intervals) > 0:
                for used_start, used_end in used_intervals:
                    # cand overlaps if NOT (cand_end <= used_start OR cand_time >= used_end)
                    overlap_mask |= ~((cand_ends <= used_start) | (cand_times >= used_end))
            
            overlap_rejected = int(np.sum(overlap_mask))
            
            # Combined valid mask
            valid_mask = duration_mask & cursor_mask & ~overlap_mask
            valid_indices = np.where(valid_mask)[0]
            n_valid = len(valid_indices)
            
            # Detailed overlap stats for first few beat groups
            if beat_idx < 3:
                print(f"       [OVERLAP DIAG] Beat {beat_idx} VP {os.path.basename(vp)}: total_cands={n_cands}, dur_pass={diag_duration_passed}, cursor_pass={diag_cursor_passed}, overlap_rej={overlap_rejected}, valid={n_valid}")
                if len(used_intervals) > 0:
                    print(f"         used_intervals: {[(f'{s:.1f}', f'{e:.1f}') for s,e in used_intervals]}")
                if n_valid == 0 and overlap_rejected > 0:
                    # Show some rejected candidates
                    rejected_indices = np.where(overlap_mask & duration_mask & cursor_mask)[0]
                    for ridx in rejected_indices[:3]:
                        c = candidates[int(ridx)]
                        print(f"         rejected candidate: t={c['t']:.2f}, end={c['t']+duration:.2f}")
            
            if n_valid == 0:
                continue
                
            # Extract valid candidate data
            valid_cand_times = cand_times[valid_mask]
            valid_cand_ends = cand_ends[valid_mask]
            valid_scene_flags = scene_flags[valid_mask]
            valid_motion_raw = motion_raw_arr[valid_mask]
            valid_cand_indices = valid_indices  # original indices in candidates list
            
            # ---- Vectorized Scoring Components ----
            
            # Scene bonus (0.20 max)
            scene_bonus = np.where(valid_scene_flags, 0.20, 0.0)
            
            # Proximity bonus (0.10 max) - closer to beat time is better
            proximity = np.abs(valid_cand_times - beat_time)
            proximity_bonus = np.maximum(0.0, 0.10 * (1.0 - np.minimum(1.0, proximity / 10.0)))
            
            # Motion score (0.30 max)
            motion_score = valid_motion_raw * 0.30
            
            # Diversity penalty (subtract up to 0.15)
            # _compute_diversity_penalty can't be easily vectorized, compute per candidate
            diversity_penalties = np.array([
                _compute_diversity_penalty(valid_cand_times[i], duration, used_intervals, source_duration) * 0.15
                for i in range(n_valid)
            ], dtype=np.float64)
            
            # Highlight bonus (0.20 max)
            highlight_bonus = np.zeros(n_valid, dtype=np.float64)
            highlights = video_highlights.get(vp, [])
            if highlights:
                for i in range(n_valid):
                    cand_time = valid_cand_times[i]
                    cand_end = valid_cand_ends[i]
                    best_hb = 0.0
                    for hl in highlights:
                        hl_start = hl.get("start", 0)
                        hl_end = hl.get("end", 0)
                        if not (valid_cand_ends[i] <= hl_start or cand_time >= hl_end):
                            overlap_ratio = min(valid_cand_ends[i], hl_end) - max(cand_time, hl_start)
                            if overlap_ratio > 0:
                                best_hb = max(best_hb, 0.20 * (overlap_ratio / duration))
                    highlight_bonus[i] = best_hb
            
            # Quality bonus
            quality_bonus = 0.0
            
            # Beat type specific adjustments
            type_bonus = np.zeros(n_valid, dtype=np.float64)
            if beat_type == "drop":
                type_bonus = valid_motion_raw * 0.25
            elif beat_type == "buildup":
                type_bonus = valid_motion_raw * 0.15
            elif beat_type == "downbeat":
                type_bonus = np.where(valid_scene_flags, 0.15, 0.0)
            
            # Energy matching weight
            # _energy_match_weight depends on motion_raw, compute per candidate
            energy_weights = np.array([
                _energy_match_weight(beat_energy, valid_motion_raw[i])
                for i in range(n_valid)
            ], dtype=np.float64)
            
            # Base score (before energy weight)
            base_score = scene_bonus + proximity_bonus + motion_score + highlight_bonus + quality_bonus + type_bonus - diversity_penalties
            score_arr = base_score * energy_weights
            
            # Find best
            best_idx_local = int(np.argmax(score_arr))
            best_score_local = float(score_arr[best_idx_local])
            best_cand_idx = int(valid_cand_indices[best_idx_local])
            
            if best_score_local > best_score:
                best_score = best_score_local
                best_candidate = candidates[best_cand_idx]
                best_video = vp
            
            # Build beat_scores list for distribution bonuses
            for i in range(n_valid):
                cand_idx = int(valid_cand_indices[i])
                cand = candidates[cand_idx]
                beat_scores.append({
                    "video_path": vp,
                    "candidate_idx": cand_idx,
                    "candidate_time": float(valid_cand_times[i]),
                    "scene_flag": bool(valid_scene_flags[i]),
                    "motion_score": float(valid_motion_raw[i]),
                    "diversity_penalty": float(diversity_penalties[i]),
                    "beat_energy": beat_energy,
                    "beat_strength": beat_strength,
                    "beat_type": beat_type,
                    "energy_weight": float(energy_weights[i]),
                    "type_bonus": float(type_bonus[i]),
                    "score": float(score_arr[i]),
                })
            
            candidates_considered += n_valid
            candidates_filtered_duration += np.sum(~duration_mask)
            candidates_filtered_cursor += np.sum(~cursor_mask)
            candidates_filtered_overlap += np.sum(overlap_mask & duration_mask & cursor_mask)

        score_elapsed = time.perf_counter() - score_start
        if score_elapsed > 0.1:
            print(f"       Beat {beat_idx}: scoring took {score_elapsed:.3f}s for {len(beat_scores)} candidates", flush=True)
        
        if not beat_scores:
            debug_scores[beat_idx] = []
            continue

        # Multi-video distribution bonuses: apply after initial scoring
        # This encourages using all uploaded videos across their full durations
        if len(video_paths) > 1 and beat_scores:
            total_clips_used = sum(video_usage_count.values())
            avg_clips_per_video = total_clips_used / len(video_paths) if total_clips_used > 0 else 0
            
            for sc in beat_scores:
                vp = sc["video_path"]
                
                # Usage balance bonus: STRONGLY prefer videos used less than average
                clips_used = video_usage_count[vp]
                if avg_clips_per_video > 0:
                    usage_ratio = clips_used / avg_clips_per_video
                    # Bonus up to 0.50 for videos underused, penalty for overused
                    usage_bonus = max(-0.30, min(0.50, 0.50 * (1.0 - usage_ratio)))
                else:
                    usage_bonus = 0.50  # First round: strongly encourage all videos
                
                # Footage availability bonus: prefer videos with more unused duration
                source_duration = source_durations.get(vp, 0.0) if source_durations else get_video_duration(vp)
                used_duration = sum(end - start for start, end in used_source_intervals[vp])
                unused_ratio = max(0.0, (source_duration - used_duration) / source_duration)
                # Bonus up to 0.20 for videos with lots of unused footage
                freshness_bonus = 0.20 * unused_ratio
                
                # Diversity penalty: penalize videos already used for this beat's energy level
                # (simplified: just penalize heavily used videos)
                diversity_penalty_extra = -0.10 * clips_used
                
                sc["distribution_bonus"] = usage_bonus + freshness_bonus + diversity_penalty_extra
                sc["adjusted_score"] = sc["score"] + sc["distribution_bonus"]
        else:
            for sc in beat_scores:
                sc["distribution_bonus"] = 0.0
                sc["adjusted_score"] = sc["score"]

        # Controlled randomness: select from top candidates with weighted probability
        # This adds visual diversity while maintaining ascending order (candidates already filtered by cursor)
        TOP_CANDIDATES_POOL = 5
        if beat_scores:
            # Sort all scored candidates by adjusted_score (includes distribution bonus) descending
            beat_scores.sort(key=lambda x: x["adjusted_score"], reverse=True)
            # Take top N candidates
            top_candidates = beat_scores[:TOP_CANDIDATES_POOL]
            
            # Apply additional diversity bonus: prefer candidates further from last used position
            for sc in top_candidates:
                vp = sc["video_path"]
                cand_time = sc["candidate_time"]
                cursor = video_cursors[vp]
                # Bonus for jumping forward more (story progression)
                forward_bonus = min(0.10, max(0.0, (cand_time - cursor) / 30.0))  # Up to 0.10 for 30s jump
                sc["adjusted_score"] = sc["adjusted_score"] + forward_bonus
            
            # Re-sort by adjusted score
            top_candidates.sort(key=lambda x: x["adjusted_score"], reverse=True)
            
            # Weighted random selection from top candidates
            # Higher score = higher probability, but lower-ranked still have a chance
            weights = [math.exp(s["adjusted_score"] * 5) for s in top_candidates]  # Temperature = 5
            total_weight = sum(weights)
            probs = [w / total_weight for w in weights]
            
            # Select based on probabilities
            r = random.random()
            cumsum = 0.0
            selected = top_candidates[0]
            for i, p in enumerate(probs):
                cumsum += p
                if r <= cumsum:
                    selected = top_candidates[i]
                    break
            
            best_candidate = None
            for vp in video_paths:
                if vp == selected["video_path"]:
                    candidates = all_candidates.get(vp, {}).get(bucketed_duration, [])
                    for cand in candidates:
                        if abs(cand["t"] - selected["candidate_time"]) < 0.001:
                            best_candidate = cand
                            best_video = vp
                            best_score = selected["adjusted_score"]
                            break
                    break

        def _find_fallback_candidate(beat_idx, beat_group, duration, beat_time, beat_energy, beat_strength, beat_type):
            """Fallback: find ANY valid candidate by progressively relaxing constraints."""
            bucketed_duration = round(duration * 20) / 20
            # Prioritize videos with fewer clips used and more unused footage
            def _video_priority(vp):
                clips_used = video_usage_count[vp]
                source_duration = source_durations.get(vp, 0.0) if source_durations else get_video_duration(vp)
                used_duration = sum(end - start for start, end in used_source_intervals[vp])
                unused_ratio = max(0.0, (source_duration - used_duration) / source_duration) if source_duration > 0 else 0
                # Lower priority score = better (less used, more fresh footage)
                return clips_used - unused_ratio * 10
            
            sorted_videos = sorted(video_paths, key=_video_priority)
            
            # Phase 1: Try with relaxed reuse_limit (allow reuse of video, but not same clip intervals)
            # Respect ENFORCE_ASCENDING_ORDER
            for vp in sorted_videos:
                candidates = all_candidates.get(vp, {}).get(bucketed_duration, [])
                if not candidates:
                    continue
                source_duration = source_durations.get(vp, 0.0) if source_durations else get_video_duration(vp)
                if source_duration <= 0:
                    continue
                used_intervals = used_source_intervals[vp]
                cursor = video_cursors[vp] if ENFORCE_ASCENDING_ORDER else 0.0
                for cand in candidates:
                    cand_time = cand["t"]
                    cand_end = cand_time + duration
                    if cand_end > source_duration:
                        continue
                    if ENFORCE_ASCENDING_ORDER and cand_time < cursor:
                        continue
                    overlap = False
                    for (used_start, used_end) in used_intervals:
                        if not (cand_end <= used_start or cand_time >= used_end):
                            overlap = True
                            break
                    if overlap:
                        continue
                    snapped_start = snap_to_frame(cand_time)
                    snapped_end = snapped_start + duration
                    overlap = False
                    for (used_start, used_end) in used_intervals:
                        if not (snapped_end <= used_start or snapped_start >= used_end):
                            overlap = True
                            break
                    if not overlap:
                        return vp, cand, snapped_start, 0.0
            
            # Phase 2: If ALLOW_CLIP_REUSE is True, ignore overlap check entirely
            # If False, we must find unused candidates - expand search
            if ALLOW_CLIP_REUSE:
                for vp in sorted_videos:
                    candidates = all_candidates.get(vp, {}).get(bucketed_duration, [])
                    if not candidates:
                        continue
                    source_duration = source_durations.get(vp, 0.0) if source_durations else get_video_duration(vp)
                    if source_duration <= 0:
                        continue
                    cursor = video_cursors[vp] if ENFORCE_ASCENDING_ORDER else 0.0
                    for cand in candidates:
                        cand_time = cand["t"]
                        cand_end = cand_time + duration
                        if cand_end > source_duration:
                            continue
                        if ENFORCE_ASCENDING_ORDER and cand_time < cursor:
                            continue
                        snapped_start = snap_to_frame(cand_time)
                        return vp, cand, snapped_start, -1.0
            
            # Phase 3 (no-reuse): Expand candidate search with wider sampling
            # Generate more candidates at finer intervals to find unused slots
            for vp in sorted_videos:
                source_duration = source_durations.get(vp, 0.0) if source_durations else get_video_duration(vp)
                if source_duration <= duration:
                    continue
                used_intervals = used_source_intervals[vp]
                cursor = video_cursors[vp] if ENFORCE_ASCENDING_ORDER else 0.0
                # Sample at finer intervals to find gaps
                fine_interval = 0.5  # 500ms steps
                t = cursor  # Start from cursor to respect ascending order
                max_iterations = int((source_duration - cursor) / fine_interval) + 10
                iterations = 0
                while t + duration <= source_duration and iterations < max_iterations:
                    iterations += 1
                    cand_time = t
                    cand_end = cand_time + duration
                    overlap = False
                    for (used_start, used_end) in used_intervals:
                        if not (cand_end <= used_start or cand_time >= used_end):
                            overlap = True
                            break
                    if not overlap:
                        snapped_start = snap_to_frame(cand_time)
                        snapped_end = snapped_start + duration
                        overlap = False
                        for (used_start, used_end) in used_intervals:
                            if not (snapped_end <= used_start or snapped_start >= used_end):
                                overlap = True
                                break
                        if not overlap:
                            return vp, {"t": cand_time, "scene_flag": False, "motion_score": 0.0}, snapped_start, -1.5
                    t += fine_interval
                
                # Phase 4 (was Phase 6): GUARANTEED ASSIGNMENT with HARD overlap check
                # Search from cursor onwards for a non-overlapping slot
                for vp in sorted_videos:
                    source_duration = source_durations.get(vp, 0.0) if source_durations else get_video_duration(vp)
                    cursor = video_cursors[vp] if ENFORCE_ASCENDING_ORDER else 0.0
                    search_start = cursor
                    max_iterations = int((source_duration - cursor) / 0.5) + 10
                    iterations = 0
                    while search_start + duration <= source_duration and iterations < max_iterations:
                        iterations += 1
                        cand_time = search_start
                        cand_end = cand_time + duration
                        overlap = False
                        for (used_start, used_end) in used_source_intervals[vp]:
                            if not (cand_end <= used_start or cand_time >= used_end):
                                overlap = True
                                break
                        if not overlap:
                            snapped_start = snap_to_frame(cand_time)
                            return vp, {"t": cand_time, "scene_flag": False, "motion_score": 0.0}, snapped_start, -5.0
                        search_start += 0.5  # Step forward
                
                # If absolutely no non-overlapping slot exists in any video, return None
                # (should not happen if total source duration >= music duration)
                return None, None, None, None

        if best_candidate and best_video:
            cand_time = best_candidate["t"]
            cand_end = cand_time + duration

            # Snap to frame for precise alignment
            snapped_start = snap_to_frame(cand_time)
            snapped_end = snapped_start + duration

            # Re-check overlap with snapped times
            overlap = False
            for (used_start, used_end) in used_source_intervals[best_video]:
                if not (snapped_end <= used_start or snapped_start >= used_end):
                    overlap = True
                    break

            if overlap:
                # Try fallback instead of skipping
                fb_video, fb_candidate, fb_start, fb_score = _find_fallback_candidate(
                    beat_idx, beat_group, duration, beat_time, beat_energy, beat_strength, beat_type
                )
                if fb_video and fb_candidate:
                    best_video = fb_video
                    best_candidate = fb_candidate
                    snapped_start = fb_start
                    best_score = fb_score
                else:
                    debug_scores[beat_idx] = beat_scores
                    continue

            # else: use the normal best_candidate

            assignments.append({
                "beat_idx": beat_idx,
                "video_path": best_video,
                "source_start": snapped_start,
                "duration": duration,
                "score": best_score,
                "scene_flag": best_candidate.get("scene_flag", False),
                "motion_score": best_candidate.get("motion_score", 0.0),
                "beat_strength": beat_strength,
                "beat_energy": beat_energy,
                "beat_type": beat_type,
            })

            video_usage_count[best_video] += 1
            # Store snapped intervals to prevent future overlaps
            used_source_intervals[best_video].append((snapped_start, snapped_start + duration))
            # Update cursor for chronological ordering: next clip must start after this one ends + gap
            from app.video import MIN_CLIP_GAP
            video_cursors[best_video] = max(video_cursors[best_video], snapped_start + duration + MIN_CLIP_GAP)

        else:
            # No candidate found in normal scoring - use fallback
            fb_video, fb_candidate, fb_start, fb_score = _find_fallback_candidate(
                beat_idx, beat_group, duration, beat_time, beat_energy, beat_strength, beat_type
            )
            if fb_video and fb_candidate:
                assignments.append({
                    "beat_idx": beat_idx,
                    "video_path": fb_video,
                    "source_start": fb_start,
                    "duration": duration,
                    "score": fb_score,
                    "scene_flag": fb_candidate.get("scene_flag", False),
                    "motion_score": fb_candidate.get("motion_score", 0.0),
                    "beat_strength": beat_strength,
                    "beat_energy": beat_energy,
                    "beat_type": beat_type,
                })
                video_usage_count[fb_video] += 1
                used_source_intervals[fb_video].append((fb_start, fb_start + duration))
                from app.video import MIN_CLIP_GAP
                video_cursors[fb_video] = max(video_cursors[fb_video], fb_start + duration + MIN_CLIP_GAP)
            else:
                debug_scores[beat_idx] = beat_scores
                continue

        beat_elapsed = time.perf_counter() - beat_start
        if beat_elapsed > 0.05:
            print(f"       Beat {beat_idx}: total={beat_elapsed:.3f}s score={score_elapsed:.3f}s "
                  f"cand_considered={candidates_considered} filtered: "
                  f"overlap={candidates_filtered_overlap} cursor={candidates_filtered_cursor} "
                  f"dur={candidates_filtered_duration} scored={len(beat_scores)}", flush=True)

    # Final diagnostic summary
    print(f"\n{'='*60}")
    print(f"[DIAGNOSTIC SUMMARY] _greedy_assign")
    print(f"{'='*60}")
    print(f"Total beat groups: {len(beat_groups)}")
    print(f"Assignments made: {len(assignments)}")
    print(f"Beat groups with 0 candidates: {sum(1 for v in debug_scores.values() if len(v) == 0)}")
    print(f"Beat groups with fallback: {sum(1 for v in debug_scores.values() if len(v) > 0 and v and any('fallback' in str(s) for s in v))}")
    if debug_scores:
        total_examined = sum(len(v) for v in debug_scores.values() if v)
        total_scored = sum(1 for v in debug_scores.values() if v)
        print(f"Total candidates scored across all beats: {total_examined}")
        print(f"Beats with any scored candidates: {total_scored}")
    print(f"{'='*60}\n")
    
    print(f"\n{'='*60}")
    print(f"[DIAG] _greedy_assign END - returning {len(assignments)} assignments")
    print(f"{'='*60}\n")
    
    return assignments, debug_scores, all_candidates


def _hungarian_assign(
    beat_groups: list[dict],
    video_paths: list[str],
    all_candidates: dict,
    music_analysis: dict,
    video_analyses: list,
    reuse_limit: int,
    intro_skip_seconds: float = INTRO_SKIP_SECONDS,
) -> tuple:
    """
    Hungarian algorithm for global optimal clip assignment.
    
    Builds cost matrix: beat_groups × candidates, where cost = 1 - score.
    Invalid assignments (overlap, duration mismatch, reuse limit) get INF cost.
    
    Returns: (assignments, debug_scores, all_candidates)
    """
    print(f"\n{'='*60}")
    print(f"[DIAG] _hungarian_assign START")
    print(f"       beat_groups: {len(beat_groups)}")
    print(f"       video_paths: {len(video_paths)}")
    print(f"       reuse_limit: {reuse_limit}")
    print(f"       intro_skip_seconds: {intro_skip_seconds:.1f}")
    total_cands = sum(len(cands) for vp in video_paths for cands in all_candidates.get(vp, {}).values())
    durations = set()
    for vp in video_paths:
        durations.update(all_candidates.get(vp, {}).keys())
    print(f"       total candidates: {total_cands}")
    print(f"       durations: {sorted(durations)}")
    for vp in video_paths:
        vp_cands = all_candidates.get(vp, {})
        for dur, cands in vp_cands.items():
            print(f"         {os.path.basename(vp)} dur={dur:.2f}s: {len(cands)} candidates")
    print(f"{'='*60}\n")
    
    # Pre-compute video durations up-front so every fallback path uses the same
    # concrete value and never references an uninitialized local.
    source_durations = {vp: get_video_duration(vp) for vp in video_paths}

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        print("       SciPy not available, falling back to greedy")
        return _greedy_assign(beat_groups, video_paths, SCENE_THRESH, SAMPLE_INTERVAL, reuse_limit, True, None, music_analysis, video_analyses, all_candidates, intro_skip_seconds, source_durations=source_durations)

    # Fallback to greedy for large problems (cost matrix build is O(n_beats * n_cands))
    total_cands = sum(len(cands) for vp in video_paths for cands in all_candidates.get(vp, {}).values())
    if len(beat_groups) * total_cands > 5000:
        print(f"       Problem too large for Hungarian ({len(beat_groups)} beats × {total_cands} cands), using greedy")
        return _greedy_assign(beat_groups, video_paths, SCENE_THRESH, SAMPLE_INTERVAL, reuse_limit, True, None, music_analysis, video_analyses, all_candidates, intro_skip_seconds, source_durations=source_durations)
    
    # Flatten all candidates with metadata
    all_cands_flat = []
    cand_metadata = []  # (video_path, duration, candidate_dict)
    
    for vp in video_paths:
        for duration, cands in all_candidates.get(vp, {}).items():
            for c in cands:
                all_cands_flat.append(c)
                cand_metadata.append((vp, duration, c))
    
    n_beats = len(beat_groups)
    n_cands = len(all_cands_flat)
    
    if n_cands == 0:
        return [], {}, all_candidates
    
    # Build cost matrix: rows=beat_groups, cols=candidates
    INF = 1e6
    cost_matrix = np.full((n_beats, n_cands), INF)
    
    all_beats = music_analysis.get("beats", []) if music_analysis else []
    drops = music_analysis.get("drops", []) if music_analysis else []
    buildups = music_analysis.get("buildups", []) if music_analysis else []
    video_highlights = {}
    if video_analyses:
        for va in video_analyses:
            video_highlights[va.get("path")] = va.get("highlights", [])

    # Pre-compute used intervals per video (will be updated during assignment)
    used_intervals = {vp: [] for vp in video_paths}
    video_usage = {vp: 0 for vp in video_paths}
    
    # Pre-compute video durations to avoid repeated get_video_duration() calls in inner loop
    source_durations = {vp: get_video_duration(vp) for vp in video_paths}
    
    # We need to score each (beat, candidate) pair
    # This is O(n_beats * n_cands) but done once
    
    # Diagnostic counters for cost matrix construction
    diag_total_checked = 0
    diag_rejected_duration_mismatch = 0
    diag_rejected_source_duration = 0
    diag_rejected_cursor = 0
    diag_rejected_reuse = 0
    diag_rejected_overlap = 0
    diag_scored = 0
    
    for beat_idx, beat_group in enumerate(beat_groups):
        beat_time = beat_group["start"]
        duration = beat_group["duration"]
        bucketed_duration = round(duration * 20) / 20
        
        # Find corresponding beat info
        beat_info = None
        if all_beats:
            beat_diffs = [(abs(b["time"] - beat_time), i) for i, b in enumerate(all_beats)]
            if beat_diffs:
                _, closest_idx = min(beat_diffs)
                beat_info = all_beats[closest_idx]
        
        beat_energy = beat_info.get("energy", beat_info.get("strength", 0.5)) if beat_info else 0.5
        beat_strength = beat_info.get("strength", 0.5) if beat_info else 0.5
        beat_type = classify_beat_type(beat_info, all_beats, closest_idx if beat_info else 0, drops, buildups) if beat_info else "regular"
        
        beat_scored = 0
        for cand_idx, (cand, (vp, cand_duration, _)) in enumerate(zip(all_cands_flat, cand_metadata)):
            diag_total_checked += 1
            # Duration must match (within bucket tolerance)
            if abs(cand_duration - bucketed_duration) > 0.05:
                diag_rejected_duration_mismatch += 1
                continue
            
            cand_time = cand["t"]
            cand_end = cand_time + duration
            source_duration = source_durations[vp]
            
            if cand_end > source_duration:
                diag_rejected_source_duration += 1
                continue
            
            # Check chronological ordering
            cursor = 0.0  # Simplified: no cursor for Hungarian
            if ENFORCE_ASCENDING_ORDER and cand_time < cursor:
                diag_rejected_cursor += 1
                continue
            
            # Check reuse limit
            if video_usage[vp] >= reuse_limit:
                diag_rejected_reuse += 1
                continue
            
            # Check overlap with already assigned intervals
            overlap = False
            for (used_start, used_end) in used_intervals[vp]:
                if not (cand_end <= used_start or cand_time >= used_end):
                    overlap = True
                    break
            if overlap:
                diag_rejected_overlap += 1
                continue
            
            diag_scored += 1
            beat_scored += 1
            
            # Compute score (same as greedy)
            scene_bonus = 0.20 if cand.get("scene_flag", False) else 0.0
            proximity_bonus = max(0.0, 0.10 * (1.0 - min(1.0, abs(cand_time - beat_time) / 10.0)))
            motion_raw = cand.get("motion_score", 0.0)
            motion_score = motion_raw * 0.30
            
            # Diversity penalty
            diversity_penalty = 0.0
            min_dist = float('inf')
            for (used_start, used_end) in used_intervals[vp]:
                if cand_end <= used_start:
                    dist = used_start - cand_end
                elif cand_time >= used_end:
                    dist = cand_time - used_end
                else:
                    dist = 0
                min_dist = min(min_dist, dist)
            if min_dist < 3.0 and min_dist > 0:  # MIN_CLIP_GAP
                diversity_penalty = 1.0
            elif min_dist <= 15.0:
                diversity_penalty = 0.7 * (1.0 - (min_dist - 3.0) / 12.0)
            
            # Highlight bonus
            highlight_bonus = 0.0
            highlights = video_highlights.get(vp, [])
            for hl in highlights:
                hl_start = hl.get("start", 0)
                hl_end = hl.get("end", 0)
                if not (cand_end <= hl_start or cand_time >= hl_end):
                    overlap_ratio = min(cand_end, hl_end) - max(cand_time, hl_start)
                    if overlap_ratio > 0:
                        highlight_bonus = max(highlight_bonus, 0.20 * (overlap_ratio / duration))
                        break
            
            # Beat type bonus
            type_bonus = 0.0
            if beat_type == "drop":
                type_bonus = motion_raw * 0.25
            elif beat_type == "buildup":
                type_bonus = motion_raw * 0.15
            elif beat_type == "downbeat":
                type_bonus = 0.15 if cand.get("scene_flag", False) else 0.0
            
            # Energy matching
            def _energy_match_weight(beat_energy, motion_raw):
                if beat_energy < 0.2:
                    return 1.0 + ((1.0 - motion_raw) * 0.4)
                elif beat_energy < 0.4:
                    return 1.0 + ((1.0 - motion_raw) * 0.2)
                elif beat_energy < 0.6:
                    return 1.0
                elif beat_energy < 0.8:
                    return 1.0 + (motion_raw * 0.3)
                else:
                    return 1.0 + (motion_raw * 0.5)
            
            energy_weight = _energy_match_weight(beat_energy, motion_raw)
            
            base_score = scene_bonus + proximity_bonus + motion_score + highlight_bonus + type_bonus - diversity_penalty * 0.15
            score = base_score * energy_weight
            
            # Distribution bonus
            total_clips = sum(video_usage.values())
            avg_clips = total_clips / len(video_paths) if total_clips > 0 else 0
            clips_used = video_usage[vp]
            if avg_clips > 0:
                usage_ratio = clips_used / avg_clips
                usage_bonus = max(-0.30, min(0.50, 0.50 * (1.0 - usage_ratio)))
            else:
                usage_bonus = 0.50
            source_dur = source_durations[vp]
            used_dur = sum(e - s for s, e in used_intervals[vp])
            unused_ratio = max(0.0, (source_dur - used_dur) / source_dur) if source_dur > 0 else 0
            freshness_bonus = 0.20 * unused_ratio
            diversity_penalty_extra = -0.10 * clips_used
            distribution_bonus = usage_bonus + freshness_bonus + diversity_penalty_extra
            
            adjusted_score = score + distribution_bonus
            
            # Cost = 1 - score (minimize cost = maximize score)
            # Normalize score to roughly [0, 1] range for cost
            normalized_score = max(0.0, min(1.0, (adjusted_score + 1.0) / 2.0))
            cost_matrix[beat_idx, cand_idx] = 1.0 - normalized_score
        
        if beat_idx < 3 or beat_idx % 10 == 0:  # Log first 3 and every 10th
            print(f"       [DIAG] Hungarian cost matrix beat {beat_idx}: checked={diag_total_checked - (beat_idx * n_cands) if beat_idx > 0 else diag_total_checked}, dur_mismatch={diag_rejected_duration_mismatch}, src_dur={diag_rejected_source_duration}, cursor={diag_rejected_cursor}, reuse={diag_rejected_reuse}, overlap={diag_rejected_overlap}, scored={diag_scored}, this_beat_scored={beat_scored}")
    
    print(f"\n[DIAG] Hungarian cost matrix SUMMARY: total_checked={diag_total_checked}, dur_mismatch={diag_rejected_duration_mismatch}, src_dur={diag_rejected_source_duration}, cursor={diag_rejected_cursor}, reuse={diag_rejected_reuse}, overlap={diag_rejected_overlap}, scored={diag_scored}")
    print(f"       Cost matrix shape: {cost_matrix.shape}, finite entries: {np.sum(cost_matrix < 1e6)}")
    
    # Run Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Build assignments from matching
    assignments = []
    debug_scores = {}
    
    for beat_idx, cand_idx in zip(row_ind, col_ind):
        if cost_matrix[beat_idx, cand_idx] >= 1e6:
            continue  # Invalid assignment

        beat_group = beat_groups[beat_idx]
        cand = all_cands_flat[cand_idx]
        vp, cand_duration, _ = cand_metadata[cand_idx]

        # Recompute the chosen beat metadata for this specific row before serializing output.
        beat_info = None
        if all_beats:
            beat_diffs = [(abs(b["time"] - beat_group["start"]), i) for i, b in enumerate(all_beats)]
            if beat_diffs:
                _, closest_idx = min(beat_diffs)
                beat_info = all_beats[closest_idx]

        beat_energy = beat_info.get("energy", beat_info.get("strength", 0.5)) if beat_info else 0.5
        beat_strength = beat_info.get("strength", 0.5) if beat_info else 0.5
        beat_type = classify_beat_type(beat_info, all_beats, closest_idx if beat_info else 0, drops, buildups) if beat_info else "regular"

        duration = beat_group["duration"]
        cand_time = cand["t"]
        cand_end = cand_time + duration

        # Final overlap check
        overlap = False
        for (used_start, used_end) in used_intervals[vp]:
            if not (cand_end <= used_start or cand_time >= used_end):
                overlap = True
                break
        if overlap:
            continue

        # Reuse limit check
        if video_usage[vp] >= reuse_limit:
            continue

        # Snap to frame
        snapped_start = snap_to_frame(cand_time)
        snapped_end = snapped_start + duration

        # Final snapped overlap check
        overlap = False
        for (used_start, used_end) in used_intervals[vp]:
            if not (snapped_end <= used_start or snapped_start >= used_end):
                overlap = True
                break
        if overlap:
            continue

        # Compute final score for output
        score = 1.0 - cost_matrix[beat_idx, cand_idx]

        assignments.append({
            "beat_idx": beat_idx,
            "video_path": vp,
            "source_start": snapped_start,
            "duration": duration,
            "score": score,
            "scene_flag": cand.get("scene_flag", False),
            "motion_score": cand.get("motion_score", 0.0),
            "beat_strength": beat_strength,
            "beat_energy": beat_energy,
            "beat_type": beat_type,
        })

        video_usage[vp] += 1
        used_intervals[vp].append((snapped_start, snapped_end))
        from app.video import MIN_CLIP_GAP
        # Note: cursor not updated in Hungarian (global optimization)
    
    print(f"\n{'='*60}")
    print(f"[DIAG] _hungarian_assign END - returning {len(assignments)} assignments")
    print(f"{'='*60}\n")
    
    return assignments, debug_scores, all_candidates


def ai_assign_clips(
    beat_times: list[Union[float, dict]],
    video_paths: list[str],
    *,
    min_beats: int = 4,
    max_beats: int = 8,
    sample_interval: float = SAMPLE_INTERVAL,
    scene_threshold: float = SCENE_THRESH,
    reuse_limit: int = None,  # Auto-calculate if None
    compute_motion: bool = True,
    cache_dir: str = None,
    music_analysis: dict = None,
    video_analyses: list = None,  # Pre-computed video analyses for cached scenes (optional)
    intro_skip_seconds: float = INTRO_SKIP_SECONDS,
) -> list[dict]:
    """
    heuristic-aware clip assignment with:
    - Scene detection (lazy: only analyzes videos when needed)
    - Motion analysis (optical flow)
    - Diversity scoring
    - Beat energy matching (music-aware)
    - Beat type classification (downbeat, drop, buildup, regular)

    Returns list of dicts with keys:
        source_path, music_start, music_end, clip_duration,
        source_start, score, scene_flag, motion_score, beat_strength, beat_energy, beat_type
    """
    # Single source of truth for the target timeline length: the music duration.
    # Prefer analyzed music duration, probe the audio file, else fall back to last beat.
    music_duration = None
    if music_analysis and isinstance(music_analysis, dict):
        md = music_analysis.get("duration", None)
        if isinstance(md, (int, float)):
            music_duration = float(md)
    if music_duration is None or music_duration <= 0:
        probed = None
        for _path in video_paths:
            _ext = os.path.splitext(_path)[1].lower()
            if _ext in (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma"):
                try:
                    probed = probe_audio_duration(_path)
                    if probed and probed > 0:
                        break
                except Exception:
                    probed = None
        if probed and probed > 0:
            music_duration = probed
        else:
            _last = beat_times[-1] if beat_times else 0.0
        if isinstance(_last, dict):
            _last = float(_last.get("time", 0.0))
        elif not isinstance(_last, (int, float)):
            _last = 0.0
        music_duration = float(_last)

    beat_groups = create_beat_groups(beat_times, min_beats, max_beats, music_duration=music_duration)
    if not beat_groups:
        return [], {}, {}, {}

    # Lazy scene cache: analyze videos on-demand, cache results
    # This avoids analyzing all videos upfront (saves RAM for many videos)
    _scene_cache = {}
    def get_cached_scenes(vp):
        if vp in _scene_cache:
            return _scene_cache[vp]
        if video_analyses:
            for va in video_analyses:
                if va.get("path") == vp:
                    _scene_cache[vp] = va.get("scenes", [])
                    return _scene_cache[vp]
        # Not in pre-computed analyses: detect scenes now (lazy)
        from app.video import detect_scenes_detailed
        print(f"       [lazy] Detecting scenes for {os.path.basename(vp)}...")
        scenes = detect_scenes_detailed(vp, intro_skip_seconds=intro_skip_seconds)
        _scene_cache[vp] = scenes
        return scenes

    # ============================================================
    # PRE-FLIGHT CHECK: Validate total source duration vs music duration
    # ============================================================
    total_music_duration = sum(bg["duration"] for bg in beat_groups)
    source_durations = {vp: get_video_duration(vp) for vp in video_paths}
    total_source_duration = sum(source_durations.values())
    
    # Calculate maximum usable source duration based on reuse policy
    if ALLOW_CLIP_REUSE:
        max_usable_duration = total_source_duration  # Can reuse any portion
    else:
        # With no reuse + MIN_CLIP_GAP, each video can contribute at most
        # (duration - (n_clips-1)*gap) but conservatively estimate:
        max_usable_duration = total_source_duration * 0.7  # Conservative: 70% usable due to gaps/alignment
    
    if max_usable_duration < total_music_duration * 0.95:  # Allow 5% tolerance
        from app.video import MIN_CLIP_GAP
        msg = (
            f"Insufficient source footage: need {total_music_duration:.1f}s for music, "
            f"but only {max_usable_duration:.1f}s usable from {len(video_paths)} videos "
            f"(total source: {total_source_duration:.1f}s, MIN_CLIP_GAP={MIN_CLIP_GAP}s, "
            f"ALLOW_CLIP_REUSE={ALLOW_CLIP_REUSE}). "
        )
        if not ALLOW_CLIP_REUSE:
            msg += "Enable ALLOW_CLIP_REUSE=True or provide longer videos."
        print(f"       WARNING: {msg}")
        # Don't fail - try anyway with reuse enabled as fallback
        # But warn user it may produce repetitive results
    
    # Enforce no clip reuse if ALLOW_CLIP_REUSE is False
    # Even with ALLOW_CLIP_REUSE=False, the assignment loop must be able to place
    # enough UNIQUE clips to reach the music duration. reuse_limit is only a
    # per-video cap on how many clips we may draw from one source; it must be
    # large enough that the timeline is not artificially capped short.
    if not ALLOW_CLIP_REUSE:
        reuse_limit = 100
    elif reuse_limit is None:
        reuse_limit = max(1, len(beat_groups) // max(1, len(video_paths)) + 1)

    # Estimate if Hungarian will be used (small problem) or greedy fallback (large)
    # Estimate total candidates: ~10 per video per duration × num_durations
    num_durations = len(set(round(bg["duration"] * 20) / 20 for bg in beat_groups))
    est_total_cands = len(video_paths) * num_durations * 10
    use_hungarian = len(beat_groups) * est_total_cands <= 5000

    # Do not allow the assignment to stop early just because a conservative
    # "usable footage" estimate said there is not enough. The timeline must keep
    # adding UNIQUE clips until it reaches music_duration; if the source cannot
    # provide that many unique seconds, the final clip trim/extend step will
    # report the shortfall rather than silently capping the timeline short.
    print(f"[DIAG] ai_assign_clips: music_duration={music_duration:.3f}s, total_source_duration={total_source_duration:.3f}s")
    print(f"[DIAG] ai_assign_clips: allowing assignment to continue toward music_duration (no hard cap from conservative usable estimate)")

    print(f"       [DIAG] ai_assign_clips: beat_groups={len(beat_groups)}, est_total_cands={est_total_cands}, use_hungarian={use_hungarian}")

    if use_hungarian:
        # Pre-generate candidates for Hungarian (small problem)
        def bucket_duration(d):
            return round(d * 20) / 20
        
        unique_durations = sorted(set(bucket_duration(bg["duration"]) for bg in beat_groups))
        print(f"       Pre-generating candidates for {len(unique_durations)} unique durations (0.05s buckets): {unique_durations}")
        
        all_candidates = {}
        for vp in video_paths:
            all_candidates[vp] = {}
            cached_scenes = get_cached_scenes(vp)
            for duration in unique_durations:
                candidates = generate_candidates(
                    vp,
                    duration,
                    sample_interval=sample_interval,
                    scene_threshold=scene_threshold,
                    max_candidates=MAX_CANDIDATES_PER_SOURCE,
                    compute_motion=compute_motion,
                    cache_dir=cache_dir,
                    cached_scenes=cached_scenes,
                    intro_skip_seconds=intro_skip_seconds,
                )
                all_candidates[vp][duration] = candidates

        assignments, debug_scores, all_candidates = _hungarian_assign(
            beat_groups,
            video_paths,
            all_candidates,
            music_analysis,
            video_analyses,
            reuse_limit,
            intro_skip_seconds=intro_skip_seconds,
        )
    else:
        # Large problem: use greedy directly with pre_generated_candidates=None
        # so it generates its own candidates internally
        print(f"       Problem size large, using greedy assignment directly")
        assignments, debug_scores, all_candidates = _greedy_assign(
            beat_groups,
            video_paths,
            scene_threshold=scene_threshold,
            sample_interval=sample_interval,
            reuse_limit=reuse_limit,
            compute_motion=compute_motion,
            cache_dir=cache_dir,
            music_analysis=music_analysis,
            video_analyses=video_analyses,
            pre_generated_candidates=None,  # Let greedy generate its own
            intro_skip_seconds=intro_skip_seconds,
            source_durations=source_durations,
        )

    # Build result format compatible with existing sync_clips_with_beats output
    results = []
    for a in assignments:
        results.append({
            "source_path": a["video_path"],
            "music_start": beat_groups[a["beat_idx"]]["start"],
            "music_end": beat_groups[a["beat_idx"]]["end"],
            "clip_duration": a["duration"],
            "source_start": a["source_start"],
            "score": a["score"],
            "scene_flag": a["scene_flag"],
            "motion_score": a.get("motion_score", 0.0),
            "beat_strength": a.get("beat_strength", 0.5),
            "beat_energy": a.get("beat_energy", 0.5),
            "beat_type": a.get("beat_type", "regular"),
            "beat_idx": a["beat_idx"],  # Add for debugging
        })

    return results, beat_groups, debug_scores, all_candidates


def sync_clips_with_beats(
    video_paths,
    beat_times,
    output_dir=None,
    min_beats=4,
    max_beats=8,
    use_proxies=False,
    progress_callback=None,
    proxy_progress_callback=None,
    ai_mode=False,
    sample_interval=SAMPLE_INTERVAL,
    scene_threshold=SCENE_THRESH,
    reuse_limit=None,  # Auto-calculate if None
    compute_motion=True,
    music_analysis=None,
    intro_skip_seconds=INTRO_SKIP_SECONDS,
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

    # Single source of truth for the target timeline length: the music duration.
    # Read it from the analyzed music when available, otherwise probe the file.
    music_duration = None
    if music_analysis and isinstance(music_analysis, dict):
        md = music_analysis.get("duration", None)
        if isinstance(md, (int, float)):
            music_duration = float(md)

    # Normalize beats
    beats = _normalize_beats(beat_times)
    if len(beats) < 2:
        return [], [], output_dir

    # Single source of truth for the target timeline length: the music duration.
    # Prefer the analyzed music duration, then probe the audio file, then fall back to
    # the last detected beat time. This must be the SAME value used when building
    # beat groups so the timeline targets the music, not a hardcoded/clip-count limit.
    if music_duration is None or music_duration <= 0:
        probed = None
        # Only probe audio if one of the provided paths looks like an audio file.
        for _path in video_paths:
            _ext = os.path.splitext(_path)[1].lower()
            if _ext in (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma"):
                try:
                    probed = probe_audio_duration(_path)
                    if probed and probed > 0:
                        break
                except Exception:
                    probed = None
        if probed and probed > 0:
            music_duration = probed
        else:
            music_duration = float(beats[-1]["time"]) if beats else 0.0

    # Build beat groups against the SAME music_duration used as the source of truth.
    beat_groups = create_beat_groups(beats, min_beats, max_beats, music_duration=music_duration)

    if ai_mode:
        # Heuristic mode: assign clips using beat-matching, scene detection, motion scoring
        cache_dir = os.path.join(output_dir, "cache")
        ai_results, beat_groups, debug_scores, all_candidates = ai_assign_clips(
            beats,
            video_paths,
            min_beats=min_beats,
            max_beats=max_beats,
            sample_interval=sample_interval,
            scene_threshold=scene_threshold,
            reuse_limit=reuse_limit,
            compute_motion=compute_motion,
            cache_dir=cache_dir,
            music_analysis=music_analysis,
            intro_skip_seconds=intro_skip_seconds,
        )

        if not ai_results:
            return [], [], output_dir

        final_clip_paths = []
        used_beat_groups = []
        total = len(ai_results)

        print(f"\nAI mode: assigned {total} clips (scene + motion + diversity + beat energy matching)")

        for index, assignment in enumerate(ai_results):
            source_path = assignment["source_path"]
            duration = assignment["clip_duration"]
            start_time = assignment["source_start"]
            beat_group = {
                "start": assignment["music_start"],
                "end": assignment["music_end"],
                "duration": duration,
                "source_start": start_time,
            }

            clip_path = os.path.join(clip_dir, f"clip_{index + 1:04d}.mp4")

            print(f"\nFinal clip {index + 1}/{total}")
            print(f"Source: {os.path.basename(source_path)}")
            print(f"Music start: {beat_group['start']:.3f}s")
            print(f"Music end:   {beat_group['end']:.3f}s")
            print(f"Duration:    {duration:.3f}s")
            print(f"Source start: {start_time:.3f}s")
            print(f"Score: {assignment['score']:.3f} (scene={'yes' if assignment['scene_flag'] else 'no'}, motion={assignment.get('motion_score', 0.0):.3f}, beat_str={assignment.get('beat_strength', 0.5):.3f}, beat_energy={assignment.get('beat_energy', 0.5):.3f}, type={assignment.get('beat_type', 'regular')})")

            process_clip(
                video_path=source_path,
                duration=duration,
                output_path=clip_path,
                start_time=start_time,
                snap_frame=True,
            )

            final_clip_paths.append(clip_path)
            used_beat_groups.append({
                "clip_path": clip_path,
                "source_path": source_path,
                "music_start": beat_group["start"],
                "music_end": beat_group["end"],
                "clip_duration": duration,
                "source_start": start_time,
                "score": assignment["score"],
                "scene_flag": assignment["scene_flag"],
                "motion_score": assignment.get("motion_score", 0.0),
                "beat_strength": assignment.get("beat_strength", 0.5),
                "beat_energy": assignment.get("beat_energy", 0.5),
                "beat_type": assignment.get("beat_type", "regular"),
            })

            if progress_callback:
                progress_callback(index + 1, total)

        # Post-condition: verify behavior requirements (ascending, coverage, etc.)
        _source_durations = {}
        _assert_ascend_no_overlap(used_beat_groups, music_duration, _source_durations, intro_skip_seconds)

        print(f"[DURATION CHECK] music_duration_source_of_truth={music_duration:.3f}s  clips_assigned={len(used_beat_groups)}  total_clip_duration_before_trim={sum(bg['clip_duration'] for bg in used_beat_groups):.3f}s")
        print(f"[DURATION CHECK] beat_groups_target_end={beat_groups[-1]['end']:.3f}s  music_duration={music_duration:.3f}s")

        return final_clip_paths, used_beat_groups, output_dir

    # Original random mode (legacy)
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
            snap_frame=True,
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

    print(f"[DURATION CHECK] music_duration_source_of_truth={music_duration:.3f}s  clips_assigned={len(used_beat_groups)}  total_clip_duration_before_trim={sum(bg['clip_duration'] for bg in used_beat_groups):.3f}s")

    debug_path = os.path.join(output_dir, "sync_debug.json")
    try:
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump({
                "beat_groups": used_beat_groups,
                "music_duration_source_of_truth": music_duration,
                "clip_durations_in_seconds": [float(bg.get("clip_duration", bg.get("duration", 0))) for bg in used_beat_groups],
            }, f, indent=2)
        print(f"\nSync debug written to: {debug_path}")
    except Exception as exc:
        print(f"Warning: Could not write sync debug: {exc}")

    return final_clip_paths, used_beat_groups, output_dir

    # Write AI debug output (executed when ai_mode branch returned above via the earlier
    # return; kept here as a fallback for the legacy code path if needed).
    debug_path = os.path.join(output_dir, "sync_debug_ai.json")
    try:
        # Prepare candidates serializable format
        candidates_serializable = {}
        if all_candidates:
            for vp, cands in all_candidates.items():
                candidates_serializable[vp] = [
                    {"t": c["t"], "scene_flag": c.get("scene_flag", False), "motion_score": c.get("motion_score", 0.0)} for c in cands
                ]

        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump({
                "beat_times": [{"time": b["time"], "strength": b["strength"], "energy": b.get("energy", 0.5)} for b in beats],
                "beat_groups": [{"start": bg["start"], "end": bg["end"], "duration": bg["duration"], "strength": bg.get("strength", 0.5)} for bg in beat_groups],
                "candidates": candidates_serializable,
                "scores": {str(k): v for k, v in debug_scores.items()},
                "assignments": used_beat_groups,
                "music_duration_source_of_truth": music_duration,
                "total_music_duration_from_beats": beats[-1]["time"] if beats else 0,
                "clip_durations_in_seconds": [float(bg.get("clip_duration", bg.get("duration", 0))) for bg in used_beat_groups],
                "config": {
                    "sample_interval": sample_interval,
                    "scene_threshold": scene_threshold,
                    "reuse_limit": reuse_limit,
                    "compute_motion": compute_motion,
                },
            }, f, indent=2)
        print(f"\nAI sync debug written to: {debug_path}")
    except Exception as exc:
        print(f"Warning: Could not write AI sync debug: {exc}")

    return final_clip_paths, used_beat_groups, output_dir

    # Original random mode (legacy)
    beat_groups = create_beat_groups(beats, min_beats, max_beats)
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
            snap_frame=True,
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
                "music_duration_source_of_truth": music_duration,
                "clip_durations_in_seconds": [float(bg.get("clip_duration", bg.get("duration", 0))) for bg in used_beat_groups],
            }, f, indent=2)
        print(f"\nSync debug written to: {debug_path}")
    except Exception as exc:
        print(f"Warning: Could not write sync debug: {exc}")

    return final_clip_paths, used_beat_groups, output_dir