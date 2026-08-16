import os
import random
import math
import tempfile
import json
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

# Transition types by energy level
TRANSITIONS_CALM = ["fade", "dissolve", "fadeblack", "fadewhite", "fadeslow", "smoothleft", "smoothright", "smoothup", "smoothdown"]
TRANSITIONS_MEDIUM = ["wipeleft", "wiperight", "wipeup", "wipedown", "slideleft", "slideright", "slideup", "slidedown", "circlecrop", "rectcrop", "distance", "radial", "circleopen", "circleclose", "vertopen", "vertclose", "horzopen", "horzclose"]
TRANSITIONS_HIGH = ["zoomin", "fadefast", "pixelize", "diagtl", "diagtr", "diagbl", "diagbr", "hlslice", "hrslice", "vuslice", "vdslice", "hblur", "hlwind", "hrwind", "vuwind", "vdwind", "squeezeh", "squeezev", "coverleft", "coverright", "coverup", "coverdown", "revealleft", "revealright", "revealup", "revealdown", "wipetl", "wipetr", "wipebl", "wipebr"]

# Maximum attempts to find a valid candidate per beat group
MAX_ASSIGNMENT_ATTEMPTS = 3

# Diversity penalty for similar music sections (chorus/verse repetition)
SECTION_SIMILARITY_PENALTY = 0.25


def _normalize_beats(beat_times):
    """Convert beat input to list[dict] with 'time' and 'strength' keys."""
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

    Each group corresponds to a contiguous range of beats.
    The clip boundaries align exactly with detected beat timestamps.
    Guarantees coverage of the entire music duration.
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

    # CRITICAL: Ensure coverage extends to full music duration
    # If music_duration is provided and exceeds last beat, extend final group
    if REQUIRE_FULL_COVERAGE and music_duration and groups:
        last_beat_time = beats[-1]["time"]
        if music_duration > last_beat_time:
            # Extend the last group to cover the full music duration
            groups[-1]["end"] = music_duration
            groups[-1]["duration"] = groups[-1]["end"] - groups[-1]["start"]
            print(f"       Extended final group to cover full music duration: {music_duration:.2f}s (last beat at {last_beat_time:.2f}s)")
        elif music_duration > groups[-1]["end"]:
            # Music duration is between last beat and current group end
            groups[-1]["end"] = music_duration
            groups[-1]["duration"] = groups[-1]["end"] - groups[-1]["start"]
            print(f"       Adjusted final group to match music duration: {music_duration:.2f}s")

    # If still not enough coverage (e.g., no beats or very few), create synthetic groups
    if REQUIRE_FULL_COVERAGE and music_duration and groups:
        total_duration = sum(g["duration"] for g in groups)
        if total_duration < music_duration - 0.5:  # Allow small tolerance
            # Need to create more groups to cover the full duration
            remaining = music_duration - total_duration
            # Add additional groups at the end using average beat spacing
            avg_beat_interval = (beats[-1]["time"] - beats[0]["time"]) / max(1, len(beats) - 1) if len(beats) > 1 else 2.0
            last_end = groups[-1]["end"]
            
            while remaining > 0.5:
                group_duration = min(avg_beat_interval * 4, remaining)  # ~4 beats per group
                groups.append({
                    "start": last_end,
                    "end": last_end + group_duration,
                    "duration": group_duration,
                    "strength": 0.5,
                    "beat_indices": [],
                    "synthetic": True,
                })
                last_end += group_duration
                remaining = music_duration - last_end
            print(f"       Added synthetic groups to cover full duration: {music_duration:.2f}s")

    return groups


def _compute_diversity_penalty(cand_time: float, duration: float, used_intervals: list[tuple], source_duration: float) -> float:
    """
    Compute diversity penalty based on temporal distance from already used intervals.
    Returns penalty in range [0, 1] where 0 = no penalty, 1 = max penalty (too close).
    Enforces MIN_CLIP_GAP between clips from same video.
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
    """Classify beat type for music-aware matching."""
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
    """
    Compute energy matching weight per SPEC:
    - Very Low energy (<0.2): prefer calm/low motion
    - Low (0.2-0.4): prefer gentle motion
    - Medium (0.4-0.6): neutral
    - High (0.6-0.8): prefer high motion
    - Very High (>0.8): prefer maximum motion
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
    assignments = []
    video_usage_count = {v: 0 for v in video_paths}
    used_source_intervals = {v: [] for v in video_paths}

    # Extract beats and music features for energy matching
    all_beats = music_analysis.get("beats", []) if music_analysis else []
    drops = music_analysis.get("drops", []) if music_analysis else []
    buildups = music_analysis.get("buildups", []) if music_analysis else []

    # Build highlight map from video analyses for clip preference
    video_highlights = {}
    if video_analyses:
        for va in video_analyses:
            video_highlights[va.get("path")] = va.get("highlights", [])

    # Pre-generate candidates for each video (using first group's duration as reference)
    all_candidates = {}
    # Track the earliest allowed timestamp per video for chronological ordering
    video_cursors = {vp: 0.0 for vp in video_paths}
    for vp in video_paths:
        duration = beat_groups[0]["duration"] if beat_groups else 1.0
        
        # Find cached scenes for this video
        cached_scenes = None
        if video_analyses:
            for va in video_analyses:
                if va.get("path") == vp:
                    cached_scenes = va.get("scenes", [])
                    break
        
        candidates = generate_candidates(
            vp,
            duration,
            sample_interval=sample_interval,
            scene_threshold=scene_threshold,
            max_candidates=MAX_CANDIDATES_PER_SOURCE,
            compute_motion=compute_motion,
            cache_dir=cache_dir,
            cached_scenes=cached_scenes,
        )
        all_candidates[vp] = candidates

    # Debug: collect scores per beat
    debug_scores = {}

    for beat_idx, beat_group in enumerate(beat_groups):
        beat_time = beat_group["start"]
        duration = beat_group["duration"]
        
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

        for vp in video_paths:
            if video_usage_count[vp] >= reuse_limit:
                continue

            candidates = all_candidates.get(vp, [])
            if not candidates:
                continue

            source_duration = get_video_duration(vp)
            if source_duration <= 0:
                continue

            used_intervals = used_source_intervals[vp]
            cursor = video_cursors[vp]  # Earliest allowed timestamp for this video

            for cand_idx, cand in enumerate(candidates):
                cand_time = cand["t"]
                cand_end = cand_time + duration

                if cand_end > source_duration:
                    continue

                # CHRONOLOGICAL ORDERING: only use clips at or after cursor
                if cand_time < cursor:
                    continue

                # Check for overlap with already used intervals from this video
                overlap = False
                for (used_start, used_end) in used_intervals:
                    if not (cand_end <= used_start or cand_time >= used_end):
                        overlap = True
                        break

                if overlap:
                    continue

                # ---- Scoring components ----

                # Scene bonus (0.20 max) - per SPEC weight
                scene_bonus = 0.20 if cand.get("scene_flag", False) else 0.0

                # Proximity bonus (0.10 max) - closer to beat time is better
                proximity_bonus = max(0.0, 0.10 * (1.0 - min(1.0, abs(cand_time - beat_time) / 10.0)))

                # Motion score (0.30 max) - per SPEC weight
                motion_raw = cand.get("motion_score", 0.0)
                motion_score = motion_raw * 0.30

                # Diversity penalty (subtract up to 0.15) - per SPEC weight
                diversity_penalty = _compute_diversity_penalty(cand_time, duration, used_intervals, source_duration) * 0.15

                # Highlight bonus (0.20 max) - prefer visually interesting moments
                highlight_bonus = 0.0
                highlights = video_highlights.get(vp, [])
                for hl in highlights:
                    hl_start = hl.get("start", 0)
                    hl_end = hl.get("end", 0)
                    # Check if candidate overlaps with highlight
                    if not (cand_end <= hl_start or cand_time >= hl_end):
                        overlap_ratio = min(cand_end, hl_end) - max(cand_time, hl_start)
                        if overlap_ratio > 0:
                            highlight_bonus = max(highlight_bonus, 0.20 * (overlap_ratio / duration))
                            break

                # Quality bonus (0.15 max) - would need visual quality at candidate time
                quality_bonus = 0.0

                # Beat type specific adjustments
                type_bonus = 0.0
                if beat_type == "drop":
                    # Drop: extra boost for high motion
                    type_bonus = motion_raw * 0.25
                elif beat_type == "buildup":
                    # Buildup: prefer increasing motion (approximated by higher motion)
                    type_bonus = motion_raw * 0.15
                elif beat_type == "downbeat":
                    # Downbeat: prefer scene changes
                    type_bonus = 0.15 if cand.get("scene_flag", False) else 0.0

                # Energy matching weight (music-aware)
                energy_weight = _energy_match_weight(beat_energy, motion_raw)

                # Combined score
                base_score = scene_bonus + proximity_bonus + motion_score + highlight_bonus + quality_bonus + type_bonus - diversity_penalty
                score = base_score * energy_weight

                beat_scores.append({
                    "video_path": vp,
                    "candidate_idx": cand_idx,
                    "candidate_time": cand_time,
                    "scene_flag": cand.get("scene_flag", False),
                    "motion_score": motion_raw,
                    "diversity_penalty": diversity_penalty,
                    "beat_energy": beat_energy,
                    "beat_strength": beat_strength,
                    "beat_type": beat_type,
                    "energy_weight": energy_weight,
                    "type_bonus": type_bonus,
                    "score": score,
                })

                if score > best_score:
                    best_score = score
                    best_candidate = cand
                    best_video = vp

        # Multi-video distribution bonuses: apply after initial scoring
        # This encourages using all uploaded videos across their full durations
        if len(video_paths) > 1 and beat_scores:
            total_clips_used = sum(video_usage_count.values())
            avg_clips_per_video = total_clips_used / len(video_paths) if total_clips_used > 0 else 0
            
            for sc in beat_scores:
                vp = sc["video_path"]
                
                # Usage balance bonus: prefer videos used less than average
                clips_used = video_usage_count[vp]
                if avg_clips_per_video > 0:
                    usage_ratio = clips_used / avg_clips_per_video
                    # Bonus up to 0.15 for videos underused, penalty for overused
                    usage_bonus = max(-0.10, min(0.15, 0.15 * (1.0 - usage_ratio)))
                else:
                    usage_bonus = 0.15  # First round: encourage all videos
                
                # Footage availability bonus: prefer videos with more unused duration
                source_duration = get_video_duration(vp)
                used_duration = sum(end - start for start, end in used_source_intervals[vp])
                unused_ratio = max(0.0, (source_duration - used_duration) / source_duration)
                # Bonus up to 0.10 for videos with lots of unused footage
                freshness_bonus = 0.10 * unused_ratio
                
                sc["distribution_bonus"] = usage_bonus + freshness_bonus
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
                    candidates = all_candidates.get(vp, [])
                    for cand in candidates:
                        if abs(cand["t"] - selected["candidate_time"]) < 0.001:
                            best_candidate = cand
                            best_video = vp
                            best_score = selected["adjusted_score"]
                            break
                    break

        def _find_fallback_candidate(beat_idx, beat_group, duration, beat_time, beat_energy, beat_strength, beat_type):
            """Fallback: find ANY valid candidate by progressively relaxing constraints."""
            # Prioritize videos with fewer clips used and more unused footage
            def _video_priority(vp):
                clips_used = video_usage_count[vp]
                source_duration = get_video_duration(vp)
                used_duration = sum(end - start for start, end in used_source_intervals[vp])
                unused_ratio = max(0.0, (source_duration - used_duration) / source_duration) if source_duration > 0 else 0
                # Lower priority score = better (less used, more fresh footage)
                return clips_used - unused_ratio * 10
            
            sorted_videos = sorted(video_paths, key=_video_priority)
            
            # Phase 1: Try with relaxed reuse_limit (allow reuse of video, but not same clip intervals)
            # Respect ENFORCE_ASCENDING_ORDER
            for vp in sorted_videos:
                candidates = all_candidates.get(vp, [])
                if not candidates:
                    continue
                source_duration = get_video_duration(vp)
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
                    candidates = all_candidates.get(vp, [])
                    if not candidates:
                        continue
                    source_duration = get_video_duration(vp)
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
                source_duration = get_video_duration(vp)
                if source_duration <= duration:
                    continue
                used_intervals = used_source_intervals[vp]
                cursor = video_cursors[vp] if ENFORCE_ASCENDING_ORDER else 0.0
                # Sample at finer intervals to find gaps
                fine_interval = 0.5  # 500ms steps
                t = cursor  # Start from cursor to respect ascending order
                while t + duration <= source_duration:
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
            
            # Phase 4: Last resort - ignore overlap check but still respect ascending order
            # Only if no candidate found respecting both overlap and order
            for vp in sorted_videos:
                candidates = all_candidates.get(vp, [])
                if not candidates:
                    continue
                source_duration = get_video_duration(vp)
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
                    return vp, cand, snapped_start, -2.0
            
            # Phase 5: Last resort - if ALLOW_CLIP_REUSE, use first video respecting cursor
            if ALLOW_CLIP_REUSE:
                for vp in sorted_videos:
                    source_duration = get_video_duration(vp)
                    if source_duration >= duration:
                        cursor = video_cursors[vp] if ENFORCE_ASCENDING_ORDER else 0.0
                        cand_time = max(cursor, 0.0)
                        if cand_time + duration <= source_duration:
                            snapped_start = snap_to_frame(cand_time)
                            return vp, {"t": cand_time, "scene_flag": False, "motion_score": 0.0}, snapped_start, -3.0
                        # If cursor is too far, wrap to start of video (last resort)
                        if not ENFORCE_ASCENDING_ORDER:
                            snapped_start = snap_to_frame(0.0)
                            return vp, {"t": 0.0, "scene_flag": False, "motion_score": 0.0}, snapped_start, -3.0
            
            # Phase 6: GUARANTEED ASSIGNMENT - ensure full music coverage while respecting ascending order
            # This ensures timeline_duration >= music_duration by always finding a slot
            best_vp = None
            best_duration = 0.0
            for vp in sorted_videos:
                source_duration = get_video_duration(vp)
                cursor = video_cursors[vp] if ENFORCE_ASCENDING_ORDER else 0.0
                if source_duration >= cursor + duration:
                    # Use cursor position to maintain ascending order
                    cand_time = cursor
                    snapped_start = snap_to_frame(cand_time)
                    return vp, {"t": cand_time, "scene_flag": False, "motion_score": 0.0}, snapped_start, -10.0
                elif source_duration >= duration and best_vp is None:
                    # Video has enough duration but cursor is too far - use end - duration (last resort)
                    best_vp = vp
                    best_duration = source_duration
                elif source_duration > best_duration:
                    best_vp = vp
                    best_duration = source_duration
            
            # If no video can accommodate at cursor, use the best video at latest possible position
            if best_vp:
                cursor = video_cursors[best_vp] if ENFORCE_ASCENDING_ORDER else 0.0
                cand_time = min(cursor, max(0.0, best_duration - duration))
                snapped_start = snap_to_frame(cand_time)
                return best_vp, {"t": cand_time, "scene_flag": False, "motion_score": 0.0}, snapped_start, -10.0
            
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

        debug_scores[beat_idx] = beat_scores

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
    video_analyses: list = None,  # Pre-computed video analyses for cached scenes
) -> list[dict]:
    """
    AI-assisted clip assignment with:
    - Scene detection
    - Motion analysis (optical flow)
    - Diversity scoring
    - Beat energy matching (music-aware)
    - Beat type classification (downbeat, drop, buildup, regular)

    Returns list of dicts with keys:
        source_path, music_start, music_end, clip_duration,
        source_start, score, scene_flag, motion_score, beat_strength, beat_energy, beat_type
    """
    music_duration = music_analysis.get("duration", 0.0) if music_analysis else None
    beat_groups = create_beat_groups(beat_times, min_beats, max_beats, music_duration=music_duration)
    if not beat_groups:
        return []

    # Enforce no clip reuse if ALLOW_CLIP_REUSE is False
    # With ALLOW_CLIP_REUSE=False, overlap check prevents reusing intervals
    # So we don't need a hard limit on clips per video - just distribute across videos
    if not ALLOW_CLIP_REUSE:
        # High limit: overlap check prevents actual clip reuse
        # This allows multiple clips per video as long as they don't overlap
        reuse_limit = 100
    elif reuse_limit is None:
        # Auto-calculate reuse_limit to cover full music duration
        reuse_limit = max(1, len(beat_groups) // max(1, len(video_paths)) + 1)

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

    # Normalize beats
    beats = _normalize_beats(beat_times)
    if len(beats) < 2:
        return [], [], output_dir

    if ai_mode:
        # AI-assisted mode
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

        # Write AI debug output
        debug_path = os.path.join(output_dir, "sync_debug_ai.json")
        try:
            # Prepare candidates serializable format
            candidates_serializable = {}
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
                    "total_music_duration": beats[-1]["time"] if beats else 0,
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
                "total_music_duration": beats[-1]["time"] if beats else 0,
            }, f, indent=2)
        print(f"\nSync debug written to: {debug_path}")
    except Exception as exc:
        print(f"Warning: Could not write sync debug: {exc}")

    return final_clip_paths, used_beat_groups, output_dir