# Full AI-Powered Automatic Video Editing System - Specification

## Overview

This document specifies the complete implementation of an AI-powered automatic video editing system that works directly with full-length source videos and music tracks - no pre-cut clips required.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BEATS VIDEO EDITOR PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐   │
│  │  Upload  │───▶│   Music      │───▶│  Beat &      │───▶│   Beat     │   │
│  │  Videos  │    │  Analysis    │    │  Energy      │    │  Timeline  │   │
│  └──────────┘    └──────────────┘    │  Detection   │    └─────┬──────┘   │
│       │                                └──────┬───────┘          │         │
│       │                                       │                  │         │
│       ▼                                       ▼                  ▼         │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐   │
│  │  Upload  │───▶│   Video      │───▶│  Scene &     │───▶│  Auto Clip │   │
│  │  Music   │    │  Analysis    │    │  Motion      │    │  Discovery │   │
│  └──────────┘    └──────────────┘    │  Detection   │    └─────┬──────┘   │
│       │                                └──────┬───────┘          │         │
│       │                                       │                  │         │
│       ▼                                       ▼                  ▼         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    CLIP SCORING & MATCHING ENGINE                    │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────────┐  │   │
│  │  │   Motion    │ │   Scene     │ │  Diversity  │ │  Visual      │  │   │
│  │  │   Score     │ │   Score     │ │   Score     │ │  Quality     │  │   │
│  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬───────┘  │   │
│  │         │               │               │               │           │   │
│  │         └───────────────┼───────────────┼───────────────┘           │   │
│  │                         ▼               ▼                           │   │
│  │              ┌─────────────────────────┐                            │   │
│  │              │  Music-Aware Matching   │                            │   │
│  │              │  (Energy ↔ Visual)      │                            │   │
│  │              └───────────┬─────────────┘                            │   │
│  │                          ▼                                           │   │
│  │              ┌─────────────────────────┐                            │   │
│  │              │  Final Timeline Build   │                            │   │
│  │              └───────────┬─────────────┘                            │   │
│  └──────────────────────────┼──────────────────────────────────────────┘   │
│                             ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
  │                    RENDERING (FFmpeg)                                  │   │
  │  Frame-Accurate Trimming → Concatenation → Transitions → Audio Mix    │   │
  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Stage 1: Music Understanding

### Input
- Audio file (MP3, WAV, M4A, AAC)

### Output: `MusicAnalysis`
```python
{
    "bpm": float,
    "beats": [
        {"time": float, "strength": float, "energy": float}
    ],
    "drops": [{"time": float, "intensity": float}],
    "buildups": [{"start": float, "end": float, "intensity": float}],
    "sections": [
        {"label": "intro|verse|chorus|drop|bridge|outro", "start": float, "end": float, "energy": float}
    ],
    "energy_curve": [(time, energy), ...],  # Normalized 0-1
    "duration": float
}
```

### Implementation Details
- Use `librosa` for beat tracking, onset detection, spectral features
- Beat strength from onset strength envelope
- Energy curve from RMS energy in sliding windows
- Drop detection: sudden energy increase + spectral flux peak
- Buildup detection: sustained energy increase over 4+ beats
- Section detection: self-similarity matrix + energy profile clustering

## Stage 2: Source Video Analysis

### Input
- List of video file paths

### Output per video: `VideoAnalysis`
```python
{
    "path": str,
    "duration": float,
    "fps": float,
    "width": int,
    "height": int,
    "scenes": [
        {"start": float, "end": float, "score": float, "type": "cut|gradual"}
    ],
    "motion_profile": [(time, motion_magnitude), ...],  # Normalized 0-1
    "highlights": [
        {"start": float, "end": float, "score": float, "type": "action|face|motion|scene_change"}
    ],
    "camera_changes": [float],  # timestamps
    "visual_quality": [(time, sharpness, brightness, contrast), ...],
    "dominant_colors": [(time, [r,g,b]), ...]
}
```

### Implementation Details
- Scene detection: FFmpeg `select='gt(scene,threshold)'` filter
- Motion analysis: Optical flow (Farneback) at 2 FPS sampling
- Highlight detection: Combine motion peaks, scene boundaries, face detection (optional)
- Camera change detection: Sudden color histogram shifts
- Visual quality: Laplacian variance (sharpness), mean brightness, RMS contrast
- Process in streaming fashion - never load full video into memory

## Stage 3: Automatic Clip Discovery

### Input
- `VideoAnalysis` for each video
- Target clip durations from beat timeline

### Output: `CandidateClips`
```python
[
    {
        "video_path": str,
        "source_start": float,
        "source_end": float,
        "duration": float,
        "scene_score": float,      # 0-1, high at scene boundaries
        "motion_score": float,     # 0-1, average motion in clip
        "quality_score": float,    # 0-1, visual quality
        "diversity_features": {...},  # For diversity scoring
        "highlight_score": float,  # 0-1, if contains highlight
        "camera_change": bool,     # True if clip contains camera cut
    },
    ...
]
```

### Generation Strategy
1. **Scene-anchored candidates**: Start clips at detected scene boundaries
2. **Uniform sampling**: Sample every N seconds as fallback
3. **Highlight-anchored**: Start clips at detected highlight moments
4. **Duration matching**: Generate candidates for each target duration needed
5. **Pruning**: Remove candidates too close to video end, deduplicate
6. **Cap**: Max 200 candidates per source video

## Stage 4: Clip Scoring

### Scoring Components (0-1 each)

| Component | Weight | Description |
|-----------|--------|-------------|
| Motion | 0.30 | Average optical flow magnitude in clip |
| Scene | 0.20 | 1.0 if clip starts at scene boundary, decayed |
| Highlight | 0.20 | Highest highlight score within clip window |
| Quality | 0.15 | Sharpness × brightness × contrast balance |
| Diversity | 0.15 | Penalty for similar content to already-used clips |

### Diversity Scoring
- Track used time intervals per source video
- Penalty increases as candidate overlaps or is near used intervals
- Visual feature similarity (color histogram, motion pattern) for cross-video diversity

## Stage 5: Music-Aware Clip Selection

### Energy Matching Rules

| Beat Energy | Visual Preference | Scoring Adjustment |
|-------------|-------------------|-------------------|
| Very Low (<0.2) | Calm, static, slow motion | Boost low-motion clips |
| Low (0.2-0.4) | Gentle motion | Slight boost to medium motion |
| Medium (0.4-0.6) | Moderate motion | Neutral |
| High (0.6-0.8) | High motion, action | Boost high-motion clips |
| Very High (>0.8) | Peak action, drops | Maximum boost to highest motion |

### Beat Classification
- **Downbeat** (strong): First beat of measure - prefer scene changes
- **Regular beat**: Standard matching
- **Drop beat**: Highest energy - best clip available
- **Buildup beat**: Rising energy - prefer increasing motion

### Assignment Algorithm
1. Create beat groups (4-8 beats each) from music analysis
2. For each group, determine target duration and energy profile
3. Score all candidates against group using energy-weighted scoring
4. Greedy assignment with constraints:
   - Max 2 clips per source video (configurable)
   - No overlapping intervals from same source
   - Prefer temporal diversity across sources
5. Snap all start times to frame boundaries

## Stage 6: Timeline Generation

### Output: `Timeline`
```python
[
    {
        "beat_group_idx": int,
        "music_start": float,
        "music_end": float,
        "duration": float,
        "video_path": str,
        "source_start": float,
        "source_end": float,
        "score": float,
        "scene_flag": bool,
        "motion_score": float,
        "beat_energy": float,
        "beat_type": "downbeat|regular|drop|buildup"
    },
    ...
]
```

## Stage 7: Frame-Accurate Trimming

### Function
```python
def snap_to_frame(time_s: float, fps: int = 30) -> float:
    """Snap timestamp to nearest frame boundary."""
    return round(time_s * fps) / fps
```

### Applied to:
- Clip start times (source_start)
- Clip end times (source_start + duration)
- Transition offsets

### FFmpeg Command Pattern
```bash
ffmpeg -ss {snapped_start:.3f} -i input -t {snapped_duration:.3f} -vf "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1,fps=30" -c:v h264_qsv -global_quality 20 -an output.mp4
```

## Stage 8: Rendering

### Pipeline
1. **Process individual clips** - Trim, scale, encode (QSV preferred, CPU fallback)
2. **Create concat groups** - Group clips (4-8 per group)
3. **Add transitions** - xfade between groups (fade, fadeblack, etc.)
4. **Concatenate groups** - Final video without audio
5. **Low-RAM re-encode** - Optional chunked encoding for quality
6. **Add audio** - Mix music track, loop if needed, -shortest

### Memory Optimization
- Stream frames through FFmpeg pipes
- Never load full video into Python memory
- Chunked encoding: split → encode sequentially → concat
- Cache motion analysis results per video

## Frontend Workflow (Streamlit)

### Inputs (Only These!)
```
┌─────────────────────────────────────┐
│  📹 Upload Videos (multiple)        │
│  🎵 Upload Music                    │
│                                     │
│  [ Generate AI Video ]              │
└─────────────────────────────────────┘
```

### Removed from UI:
- Video folder path input
- Min/max beats per clip
- Resolution/quality selectors (use defaults)
- Transition controls
- AI matching toggle (always on)
- Low-RAM encoding toggle (always on)
- All advanced settings

### Progress Display
```
[1/8] Analyzing music... ████████░░ 45%
[2/8] Detecting beats... ██████████ 100%
[3/8] Analyzing videos... ██████░░░░ 60%
[4/8] Detecting scenes... ████████░░ 80%
[5/8] Generating clips... ██████████ 100%
[6/8] Matching to beats... ████████░░ 75%
[7/8] Building timeline... ██████████ 100%
[8/8] Rendering final... ███████░░░ 70%
```

### Output
- Final video download button
- Debug report download (JSON)
- Processing log

## Debug Report: `output/debug_report.json`

```json
{
  "music_analysis": { ... },
  "video_analyses": [ { ... }, ... ],
  "candidate_clips": [ { ... }, ... ],
  "clip_scores": [ { ... }, ... ],
  "timeline": [ { ... }, ... ],
  "render_info": {
    "total_clips": int,
    "total_duration": float,
    "source_videos_used": int,
    "encoding_time": float
  },
  "config": {
    "target_resolution": "1920x1080",
    "target_fps": 30,
    "max_clips_per_source": 2,
    "transition_duration": 0.35
  }
}
```

## Configuration Constants

```python
# Video settings
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
TARGET_FPS = 30

# Analysis settings
SCENE_THRESHOLD = 0.3
SAMPLE_INTERVAL = 0.5
MOTION_SAMPLE_FPS = 2
MAX_CANDIDATES_PER_SOURCE = 200
MAX_CLIPS_PER_SOURCE = 2

# Clip duration bounds (beats)
MIN_BEATS_PER_CLIP = 4
MAX_BEATS_PER_CLIP = 8

# Transitions
TRANSITION_DURATION = 0.35
TRANSITION_TYPES = ["fade", "fadeblack"]

# Encoding
QSV_QUALITY = 20
CPU_CRF = 20
CPU_PRESET = "veryfast"
CHUNK_SIZE_SECONDS = 60
```

## Error Handling

- Graceful degradation: if music analysis fails, use basic beat detection
- If video analysis fails for one source, skip it and continue
- If clip generation fails, try next best candidate
- If FFmpeg QSV fails, fallback to CPU encoding
- Always cleanup temp directories on exit

## Testing Criteria

1. **End-to-end test**: Upload 2+ videos + music → get beat-synced output
2. **No manual clips**: System never asks for pre-cut clips
3. **Beat alignment**: Visual cuts align to beats within 1 frame (33ms at 30fps)
4. **Energy matching**: Drop sections get highest motion clips
5. **Diversity**: No source video dominates; no repeated scenes
6. **Memory**: Peak RAM < 2GB for 10-min sources
7. **Speed**: < 5 min for 3 min output from 10 min sources