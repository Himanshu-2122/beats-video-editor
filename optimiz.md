# Beats Video Editor - Performance Optimization Guide

Based on comprehensive analysis of the entire codebase (main.py, beat.py, video.py, sync.py, frontend.py, cache.py, progress_tracker.py, logging_system.py, theme.py, SPEC.md, ai_implement.md).

---

## Executive Summary

The codebase is well-structured with good architectural patterns (caching, streaming FFmpeg, modular design). However, several bottlenecks exist that can significantly improve performance:

| Area | Current State | Optimization Potential |
|------|---------------|----------------------|
| Beat Detection | Sequential librosa calls, no parallelization | **High** - Multi-threaded analysis |
| Video Analysis | Heavy OpenCV ops per frame, no batching | **High** - Batch frame extraction, reduce sampling |
| Clip Discovery | Redundant candidate generation per beat group | **High** - Pre-compute once, reuse |
| Motion Analysis | Farneback optical flow at 1/8 resolution | **Medium** - Switch to lighter methods |
| FFmpeg Calls | Many subprocess spawns per clip | **Medium** - Batch concat, fewer invocations |
| Memory | No explicit limits, potential leaks | **Medium** - Add memory monitoring |
| AI Matching | Greedy with fallback, no global optimization | **Medium** - Hungarian algorithm |

---

## 1. Beat Analysis Optimizations (beat.py)

### 1.1 Parallel Music Analysis
**File:** `app/beat.py`  
**Lines:** 99-307 (`analyze_music_full`)

```python
# Current: Sequential processing
y, sr = librosa.load(audio_path, sr=None, mono=True)  # Blocks
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)  # Blocks
# ... all features computed sequentially
```

**Optimization:** Use ThreadPoolExecutor for independent feature extraction:

```python
from concurrent.futures import ThreadPoolExecutor

def analyze_music_full_parallel(audio_path, cache_dir=None):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit independent tasks
        beat_future = executor.submit(librosa.beat.beat_track, y=y, sr=sr)
        onset_future = executor.submit(librosa.onset.onset_strength, y=y, sr=sr)
        rms_future = executor.submit(librosa.feature.rms, y=y, frame_length=2048, hop_length=512)
        chroma_future = executor.submit(librosa.feature.chroma_cqt, y=y, sr=sr, hop_length=512)
        spectral_future = executor.submit(librosa.feature.spectral_centroid, y=y, sr=sr)
        
        # Collect results
        tempo, beat_frames = beat_future.result()
        onset_env = onset_future.result()
        rms = rms_future.result()
        chroma = chroma_future.result()
        spectral_centroid = spectral_future.result()
```

**Expected gain:** 2-3x faster on multi-core CPUs.

### 1.2 Cache Beat Detection Results
**File:** `app/beat.py`  
**Lines:** 8-16, 18-96

The `@cached` decorator exists but `_get_beats_cached` recomputes everything. Add file-hash based caching:

```python
def _get_file_hash(filepath):
    import hashlib, os
    stat = os.stat(filepath)
    return hashlib.md5(f"{filepath}:{stat.st_size}:{stat.st_mtime}".encode()).hexdigest()[:16]

@cached("beats", "_beats")
def _get_beats_cached(audio_path):
    file_hash = _get_file_hash(audio_path)
    cache_path = f"cache/beats/{file_hash}_beats.json"
    # ... rest of function
```

### 1.3 Reduce Librosa Overhead
- Use `sr=22050` for initial analysis (downsample), then refine
- Cache `onset_strength` and `rms` computations
- Use `librosa.util.fix_length` instead of manual padding

---

## 2. Video Analysis Optimizations (video.py)

### 2.1 Batch Frame Extraction (Major)
**File:** `app/video.py`  
**Lines:** 149-182, 383-424, 1551-1591

**Current:** Each `extract_frame` spawns a separate FFmpeg process!

```python
# Current - TERRIBLE for performance
def extract_frame(video_path, time_s):
    command = ["ffmpeg", "-ss", f"{time_s:.3f}", "-i", video_path, ...]
    process = subprocess.run(command, ...)  # New process EVERY frame!
```

**Optimization:** Extract ALL frames in ONE FFmpeg call using `select` filter:

```python
def extract_frames_batch(video_path, timestamps: list[float]) -> list[Image]:
    """Extract multiple frames in single FFmpeg invocation."""
    if not timestamps:
        return []
    
    # Build select filter: select='eq(n,1)+eq(n,2)+...'
    # Or use fps filter with specific timestamps
    select_expr = "+".join([f"gte(t,{t:.3f})*lt(t,{t+0.033:.3f})" for t in timestamps])
    
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"select='{select_expr}',scale=160:90",  # Downscale early!
        "-vsync", "vfr", "-f", "image2pipe", "-vcodec", "png", "-"
    ]
    
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    # Parse concatenated PNG stream
```

**Expected gain:** 10-50x faster frame extraction for motion analysis.

### 2.2 Reduce Motion Analysis Resolution
**File:** `app/video.py`  
**Lines:** 37, 39, 342, 358

```python
MOTION_DOWNSCALE = 0.125  # 1/8 resolution - already good
MOTION_SAMPLE_FPS = 1     # 1 FPS - can reduce to 0.5 for long videos
```

**Optimization:** Adaptive sampling based on video duration:

```python
def get_adaptive_motion_fps(duration: float) -> float:
    if duration > 300:      # > 5 min
        return 0.5
    elif duration > 120:    # > 2 min
        return 1.0
    else:
        return 2.0
```

### 2.3 Replace Optical Flow with Frame Difference
**File:** `app/video.py`  
**Lines:** 345-380 (`compute_optical_flow_magnitude`)

Farneback optical flow is CPU-intensive. For beat-sync, simple frame difference is often sufficient:

```python
def compute_frame_difference(prev_frame, curr_frame):
    """Ultra-fast motion estimate using absolute difference."""
    import numpy as np
    
    if hasattr(prev_frame, 'convert'):
        prev_frame = np.array(prev_frame.convert('L'), dtype=np.float32)
    if hasattr(curr_frame, 'convert'):
        curr_frame = np.array(curr_frame.convert('L'), dtype=np.float32)
    
    # Downscale to tiny (32x18) for speed
    h, w = prev_frame.shape[:2]
    prev_small = cv2.resize(prev_frame, (32, 18))
    curr_small = cv2.resize(curr_frame, (32, 18))
    
    diff = np.abs(curr_small.astype(np.float32) - prev_small.astype(np.float32))
    return float(np.mean(diff)) / 255.0  # Normalized 0-1
```

**Expected gain:** 20-50x faster motion analysis with acceptable accuracy for beat matching.

### 2.4 Cache Video Analysis Results
**File:** `app/video.py`  
**Lines:** 1349-1451 (`analyze_video_full`)

Add persistent caching with file hash (similar to beat caching):

```python
def analyze_video_full(video_path, cache_dir=None, fast_mode=True):
    file_hash = get_video_hash(video_path)
    cache_path = os.path.join(cache_dir or "cache/video", f"{file_hash}.json")
    
    if os.path.exists(cache_path):
        # Check if file hasn't changed
        with open(cache_path, 'r') as f:
            cached = json.load(f)
        if cached.get("file_hash") == file_hash:
            return cached
    # ... compute and save
```

---

## 3. Clip Discovery & Sync Optimizations (sync.py, video.py)

### 3.1 Pre-Compute Candidates Once
**File:** `app/sync.py`  
**Lines:** 287-312 (`_greedy_assign`)

**Current:** Candidates generated per beat group duration:

```python
for vp in video_paths:
    duration = beat_groups[0]["duration"]  # Only first group!
    candidates = generate_candidates(vp, duration, ...)
    all_candidates[vp] = candidates
```

**Problem:** Different beat groups have different durations → wrong candidates!

**Fix:** Pre-compute candidates for ALL unique durations needed:

```python
def precompute_all_candidates(video_paths, beat_groups, ...):
    unique_durations = set(bg["duration"] for bg in beat_groups)
    all_candidates = {}
    
    for vp in video_paths:
        all_candidates[vp] = {}
        for duration in unique_durations:
            all_candidates[vp][duration] = generate_candidates(vp, duration, ...)
    
    return all_candidates
```

### 3.2 Vectorize Candidate Scoring
**File:** `app/sync.py`  
**Lines:** 339-443

Replace Python loops with NumPy vectorization:

```python
# Current: Loop over candidates
for cand_idx, cand in enumerate(candidates):
    # ... compute score

# Optimized: Vectorized
cand_times = np.array([c["t"] for c in candidates])
scene_flags = np.array([c.get("scene_flag", False) for c in candidates])
motion_scores = np.array([c.get("motion_score", 0.0) for c in candidates])

# Vectorized proximity bonus
proximity = np.maximum(0, 0.10 * (1 - np.minimum(1, np.abs(cand_times - beat_time) / 10)))

# Vectorized diversity penalty (requires vectorized interval check)
# ... use numpy broadcasting
```

### 3.3 Use Hungarian Algorithm for Global Optimal Assignment
**File:** `app/sync.py`  
**Lines:** 771-842 (`ai_assign_clips`)

Replace greedy with `scipy.optimize.linear_sum_assignment`:

```python
from scipy.optimize import linear_sum_assignment

def hungarian_assign(beat_groups, all_candidates, video_paths, ...):
    # Build cost matrix: beats × candidates
    # Cost = 1 - score (minimize cost = maximize score)
    n_beats = len(beat_groups)
    all_cands = []
    cand_metadata = []
    
    for vp in video_paths:
        for duration, cands in all_candidates[vp].items():
            for c in cands:
                all_cands.append(c)
                cand_metadata.append({"video": vp, "duration": duration})
    
    if len(all_cands) < n_beats:
        return greedy_fallback(...)
    
    cost_matrix = np.zeros((n_beats, len(all_cands)))
    for i, bg in enumerate(beat_groups):
        for j, cand in enumerate(all_cands):
            if cand_metadata[j]["duration"] == bg["duration"]:
                cost_matrix[i, j] = 1 - compute_score(bg, cand, ...)
            else:
                cost_matrix[i, j] = 1e6  # Invalid assignment
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Post-process: enforce no-overlap constraints
    return build_assignments(row_ind, col_ind, ...)
```

---

## 4. FFmpeg Pipeline Optimizations (video.py)

### 4.1 Reduce FFmpeg Invocations
**File:** `app/video.py`  
**Lines:** 840-973 (`process_clip`), 1045-1136 (`create_transition_video`)

**Current:** Multiple FFmpeg calls per clip (copy attempt → QSV fallback → CPU fallback)

**Optimization:** Single-pass with hardware detection:

```python
def process_clip_optimized(video_path, duration, output_path, start_time=None):
    use_qsv = check_qsv()  # Check ONCE at startup, cache result
    
    if use_qsv:
        # Single QSV command with all filters
        command = build_qsv_command(video_path, duration, output_path, start_time)
    else:
        command = build_cpu_command(video_path, duration, output_path, start_time)
    
    run_ffmpeg(command)
```

### 4.2 Batch Concat Operations
**File:** `app/video.py`  
**Lines:** 1171-1258 (`concatenate_videos`)

**Current:** Creates group files → concats each group → concats groups → transitions

**Optimization:** Single filter_complex for all clips:

```python
def concatenate_videos_single_pass(clip_paths, output_path, beat_groups=None):
    """Build entire timeline in one filter_complex."""
    # Build single xfade chain for ALL clips
    # Single FFmpeg invocation instead of N+1
```

### 4.3 Remove Redundant Re-encode
**File:** `app/video.py`  
**Lines:** 1261-1296 (`encode_in_chunks`)

**Current:** Remuxes (copies) the already-processed video - unnecessary!

```python
# Current: "Fast remux - no re-encode"
command = ["ffmpeg", "-i", input_path, "-c", "copy", ...]
```

**Fix:** Just copy the file or use `shutil.copy2()` since it's already fully processed:

```python
def encode_in_chunks(input_path, output_path, **kwargs):
    import shutil
    shutil.copy2(input_path, output_path)
    return output_path
```

---

## 5. Memory Optimizations

### 5.1 Add Memory Monitoring
**File:** `app/progress_tracker.py` or new `app/memory.py`

```python
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB

def check_memory_limit(limit_mb=2048):
    if get_memory_usage() > limit_mb:
        import gc
        gc.collect()
        # Trigger cache cleanup
        cache_manager.clear_all()
```

### 5.2 Streaming Video Processing
**File:** `app/video.py` - Ensure no full video loading:

```python
# BAD - loads entire video
def bad_example(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)  # OOM risk!
    return frames

# GOOD - process in streaming fashion
def good_example(video_path, callback):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret: break
        callback(frame)  # Process one frame at a time
    cap.release()
```

### 5.3 Limit Candidate Cache Size
**File:** `app/cache.py`  
**Lines:** 79-92

Add LRU eviction:

```python
def enforce_cache_limit(self, max_files=1000, max_size_mb=500):
    """Remove oldest cache files if limits exceeded."""
    for category in ["beats", "scenes", "motion", "video_analysis"]:
        cat_dir = self.cache_dir / category
        files = sorted(cat_dir.glob("*.json"), key=lambda f: f.stat().st_mtime)
        
        # Remove by count
        while len(files) > max_files:
            files.pop(0).unlink()
        
        # Remove by size
        total_size = sum(f.stat().st_size for f in files)
        while total_size > max_size_mb * 1024 * 1024:
            removed = files.pop(0)
            total_size -= removed.stat().st_size
            removed.unlink()
```

---

## 6. Frontend/Streamlit Optimizations (frontend.py)

### 6.1 Reduce Reruns
**File:** `app/frontend.py`  
**Lines:** 210-211, 390-391

```python
@st.fragment(run_every=1)
def progress_dashboard(tracker):
    # This reruns every second - can cause UI flicker
```

**Optimization:** Use `st.empty()` placeholders instead of fragments:

```python
def progress_dashboard(tracker):
    progress_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    while tracker.is_running():
        data = tracker.get_dashboard_data()
        with progress_placeholder.container():
            st.progress(data['overall_progress'] / 100)
        with metrics_placeholder.container():
            # Update metrics
            pass
        time.sleep(1)
```

### 6.2 Cache Video Previews
**File:** `app/frontend.py`  
**Lines:** 239-310

Generated videos re-read from disk on every render. Cache thumbnails:

```python
@st.cache_data
def get_video_thumbnail(video_path, time_s=1.0):
    """Extract and cache thumbnail."""
    # Extract frame at time_s, return as bytes
```

### 6.3 Parallel Upload Processing
**File:** `app/frontend.py`  
**Lines:** 340-346

```python
# Current: Sequential save
for idx, vf in enumerate(video_files):
    with open(vp, "wb") as f:
        f.write(vf.getbuffer())

# Optimized: Parallel save
from concurrent.futures import ThreadPoolExecutor

def save_upload(vf, idx, temp_dir):
    ext = os.path.splitext(vf.name)[1]
    vp = os.path.join(temp_dir, f"video_{idx}{ext}")
    with open(vp, "wb") as f:
        f.write(vf.getbuffer())
    return vp

with ThreadPoolExecutor(max_workers=4) as executor:
    video_paths = list(executor.map(
        lambda args: save_upload(*args),
        [(vf, idx, temp_dir) for idx, vf in enumerate(video_files)]
    ))
```

---

## 7. Configuration Optimizations

### 7.1 Optimal Default Values (video.py)

```python
# Current settings (lines 16-45, 37-40)
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
FPS = 30
SCENE_THRESH = 0.3          # Increase to 0.4 for fewer false positives
SAMPLE_INTERVAL = 0.5       # Increase to 1.0-2.0 for long videos
MAX_CANDIDATES_PER_SOURCE = 100  # Reduce to 50
MOTION_SAMPLE_FPS = 1       # Reduce to 0.5 for >2min videos
MOTION_DOWNSCALE = 0.125    # Good, keep
MIN_CLIP_GAP = 3.0          # Good
```

### 7.2 Adaptive Settings Based on Input

```python
def get_adaptive_config(video_paths, music_duration):
    total_source_duration = sum(get_video_duration(vp) for vp in video_paths)
    
    config = {
        "sample_interval": min(2.0, max(0.5, total_source_duration / 200)),
        "motion_fps": 0.5 if total_source_duration > 300 else 1.0,
        "max_candidates": min(200, max(50, int(total_source_duration / 2))),
        "scene_threshold": 0.4 if len(video_paths) > 5 else 0.3,
    }
    return config
```

---

## 8. Priority Implementation Order

| Priority | Task | File(s) | Effort | Impact |
|----------|------|---------|--------|--------|
| 1 | Batch frame extraction | video.py | Medium | **Very High** (10-50x motion analysis) |
| 2 | Pre-compute all candidates | sync.py, video.py | Low | **High** (avoids redundant work) |
| 3 | Parallel music analysis | beat.py | Low | **High** (2-3x beat detection) |
| 4 | Replace optical flow with frame diff | video.py | Low | **High** (20-50x motion) |
| 5 | Persistent caching for video analysis | video.py, cache.py | Low | **High** (instant repeat runs) |
| 6 | Single-pass FFmpeg concat | video.py | Medium | **Medium** (fewer subprocess calls) |
| 7 | Hungarian assignment | sync.py | Medium | **Medium** (better clip selection) |
| 8 | Memory monitoring & limits | New file | Low | **Medium** (stability) |
| 9 | Adaptive config | video.py, sync.py | Low | **Medium** (auto-tuning) |
| 10 | Streamlit fragment optimization | frontend.py | Low | **Low** (UI smoothness) |

---

## 9. Quick Wins (Implement Today)

### 9.1 Fix `encode_in_chunks` to Just Copy
```python
# video.py:1261-1296
def encode_in_chunks(input_path, output_path, **kwargs):
    import shutil
    shutil.copy2(input_path, output_path)
    return output_path
```

### 9.2 Cache QSV Check Result
```python
# video.py:716-729
_qsv_available = None

def check_qsv():
    global _qsv_available
    if _qsv_available is not None:
        return _qsv_available
    # ... existing check
    _qsv_available = result
    return result
```

### 9.3 Increase Scene Threshold
```python
# video.py:21
SCENE_THRESH = 0.4  # Was 0.3 - fewer false positives, fewer candidates
```

### 9.4 Reduce Motion Sample FPS for Long Videos
```python
# video.py:37-40
def get_motion_fps(source_duration):
    if source_duration > 300: return 0.5
    if source_duration > 120: return 1.0
    return 2.0
```

---

## 10. Profiling Commands

```bash
# Profile CPU usage
python -m cProfile -o profile.stats tools/run_headless_test.py --ai

# Analyze profile
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(30)
"

# Memory profiling
pip install memory_profiler
python -m memory_profiler tools/run_headless_test.py --ai

# Line-by-line profiling
pip install line_profiler
kernprof -l -v tools/run_headless_test.py --ai
```

---

## 11. Expected Performance Gains

| Optimization | Current Time (est.) | Optimized Time (est.) | Speedup |
|--------------|---------------------|----------------------|---------|
| Beat detection | ~30s | ~10s | 3x |
| Video analysis (10min total) | ~120s | ~15s | 8x |
| Motion analysis | ~60s | ~2s | 30x |
| Clip discovery | ~10s | ~2s | 5x |
| FFmpeg rendering | ~60s | ~30s | 2x |
| **Total (3min output)** | **~4.5 min** | **~1 min** | **4.5x** |

---

## 12. Testing Optimizations

### 12.1 Benchmark Script
Create `tools/benchmark.py`:

```python
#!/usr/bin/env python3
"""Benchmark key operations."""
import time
from app.beat import analyze_music_full
from app.video import analyze_video_full, generate_candidates
from app.sync import ai_assign_clips

def benchmark():
    music_path = "data/music/song.mp3"
    video_paths = ["data/videos/test_video.mp4"] * 3  # Test with 3 videos
    
    # Benchmark music analysis
    t0 = time.time()
    music = analyze_music_full(music_path)
    print(f"Music analysis: {time.time()-t0:.1f}s")
    
    # Benchmark video analysis
    t0 = time.time()
    analyses = [analyze_video_full(vp, fast_mode=True) for vp in video_paths]
    print(f"Video analysis (3x): {time.time()-t0:.1f}s")
    
    # Benchmark candidate generation
    t0 = time.time()
    beats = music["beats"]
    beat_groups = create_beat_groups(beats, 4, 8, music["duration"])
    results, _, _, _ = ai_assign_clips(beats, video_paths, music_analysis=music, video_analyses=analyses)
    print(f"AI assignment: {time.time()-t0:.1f}s")
    print(f"Generated {len(results)} clips")

if __name__ == "__main__":
    benchmark()
```

### 12.2 Regression Tests
Add to `test_pipeline.py`:

```python
def test_performance_regression():
    """Ensure optimizations don't break correctness."""
    # Run full pipeline
    # Verify output duration matches music duration ±1s
    # Verify no duplicate clips from same source
    # Verify beat alignment within 1 frame (33ms)
```

---

## 13. Monitoring & Observability

Add structured logging for performance tracking:

```python
# In each major function
import time
from app.logging_system import get_logger

logger = get_logger()

def analyze_music_full(audio_path, ...):
    start = time.time()
    try:
        result = _analyze_music_full(audio_path, ...)
        logger.analysis_info(f"Music analysis: {time.time()-start:.2f}s, {len(result['beats'])} beats")
        return result
    except Exception as e:
        logger.analysis_error(f"Music analysis failed: {e}")
        raise
```

---

## 14. Future Architecture Considerations

### 14.1 Async Pipeline
Consider `asyncio` for I/O-bound operations (FFmpeg, file I/O):

```python
async def process_clips_async(clip_specs):
    semaphore = asyncio.Semaphore(4)  # Limit concurrent FFmpeg
    
    async def process_one(spec):
        async with semaphore:
            return await run_ffmpeg_async(spec)
    
    return await asyncio.gather(*[process_one(s) for s in clip_specs])
```

### 14.2 GPU Acceleration
If hardware allows:
- Use `h264_nvenc` / `hevc_nvenc` for encoding
- Use CUDA-accelerated OpenCV for optical flow
- Consider `torch` with CUDA for embeddings

### 14.3 Distributed Processing
For very large inputs:
- Split video analysis across workers
- Use Redis/RabbitMQ for task queue
- Shared cache via network filesystem

---

## 15. Summary Checklist

- [ ] Batch frame extraction (single FFmpeg call)
- [ ] Replace Farneback with frame difference
- [ ] Pre-compute candidates for all durations
- [ ] Parallel music feature extraction
- [ ] Persistent caching with file hashes
- [ ] Single-pass FFmpeg concat
- [ ] Remove redundant remux in `encode_in_chunks`
- [ ] Cache QSV availability check
- [ ] Adaptive config based on input duration
- [ ] Memory monitoring and limits
- [ ] Hungarian algorithm for assignment
- [ ] Benchmark suite for regression detection

---

*Generated from codebase analysis on 2026-08-22*