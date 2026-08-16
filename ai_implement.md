# AI Implementation Plan for beats-video-editor

## Goal
Design and implement an AI-assisted clip→beat matching system so the editor automatically selects and arranges video clips to match beats and musical features, producing engaging, well-timed videos without hardcoded rules.

This document outlines architecture, components, integration points, configuration, dependencies, tests, and an initial MVP (heuristic AI) to get practical gains quickly on CPU-only machines.

---

## High-level architecture

- Input: list of source videos (data/videos/), audio track with beat timestamps (from `app.beat.get_beats`).
- Output: beat-synchronized clips and final edited video (existing pipeline using `app/sync.py` / `app/video.py`).
- Components:
  - Feature extractor (visual + audio)
  - Candidate generator (scene anchors + uniform sampling)
  - Scorer (compute similarity of beat vs candidate)
  - Assigner (solve assignment under constraints)
  - Trimmer/encoder (use `process_clip`, snap-to-frame)
  - UI/CLI integration (Streamlit `AI mode`, headless `--ai` flag)

---

## Design principles

- Keep memory usage low: reuse existing chunked encode approach; batch feature extraction; limit candidate count.
- CPU-only friendly: prefer CPU models and lightweight methods (ResNet50, MFCC) rather than large GPU-only transformer models.
- Explainable: save `sync_debug_ai.json` with beats, candidates, scores and chosen assignments.
- Fallbacks: if ML models not available, use scene-detection + snap-to-frame greedy assignment.

---

## Component details

### 1) Feature extractor

- Visual features:
  - Option A (recommended CPU): `torchvision.models.resnet50(pretrained=True)` feature vector (pool5) for a frame.
  - Sample 3 frames per candidate (start, -0.2s, +0.2s), average L2-normalized vectors.
  - Extraction method: `ffmpeg -ss <t> -frames:v 1 -f image2pipe -vcodec png -` piped into PIL/OpenCV and then to model.

- Audio features:
  - Use `librosa` to extract Log-Mel or MFCC from a small window around each beat (e.g., 0.5s window).
  - Compute mean-pooled feature vector and L2-normalize.

### 2) Candidate generation

- Scene anchors:
  - Use `ffmpeg` scene detection filter: `-vf select='gt(scene,SCENE_THRESH)'` to list scene-change timestamps.
  - Add those timestamps as high-priority candidates.

- Uniform sampling:
  - Sample candidate starts every `sample_interval` seconds (configurable, default 0.5s).
  - Prune candidates where `source_duration - start < clip_duration`.

- Cap candidates per source (e.g., 200) to bound computation.

### 3) Scoring

- For each beat b and candidate c compute score S(b,c):
  - Visual similarity: cosine( visual_embed(c), visual_embed_target(b) )
  - Audio-visual heuristic: compare audio embed around beat vs candidate's visual for cross-modal alignment (simple weighting) — if no cross-modal model available, use a weighted sum: alpha*visual + beta*scene_bonus + gamma*proximity_bonus.

- Normalize scores into [0,1]; cost = 1 - score.

### 4) Assignment

- Use constrained assignment:
  - Solve minimal-cost assignment using `scipy.optimize.linear_sum_assignment` on a rectangular cost matrix (beats × candidates). If number of candidates < beats, fall back to greedy.
  - Constraint handling: disallow assignments that cause overlapping source intervals (postprocess; if conflict, choose next-best candidate).

### 5) Trimming & encoding

- Snap `-ss` to nearest frame using project FPS (30 by default):
  - snapped = round(start_time * FPS) / FPS
  - Implement helper `snap_to_frame(time_s, fps=FPS)` in `app/video.py` and use it before calling `process_clip`.
- Use `process_clip()` to create clips, then run existing concat/transition/encode pipeline.

---

## Integration points (files & functions to change)

- `app/sync.py`
  - Add parameter `ai_mode=False` to `sync_clips_with_beats()`.
  - Create `ai_assign_clips(beat_times, video_paths, cfg)` returning `used_beat_groups` with chosen `source_start` and `score`.
  - When `ai_mode=True`, call `ai_assign_clips()` instead of random sampling.

- `app/video.py`
  - Add helpers:
    - `snap_to_frame(time_s, fps=FPS)`
    - `extract_frame(video_path, time_s) -> PIL.Image` (ffmpeg pipe)
    - `extract_audio_segment(video_path, start_s, duration_s) -> np.ndarray` (ffmpeg or librosa)
  - Update `process_clip()` to accept `snap_frame=True` and apply `-ss` after snapping.

- `app/frontend.py`
  - Add `Use AI matching` checkbox and a small `AI settings` panel (sampling interval, scene threshold, reuse limit).
  - Add preview UI for assignment results: thumbnail + beat time + score; allow manual override.

- `tools/run_headless_test.py`
  - Add `--ai` flag and `--debug` to write `sync_debug_ai.json` with beat→candidate mapping.

---

## Dependencies (add to `requirements.txt`)

- Minimal ML/processing libs (CPU-friendly):
```
torch
torchvision
librosa
opencv-python
scipy
Pillow
numpy
```

Note: `torch` CPU install will be slower but works without CUDA.

---

## Commands & developer workflow

- Install dependencies:
```bash
python -m pip install -r requirements.txt
```

- Run headless AI pipeline (debug):
```bash
.\\.pmv\\Scripts\\python.exe tools\\run_headless_test.py --ai --debug
```

- Run Streamlit UI and toggle `AI Mode`:
```bash
streamlit run app/frontend.py
```

---

## MVP plan (fast delivery)

1. Implement `snap_to_frame()` and wire `process_clip()` to snap starts (low effort, high impact).
2. Implement scene-anchor candidate generation using ffmpeg scene-detect. Build greedy scorer that prioritizes scene anchors and proximity to beat.
3. Add `--ai` flag to `tools/run_headless_test.py` and `ai_mode` toggle in Streamlit that runs heuristic assignment.
4. Produce `sync_debug_ai.json` for inspection.

This MVP runs on CPU and avoids heavy ML models while significantly improving perceptual alignment.

---

## Testing & evaluation

- Unit tests:
  - Feature extractor returns consistent shape.
  - Candidate generator prunes invalid timestamps.
  - Assigner respects no-overlap constraint.

- Integration tests:
  - Headless `--ai --debug` run produces `sync_debug_ai.json` and final video in `output/`.
  - Manual review: verify that assigned clip frames align visually to beat events.

---

## Explainability & logs

- Save `sync_debug_ai.json` containing:
  - `beat_times`: list
  - `candidates`: per-video list of timestamps + features (hash) + scene_flag
  - `scores`: matrix or top-k recommendations per beat
  - `assignments`: chosen candidate index, score, snapped_start

- UI: show top-3 candidate thumbnails for each beat with scores and reason (scene/motion/proximity).

---

## Performance tuning

- Reduce `sample_interval` or `candidates_per_video` to trade accuracy for speed.
- Cache extracted embeddings in `tmp/embeddings/<video>.npz` to avoid recomputation.
- Batch frame extraction via ffmpeg to reduce process overhead.

---

## Future improvements (post-MVP)

- Add cross-modal models (e.g., CLAP, AudioCLIP) for true audio↔visual similarity on CPU (smaller distillation models), or use cloud GPU for heavy training.
- Implement Hungarian assignment with soft constraints and global continuity optimization (preserve story / reuse rules).
- Add motion/optical-flow detectors to prefer high-energy visual moments for beats.

---

## Example: function signatures

```python
# app/sync.py
def ai_assign_clips(beat_times: list[float], video_paths: list[str], *, sample_interval=0.5, scene_thresh=0.3, reuse_limit=2) -> list[dict]:
    """Return used_beat_groups: list of dicts with keys: clip_path, source_path, music_start, music_end, clip_duration, source_start, score
    """

# app/video.py
def snap_to_frame(time_s: float, fps: int = FPS) -> float:
    return round(time_s * fps) / fps
```

---

If you'd like, I can now implement the MVP steps (snap-to-frame + scene-anchor greedy assignment + `--ai` flag) and run the headless test on your machine. Reply "start MVP" to proceed, or tell me if you want the document in Hindi or to include more code samples.
