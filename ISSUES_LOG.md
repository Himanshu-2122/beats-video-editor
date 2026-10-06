# Project Issue Log

This is the cumulative troubleshooting record for issues encountered while reviewing and attempting to inspect this repository. It was initialized on 2026-10-01 after checking for an existing `ISSUES_LOG.md`; none existed. IDs begin at `ISSUE-001` and must not be reused.

## Scope and Evidence

- The application was not run during the review. The checked-in repository has no `data/videos/` directory, and the standalone tests require additional video files. Code-path issues below are explicitly marked as static findings unless actual output is cited.
- A historical benchmark transcript in `benchmark.txt` is recorded as historical evidence, not as a fresh run.
- Potential issues are explicitly identified with the words **Potential Issue**. Their root causes or impact have not been demonstrated at runtime.
- No application source files were changed to resolve these items. For open items, “Solution” means no code fix was applied; suggested actions are listed as remediation/prevention, not represented as completed work.
- Machine-local paths and media names found in logs and benchmark data are intentionally redacted. No credentials are included.

---

## ISSUE-001 — Progress Callback Can Deadlock

**Status:** Open  
**Severity:** High  
**Category:** Runtime

### Problem

Static finding: progress methods call `_notify()` while holding a regular, non-reentrant `threading.Lock`. When a `frontend_callback` is configured, `_notify()` calls `get_dashboard_data()`, which attempts to acquire the same lock. The current Streamlit UI does not pass a callback when constructing `ProgressTracker`, so this specific deadlock path is conditional and was not observed during a run.

### Error

Not reproduced at runtime. The confirmed call chain is:

```text
start_stage() [lock held] -> _notify() -> frontend_callback(get_dashboard_data())
-> get_dashboard_data() [acquires same lock]
```

### Where It Occurred

- File: `app/progress_tracker.py`
- Functions: `start_stage()` (line 57), `update_stage_progress()` (line 66), `_notify()` (line 151), `get_dashboard_data()` (line 158)
- Trigger: any caller constructing `ProgressTracker(..., frontend_callback=...)` and then updating progress
- Command: none; static inspection only

### Root Cause

`self.lock` is created with `threading.Lock()` at line 42. Progress update methods invoke `_notify()` inside `with self.lock`; the callback path re-enters a method that acquires that same lock.

### Investigation

1. Checked lock creation and all progress update methods.
2. Followed `_notify()` into `get_dashboard_data()`.
3. Confirmed `get_dashboard_data()` acquires `self.lock` again.
4. Confirmed the current frontend constructs the tracker without a callback, so the issue is latent in the optional callback API.

### Solution

No source fix was applied. A fix should call callbacks after releasing the lock, or otherwise avoid lock re-entry; changing lock type alone should be considered carefully because callbacks may run arbitrary code.

### Verification

Static call-path verification only. No reproducer was run because source changes were out of scope.

### Prevention

Keep callbacks outside critical sections and add a timeout-based test using a non-`None` callback.

### Related Files

- `app/progress_tracker.py`
- `app/frontend.py`

---

## ISSUE-002 — Hungarian Fallback Uses an Uninitialized Variable

**Status:** Open  
**Severity:** High  
**Category:** Logic

### Problem

Static finding: `_hungarian_assign()` passes `source_durations` to `_greedy_assign()` from two fallback branches before assigning that local variable later in the function. If either fallback branch is entered, evaluating the argument can raise `UnboundLocalError` before greedy assignment begins.

### Error

Not reproduced at runtime. The source order is the relevant evidence:

```text
app/sync.py:1007, 1013  -> source_durations=source_durations
app/sync.py:1048        -> source_durations is assigned afterward
```

### Where It Occurred

- File: `app/sync.py`
- Function: `_hungarian_assign()`; fallback branches around lines 1004-1013, assignment around line 1048
- Trigger: SciPy import failure or an oversized cost matrix when `_hungarian_assign()` is called
- Command: none; static inspection only

### Root Cause

The `source_durations` local is initialized after both early fallback branches. Python treats it as a local because it is assigned later in the same function.

### Investigation

1. Located both `_greedy_assign()` fallback calls.
2. Located the only local initialization of `source_durations`.
3. Confirmed the initialization appears below those calls.
4. Did not invoke this internal path because the task prohibited application changes and no video fixtures are present.

### Solution

No source fix was applied. Compute `source_durations` before either branch, or pass an explicit value that does not reference the uninitialized local.

### Verification

Static source-order check only; no runtime reproduction.

### Prevention

Add tests for `_hungarian_assign()` with SciPy unavailable and with a matrix above the fallback threshold.

### Related Files

- `app/sync.py`

---

## ISSUE-003 — Hungarian Assignments Can Carry Incorrect Metadata

**Status:** Open  
**Severity:** Medium  
**Category:** Logic

### Problem

Static finding: the assignment-building loop writes `beat_strength`, `beat_energy`, and `beat_type` into every result, but those variables were last assigned in the preceding cost-matrix loop. Consequently, returned rows can inherit metadata from the last beat group instead of the beat group represented by that assignment. The Hungarian route also computes candidate costs using initially empty used-interval state and does not update a chronological cursor; it rejects some conflicts after solving without re-solving those rows.

### Error

Not reproduced at runtime. Relevant implementation behavior:

```text
The solver returns (beat_idx, candidate_idx) pairs, but output metadata
uses beat_strength / beat_energy / beat_type left over from matrix construction.
Rejected overlaps/reuse limits are skipped after solving; no replacement is selected.
```

### Where It Occurred

- File: `app/sync.py`
- Function: `_hungarian_assign()`; per-beat metadata is assigned around lines 1075-1077; results are built around lines 1230-1255
- Trigger: a Hungarian assignment containing more than one beat group or conflicting candidates
- Command: none; static inspection only

### Root Cause

Metadata is not recomputed from `beat_idx` in the output loop. Constraint checks are not encoded consistently in the global assignment and post-solver validation does not request alternate assignments.

### Investigation

1. Followed metadata variables from matrix construction to result serialization.
2. Confirmed their values are scoped to the last completed matrix-loop iteration.
3. Followed overlap/reuse checks after `linear_sum_assignment()` and confirmed rejected rows are simply skipped.

### Solution

No source fix was applied. Recompute beat metadata from the current result row and encode/reconcile assignment constraints so a rejected pair can be replaced.

### Verification

Static control-flow review only. No media-backed test was run.

### Prevention

Test multiple beats with deliberately different energy/type values, overlapping source intervals, and per-source reuse limits. Compare greedy and Hungarian output invariants.

### Related Files

- `app/sync.py`

---

## ISSUE-004 — AI Debug Serialization Assumes the Wrong Candidate Shape

**Status:** Open  
**Severity:** Medium  
**Category:** Logic

### Problem

Static finding: `all_candidates[video_path]` is a mapping from duration to candidate lists. The debug serializer iterates that mapping as though each item were a candidate dictionary. It then indexes a duration key with `c["t"]`, which is not valid for the numeric duration keys produced by candidate generation. The surrounding exception handler suppresses the failure and reports only a warning, so the expected debug JSON may not be created.

### Error

Not reproduced at runtime. Relevant shape mismatch:

```text
all_candidates[video_path] = {duration: [candidate, ...], ...}
serializer: for c in all_candidates[video_path]: c["t"]
```

### Where It Occurred

- File: `app/sync.py`
- Function: `sync_clips_with_beats()`; serializer around lines 1567-1577
- Trigger: AI mode reaches the debug-output writer
- Command: a successful `sync_clips_with_beats(..., ai_mode=True)` call

### Root Cause

The serializer does not traverse the nested duration-to-candidates mapping used by both assignment algorithms.

### Investigation

1. Inspected candidate construction in `_greedy_assign()` and `ai_assign_clips()`.
2. Confirmed each video maps to duration buckets, and each bucket maps to a list of candidate dictionaries.
3. Compared that shape with the flat iteration in the debug writer.
4. Confirmed the writer catches exceptions, which can make the missing file look like a successful sync with a warning.

### Solution

No source fix was applied. Serialize each duration bucket and candidate list explicitly, or define one consistent flat debug schema.

### Verification

Static data-shape verification only; debug generation was not run.

### Prevention

Add a test that runs the serializer against the actual nested candidate structure and asserts the output JSON parses and includes candidates.

### Related Files

- `app/sync.py`

---

## ISSUE-005 — `ai_assign_clips()` Has Inconsistent Return Shapes

**Status:** Open  
**Severity:** Medium  
**Category:** API

### Problem

Static finding: `ai_assign_clips()` normally returns a four-item tuple `(results, beat_groups, debug_scores, all_candidates)`, but returns a bare empty list when `create_beat_groups()` produces no groups. Callers that unpack four values can fail on that path.

### Error

Not reproduced at runtime. Relevant return statements:

```text
app/sync.py:1330  return []
normal path       return results, beat_groups, debug_scores, all_candidates
```

### Where It Occurred

- File: `app/sync.py`
- Function: `ai_assign_clips()` around line 1330
- Callers: `run_generation()` in `app/frontend.py`; AI branch of `sync_clips_with_beats()`
- Trigger: a valid call that results in no beat groups

### Root Cause

The early-return branch does not follow the function's normal tuple contract.

### Investigation

1. Compared all return paths in `ai_assign_clips()`.
2. Checked callers, which unpack four values.
3. Confirmed the bare-list path is reachable if beat grouping returns empty.

### Solution

No source fix was applied. Return a four-item empty result consistently and document the contract in the function annotation.

### Verification

Static API/call-site review only.

### Prevention

Add an empty-input/empty-group test and assert the return structure at every call site.

### Related Files

- `app/sync.py`
- `app/frontend.py`

---

## ISSUE-006 — Beat-Group Coverage Can Extend Beyond the Music Timeline

**Status:** Open  
**Severity:** Medium  
**Category:** Logic

### Problem

Static finding: beat groups start at the first detected beat, not necessarily at time zero. After extending the final group to `music_duration`, the code sums group durations and may interpret the omitted initial interval as remaining coverage. It can then append synthetic groups starting at the already-covered music end, extending the planned timeline beyond the track.

### Error

The checked-in historical benchmark contains inconsistent coverage diagnostics; this is not a fresh reproduction:

```text
Extended final group to cover full music duration: 173.43s
Added synthetic groups to cover full duration: 173.43s
Total timeline duration: 172.63s
```

### Where It Occurred

- File: `app/sync.py`
- Function: `create_beat_groups()` around lines 66-160
- Trigger: music duration is supplied, first beat is after zero, and coverage is extended
- Command: benchmark transcript in `benchmark.txt`; no new command run

### Root Cause

Coverage is checked by summing durations of beat groups rather than by validating a continuous union of intervals against `[0, music_duration]`. The final extension and synthetic-group logic therefore do not share a consistent coverage model.

### Investigation

1. Followed beat-group start/end construction and the final-group extension.
2. Followed the later summed-duration comparison and synthetic-group start time.
3. Compared it with the historical benchmark's coverage messages and total.
4. Did not run a controlled fixture, so exact output for current inputs remains unverified.

### Solution

No source fix was applied. Build timeline coverage from explicit intervals beginning at zero, then validate that the final interval ends exactly at the chosen music boundary without overlap or overshoot.

### Verification

Static control-flow review plus historical transcript review; no fresh end-to-end verification.

### Prevention

Add deterministic unit tests with a known first-beat offset, final-beat offset, empty beats, and tracks both shorter and longer than detected beat span. Assert rendered duration against music duration.

### Related Files

- `app/sync.py`
- `benchmark.txt`

---

## ISSUE-007 — Historical Benchmark Produced Zero Clip Assignments

**Status:** Open  
**Severity:** High  
**Category:** Runtime

### Problem

The checked-in benchmark transcript shows candidate generation and assignment completing, but no assignments were returned. This is a real recorded failure in the artifact, but it is historical and was not reproduced during this review.

### Error

Actual output recorded in `benchmark.txt` (local media identifiers omitted):

```text
Beat groups (assigned): 0
Clips assigned: 0
Total music coverage (assigned only): 0.00s / 172.63s planned
Unique sources used: 0 / 3
```

### Where It Occurred

- File: `benchmark.txt`
- Runtime components: `benchmark_clip_assignment()` in `tools/benchmark.py`, `ai_assign_clips()` and assignment implementations in `app/sync.py`
- Command that originally produced the output: not recorded in the transcript
- Current reproduction command: none; the original video inputs are not present in the repository

### Root Cause

Cannot be confirmed from the checked-in transcript alone. It records input sizes, candidate generation, and zero assignments, but does not preserve enough diagnostic state to identify which constraint or matching branch caused the result.

### Investigation

1. Read the complete benchmark transcript after decoding its UTF-16LE encoding.
2. Confirmed candidate generation reported candidates for the videos.
3. Confirmed the assignment summary nevertheless reported zero results and zero coverage.
4. Checked current repository assets; the historical source videos are absent, so the run could not be reproduced.

### Solution

No fix can be claimed. Re-run the benchmark with portable fixtures and retain per-beat rejection diagnostics before changing assignment logic.

### Verification

Verified only that the checked-in transcript contains the zero-assignment output. Current code behavior remains unverified on those inputs.

### Prevention

Make the benchmark fail nonzero when assignments or coverage are unexpectedly zero, and preserve a small generated-media regression case.

### Related Files

- `benchmark.txt`
- `tools/benchmark.py`
- `app/sync.py`

---

## ISSUE-008 — Music Section Detection Produces Excessive Boundaries

**Status:** Open  
**Severity:** Medium  
**Category:** Logic

### Problem

The chroma-difference threshold can mark many individual feature frames as section boundaries. The checked-in benchmark reported 2,241 sections for a track of roughly 173 seconds, which is not a useful musical-section segmentation.

### Error

Actual historical benchmark output:

```text
Music analysis complete: 215 beats, 25 drops, 3 buildups, 2241 sections
Sections: 2241
```

### Where It Occurred

- File: `app/beat.py`
- Function: `analyze_music_full()`; `chroma_diff` and percentile boundary selection around lines 234-247
- Evidence: historical output in `benchmark.txt`; not a fresh run

### Root Cause

The implementation selects every frame whose chroma difference exceeds the 85th percentile and does not merge nearby boundaries or impose a minimum section duration. Labels are assigned from energy, not a learned or semantic section classifier.

### Investigation

1. Inspected the chroma feature and difference calculation.
2. Inspected the per-frame percentile threshold and section-construction loop.
3. Compared the algorithm with the historical section count.
4. No new audio analysis was run.

### Solution

No source fix was applied. Merge nearby candidates, enforce minimum section lengths, and validate section counts against representative audio before using labels downstream.

### Verification

The historical transcript verifies the observed count; no updated algorithm exists to verify.

### Prevention

Add tests for boundary count, minimum section length, and coverage; inspect outputs on a short fixed audio fixture.

### Related Files

- `app/beat.py`
- `benchmark.txt`

---

## ISSUE-009 — Cache Configuration and Invalidation Are Inconsistent

**Status:** Open  
**Severity:** Medium  
**Category:** Configuration

### Problem

Static findings:

- `analyze_music_full()` and `analyze_video_full()` accept `cache_dir`, but use the global `CacheManager` rooted at repository-relative `cache/`; the supplied directory is not used.
- The in-memory file-hash map returns an old hash for a path without rechecking size/mtime, so replacing a file at that path during the process can reuse stale results.
- `clear_all()` omits the `music_analysis` category, so it does not clear all data written by the music analyzer.
- `cache/motion/7af18bf36641f6aa_motion_0.8.json` is an empty, invalid JSON file. Why it is empty cannot be determined from the repository.

### Error

No runtime error was reproduced. Relevant source behavior:

```text
CacheManager.cache_dir = Path("cache")
clear_all() categories = beats, scenes, motion, video_analysis, embeddings
music analysis category = music_analysis
```

### Where It Occurred

- Files: `app/cache.py` lines 26-42 and 89-92; `app/beat.py` lines 99-110; `app/video.py` around line 1573 onward
- Trigger: passing a temporary `cache_dir`, modifying a source file in place, calling `clear_all()`, or loading the empty motion cache
- Command: static review and JSON parsing during repository inspection

### Root Cause

Cache path selection and invalidation are centralized in a global manager, while public analysis APIs imply per-call cache directories. The hash cache is keyed only by path and never refreshed. Cache clearing uses a hard-coded category list that is incomplete.

### Investigation

1. Traced all `cache_dir` parameters to the actual calls to `cache_manager.load/save()`.
2. Inspected `get_file_hash()` and found it returns a memoized value before restatting the path.
3. Compared saved categories with `clear_all()`'s category list.
4. Parsed all cache JSON files; the motion cache file was zero bytes and JSON parsing is caught by `load()`.

### Solution

No source fix was applied. Align API parameters with cache-root selection; include file identity/version in cache invalidation; include every active category in `clear_all()`; remove or regenerate the empty cache entry.

### Verification

Static inspection and JSON parsing only. Cache invalidation was not exercised with a changed media file.

### Prevention

Test per-call cache isolation, replacement of a file at the same path, clear-all behavior for every category, and recovery from malformed cache JSON.

### Related Files

- `app/cache.py`
- `app/beat.py`
- `app/video.py`
- `cache/motion/7af18bf36641f6aa_motion_0.8.json`

---

## ISSUE-010 — Checked-In Tests Depend on Missing, Machine-Specific Media

**Status:** Open  
**Severity:** High  
**Category:** Setup

### Problem

The checkout contains `data/music/song.mp3` but no `data/videos/` directory or test video. `test_pipeline.py` requires `data/videos/test_video.mp4`. `test_multi_video_greedy_path.py` requires external output videos and a temporary upload file, and inserts a hard-coded Windows project path into `sys.path`. The pipeline test returns `False` when files are missing; it does not assert, so pytest may not fail in the intended way.

### Error

No test command was run. Repository inventory showed:

```text
data/music/song.mp3
data/videos/: missing
```

The regression test's required assets are named only in code and are absent from the checkout.

### Where It Occurred

- Files: `test_pipeline.py` lines 19-37; `test_multi_video_greedy_path.py` lines 19 and 35-43
- Commands expected by the test scripts: `python test_pipeline.py`, `python test_multi_video_greedy_path.py`
- Current environment: Linux; no required video fixtures present

### Root Cause

The tests are coupled to local media/output state rather than checked-in or generated fixtures. The pipeline test's missing-input path returns a boolean instead of asserting or raising a test failure.

### Investigation

1. Read both complete test scripts.
2. Searched the repository inventory for their expected video/output/temp-upload paths.
3. Confirmed only the MP3 sample is present under `data/`.
4. Did not run the tests because required files are unavailable.

### Solution

No test or source changes were applied. Use generated short video/audio fixtures, remove hard-coded paths, and assert failures explicitly.

### Verification

Verified missing assets through filesystem inventory; no test execution was attempted.

### Prevention

Make tests self-contained and ensure `pytest` collection reports failures rather than returning false from a test function.

### Related Files

- `test_pipeline.py`
- `test_multi_video_greedy_path.py`
- `.gitignore`
- `data/music/song.mp3`

---

## ISSUE-011 — Headless Runner Cannot Create Its Video Fixture in a Fresh Checkout

**Status:** Open  
**Severity:** Medium  
**Category:** Runtime

### Problem

Static finding: `tools/run_headless_test.py` tries to create `data/videos/test_video.mp4` when it is missing, but it does not create the `data/videos/` parent directory before invoking FFmpeg. That directory is absent in the current checkout, so the synthetic-video command is expected to fail unless the user creates the directory first.

### Error

Not reproduced at runtime. Current inventory evidence:

```text
data/videos/: missing
```

The script invokes `subprocess.run(cmd, check=True)` with the missing-directory output path.

### Where It Occurred

- File: `tools/run_headless_test.py`
- Function: `main()`; synthetic test video setup around lines 39-48
- Command: `python tools/run_headless_test.py` or `python tools/run_headless_test.py --ai`

### Root Cause

The script creates `output/` but not `DATA_VIDEOS` before directing FFmpeg to write inside it.

### Investigation

1. Inspected the runner's directory setup and synthetic-video branch.
2. Checked the current filesystem and confirmed `data/videos/` is absent.
3. Did not run FFmpeg, so no FFmpeg error text was captured.

### Solution

No source fix was applied. Create `DATA_VIDEOS` before FFmpeg runs or require and clearly validate a fixture path.

### Verification

Static path review and filesystem inventory only.

### Prevention

Add a smoke test that starts from a clean checkout with only the checked-in sample music and verifies fixture creation.

### Related Files

- `tools/run_headless_test.py`
- `data/music/song.mp3`

---

## ISSUE-012 — Headless `--debug` Option Is Not Used

**Status:** Open  
**Severity:** Low  
**Category:** API

### Problem

The runner advertises `--debug` as writing `sync_debug_ai.json`, parses the flag, and then never reads `args.debug`. The option does not control debug output. In addition, the AI branch does not pass `music_analysis` to the sync API, so this runner does not supply the analyzed music features used for energy/type matching.

### Error

Not reproduced at runtime. Source inspection confirms:

```text
parse_args(): defines --debug
main(): args.debug has no references
AI sync kwargs: no music_analysis value
```

### Where It Occurred

- File: `tools/run_headless_test.py`
- Functions: `parse_args()` around line 19 and `main()` around lines 30-75
- Command: `python tools/run_headless_test.py --ai --debug`

### Root Cause

The CLI option was added to the parser but not wired to conditional debug behavior. The caller only supplies beat timestamps, not the richer music analysis object.

### Investigation

1. Read the argument parser and entire `main()` implementation.
2. Searched references to `args.debug` and found none.
3. Compared AI kwargs with the `sync_clips_with_beats()` signature.

### Solution

No source fix was applied. Wire the flag to the debug writer and pass `music_analysis` if music-aware matching is intended.

### Verification

Static inspection only; no headless run was performed.

### Prevention

Add CLI tests asserting each option changes the arguments/output behavior it documents.

### Related Files

- `tools/run_headless_test.py`
- `app/sync.py`

---

## ISSUE-013 — Streamlit XSRF Protection Is Disabled

**Status:** Open  
**Severity:** High  
**Category:** Security

### Problem

The Streamlit configuration explicitly disables XSRF protection. This is a confirmed insecure setting if the UI is reachable by untrusted clients. No authentication layer is configured in this repository.

### Error

Configuration evidence:

```toml
enableXsrfProtection = false
```

### Where It Occurred

- File: `.streamlit/config.toml`, line 3
- Component: Streamlit server configuration
- Trigger: launching the UI with `streamlit run app/frontend.py` and exposing it beyond a trusted local context

### Root Cause

XSRF protection was explicitly disabled in the checked-in configuration. The repository does not explain why this is required.

### Investigation

1. Read the complete Streamlit TOML configuration.
2. Confirmed the setting is `false`.
3. Checked the app configuration and found no authentication implementation.
4. No network exposure test was run.

### Solution

No configuration change was applied. Re-enable XSRF protection unless there is a documented, narrowly scoped reason not to; do not expose the unauthenticated app publicly.

### Verification

Static config verification only.

### Prevention

Keep security defaults enabled and review Streamlit configuration before deployment or network binding changes.

### Related Files

- `.streamlit/config.toml`
- `app/frontend.py`
- `README.md`

---

## ISSUE-014 — Documented Python Minimum Conflicts With Type Annotations

**Status:** Open  
**Severity:** Medium  
**Category:** Dependency

### Problem

`README.md` says Python 3.8+ is supported, but source annotations use built-in generic forms such as `list[dict]` and `list[tuple]`, which require Python 3.9+ without postponed annotation evaluation. Python 3.8 compatibility has not been runtime-tested.

### Error

Static compatibility evidence:

```text
README.md: Python 3.8+
app/sync.py: list[tuple] and list[dict] annotations
```

### Where It Occurred

- Files: `README.md` line 4; `app/sync.py` lines 165 and 257 onward; also annotations in `app/video.py`
- Command: setup/install on Python 3.8; not attempted

### Root Cause

The declared minimum version does not match the runtime evaluation requirements of the annotation syntax used in the code.

### Investigation

1. Read the README's prerequisite statement.
2. Searched Python sources for built-in generic annotations.
3. Did not install or execute the project on Python 3.8.

### Solution

No source/documentation change was applied. Declare Python 3.9+ as the minimum or use compatible typing syntax/postponed annotations and test Python 3.8 explicitly.

### Verification

Static language-version review only.

### Prevention

Test the declared Python minimum in CI and keep package Python constraints aligned with source syntax.

### Related Files

- `README.md`
- `app/sync.py`
- `app/video.py`
- `requirements.txt`

---

## ISSUE-015 — Setup Instructions Are Windows-Oriented

**Status:** Open  
**Severity:** Medium  
**Category:** Setup

### Problem

The only setup script is PowerShell-specific, and the README's manual activation/install example also uses Windows paths. On the current Linux environment, those commands are not directly usable. A POSIX equivalent can be run manually, but is not documented in the repository.

### Error

No setup command was attempted. The mismatch is visible in the documented commands:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Where It Occurred

- Files: `setup_env.ps1`; `README.md` setup section
- Command: documented setup commands on Linux/macOS

### Root Cause

Setup documentation and automation target Windows PowerShell. No Unix shell setup instructions are provided.

### Investigation

1. Read the full setup script and README setup instructions.
2. Confirmed both use PowerShell/Windows virtual-environment paths.
3. No installation attempt was made; Linux commands were supplied during documentation only.

### Solution

No project setup files were changed. Document POSIX activation and installation commands or add a portable setup guide.

### Verification

Static documentation review only.

### Prevention

Document and test setup on every OS the project claims to support.

### Related Files

- `setup_env.ps1`
- `README.md`
- `requirements.txt`

---

## ISSUE-016 — `run_in_background()` Waits Before Returning

**Status:** Open  
**Severity:** Medium  
**Category:** API

### Problem

The helper name/docstring promise background execution and a returned future, but the executor is created in a `with` block. Exiting that block calls executor shutdown with waiting enabled, so the function waits for the submitted work before returning the future.

### Error

Not reproduced with a timed call. The implementation pattern is:

```python
with ThreadPoolExecutor(max_workers=1) as executor:
    return executor.submit(func, *args, **kwargs)
```

### Where It Occurred

- File: `app/cache.py`
- Function: `run_in_background()` around lines 152-155
- Trigger: any call with a long-running `func`

### Root Cause

The executor context manager waits for submitted tasks during shutdown before control returns to the caller.

### Investigation

1. Read the helper body and docstring.
2. Checked the `ThreadPoolExecutor` context-manager lifecycle.
3. No timing test or caller was found/run during this review.

### Solution

No source fix was applied. Keep the executor alive for the lifetime of the future or use an application-owned executor with explicit shutdown.

### Verification

Static lifecycle review only; no runtime timing reproduction.

### Prevention

Test that the function returns before a deliberately blocked worker completes, and test executor shutdown separately.

### Related Files

- `app/cache.py`

---

## ISSUE-017 — `encode_in_chunks()` Does Not Encode in Chunks

**Status:** Open  
**Severity:** Medium  
**Category:** API

### Problem

The function name and UI option say “chunked encoding,” but the implementation remuxes the input with `-c copy`. It does not segment, re-encode, or use its `segment_time`, CRF, preset, QSV, or progress parameters. The actual behavior is documented in its docstring as a remux.

### Error

No runtime failure; behavior mismatch is confirmed by the command construction:

```text
ffmpeg -i INPUT -c copy -movflags +faststart OUTPUT
```

### Where It Occurred

- File: `app/video.py`
- Function: `encode_in_chunks()` around lines 1485-1520
- UI caller/label: `app/frontend.py`, optional “Low-RAM chunked encoding” fallback path
- Commands: `python -m app.main`; `python tools/run_headless_test.py`

### Root Cause

The implementation was changed to a fast remux while public naming, arguments, and UI terminology still imply chunked re-encoding.

### Investigation

1. Read the complete function body.
2. Confirmed it invokes one FFmpeg command with stream copy and does not reference the chunk/progress arguments.
3. Compared the name and UI label with actual behavior.

### Solution

No source or UI change was applied. Rename/document the operation as remuxing, or implement and test real chunked encoding if required.

### Verification

Static command review only; no render was run.

### Prevention

Test command construction and ensure UI controls describe actual processing behavior.

### Related Files

- `app/video.py`
- `app/frontend.py`
- `README.md`
- `tools/run_headless_test.py`

---

## ISSUE-018 — `process_clip()` Contains Unreachable Duplicate Code

**Status:** Open  
**Severity:** Low  
**Category:** Maintainability

### Problem

Static finding: after `process_clip()` returns at approximately line 1132, a second QSV/CPU command construction, verification, and return block remains in the function. That duplicate block is unreachable.

### Error

No runtime error. Control-flow evidence:

```text
app/video.py:1132  return output_path
subsequent lines    duplicate QSV/CPU render logic
```

### Where It Occurred

- File: `app/video.py`
- Function: `process_clip()` around lines 1004-1195
- Trigger: none; dead code found by static review

### Root Cause

An earlier implementation block appears to have remained after a return when the function was revised. The history/reason cannot be determined from current source alone.

### Investigation

1. Read the full `process_clip()` implementation.
2. Located the first unconditional return.
3. Confirmed equivalent command construction continues below it.

### Solution

No source fix was applied. Remove the unreachable duplicate in a separately reviewed code change.

### Verification

Static control-flow inspection only.

### Prevention

Use linting for unreachable code and keep render fallback logic in a single tested path.

### Related Files

- `app/video.py`

---

## ISSUE-019 — Potential Issue: Single-Pass Render Does Not Loop Short Music

**Status:** Open  
**Severity:** Medium  
**Category:** Runtime

### Problem

**Potential Issue.** `render_video_single_pass()` maps the music input and uses `-shortest` but does not use `-stream_loop -1`. If assignment video duration exceeds the music duration, FFmpeg can end output with the shorter audio stream. Whether this is reached for intended inputs has not been reproduced.

### Error

No runtime output captured. Relevant command difference:

```text
render_video_single_pass(): maps music input + -shortest; no -stream_loop
add_audio(): uses -stream_loop -1
```

### Where It Occurred

- File: `app/video.py`
- Function: `render_video_single_pass()` around lines 1997-2159; audio mapping around lines 2132-2138
- Comparison: `add_audio()` around lines 1523-1566
- Trigger: an assigned timeline longer than its music input

### Root Cause

The single-pass path does not share the legacy audio-looping option. Actual impact depends on assignment duration and FFmpeg stream lengths, which were not tested.

### Investigation

1. Compared audio arguments in single-pass and legacy render paths.
2. Confirmed the legacy path loops music while single-pass does not.
3. No media render was run.

### Solution

No source fix was applied. Confirm intended duration behavior, then either loop music or explicitly reject/trim timelines that exceed the music stream.

### Verification

Static command comparison only; mark as potential until a short-audio/long-timeline render is tested.

### Prevention

Add duration-mismatch integration tests for both render paths.

### Related Files

- `app/video.py`
- `app/frontend.py`
- `SPEC.md`

---

## ISSUE-020 — Potential Issue: QSV Encoder Availability Is Not a Runtime Capability Check

**Status:** Open  
**Severity:** Medium  
**Category:** Runtime

### Problem

**Potential Issue.** `check_qsv()` tests whether FFmpeg lists `h264_qsv`, not whether a usable Intel QSV device is available. `process_clip()` retries with CPU encoding, but the single-pass and transition render paths select QSV based on this check and have no CPU retry. A machine with the encoder compiled in but no working device may fail rendering; this was not reproduced.

### Error

No runtime error captured. Static evidence:

```text
check_qsv(): searches ffmpeg -encoders output for "h264_qsv"
render paths: choose QSV; no retry around the final command
```

### Where It Occurred

- File: `app/video.py`
- Functions: `check_qsv()` around line 822; `create_transition_video()` around line 1267; `render_video_single_pass()` around line 1997; CPU retry exists in `process_clip()` around line 1100
- Trigger: FFmpeg advertises QSV but device initialization/encoding fails

### Root Cause

Encoder-list presence is used as a proxy for actual hardware availability, and fallback coverage differs by rendering path.

### Investigation

1. Read the QSV detection function.
2. Compared how each renderer chooses an encoder and handles errors.
3. No QSV hardware test or render was run.

### Solution

No source fix was applied. Test a short QSV encode at startup or catch render failure and retry the complete render with libx264.

### Verification

Static path comparison only; this remains a potential issue until tested on hardware without QSV.

### Prevention

Test CPU-only, encoder-listed/no-device, and working-QSV environments in render integration tests.

### Related Files

- `app/video.py`

---

## ISSUE-021 — Potential Issue: Streamlit State Is Updated From a Worker Thread

**Status:** Open  
**Severity:** Medium  
**Category:** Runtime

### Problem

**Potential Issue.** `run_generation()` is started in a daemon thread and writes `st.session_state.generation_result` and `generation_running` from that worker. This may depend on Streamlit's thread context/version behavior. The UI suppresses a missing-`ScriptRunContext` warning, but no runtime failure was reproduced.

### Error

No runtime error captured. Static execution path:

```text
main() starts threading.Thread(target=run_generation, daemon=True)
run_generation() mutates st.session_state in that worker
```

### Where It Occurred

- File: `app/frontend.py`
- Functions: `main()` around lines 752-764; `run_generation()` around lines 194-443
- Command: `streamlit run app/frontend.py`

### Root Cause

The worker function directly accesses Streamlit session state rather than transferring completion data through a supported thread-safe mechanism. Actual impact is unconfirmed for the installed Streamlit version.

### Investigation

1. Followed thread construction and target invocation.
2. Located session-state writes in the worker.
3. Confirmed the module suppresses a missing-context warning.
4. Did not start the Streamlit app.

### Solution

No source fix was applied. Use a thread-safe result channel and update Streamlit state from the script thread, following the supported Streamlit concurrency model.

### Verification

Static inspection only; this remains potential.

### Prevention

Exercise success and exception completion through the UI on supported Streamlit versions.

### Related Files

- `app/frontend.py`
- `requirements.txt`

---

## ISSUE-022 — Potential Issue: Large Media Is Read Fully Into UI Memory

**Status:** Open  
**Severity:** Medium  
**Category:** Performance

### Problem

**Potential Issue.** Download-button data for previous and current outputs is passed using `f.read()`, loading each complete MP4 into process memory. Large renders may create high memory use. The code also invokes FFmpeg once per sampled frame during motion/video analysis, which may add substantial process overhead for long videos. Neither impact was measured during this review.

### Error

No performance failure or memory measurement captured. Relevant source calls:

```text
app/frontend.py:682 and 776 -> data=f.read()
app/video.py: extract_frame() -> subprocess.run([... "ffmpeg" ...])
```

### Where It Occurred

- Files: `app/frontend.py` around lines 682 and 776; `app/video.py` around lines 192-226 and motion-analysis callers
- Trigger: displaying/downloading large rendered videos or enabling dense motion/video analysis
- Command: Streamlit UI or a media analysis run; not executed

### Root Cause

The UI passes whole file contents to Streamlit, while analysis requests frame extraction through separate FFmpeg subprocesses. Performance impact depends on media size, sampling count, and host resources.

### Investigation

1. Inspected both download-button call sites.
2. Followed frame-analysis functions to per-frame FFmpeg invocation.
3. No memory profile or frame-extraction benchmark was run in this review.

### Solution

No source fix was applied. Consider file-backed/download streaming support and batched or persistent frame decoding after profiling representative media.

### Verification

Static call-site inspection only; this remains potential until measured.

### Prevention

Add memory/performance benchmarks with long media and track peak RSS and FFmpeg process count.

### Related Files

- `app/frontend.py`
- `app/video.py`
- `app/progress_tracker.py`

---

## ISSUE-023 — Local Media Identifiers Are Stored in Tracked Artifacts

**Status:** Open  
**Severity:** Medium  
**Category:** Security

### Problem

The checked-in `benchmark.txt`, video-analysis cache JSON, and `logs/app.log` contain machine-local paths or user-provided media names. Those values can reveal private filesystem layout and media identifiers when the repository is shared. Specific values are deliberately redacted from this log.

### Error

The issue is evidenced by the artifact contents, but values are withheld:

```text
Historical benchmark/cache/log records contain absolute local paths and input names.
Exact values omitted to avoid redisclosing them.
```

### Where It Occurred

- Files: `benchmark.txt`, `cache/video_analysis/*.json`, `logs/app.log`
- Trigger: committing or distributing the repository artifacts
- Command: artifact inspection during repository analysis

### Root Cause

The benchmark transcript and runtime artifacts preserve raw user-provided input paths/names, and these generated artifacts are tracked in Git.

### Investigation

1. Inspected the decoded benchmark transcript, cache metadata, and log records.
2. Confirmed local path/name fields are present.
3. Omitted the values from `PROJECT_ANALYSIS.md` and this issue log.

### Solution

No tracked artifact was edited or deleted. The documentation created during the review redacts those values. Repository artifacts remain unchanged and may still contain them.

### Verification

Confirmed the analysis documents do not quote the discovered local values. No repository-wide secret-scanning tool was run.

### Prevention

Avoid committing machine-specific logs/caches/benchmark output, sanitize existing artifacts before sharing, and keep generated data ignored and untracked.

### Related Files

- `benchmark.txt`
- `cache/video_analysis/`
- `logs/app.log`
- `.gitignore`

---

## ISSUE-024 — Tracked Bytecode Does Not Match Current Source Sizes

**Status:** Open  
**Severity:** Low  
**Category:** Maintainability

### Problem

The repository tracks CPython 3.12 `.pyc` files even though `.gitignore` excludes `*.pyc`. Several bytecode headers record source sizes substantially different from the current `.py` files, indicating stale generated artifacts. The Python importer normally checks source metadata, but leaving stale bytecode tracked makes the checkout harder to reason about and can confuse debugging.

### Error

No import error was reproduced. Inspection showed stale cached code-object inventories and mismatching source sizes; for example, compiled `sync` bytecode contains a small set of older functions while current `app/sync.py` is much larger.

### Where It Occurred

- Files: `app/__pycache__/*.pyc` (six tracked CPython 3.12 artifacts)
- Configuration: `.gitignore` already contains `__pycache__/` and `*.pyc`
- Trigger: reviewing/importing a checkout containing tracked stale bytecode

### Root Cause

Generated bytecode was committed before the ignore rules took effect or was force-added. Exact history is not established by the current tree.

### Investigation

1. Listed all tracked files and identified six `.pyc` files.
2. Inspected bytecode headers and embedded code-object names.
3. Compared their recorded source sizes/contents with current Python modules.
4. Confirmed ignore patterns exist but do not untrack already committed files.

### Solution

No bytecode or source file was changed. Remove generated bytecode from version control in a separately approved repository cleanup; regenerated cache files should remain ignored.

### Verification

Static metadata comparison only; no import was tested against a clean Python cache.

### Prevention

Do not commit Python bytecode; verify tracked files against ignore rules during review.

### Related Files

- `app/__pycache__/`
- `.gitignore`
- `app/*.py`

---

## ISSUE-025 — Potential Issue: Empty Motion Cache Entry Is Invalid JSON

**Status:** Open  
**Severity:** Low  
**Category:** Other

### Problem

**Potential Issue.** `cache/motion/7af18bf36641f6aa_motion_0.8.json` is a zero-byte file and therefore cannot be parsed as JSON. `CacheManager.load()` catches parse exceptions and treats the cache as absent, so the observed code path falls back to recomputation rather than propagating an error. The reason the file was left empty is unknown.

### Error

Actual artifact inspection:

```text
cache/motion/7af18bf36641f6aa_motion_0.8.json: empty
```

### Where It Occurred

- Files: `cache/motion/7af18bf36641f6aa_motion_0.8.json`; `app/cache.py` lines 53-61
- Trigger: loading the corresponding motion cache entry
- Command: JSON artifact inspection; the application loader was not run

### Root Cause

Cannot be confirmed from the repository. An interrupted write is possible but not established.

### Investigation

1. Classified the file and confirmed its size is zero.
2. Parsed all non-empty JSON cache artifacts.
3. Read `CacheManager.load()` and confirmed JSON errors are caught and return `None`.

### Solution

No cache file was deleted or rewritten. The cache loader should recompute on a miss; regenerate the artifact only if a runtime issue is observed.

### Verification

Confirmed the file is empty and the loader catches parse errors by static inspection. No recomputation was run.

### Prevention

Write cache JSON atomically via a temporary file and rename; add malformed/empty-cache recovery tests.

### Related Files

- `cache/motion/7af18bf36641f6aa_motion_0.8.json`
- `app/cache.py`
- `app/video.py`

---

## ISSUE-026 — Repository Inspection Utilities Were Missing

**Status:** Partially Resolved  
**Severity:** Low  
**Category:** Setup

### Problem

The analysis shell did not have the optional `rg` or `jq` executables. This initially blocked fast source-anchor searching and JSON key inspection, but did not block completion of the repository review.

### Error

Actual shell output:

```text
bash: rg: command not found
bash: jq: command not found
```

### Where It Occurred

- Environment: current Linux shell
- Commands: `rg -n ...`; `jq -r ...`
- Project files affected: none; these tools are not listed as application dependencies

### Root Cause

The shell environment does not have ripgrep or jq installed. This is an inspection-environment limitation, not an application runtime defect.

### Investigation

1. Attempted `rg` to locate source lines; the command was unavailable.
2. Attempted `jq` to summarize cache JSON; the command was unavailable.
3. Used `grep -nE` for source locations and Python's standard `json` parser for cache inspection.

### Solution

The repository review was completed using installed portable alternatives. The environment still lacks `rg` and `jq`; no packages were installed.

### Verification

`grep` produced source line references, and Python successfully parsed all non-empty cache JSON files and summarized their schemas.

### Prevention

For this project, keep optional inspection commands documented with fallbacks; install `ripgrep`/`jq` only if desired for local convenience.

### Related Files

- `cache/`
- `app/`

---

## ISSUE-027 — “AI” Name Overstates the Current Matching Implementation

**Status:** Open  
**Severity:** Low  
**Category:** Other

### Problem

The UI and APIs use “AI” terminology, but checked-in runtime code contains heuristic signal processing and scoring only. It does not load a trained model, use prompts/agents/embeddings, or call an AI service. This can create an incorrect expectation about semantic understanding.

### Error

No runtime error. Repository evidence:

```text
Runtime implementation: librosa features + scene/motion heuristics + weighted scoring
Torch/torchvision appear in planning text, not requirements.txt or runtime imports
```

### Where It Occurred

- Files: `app/frontend.py`, `app/sync.py`, `app/beat.py`, `app/video.py`, `requirements.txt`, `ai_implement.md`
- Trigger: interpreting “Generate AI Video” or `ai_assign_clips()` as learned semantic matching

### Root Cause

The project name/UI terminology persisted while the proposed ML feature plan was not implemented in the checked-in runtime.

### Investigation

1. Read all runtime modules and dependency declarations.
2. Searched code paths for model loading, prompts, embeddings, and remote AI calls; none were found.
3. Compared implementation with `ai_implement.md`, which describes proposed features/dependencies.

### Solution

No UI or code change was applied. The project analysis documents the distinction; product wording or implementation can be updated separately.

### Verification

Repository inspection only; no model/API run was expected or performed.

### Prevention

Describe current matching as heuristic automatic editing unless learned model components are actually integrated.

### Related Files

- `app/frontend.py`
- `app/sync.py`
- `app/beat.py`
- `app/video.py`
- `requirements.txt`
- `ai_implement.md`

---

## ISSUE-028 — Potential Issue: Video Analysis Spawns FFmpeg Per Sampled Frame

**Status:** Open  
**Severity:** Medium  
**Category:** Performance

### Problem

**Potential Issue.** Frame-analysis helpers call `extract_frame()`, which launches a new FFmpeg subprocess for one frame. Motion, camera-change, visual-quality, and color analysis loop over multiple timestamps. This may cause substantial process startup/seek overhead on long videos; no current timing comparison was performed.

### Error

No performance error captured. Static evidence:

```text
extract_frame() -> subprocess.run(ffmpeg ... -frames:v 1 ...)
analysis functions call extract_frame() in timestamp loops
```

### Where It Occurred

- File: `app/video.py`
- Functions: `extract_frame()` around lines 192-226; motion and full-analysis functions later in the module
- Trigger: non-fast video analysis or `compute_motion=True`

### Root Cause

The implementation uses one FFmpeg process per requested timestamp instead of decoding a stream or batching frame extraction.

### Investigation

1. Followed frame-extraction callers in motion and profile-analysis loops.
2. Confirmed each call invokes `subprocess.run()` with a single-frame output.
3. Did not run profiling or benchmark frame extraction.

### Solution

No source fix was applied. Profile first; if material, use a persistent decoder or batch extraction while preserving timestamp accuracy and bounded memory.

### Verification

Static call-count risk only; impact remains potential pending measurement.

### Prevention

Benchmark representative video durations, record subprocess counts and wall time, and set a regression target.

### Related Files

- `app/video.py`
- `tools/benchmark.py`
- `optimiz.md`

---

## ISSUE-029 — Potential Issue: Streamlit Upload/Session State May Retain Deleted Paths

**Status:** Open  
**Severity:** Low  
**Category:** Runtime

### Problem

**Potential Issue.** `run_generation()` deletes its temporary directory and uploaded music but leaves associated upload paths in session state. Video change detection compares basenames only, so different files with the same basename can be treated as unchanged. A subsequent rerun/generation could therefore reference deleted files or stale video paths. This lifecycle was not exercised in a running Streamlit session.

### Error

No runtime error captured. Relevant behavior:

```text
run_generation(): removes music and temp_dir in finally
main(): stores uploaded_music_path/uploaded_video_paths and compares video basenames
```

### Where It Occurred

- File: `app/frontend.py`
- Functions: `run_generation()` around lines 435-443; `main()` upload handling around lines 494-670
- Command: interactive Streamlit run, not performed

### Root Cause

Cleanup removes files without consistently clearing or validating the corresponding session-state values; path identity is reduced to basename for change detection. Actual user-visible impact is unconfirmed.

### Investigation

1. Followed temporary-directory creation and worker cleanup.
2. Followed session-state initialization and upload change detection.
3. Confirmed deleted-path references can remain in state by static review.
4. Did not start the UI or repeat a generation.

### Solution

No source fix was applied. Clear or revalidate state after cleanup and compare normalized full paths or stable file identities.

### Verification

Static lifecycle review only; remains potential.

### Prevention

Add UI integration tests for generating twice, replacing music, same-basename videos from separate folders, and failed uploads.

### Related Files

- `app/frontend.py`
