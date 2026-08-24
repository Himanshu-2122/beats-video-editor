import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List
from collections import deque
import statistics


@dataclass
class StageProgress:
    name: str
    weight: float
    started_at: float = 0
    completed_at: float = 0
    progress: float = 0.0
    details: str = ""


@dataclass
class ProcessingStats:
    videos_analyzed: int = 0
    total_videos: int = 0
    scenes_detected: int = 0
    clips_generated: int = 0
    timeline_progress: float = 0.0
    generated_duration: float = 0.0
    target_duration: float = 0.0
    elapsed_time: float = 0.0
    estimated_remaining: float = 0.0


class ProgressTracker:
    def __init__(
        self,
        stages: List[StageProgress],
        total_duration: float,
        frontend_callback: Optional[Callable] = None,
    ):
        self.stages = stages
        self.total_duration = total_duration
        self.frontend_callback = frontend_callback
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.current_stage_idx = 0
        self.stage_history: List[Dict] = []
        self.speed_samples: deque = deque(maxlen=20)
        self.last_update = time.time()
        self.total_weight = sum(s.weight for s in stages)
        self._stats = ProcessingStats()

    @property
    def stats(self) -> ProcessingStats:
        with self.lock:
            self._stats.elapsed_time = time.time() - self.start_time
            return self._stats

    def start_stage(self, stage_idx: int, details: str = ""):
        with self.lock:
            if stage_idx < len(self.stages):
                self.current_stage_idx = stage_idx
                self.stages[stage_idx].started_at = time.time()
                self.stages[stage_idx].progress = 0.0
                self.stages[stage_idx].details = details
                self._notify()

    def update_stage_progress(self, stage_idx: int, progress: float, details: str = ""):
        with self.lock:
            if stage_idx < len(self.stages):
                self.stages[stage_idx].progress = max(0.0, min(1.0, progress))
                self.stages[stage_idx].details = details
                now = time.time()
                if now - self.last_update > 0.5:
                    self._calculate_eta()
                    self.last_update = now
                self._notify()

    def complete_stage(self, stage_idx: int):
        with self.lock:
            if stage_idx < len(self.stages):
                self.stages[stage_idx].completed_at = time.time()
                self.stages[stage_idx].progress = 1.0
                duration = self.stages[stage_idx].completed_at - self.stages[stage_idx].started_at
                self.stage_history.append({
                    "name": self.stages[stage_idx].name,
                    "duration": duration,
                    "weight": self.stages[stage_idx].weight,
                })
                self._calculate_eta()
                self._notify()

    def update_stats(self, **kwargs):
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self._stats, key):
                    setattr(self._stats, key, value)
            if "generated_duration" in kwargs and "target_duration" in kwargs:
                if kwargs["target_duration"] > 0:
                    self._stats.timeline_progress = kwargs["generated_duration"] / kwargs["target_duration"] * 100
            self._notify()

    def _calculate_eta(self):
        elapsed = time.time() - self.start_time
        overall_progress = self.get_overall_progress()

        if overall_progress > 0.01:
            estimated_total = elapsed / overall_progress
            self._stats.estimated_remaining = max(0, estimated_total - elapsed)

            # Update speed samples for smoother ETA
            if elapsed > 0:
                speed = overall_progress / elapsed
                self.speed_samples.append(speed)

    def get_overall_progress(self) -> float:
        completed_weight = 0.0
        for i, stage in enumerate(self.stages):
            if i < self.current_stage_idx:
                completed_weight += stage.weight
            elif i == self.current_stage_idx:
                completed_weight += stage.weight * stage.progress
        return completed_weight / self.total_weight

    def get_progress_percent(self) -> int:
        return int(self.get_overall_progress() * 100)

    def get_current_stage_name(self) -> str:
        if self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx].name
        return "Complete"

    def get_eta_string(self) -> str:
        remaining = self._stats.estimated_remaining
        if remaining <= 0:
            return "--:--"
        mins = int(remaining // 60)
        secs = int(remaining % 60)
        return f"{mins}m {secs}s"

    def get_elapsed_string(self) -> str:
        elapsed = time.time() - self.start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        return f"{mins}m {secs}s"

    def get_progress_bar(self, width: int = 20) -> str:
        progress = self.get_overall_progress()
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        return f"{bar} {int(progress * 100)}%"

    def _notify(self):
        if self.frontend_callback:
            try:
                self.frontend_callback(self.get_dashboard_data())
            except Exception:
                pass

    def get_dashboard_data(self) -> Dict:
        with self.lock:
            return {
                "current_stage": self.get_current_stage_name(),
                "stage_progress": self.stages[self.current_stage_idx].progress if self.current_stage_idx < len(self.stages) else 1.0,
                "stage_details": self.stages[self.current_stage_idx].details if self.current_stage_idx < len(self.stages) else "",
                "overall_progress": self.get_progress_percent(),
                "progress_bar": self.get_progress_bar(),
                "eta": self.get_eta_string(),
                "elapsed": self.get_elapsed_string(),
                "stats": {
                    "videos_analyzed": f"{self._stats.videos_analyzed} / {self._stats.total_videos}",
                    "scenes_detected": self._stats.scenes_detected,
                    "clips_generated": self._stats.clips_generated,
                    "timeline_progress": f"{self._stats.timeline_progress:.0f}%",
                    "generated_duration": f"{self._stats.generated_duration:.1f}s",
                    "target_duration": f"{self._stats.target_duration:.1f}s",
                }
            }


def create_default_stages(total_duration: float) -> List[StageProgress]:
    """Create standard pipeline stages with weights."""
    return [
        StageProgress(name="Analyzing Music", weight=5),
        StageProgress(name="Analyzing Videos", weight=15),
        StageProgress(name="Discovering Clips", weight=10),
        StageProgress(name="Rendering Video (Single-Pass)", weight=60),
        StageProgress(name="Finalizing", weight=10),
    ]