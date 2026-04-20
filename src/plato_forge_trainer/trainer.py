"""GPU training job management."""
import time
from dataclasses import dataclass, field
from enum import Enum

class TrainingMode(Enum):
    LORA = "lora"
    EMBEDDING = "embedding"
    GENOME = "genome"

@dataclass
class GpuBudget:
    total_vram_mb: int = 6144
    model_mb: int = 3500
    batch_size: int = 1
    max_seq_len: int = 512

    def available_vram(self) -> int:
        return self.total_vram_mb - self.model_mb

    def estimate_batch_vram(self, batch_size: int, seq_len: int) -> int:
        return batch_size * seq_len * 4  # rough estimate

@dataclass
class TrainingJob:
    name: str
    mode: TrainingMode
    steps: int = 100
    learning_rate: float = 2e-5
    status: str = "queued"
    started_at: float = 0.0
    completed_at: float = 0.0
    final_loss: float = 0.0
    steps_per_sec: float = 0.0

class ForgeTrainer:
    def __init__(self, budget: GpuBudget = None):
        self.budget = budget or GpuBudget()
        self._queue: list[TrainingJob] = []
        self._history: list[TrainingJob] = []

    def submit(self, name: str, mode: str = "lora", steps: int = 100, lr: float = 2e-5) -> TrainingJob:
        job = TrainingJob(name=name, mode=TrainingMode(mode), steps=steps, learning_rate=lr)
        self._queue.append(job)
        return job

    def start_next(self) -> TrainingJob:
        for job in self._queue:
            if job.status == "queued":
                job.status = "running"
                job.started_at = time.time()
                return job
        return None

    def complete(self, name: str, final_loss: float = 0.0, steps_per_sec: float = 0.0):
        for job in self._queue:
            if job.name == name and job.status == "running":
                job.status = "completed"
                job.completed_at = time.time()
                job.final_loss = final_loss
                job.steps_per_sec = steps_per_sec
                self._history.append(job)
                return

    def active_jobs(self) -> list[TrainingJob]:
        return [j for j in self._queue if j.status == "running"]

    def completed_jobs(self) -> list[TrainingJob]:
        return self._history

    @property
    def stats(self) -> dict:
        return {"queued": len([j for j in self._queue if j.status == "queued"]),
                "running": len(self.active_jobs()),
                "completed": len(self._history),
                "vram_budget_mb": self.budget.total_vram_mb}
