"""Forge trainer — training loop with epochs, loss tracking, checkpointing, and evaluation."""
import time
import json
import os
import math
from dataclasses import dataclass, field
from typing import Optional, Callable
from collections import defaultdict

@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    epochs: int = 10
    batch_size: int = 32
    checkpoint_interval: int = 5
    early_stop_patience: int = 3
    min_delta: float = 0.001
    warmup_steps: int = 100
    weight_decay: float = 0.0001

@dataclass
class EpochResult:
    epoch: int
    train_loss: float = 0.0
    val_loss: float = 0.0
    accuracy: float = 0.0
    learning_rate: float = 0.0
    duration_s: float = 0.0
    samples: int = 0

@dataclass
class Checkpoint:
    epoch: int
    train_loss: float
    val_loss: float
    path: str = ""
    timestamp: float = field(default_factory=time.time)

class ForgeTrainer:
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self._history: list[EpochResult] = []
        self._checkpoints: list[Checkpoint] = []
        self._best_val_loss = float('inf')
        self._patience_counter = 0
        self._current_lr = self.config.learning_rate
        self._step = 0
        self._eval_fn: Optional[Callable] = None
        self._stats = {"total_steps": 0, "total_samples": 0, "start_time": 0.0}

    def set_evaluator(self, fn: Callable):
        self._eval_fn = fn

    def train_step(self, loss_fn: Callable, batch: list) -> float:
        """Execute one training step. loss_fn(batch) -> float."""
        loss = loss_fn(batch)
        self._step += 1
        # Learning rate warmup
        if self._step < self.config.warmup_steps:
            self._current_lr = self.config.learning_rate * (self._step / self.config.warmup_steps)
        else:
            self._current_lr = self.config.learning_rate
        self._stats["total_steps"] += 1
        self._stats["total_samples"] += len(batch)
        return loss

    def epoch(self, train_fn: Callable, val_fn: Callable = None,
              data: list = None, epoch_num: int = 0) -> EpochResult:
        """Run one full epoch."""
        start = time.time()
        if not self._stats["start_time"]:
            self._stats["start_time"] = time.time()

        train_loss = train_fn(epoch_num)
        val_loss = val_fn(epoch_num) if val_fn else 0.0
        accuracy = self._eval_fn(data) if self._eval_fn and data else 0.0

        result = EpochResult(epoch=epoch_num, train_loss=train_loss,
                            val_loss=val_loss, accuracy=accuracy,
                            learning_rate=self._current_lr,
                            duration_s=round(time.time() - start, 2),
                            samples=len(data) if data else 0)
        self._history.append(result)

        # Early stopping check
        if val_loss > 0:
            if val_loss < self._best_val_loss - self.config.min_delta:
                self._best_val_loss = val_loss
                self._patience_counter = 0
            else:
                self._patience_counter += 1

        # Auto-checkpoint
        if (epoch_num + 1) % self.config.checkpoint_interval == 0:
            self._checkpoints.append(Checkpoint(epoch=epoch_num,
                                                train_loss=train_loss,
                                                val_loss=val_loss))
        return result

    def should_stop(self) -> bool:
        if self._patience_counter >= self.config.early_stop_patience:
            return True
        if len(self._history) >= self.config.epochs:
            return True
        return False

    def best_epoch(self) -> Optional[EpochResult]:
        if not self._history:
            return None
        valid = [e for e in self._history if e.val_loss > 0]
        return min(valid, key=lambda e: e.val_loss) if valid else self._history[-1]

    def learning_curve(self) -> dict:
        return {"train_loss": [e.train_loss for e in self._history],
                "val_loss": [e.val_loss for e in self._history],
                "accuracy": [e.accuracy for e in self._history],
                "epochs": [e.epoch for e in self._history]}

    def save_checkpoint(self, path: str, epoch: int, metadata: dict = None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        latest = self._history[-1] if self._history else None
        data = {"epoch": epoch, "step": self._step, "lr": self._current_lr,
                "best_val_loss": self._best_val_loss,
                "latest_loss": latest.train_loss if latest else 0,
                "metadata": metadata or {}, "timestamp": time.time()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_checkpoint(self, path: str) -> dict:
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            data = json.load(f)
        self._step = data.get("step", 0)
        self._best_val_loss = data.get("best_val_loss", float('inf'))
        self._current_lr = data.get("lr", self.config.learning_rate)
        return data

    def summary(self) -> dict:
        total_time = time.time() - self._stats["start_time"] if self._stats["start_time"] else 0
        losses = [e.train_loss for e in self._history]
        return {"epochs_completed": len(self._history),
                "best_train_loss": min(losses) if losses else 0,
                "best_val_loss": self._best_val_loss,
                "total_steps": self._step,
                "total_samples": self._stats["total_samples"],
                "total_time_s": round(total_time, 1),
                "current_lr": self._current_lr,
                "checkpoints": len(self._checkpoints),
                "early_stopped": self.should_stop()}

    @property
    def stats(self) -> dict:
        return {**self.summary(), "config": {"lr": self.config.learning_rate,
                "epochs": self.config.epochs, "batch_size": self.config.batch_size,
                "patience": self.config.early_stop_patience}}
