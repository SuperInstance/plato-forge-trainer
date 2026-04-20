"""Forge trainer — GPU training job manager, LoRA/embedding/genome modes.
Part of the PLATO framework."""
from .trainer import ForgeTrainer, TrainingJob, GpuBudget
__version__ = "0.1.0"
__all__ = ["ForgeTrainer", "TrainingJob", "GpuBudget"]
