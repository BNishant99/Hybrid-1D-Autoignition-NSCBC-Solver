from dataclasses import dataclass
from itertools import product
from typing import Iterator, Sequence

import numpy as np


ATM_PRESSURE = 101325.0


@dataclass(frozen=True)
class TrainingCase:
    p_target: float
    damp1: float
    damp2: float
    damp3: float


@dataclass(frozen=True)
class TrainingConfig:
    p_targets: Sequence[float]
    damp1_values: Sequence[float]
    damp2_values: Sequence[float]
    damp3_values: Sequence[float]
    history_length: int = 8
    prediction_horizon: int = 25
    batch_size: int = 256
    train_epochs_per_run: int = 8
    learning_rate: float = 1.0e-3
    end_count: int = 200000
    max_cases: int | None = None
    model_path: str = "pressure_response_model.pt"


DEFAULT_CONFIG = TrainingConfig(
    p_targets=np.linspace(0.95 * ATM_PRESSURE, 1.15 * ATM_PRESSURE, 5),
    damp1_values=np.linspace(0.05, 0.60, 4),
    damp2_values=np.linspace(0.05, 0.60, 4),
    damp3_values=np.linspace(0.05, 0.60, 4),
)


def iter_training_cases(config: TrainingConfig = DEFAULT_CONFIG) -> Iterator[TrainingCase]:
    cases = product(
        config.p_targets,
        config.damp1_values,
        config.damp2_values,
        config.damp3_values,
    )
    for idx, values in enumerate(cases):
        if config.max_cases is not None and idx >= config.max_cases:
            break
        yield TrainingCase(*map(float, values))
