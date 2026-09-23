from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import importlib.util
import os
from pathlib import Path

import numpy as np

from control_setup import DEFAULT_CONFIG, TrainingCase, TrainingConfig, iter_training_cases
from ml_controller import PRESSURE_SCALE, PressureResponseModel, TrainingBatch


@dataclass
class SimulationResult:
    pressure_probe: np.ndarray
    max_temperature: np.ndarray
    damp_history: np.ndarray
    autoignition_step: int
    dt: float


def run_simulation_case(case: TrainingCase, config: TrainingConfig) -> SimulationResult:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    base_path = Path(__file__).resolve().parent
    solver_candidates = [
        base_path / "1DSolver_autoignition.py",
        base_path / "1DSolver_autoignition 1.py",
    ]
    solver_path = next((path for path in solver_candidates if path.exists()), solver_candidates[0])
    spec = importlib.util.spec_from_file_location("autoignition_solver", solver_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import solver from {solver_path}")

    solver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver)

    result = solver.run_simulation(
        p_target=case.p_target,
        damp_init=(case.damp1, case.damp2, case.damp3),
        controller=None,
        end_count=config.end_count,
        write_files=False,
        make_plots=False,
        verbose=False,
    )
    return SimulationResult(
        pressure_probe=result.pressure_probe,
        max_temperature=result.max_temperature,
        damp_history=result.damp_history,
        autoignition_step=result.autoignition_step,
        dt=result.dt,
    )


def make_candidate_damps(config: TrainingConfig) -> list[tuple[float, float, float]]:
    return [
        tuple(map(float, values))
        for values in product(config.damp1_values, config.damp2_values, config.damp3_values)
    ]


def build_training_batch(
    result: SimulationResult,
    case: TrainingCase,
    config: TrainingConfig,
) -> TrainingBatch:
    features: list[np.ndarray] = []
    targets: list[float] = []

    start = max(result.autoignition_step, config.history_length)
    stop = len(result.pressure_probe) - config.prediction_horizon

    if result.autoignition_step < 0:
        raise ValueError("Autoignition was not detected in this simulation.")

    controller = PressureResponseModel(config.history_length)

    for step in range(start, stop):
        pressure_history = result.pressure_probe[step - config.history_length : step]
        current_damp = tuple(float(v) for v in result.damp_history[step])
        actual_future_pressure = result.pressure_probe[step + config.prediction_horizon]
        time_since_ignition = (step - result.autoignition_step) * result.dt

        feature = controller.make_feature(
            pressure_history=pressure_history,
            p_target=case.p_target,
            max_temperature=float(result.max_temperature[step]),
            time_since_ignition=time_since_ignition,
            current_damp=current_damp,
            candidate_damp=current_damp,
        )
        features.append(feature)
        targets.append(actual_future_pressure / PRESSURE_SCALE)

    if not features:
        raise ValueError("Simulation did not produce enough post-ignition samples.")

    return TrainingBatch(
        features=np.stack(features).astype(np.float32),
        targets=np.asarray(targets, dtype=np.float32),
    )


def train_sequentially(config: TrainingConfig = DEFAULT_CONFIG) -> PressureResponseModel:
    model = PressureResponseModel(
        history_length=config.history_length,
        learning_rate=config.learning_rate,
    )

    for case_id, case in enumerate(iter_training_cases(config), start=1):
        print(
            f"[case {case_id}] p_target={case.p_target:.1f}, "
            f"damp=({case.damp1:.3f}, {case.damp2:.3f}, {case.damp3:.3f})"
        )
        result = run_simulation_case(case, config)
        batch = build_training_batch(result, case, config)
        loss = model.train_on_batch(
            batch,
            epochs=config.train_epochs_per_run,
            batch_size=config.batch_size,
        )
        print(f"[case {case_id}] samples={len(batch.targets)}, loss={loss:.6e}")
        model.save(config.model_path)

    return model


if __name__ == "__main__":
    train_sequentially()
