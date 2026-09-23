from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PRESSURE_SCALE = 101325.0
TEMPERATURE_SCALE = 2000.0
TIME_SCALE = 1.0e-3


@dataclass
class TrainingBatch:
    features: np.ndarray
    targets: np.ndarray


class PressureResponseNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PressureResponseModel:
    """
    Learns:
        current state + candidate damp1/2/3 + p_target -> future probe pressure.

    During deployment, evaluate many candidate damping triplets and choose the one
    whose predicted future pressure is closest to p_target.
    """

    def __init__(self, history_length: int, learning_rate: float = 1.0e-3):
        self.history_length = history_length
        self.input_dim = history_length + 11
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PressureResponseNet(self.input_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def make_feature(
        self,
        pressure_history: Iterable[float],
        p_target: float,
        max_temperature: float,
        time_since_ignition: float,
        current_damp: tuple[float, float, float],
        candidate_damp: tuple[float, float, float],
    ) -> np.ndarray:
        p_hist = np.asarray(list(pressure_history), dtype=np.float32)
        if p_hist.size != self.history_length:
            raise ValueError(f"Expected {self.history_length} pressure-history values.")

        p_now = p_hist[-1]
        dp = p_hist[-1] - p_hist[-2] if p_hist.size >= 2 else 0.0

        return np.asarray(
            [
                *(p_hist / PRESSURE_SCALE),
                p_target / PRESSURE_SCALE,
                (p_target - p_now) / PRESSURE_SCALE,
                dp / PRESSURE_SCALE,
                max_temperature / TEMPERATURE_SCALE,
                time_since_ignition / TIME_SCALE,
                *current_damp,
                *candidate_damp,
            ],
            dtype=np.float32,
        )

    def train_on_batch(self, batch: TrainingBatch, epochs: int, batch_size: int) -> float:
        x = torch.as_tensor(batch.features, dtype=torch.float32)
        y = torch.as_tensor(batch.targets, dtype=torch.float32).reshape(-1, 1)
        loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

        self.model.train()
        last_loss = 0.0
        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.loss_fn(self.model(xb), yb)
                loss.backward()
                self.optimizer.step()
                last_loss = float(loss.detach().cpu())
        return last_loss

    @torch.no_grad()
    def predict_future_pressure(self, features: np.ndarray) -> np.ndarray:
        self.model.eval()
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        pred = self.model(x).detach().cpu().numpy().reshape(-1)
        return pred * PRESSURE_SCALE

    @torch.no_grad()
    def choose_damping(
        self,
        pressure_history: Iterable[float],
        p_target: float,
        max_temperature: float,
        time_since_ignition: float,
        current_damp: tuple[float, float, float],
        candidate_damps: Iterable[tuple[float, float, float]],
        rate_limit: float = 0.05,
    ) -> tuple[float, float, float]:
        candidates = list(candidate_damps)
        features = np.stack(
            [
                self.make_feature(
                    pressure_history,
                    p_target,
                    max_temperature,
                    time_since_ignition,
                    current_damp,
                    damp,
                )
                for damp in candidates
            ]
        )
        predicted = self.predict_future_pressure(features)

        current = np.asarray(current_damp, dtype=np.float32)
        changes = np.asarray(candidates, dtype=np.float32) - current
        penalty = 0.05 * np.sum(changes * changes, axis=1) * PRESSURE_SCALE
        best_idx = int(np.argmin(np.abs(predicted - p_target) + penalty))

        selected = np.asarray(candidates[best_idx], dtype=np.float32)
        selected = current + np.clip(selected - current, -rate_limit, rate_limit)
        return tuple(float(v) for v in selected)

    def save(self, path: str) -> None:
        torch.save(
            {
                "history_length": self.history_length,
                "input_dim": self.input_dim,
                "state_dict": self.model.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, learning_rate: float = 1.0e-3) -> "PressureResponseModel":
        checkpoint = torch.load(path, map_location="cpu")
        controller = cls(checkpoint["history_length"], learning_rate=learning_rate)
        controller.model.load_state_dict(checkpoint["state_dict"])
        return controller
