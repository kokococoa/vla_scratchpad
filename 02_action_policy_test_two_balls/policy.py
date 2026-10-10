"""Neural-network policy and training utilities for the two-ball game."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_DATA_DIRECTORY = PROJECT_DIRECTORY / "training_data"
DEFAULT_WEIGHT_DIRECTORY = PROJECT_DIRECTORY / "weight"
STATE_FIELDS = ("x_b", "y_b", "x_r", "y_r", "picked_up")
ACTION_FIELDS = ("dx_b", "dy_b", "pick_up")

# Training settings
TRAINING_EPOCHS = 200
TRAINING_BATCH_SIZE = 64
TRAINING_LEARNING_RATE = 1e-3
TRAINING_RANDOM_SEED = 110
TRAINING_PROGRESS_REPORTS = 100
FILTER_UNPICKED_IDLE_SAMPLES = True
RELEASE_LOSS_WEIGHT = 8.0


class BallPolicy(nn.Module):
    """A 5 -> 64 -> 64 -> 3 multilayer perceptron."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


@dataclass(frozen=True)
class TrainingResult:
    weight_path: Path
    data_directory: Path
    samples: int
    final_loss: float


@dataclass
class PolicyController:
    """A trained policy together with the normalization used during training."""

    model: BallPolicy
    state_mean: torch.Tensor
    state_std: torch.Tensor
    movement_mean: torch.Tensor
    movement_std: torch.Tensor
    weight_path: Path

    def action(
        self, state: tuple[float, float, float, float, float]
    ) -> tuple[float, float, float]:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        normalized_state = (state_tensor - self.state_mean) / self.state_std
        with torch.no_grad():
            output = self.model(normalized_state)
        movement = output[:2] * self.movement_std + self.movement_mean
        pick_up_probability = torch.sigmoid(output[2])
        return float(movement[0]), float(movement[1]), float(pick_up_probability)


def latest_data_directory(data_root: Path = DEFAULT_DATA_DIRECTORY) -> Path:
    """Return the newest timestamp-named training data directory."""
    candidates = sorted(
        (path for path in data_root.glob("data_*") if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No training data folders found in {data_root}")
    return candidates[0]


def load_latest_policy(
    weight_root: Path = DEFAULT_WEIGHT_DIRECTORY,
) -> PolicyController:
    """Load the newest saved checkpoint for inference."""
    candidates = sorted(
        (path for path in weight_root.glob("weight_*.pt") if path.is_file()),
        key=lambda path: path.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No model weights found in {weight_root}")

    weight_path = candidates[0]
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
    required_keys = {
        "model_state_dict",
        "state_mean",
        "state_std",
        "movement_mean",
        "movement_std",
    }
    if not required_keys.issubset(checkpoint):
        raise ValueError(f"Invalid policy checkpoint: {weight_path.name}")

    model = BallPolicy()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return PolicyController(
        model=model,
        state_mean=checkpoint["state_mean"].to(dtype=torch.float32),
        state_std=checkpoint["state_std"].to(dtype=torch.float32).clamp_min(1e-6),
        movement_mean=checkpoint["movement_mean"].to(dtype=torch.float32),
        movement_std=checkpoint["movement_std"]
        .to(dtype=torch.float32)
        .clamp_min(1e-6),
        weight_path=weight_path,
    )


def load_training_data(data_directory: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load all numbered CSV rounds from one training session."""
    csv_paths = sorted(
        (path for path in data_directory.glob("*.csv") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    if not csv_paths:
        raise ValueError(f"No round CSV files found in {data_directory}")

    states: list[list[float]] = []
    actions: list[list[float]] = []
    required_fields = set(STATE_FIELDS + ACTION_FIELDS)
    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8") as data_file:
            reader = csv.DictReader(data_file)
            if not required_fields.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing required columns in {csv_path.name}")
            for line_number, row in enumerate(reader, start=2):
                try:
                    states.append([float(row[field]) for field in STATE_FIELDS])
                    actions.append([float(row[field]) for field in ACTION_FIELDS])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid numeric value in {csv_path.name}, line {line_number}"
                    ) from error

    if not states:
        raise ValueError(f"The CSV files in {data_directory} contain no samples")
    return torch.tensor(states, dtype=torch.float32), torch.tensor(
        actions, dtype=torch.float32
    )


def train_latest_policy(
    data_root: Path = DEFAULT_DATA_DIRECTORY,
    weight_root: Path = DEFAULT_WEIGHT_DIRECTORY,
    epochs: int = TRAINING_EPOCHS,
    batch_size: int = TRAINING_BATCH_SIZE,
    learning_rate: float = TRAINING_LEARNING_RATE,
) -> TrainingResult:
    """Train from the latest data session and save a timestamped checkpoint."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    started_at = time.perf_counter()
    data_directory = latest_data_directory(data_root)
    csv_count = sum(
        1
        for path in data_directory.glob("*.csv")
        if path.is_file() and path.stem.isdigit()
    )
    print("\n[training] Starting policy training", flush=True)
    print(f"[training] Data folder: {data_directory}", flush=True)
    print(f"[training] Round CSV files: {csv_count}", flush=True)
    states, actions = load_training_data(data_directory)
    print(f"[training] Loaded samples: {len(states)}", flush=True)
    if FILTER_UNPICKED_IDLE_SAMPLES:
        idle_unpicked = (
            (states[:, 4] < 0.5)
            & (actions[:, 2] < 0.5)
            & (actions[:, 0].abs() < 1e-6)
            & (actions[:, 1].abs() < 1e-6)
        )
        removed_samples = int(idle_unpicked.sum().item())
        states = states[~idle_unpicked]
        actions = actions[~idle_unpicked]
        print(
            f"[training] Removed idle unpicked samples: {removed_samples}",
            flush=True,
        )
        print(f"[training] Samples used: {len(states)}", flush=True)
        if not len(states):
            raise ValueError("No actionable samples remain after idle filtering")
    state_mean = states.mean(dim=0)
    state_std = states.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized_states = (states - state_mean) / state_std
    movements = actions[:, :2]
    movement_mean = movements.mean(dim=0)
    movement_std = movements.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized_movements = (movements - movement_mean) / movement_std
    print(f"[training] State mean: {state_mean.tolist()}", flush=True)
    print(f"[training] State std:  {state_std.tolist()}", flush=True)
    print(f"[training] Movement mean: {movement_mean.tolist()}", flush=True)
    print(f"[training] Movement std:  {movement_std.tolist()}", flush=True)

    torch.manual_seed(TRAINING_RANDOM_SEED)
    model = BallPolicy()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"[training] Model parameters: {parameter_count:,}", flush=True)
    print(
        f"[training] Epochs: {epochs} | batch size: "
        f"{min(batch_size, len(states))} | learning rate: {learning_rate}",
        flush=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    movement_loss_function = nn.MSELoss()
    pickup_loss_function = nn.BCEWithLogitsLoss(reduction="none")
    dataset = TensorDataset(normalized_states, normalized_movements, actions[:, 2])
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
    )

    model.train()
    final_loss = 0.0
    report_interval = max(1, epochs // max(1, TRAINING_PROGRESS_REPORTS))
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_states, batch_movements, batch_pickups in loader:
            optimizer.zero_grad()
            predictions = model(batch_states)
            movement_loss = movement_loss_function(
                predictions[:, :2], batch_movements
            )
            pickup_losses = pickup_loss_function(predictions[:, 2], batch_pickups)
            pickup_weights = torch.where(
                batch_pickups < 0.5,
                torch.full_like(batch_pickups, RELEASE_LOSS_WEIGHT),
                torch.ones_like(batch_pickups),
            )
            pickup_loss = (pickup_losses * pickup_weights).mean()
            loss = movement_loss + pickup_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_states)
        final_loss = total_loss / len(dataset)
        if epoch == 1 or epoch % report_interval == 0 or epoch == epochs:
            elapsed = time.perf_counter() - started_at
            print(
                f"[training] Epoch {epoch:>{len(str(epochs))}}/{epochs} | "
                f"loss: {final_loss:.6f} | elapsed: {elapsed:.2f}s",
                flush=True,
            )

    weight_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    weight_path = weight_root / f"weight_{timestamp}.pt"
    suffix = 1
    while weight_path.exists():
        weight_path = weight_root / f"weight_{timestamp}_{suffix}.pt"
        suffix += 1

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_mean": state_mean,
            "state_std": state_std,
            "movement_mean": movement_mean,
            "movement_std": movement_std,
            "state_fields": STATE_FIELDS,
            "action_fields": ACTION_FIELDS,
            "source_data_directory": data_directory.name,
            "samples": len(dataset),
            "epochs": epochs,
            "final_loss": final_loss,
        },
        weight_path,
    )
    elapsed = time.perf_counter() - started_at
    print(f"[training] Saved checkpoint: {weight_path}", flush=True)
    print(
        f"[training] Complete | final loss: {final_loss:.6f} | "
        f"total time: {elapsed:.2f}s\n",
        flush=True,
    )
    return TrainingResult(weight_path, data_directory, len(dataset), final_loss)


if __name__ == "__main__":
    result = train_latest_policy()
    print(
        f"Saved {result.weight_path} from {result.samples} samples "
        f"(loss={result.final_loss:.6f})"
    )
