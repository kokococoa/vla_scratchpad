"""CSV recording utilities for human demonstration data."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import TextIO


CSV_FIELDS = (
    "x_b", "y_b", "x_r", "y_r", "picked_up", "dx_b", "dy_b", "pick_up"
)


class TrainingDataCollector:
    """Manage one timestamped training session and its numbered rounds."""

    def __init__(self, parent_directory: Path | None = None) -> None:
        self.parent_directory = parent_directory or (
            Path(__file__).resolve().parent / "training_data"
        )
        self.session_directory: Path | None = None
        self.completed_rounds = 0
        self._round_number = 0
        self._file: TextIO | None = None
        self._writer: csv.DictWriter | None = None
        self._previous_blue: tuple[float, float] | None = None
        self._previous_picked_up = False
        self._demonstration_started = False

    @property
    def active(self) -> bool:
        return self.session_directory is not None

    @property
    def recording_round(self) -> bool:
        return self._writer is not None

    def begin_session(
        self,
        blue_position: tuple[float, float],
        red_position: tuple[float, float],
        picked_up: bool = False,
    ) -> Path:
        """Create the timestamped session folder and begin its first round."""
        folder_name = datetime.now().strftime("data_%y%m%d_%H%M")
        self.session_directory = self.parent_directory / folder_name
        self.session_directory.mkdir(parents=True, exist_ok=True)

        existing_numbers = [
            int(path.stem)
            for path in self.session_directory.glob("*.csv")
            if path.stem.isdigit()
        ]
        self._round_number = max(existing_numbers, default=0)
        self.completed_rounds = self._round_number
        self.begin_round(blue_position, red_position, picked_up)
        return self.session_directory

    def continue_latest_session(
        self,
        blue_position: tuple[float, float],
        red_position: tuple[float, float],
        picked_up: bool = False,
    ) -> Path:
        """Continue the newest session by opening its next numbered round."""
        candidates = sorted(
            (
                path
                for path in self.parent_directory.glob("data_*")
                if path.is_dir()
            ),
            key=lambda path: path.name,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No training data folders found in {self.parent_directory}"
            )

        self.session_directory = candidates[0]
        existing_numbers = [
            int(path.stem)
            for path in self.session_directory.glob("*.csv")
            if path.stem.isdigit()
        ]
        self._round_number = max(existing_numbers, default=0)
        self.completed_rounds = self._round_number
        self.begin_round(blue_position, red_position, picked_up)
        return self.session_directory

    def begin_round(
        self,
        blue_position: tuple[float, float],
        red_position: tuple[float, float],
        picked_up: bool = False,
    ) -> Path:
        """Open the next sequential CSV and initialize tick tracking."""
        if self.session_directory is None:
            raise RuntimeError("Begin a training session before beginning a round")

        self._close_file()
        self._round_number += 1
        csv_path = self.session_directory / f"{self._round_number}.csv"
        self._file = csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._file.flush()
        self._previous_blue = (float(blue_position[0]), float(blue_position[1]))
        self._previous_picked_up = picked_up
        self._demonstration_started = picked_up
        return csv_path

    def record_tick(
        self,
        blue_position: tuple[float, float],
        red_position: tuple[float, float],
        picked_up: bool,
    ) -> None:
        """Record the prior state and the blue-ball action during this tick."""
        if self._writer is None or self._previous_blue is None:
            return

        # Waiting before the human first picks up the ball is not a useful
        # control demonstration. Keep the initial state, but do not write idle
        # rows; the first written row will be the pickup transition itself.
        if not self._demonstration_started:
            if not picked_up:
                return
            self._demonstration_started = True

        current_blue = (float(blue_position[0]), float(blue_position[1]))
        previous_x, previous_y = self._previous_blue
        self._writer.writerow(
            {
                "x_b": previous_x,
                "y_b": previous_y,
                "x_r": float(red_position[0]),
                "y_r": float(red_position[1]),
                "picked_up": float(self._previous_picked_up),
                "dx_b": current_blue[0] - previous_x,
                "dy_b": current_blue[1] - previous_y,
                "pick_up": float(picked_up),
            }
        )
        self._file.flush()
        self._previous_blue = current_blue
        self._previous_picked_up = picked_up

    def finish_round(self) -> None:
        """Close the current CSV and count it as a completed round."""
        if not self.recording_round:
            return
        self._close_file()
        self.completed_rounds += 1

    def close(self) -> None:
        """Close an in-progress file when the application exits."""
        self._close_file()

    def _close_file(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None
        self._previous_blue = None
        self._previous_picked_up = False
        self._demonstration_started = False
