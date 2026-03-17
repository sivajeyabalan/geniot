"""PPO-based IoT optimizer wrapper."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import PPO


class IoTOptimizer:
    """Wrapper around a trained PPO model for configuration recommendations."""

    def __init__(self, weights_path: str | Path) -> None:
        """Load a trained PPO model from disk.

        Args:
            weights_path: Path to PPO checkpoint (`.zip`).

        Raises:
            FileNotFoundError: If checkpoint path does not exist.
        """
        self.weights_path = self._resolve_weights_path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"PPO weights not found: {self.weights_path}")
        self._patch_numpy_pickle_compat()
        self.model = PPO.load(str(self.weights_path))

    def _patch_numpy_pickle_compat(self) -> None:
        """Patch module alias needed by older Stable-Baselines pickles."""
        import numpy.core.numeric as np_numeric

        sys.modules.setdefault("numpy._core.numeric", np_numeric)

    def _resolve_weights_path(self, weights_path: str | Path) -> Path:
        """Resolve model path across common project-relative locations.

        Args:
            weights_path: User-provided path.

        Returns:
            Resolved absolute or project-relative model path.
        """
        requested = Path(weights_path)
        if requested.exists():
            return requested

        project_root = Path(__file__).resolve().parents[2]
        candidates = [
            project_root / requested,
            project_root / "backend" / "models" / "weights" / requested.name,
            project_root / "weights" / requested.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return requested

    def recommend_config(self, network_state: np.ndarray) -> dict[str, float]:
        """Recommend an action configuration for a given network state.

        Args:
            network_state: 12-dimensional normalized network state vector.

        Returns:
            Dictionary with action components and expected reward estimate:
            ``{"routing", "sleep", "power", "buffer", "expected_reward"}``.

        Raises:
            ValueError: If input state shape is not ``(12,)``.
        """
        state = np.asarray(network_state, dtype=np.float32)
        if state.shape != (12,):
            raise ValueError(
                "network_state must have shape (12,), "
                f"got {state.shape}"
            )

        action, _ = self.model.predict(state, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(4,)
        action = np.clip(action, 0.0, 1.0)

        latency = float(state[0])
        throughput = float(state[1])
        energy = float(state[2])
        qos = float(state[4])
        expected_reward = (
            0.3 * (1.0 - latency)
            + 0.3 * throughput
            + 0.2 * (1.0 - energy)
            + 0.2 * qos
        )

        return {
            "routing": float(action[0]),
            "sleep": float(action[1]),
            "power": float(action[2]),
            "buffer": float(action[3]),
            "expected_reward": float(expected_reward),
        }
