"""PPO-based IoT optimizer wrapper."""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any

from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO


LOGGER = logging.getLogger(__name__)


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
        self.model = self._load_model_with_compat()

    def _load_model_with_compat(self) -> PPO:
        """Load PPO model with compatibility fallback for legacy NumPy pickles."""
        custom_objects = self._build_load_custom_objects()
        try:
            return PPO.load(str(self.weights_path), custom_objects=custom_objects)
        except ValueError as exc:
            error_text = str(exc)
            if "not a known BitGenerator module" not in error_text:
                raise

            LOGGER.warning(
                "Retrying PPO load with NumPy BitGenerator compatibility patch: %s",
                error_text,
            )
            self._patch_numpy_bitgenerator_ctor()
            return PPO.load(str(self.weights_path), custom_objects=custom_objects)

    def _build_load_custom_objects(self) -> dict[str, Any]:
        """Build Stable-Baselines custom object overrides for robust model loading."""
        observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32,
        )
        action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )
        return {
            "observation_space": observation_space,
            "action_space": action_space,
        }

    def _patch_numpy_pickle_compat(self) -> None:
        """Patch module alias needed by older Stable-Baselines pickles."""
        import numpy.core.numeric as np_numeric

        sys.modules.setdefault("numpy._core.numeric", np_numeric)

    def _patch_numpy_bitgenerator_ctor(self) -> None:
        """Patch NumPy BitGenerator constructor for cross-version pickle payloads."""
        import numpy.random._pickle as np_pickle
        import numpy.random as np_random

        bit_generators = getattr(np_pickle, "BitGenerators", None)
        if isinstance(bit_generators, dict):
            for name, generator_cls in list(bit_generators.items()):
                bit_generators.setdefault(generator_cls, generator_cls)
                bit_generators.setdefault(str(generator_cls), generator_cls)
                bit_generators.setdefault(
                    f"<class '{generator_cls.__module__}.{generator_cls.__name__}'>",
                    generator_cls,
                )

            known_names = [
                "MT19937",
                "PCG64",
                "PCG64DXSM",
                "Philox",
                "SFC64",
            ]
            for name in known_names:
                generator_cls = getattr(np_random, name, None)
                if generator_cls is None:
                    continue
                bit_generators.setdefault(name, generator_cls)
                bit_generators.setdefault(generator_cls, generator_cls)
                bit_generators.setdefault(str(generator_cls), generator_cls)
                bit_generators.setdefault(
                    f"<class '{generator_cls.__module__}.{generator_cls.__name__}'>",
                    generator_cls,
                )

        original_ctor = getattr(np_pickle, "__bit_generator_ctor", None)
        if original_ctor is None:
            return

        if getattr(original_ctor, "_geniot_compat_patched", False):
            return

        def _compat_ctor(bit_generator_name: Any) -> Any:
            normalized_name = bit_generator_name
            if isinstance(normalized_name, type):
                normalized_name = normalized_name.__name__
            elif not isinstance(normalized_name, str):
                normalized_name = str(normalized_name)

            if (
                isinstance(normalized_name, str)
                and normalized_name.startswith("<class '")
                and normalized_name.endswith("'>")
            ):
                normalized_name = normalized_name.split("'")[1].split(".")[-1]

            return original_ctor(normalized_name)

        setattr(_compat_ctor, "_geniot_compat_patched", True)
        np_pickle.__bit_generator_ctor = _compat_ctor

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
