"""Network simulator utilities built on top of IoTNetworkEnv."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.environment.iot_network_env import IoTNetworkEnv


@dataclass
class SimulationResult:
    """Container for simulation outputs."""

    observations: np.ndarray
    rewards: np.ndarray
    infos: list[dict[str, Any]]


class NetworkSimulator:
    """Wrapper for running rollouts in the IoT network environment."""

    def __init__(self, env: IoTNetworkEnv | None = None) -> None:
        """Initialize simulator.

        Args:
            env: Optional pre-initialized environment instance.
        """
        self.env = env or IoTNetworkEnv()

    def run_random_policy(
        self,
        n_steps: int = 100,
        seed: int | None = None,
    ) -> SimulationResult:
        """Run rollout with random actions sampled from action space.

        Args:
            n_steps: Maximum number of transition steps.
            seed: Optional seed for reset reproducibility.

        Returns:
            SimulationResult with observations, rewards, and step info.
        """
        obs, info = self.env.reset(seed=seed)
        observations = [obs.copy()]
        rewards: list[float] = []
        infos: list[dict[str, Any]] = [info]

        for _ in range(n_steps):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, step_info = self.env.step(action)
            observations.append(obs.copy())
            rewards.append(float(reward))
            infos.append(step_info)
            if terminated or truncated:
                break

        return SimulationResult(
            observations=np.asarray(observations, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            infos=infos,
        )
