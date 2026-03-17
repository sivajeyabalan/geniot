"""Custom Gymnasium environment for IoT network optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import gymnasium as gym
import numpy as np
from gymnasium import spaces

warnings.filterwarnings(
    "ignore",
    message="We recommend you to use a symmetric and normalized Box action space*",
    module="stable_baselines3.common.env_checker",
)

DEFAULT_CONFIG: dict[str, float | int] = {
    "n_nodes": 100,
    "max_steps": 200,
    "noise_std": 0.02,
    "anomaly_probability": 0.05,
}


@dataclass(frozen=True)
class StateIndex:
    """Index map for IoT network state vector."""

    latency: int = 0
    throughput: int = 1
    energy: int = 2
    packet_loss: int = 3
    qos_score: int = 4
    congestion: int = 5
    n_active_nodes: int = 6
    collision_rate: int = 7
    hop_count: int = 8
    buffer_occupancy: int = 9
    anomaly_score: int = 10
    channel_quality: int = 11


STATE_INDEX = StateIndex()


class IoTNetworkEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium environment to simulate and optimize IoT network behavior.

    Observation space contains 12 normalized network metrics. Action space contains
    4 normalized control signals that represent network configuration levers.

    Attributes:
        observation_space: 12-dimensional normalized metric vector.
        action_space: 4-dimensional normalized action vector.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict[str, float | int] | None = None) -> None:
        """Initialize IoT network environment.

        Args:
            config: Optional configuration dictionary overriding defaults.
        """
        super().__init__()
        merged = dict(DEFAULT_CONFIG)
        if config is not None:
            merged.update(config)
        self.config = merged

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        self._step_count = 0
        self._state = self._initial_state()

    def _initial_state(self) -> np.ndarray:
        """Create a randomized initial state near the baseline operating point.

        Returns:
            Initial observation vector clipped to ``[0, 1]``.
        """
        baseline = np.array(
            [0.6, 0.4, 0.6, 0.1, 0.7, 0.3, 0.9, 0.1, 0.4, 0.3, 0.1, 0.8],
            dtype=np.float32,
        )
        noise = self.np_random.normal(loc=0.0, scale=0.03, size=baseline.shape)
        return np.clip(baseline + noise, 0.0, 1.0).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment state.

        Args:
            seed: Optional random seed used by Gymnasium.
            options: Optional reset options.

        Returns:
            Tuple of ``(observation, info)``.
        """
        del options
        super().reset(seed=seed)
        self._step_count = 0
        self._state = self._initial_state()
        return self._state.copy(), {"step": self._step_count}

    def _apply_action(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Apply control action to current network state.

        Args:
            state: Current environment state.
            action: Action vector with 4 normalized controls.

        Returns:
            Updated state after deterministic action effects.
        """
        routing_aggressiveness, sleep_duty_cycle, transmission_power, buffer_size = action

        next_state = state.copy()

        next_state[STATE_INDEX.latency] -= 0.15 * routing_aggressiveness
        next_state[STATE_INDEX.hop_count] -= 0.10 * routing_aggressiveness
        next_state[STATE_INDEX.throughput] += 0.12 * routing_aggressiveness

        next_state[STATE_INDEX.energy] -= 0.14 * sleep_duty_cycle
        next_state[STATE_INDEX.n_active_nodes] -= 0.10 * sleep_duty_cycle
        next_state[STATE_INDEX.latency] += 0.05 * sleep_duty_cycle

        next_state[STATE_INDEX.throughput] += 0.10 * transmission_power
        next_state[STATE_INDEX.channel_quality] += 0.06 * transmission_power
        next_state[STATE_INDEX.energy] += 0.12 * transmission_power
        next_state[STATE_INDEX.collision_rate] += 0.08 * transmission_power

        next_state[STATE_INDEX.packet_loss] -= 0.12 * buffer_size
        next_state[STATE_INDEX.buffer_occupancy] += 0.10 * buffer_size

        qos = (
            0.45 * (1.0 - next_state[STATE_INDEX.latency])
            + 0.35 * next_state[STATE_INDEX.throughput]
            + 0.20 * (1.0 - next_state[STATE_INDEX.packet_loss])
        )
        next_state[STATE_INDEX.qos_score] = qos

        return np.clip(next_state, 0.0, 1.0).astype(np.float32)

    def _simulate_dynamics(self, state: np.ndarray) -> np.ndarray:
        """Simulate stochastic network dynamics.

        Adds Gaussian noise to all dimensions and introduces a congestion spike
        when throughput exceeds 0.85.

        Args:
            state: Deterministic post-action state.

        Returns:
            Stochastic next state clipped to ``[0, 1]``.
        """
        noise_std = float(self.config["noise_std"])
        noisy_state = state + self.np_random.normal(0.0, noise_std, size=state.shape)

        if noisy_state[STATE_INDEX.throughput] > 0.85:
            spike = float(self.np_random.uniform(0.05, 0.15))
            noisy_state[STATE_INDEX.congestion] += spike
            noisy_state[STATE_INDEX.packet_loss] += 0.5 * spike
            noisy_state[STATE_INDEX.qos_score] -= 0.4 * spike

        anomaly_probability = float(self.config["anomaly_probability"])
        if float(self.np_random.random()) < anomaly_probability:
            anomaly_jump = float(self.np_random.uniform(0.2, 0.6))
            noisy_state[STATE_INDEX.anomaly_score] += anomaly_jump
            noisy_state[STATE_INDEX.channel_quality] -= 0.2 * anomaly_jump

        noisy_state[STATE_INDEX.qos_score] = np.clip(
            0.45 * (1.0 - noisy_state[STATE_INDEX.latency])
            + 0.35 * noisy_state[STATE_INDEX.throughput]
            + 0.20 * (1.0 - noisy_state[STATE_INDEX.packet_loss]),
            0.0,
            1.0,
        )

        return np.clip(noisy_state, 0.0, 1.0).astype(np.float32)

    def _compute_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """Compute scalar reward for a state-action pair.

        Reward equation:
            ``R = 0.3*(1-latency) + 0.3*throughput + 0.2*(1-energy) + 0.2*qos``

        Args:
            state: Current state after transition.
            action: Action used for the transition.

        Returns:
            Reward value as float.
        """
        del action
        reward = (
            0.3 * (1.0 - state[STATE_INDEX.latency])
            + 0.3 * state[STATE_INDEX.throughput]
            + 0.2 * (1.0 - state[STATE_INDEX.energy])
            + 0.2 * state[STATE_INDEX.qos_score]
        )
        return float(reward)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Run one environment transition.

        Args:
            action: Action vector in ``[0, 1]^4``.

        Returns:
            Tuple of ``(obs, reward, terminated, truncated, info)``.
        """
        clipped_action = np.clip(action, 0.0, 1.0).astype(np.float32)

        action_state = self._apply_action(self._state, clipped_action)
        next_state = self._simulate_dynamics(action_state)
        reward = self._compute_reward(next_state, clipped_action)

        self._state = next_state
        self._step_count += 1

        terminated = False
        truncated = self._step_count >= int(self.config["max_steps"])

        info: dict[str, Any] = {
            "step": self._step_count,
            "latency": float(next_state[STATE_INDEX.latency]),
            "throughput": float(next_state[STATE_INDEX.throughput]),
            "energy": float(next_state[STATE_INDEX.energy]),
            "qos_score": float(next_state[STATE_INDEX.qos_score]),
        }
        return next_state.copy(), reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        """Render current environment metrics as a dictionary.

        Args:
            mode: Render mode. Only ``"human"`` is supported.
        """
        if mode != "human":
            raise NotImplementedError(f"Unsupported render mode: {mode}")

        metrics = {
            "latency": float(self._state[STATE_INDEX.latency]),
            "throughput": float(self._state[STATE_INDEX.throughput]),
            "energy": float(self._state[STATE_INDEX.energy]),
            "packet_loss": float(self._state[STATE_INDEX.packet_loss]),
            "qos_score": float(self._state[STATE_INDEX.qos_score]),
            "congestion": float(self._state[STATE_INDEX.congestion]),
            "n_active_nodes": float(self._state[STATE_INDEX.n_active_nodes]),
            "collision_rate": float(self._state[STATE_INDEX.collision_rate]),
            "hop_count": float(self._state[STATE_INDEX.hop_count]),
            "buffer_occupancy": float(self._state[STATE_INDEX.buffer_occupancy]),
            "anomaly_score": float(self._state[STATE_INDEX.anomaly_score]),
            "channel_quality": float(self._state[STATE_INDEX.channel_quality]),
        }
        print(metrics)
