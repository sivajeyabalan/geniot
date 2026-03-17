# IoT Network Gymnasium Environment

> Copilot context file for `backend/environment/` directory.

---

## iot_network_env.py

This is a **simulated** IoT network environment. It does NOT connect to real devices.
It models a network of N IoT nodes and simulates how configuration changes affect performance.

### Class skeleton

```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class IoTNetworkEnv(gym.Env):
    """
    Custom Gymnasium environment simulating an IoT network.

    Observation: current network metrics vector
    Action:      configuration parameter adjustments
    Reward:      weighted combination of latency, throughput, energy, QoS
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: dict = None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG
        self.n_nodes = self.config.get("n_nodes", 100)
        self.max_steps = self.config.get("max_steps", 200)
        self.current_step = 0
        self.state = None

        # Observation space: 12 network metrics, all normalized [0,1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32
        )

        # Action space: 4 continuous control parameters [0,1]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = self._initial_state()
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        self.state = self._apply_action(self.state, action)
        self.state = self._simulate_dynamics(self.state)
        reward = self._compute_reward(self.state, action)
        terminated = False
        truncated = self.current_step >= self.max_steps
        info = self._get_info()
        return self.state, reward, terminated, truncated, info

    def _initial_state(self) -> np.ndarray:
        # Returns normalized initial metrics vector
        # Add small noise for diversity across episodes
        ...

    def _apply_action(self, state, action) -> np.ndarray:
        # action[0] → routing aggressiveness (0=shortest path, 1=load balanced)
        # action[1] → sleep duty cycle (0=always on, 1=max sleep)
        # action[2] → transmission power (0=min, 1=max)
        # action[3] → buffer size (0=small, 1=large)
        ...

    def _simulate_dynamics(self, state) -> np.ndarray:
        # Simulate realistic network dynamics:
        # - Add Gaussian noise to all metrics
        # - Simulate congestion: if throughput > 0.85, latency spikes
        # - Simulate energy drain based on duty cycle
        # - Simulate random packet injection events
        ...

    def _compute_reward(self, state, action) -> float:
        # R = α1*R_latency + α2*R_throughput + α3*R_energy + α4*R_QoS
        # R_latency   = 1 - state[LATENCY_IDX]       (lower is better)
        # R_throughput = state[THROUGHPUT_IDX]         (higher is better)
        # R_energy    = 1 - state[ENERGY_IDX]         (lower is better)
        # R_QoS       = state[QOS_IDX]                (higher is better)
        alpha = [0.3, 0.3, 0.2, 0.2]
        ...

    def _get_info(self) -> dict:
        # Returns human-readable metrics dict for logging
        ...
```

---

## Observation space — 12 metrics

| Index | Metric | Description | Range |
|-------|--------|-------------|-------|
| 0 | latency | Avg end-to-end delay (normalized) | [0,1] |
| 1 | throughput | Data rate (normalized) | [0,1] |
| 2 | energy | Energy per bit (normalized) | [0,1] |
| 3 | packet_loss | % packets dropped | [0,1] |
| 4 | qos_score | SLA satisfaction rate | [0,1] |
| 5 | congestion | Network congestion level | [0,1] |
| 6 | n_active_nodes | Fraction of active nodes | [0,1] |
| 7 | collision_rate | MAC layer collision rate | [0,1] |
| 8 | hop_count | Avg hops per packet (normalized) | [0,1] |
| 9 | buffer_occupancy | Avg buffer fill level | [0,1] |
| 10 | anomaly_score | VAE reconstruction error (normalized) | [0,1] |
| 11 | channel_quality | Wireless channel SNR (normalized) | [0,1] |

---

## network_simulator.py

Provides the underlying simulation logic used by the Gym environment.
Keeps network state update equations separate from Gym API boilerplate.

```python
class NetworkSimulator:
    """
    Simulates IoT network dynamics without any Gym dependency.
    Can be used independently for generating training data.
    """
    def __init__(self, n_nodes: int = 100, seed: int = 42): ...
    def tick(self, action: np.ndarray) -> dict: ...
    def inject_anomaly(self, anomaly_type: str = "ddos"): ...
    def get_state_vector(self) -> np.ndarray: ...
    def generate_traffic_trace(self, n_steps: int = 1000) -> np.ndarray: ...
    # Returns (n_steps, n_features) array — used as real data for GAN training
```

---

## DEFAULT_CONFIG

```python
DEFAULT_CONFIG = {
    "n_nodes": 100,
    "max_steps": 200,
    "reward_weights": [0.3, 0.3, 0.2, 0.2],  # latency, throughput, energy, qos
    "noise_std": 0.02,
    "congestion_threshold": 0.85,
    "anomaly_probability": 0.05,   # 5% chance of injecting anomaly per step
    "seed": 42,
}
```

---

## Implemented index quick reference

### State vector layout (`shape=(12,)`)

| Index | Name |
|-------|------|
| 0 | latency |
| 1 | throughput |
| 2 | energy |
| 3 | packet_loss |
| 4 | qos_score |
| 5 | congestion |
| 6 | n_active_nodes |
| 7 | collision_rate |
| 8 | hop_count |
| 9 | buffer_occupancy |
| 10 | anomaly_score |
| 11 | channel_quality |

### Action vector layout (`shape=(4,)`)

| Index | Name |
|-------|------|
| 0 | routing_aggressiveness |
| 1 | sleep_duty_cycle |
| 2 | transmission_power |
| 3 | buffer_size |

### Reward used in environment

\[
R = 0.3\cdot(1-\text{latency}) + 0.3\cdot\text{throughput} + 0.2\cdot(1-\text{energy}) + 0.2\cdot\text{qos}
\]