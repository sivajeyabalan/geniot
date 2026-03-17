"""Train and evaluate PPO agent on the custom IoT network environment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from backend.environment.iot_network_env import DEFAULT_CONFIG, IoTNetworkEnv


LOGGER = logging.getLogger(__name__)


class EpisodicRewardCallback(BaseCallback):
    """Collect episodic rewards during PPO learning."""

    def __init__(self) -> None:
        """Initialize reward collector callback."""
        super().__init__()
        self.episode_rewards: list[float] = []
        self._running_reward = 0.0

    def _on_step(self) -> bool:
        """Accumulate rewards and store completed episode returns.

        Returns:
            Always True to continue training.
        """
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        if rewards is None or dones is None:
            return True

        reward = float(rewards[0]) if np.ndim(rewards) > 0 else float(rewards)
        done = bool(dones[0]) if np.ndim(dones) > 0 else bool(dones)

        self._running_reward += reward
        if done:
            self.episode_rewards.append(self._running_reward)
            self._running_reward = 0.0
        return True


def _run_policy_episodes(
    env: IoTNetworkEnv,
    n_episodes: int,
    model: PPO | None = None,
    seed: int = 123,
) -> dict[str, float]:
    """Evaluate PPO policy or random baseline over multiple episodes.

    Args:
        env: Environment instance for evaluation.
        n_episodes: Number of episodes to run.
        model: PPO model for deterministic actions. If None, random policy is used.
        seed: Seed offset for reproducible episode resets.

    Returns:
        Summary with mean reward, mean latency, and mean throughput.
    """
    episode_rewards: list[float] = []
    episode_latencies: list[float] = []
    episode_throughputs: list[float] = []

    for episode_idx in range(n_episodes):
        observation, _ = env.reset(seed=seed + episode_idx)
        terminated = False
        truncated = False

        cumulative_reward = 0.0
        latencies: list[float] = []
        throughputs: list[float] = []

        while not (terminated or truncated):
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(observation, deterministic=True)

            observation, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += float(reward)
            latencies.append(float(info.get("latency", observation[0])))
            throughputs.append(float(info.get("throughput", observation[1])))

        episode_rewards.append(cumulative_reward)
        episode_latencies.append(float(np.mean(latencies)) if latencies else 0.0)
        episode_throughputs.append(float(np.mean(throughputs)) if throughputs else 0.0)

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "mean_latency": float(np.mean(episode_latencies)),
        "mean_throughput": float(np.mean(episode_throughputs)),
    }


def _comparison_table(ppo_metrics: dict[str, float], random_metrics: dict[str, float]) -> str:
    """Build printable comparison table string.

    Args:
        ppo_metrics: Aggregated PPO metrics.
        random_metrics: Aggregated random policy metrics.

    Returns:
        Formatted table string.
    """
    header = (
        "Policy      | Mean Reward | Mean Latency | Mean Throughput\n"
        "------------|-------------|--------------|----------------"
    )
    ppo_row = (
        f"PPO         | {ppo_metrics['mean_reward']:.4f}      | "
        f"{ppo_metrics['mean_latency']:.4f}       | {ppo_metrics['mean_throughput']:.4f}"
    )
    random_row = (
        f"Random      | {random_metrics['mean_reward']:.4f}      | "
        f"{random_metrics['mean_latency']:.4f}       | {random_metrics['mean_throughput']:.4f}"
    )
    return "\n".join([header, ppo_row, random_row])


def _plot_reward_curve(episode_rewards: list[float], output_path: Path) -> Path:
    """Plot episodic reward curve and save image.

    Args:
        episode_rewards: Sequence of episode returns collected during training.
        output_path: Output PNG path.

    Returns:
        Absolute path to the saved figure.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(episode_rewards, label="Episode reward")
    plt.title("PPO Training Episodic Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path.resolve()


def train_rl(
    total_timesteps: int = 500_000,
    model_out: str | Path = "backend/models/weights/ppo_iot.zip",
    tensorboard_log: str | Path = "./logs/ppo_iot",
    eval_episodes: int = 100,
) -> dict[str, Any]:
    """Train PPO agent on IoTNetworkEnv and save the model.

    Args:
        total_timesteps: Number of environment interaction timesteps.
        model_out: Output path for trained PPO checkpoint.
        tensorboard_log: TensorBoard logging directory.
        eval_episodes: Number of episodes for PPO and random baseline evaluation.

    Returns:
        Dictionary containing model path, reward-curve path, and evaluation metrics.
    """
    env = IoTNetworkEnv(config=DEFAULT_CONFIG)
    check_env(env, warn=True)

    monitored_env = Monitor(env)
    callback = EpisodicRewardCallback()

    model = PPO(
        policy="MlpPolicy",
        env=monitored_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        tensorboard_log=str(tensorboard_log),
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)

    output_path = Path(model_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))

    ppo_eval_env = IoTNetworkEnv(config=DEFAULT_CONFIG)
    random_eval_env = IoTNetworkEnv(config=DEFAULT_CONFIG)

    ppo_metrics = _run_policy_episodes(
        env=ppo_eval_env,
        n_episodes=eval_episodes,
        model=model,
        seed=123,
    )
    random_metrics = _run_policy_episodes(
        env=random_eval_env,
        n_episodes=eval_episodes,
        model=None,
        seed=123,
    )

    table = _comparison_table(ppo_metrics=ppo_metrics, random_metrics=random_metrics)
    LOGGER.info("\n%s", table)

    reward_curve_path = _plot_reward_curve(
        callback.episode_rewards,
        output_path=output_path.parent / "ppo_training_rewards.png",
    )

    return {
        "model_path": output_path.resolve(),
        "reward_curve_path": reward_curve_path,
        "ppo_metrics": ppo_metrics,
        "random_metrics": random_metrics,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for PPO training."""
    parser = argparse.ArgumentParser(description="Train PPO on IoTNetworkEnv")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total timesteps for PPO training.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="backend/models/weights/ppo_iot.zip",
        help="Output path for saved PPO model.",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default="./logs/ppo_iot",
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Number of episodes for PPO and random baseline evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    result = train_rl(
        total_timesteps=args.timesteps,
        model_out=args.out,
        tensorboard_log=args.tensorboard_log,
        eval_episodes=args.eval_episodes,
    )
    LOGGER.info("Saved PPO model to: %s", result["model_path"])
    LOGGER.info("Saved reward curve to: %s", result["reward_curve_path"])


if __name__ == "__main__":
    main()
