"""WGAN-GP model for IoT traffic sequence generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.utils import spectral_norm


logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CONFIG: dict[str, Any] = {
    "latent_dim": 128,
    "seq_len": 50,
    "n_features": 41,
    "hidden_dim": 256,
    "lstm_layers": 2,
    "critic_lr": 1e-4,
    "generator_lr": 1e-4,
    "betas": (0.0, 0.9),
    "lambda_gp": 10.0,
    "n_critic": 5,
    "debug_update_counter": False,
}


class Generator(nn.Module):
    """Generator network mapping latent noise to IoT traffic sequences.

    Architecture:
        Linear(latent_dim -> hidden_dim) -> reshape -> repeat over seq_len ->
        2-layer LSTM(hidden_dim) -> Linear(hidden_dim -> n_features) -> Tanh.

    Args:
        config: Generator hyperparameters.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize generator layers."""
        super().__init__()
        self.config = config
        self.latent_dim = int(config["latent_dim"])
        self.seq_len = int(config["seq_len"])
        self.hidden_dim = int(config["hidden_dim"])
        self.n_features = int(config["n_features"])
        self.lstm_layers = int(config["lstm_layers"])

        self.noise_projection = nn.Linear(self.latent_dim, self.hidden_dim)
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(self.hidden_dim, self.n_features)
        self.output_activation = nn.Tanh()

    def forward(self, noise: Tensor) -> Tensor:
        """Generate synthetic traffic sequences.

        Args:
            noise: Latent noise tensor with shape ``(batch, latent_dim)``.

        Returns:
            Generated traffic tensor with shape ``(batch, seq_len, n_features)``.
        """
        projected = self.noise_projection(noise)
        repeated = projected.unsqueeze(1).repeat(1, self.seq_len, 1)
        temporal_features, _ = self.lstm(repeated)
        raw_output = self.output_layer(temporal_features)
        return self.output_activation(raw_output)


class Critic(nn.Module):
    """Critic network scoring realism of IoT traffic sequences.

    Args:
        config: Critic hyperparameters.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize critic layers."""
        super().__init__()
        self.config = config
        self.n_features = int(config["n_features"])
        self.hidden_dim = int(config["hidden_dim"])
        self.lstm_layers = int(config["lstm_layers"])

        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
        )
        self.score_head = spectral_norm(nn.Linear(self.hidden_dim, 1))

    def forward(self, traffic: Tensor) -> Tensor:
        """Compute critic scores for traffic sequences.

        Args:
            traffic: Input tensor with shape ``(batch, seq_len, n_features)``.

        Returns:
            Critic score tensor with shape ``(batch, 1)``.
        """
        _, (hidden_state, _) = self.lstm(traffic)
        final_hidden = hidden_state[-1]
        return self.score_head(final_hidden)


class WGANGP:
    """Wrapper for WGAN-GP training and inference.

    Args:
        config: Optional configuration overrides.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize models, optimizers, and training settings."""
        merged_config = DEFAULT_CONFIG.copy()
        if config is not None:
            merged_config.update(config)

        self.config = merged_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = Generator(config=self.config).to(self.device)
        self.critic = Critic(config=self.config).to(self.device)

        beta1, beta2 = self.config["betas"]
        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=float(self.config["generator_lr"]),
            betas=(float(beta1), float(beta2)),
        )
        self.optimizer_c = torch.optim.Adam(
            self.critic.parameters(),
            lr=float(self.config["critic_lr"]),
            betas=(float(beta1), float(beta2)),
        )

    def compute_gradient_penalty(self, real: Tensor, fake: Tensor) -> Tensor:
        """Compute WGAN-GP gradient penalty term.

        Args:
            real: Real batch with shape ``(batch, seq_len, n_features)``.
            fake: Fake batch with shape ``(batch, seq_len, n_features)``.

        Returns:
            Scalar gradient penalty tensor.
        """
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, 1, device=self.device)
        interpolated = alpha * real + (1.0 - alpha) * fake
        interpolated.requires_grad_(True)

        critic_interpolated = self.critic(interpolated)
        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.reshape(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        return ((gradient_norm - 1.0) ** 2).mean()

    def train_step(self, real_batch: Tensor) -> dict[str, float]:
        """Run one WGAN-GP training step.

        Critic is updated ``n_critic`` times before one generator update.

        Args:
            real_batch: Real data batch with shape ``(batch, seq_len, n_features)``.

        Returns:
            Dictionary containing training loss values.
        """
        real_batch = real_batch.to(self.device)
        batch_size = real_batch.size(0)

        n_critic = int(self.config["n_critic"])
        lambda_gp = float(self.config["lambda_gp"])
        latent_dim = int(self.config["latent_dim"])
        debug_counter = bool(self.config.get("debug_update_counter", False))

        critic_loss_value = 0.0
        gradient_penalty_value = 0.0
        wasserstein_value = 0.0

        for _ in range(n_critic):
            self.optimizer_c.zero_grad(set_to_none=True)

            noise = torch.randn(batch_size, latent_dim, device=self.device)
            fake_batch = self.generator(noise).detach()

            real_score = self.critic(real_batch)
            fake_score = self.critic(fake_batch)

            wasserstein = fake_score.mean() - real_score.mean()
            gradient_penalty = self.compute_gradient_penalty(real_batch, fake_batch)
            critic_loss = wasserstein + lambda_gp * gradient_penalty

            critic_loss.backward()
            self.optimizer_c.step()

            critic_loss_value += critic_loss.detach().item()
            gradient_penalty_value += gradient_penalty.detach().item()
            wasserstein_value += wasserstein.detach().item()

        if debug_counter:
            print(f"critic_updates={n_critic}, generator_updates=1")

        critic_loss_value /= n_critic
        gradient_penalty_value /= n_critic
        wasserstein_value /= n_critic

        self.optimizer_g.zero_grad(set_to_none=True)
        noise = torch.randn(batch_size, latent_dim, device=self.device)
        generated_batch = self.generator(noise)
        fake_score = self.critic(generated_batch)
        generator_loss = -fake_score.mean()

        generator_loss.backward()
        self.optimizer_g.step()

        return {
            "critic_loss": float(critic_loss_value),
            "generator_loss": float(generator_loss.detach().item()),
            "gradient_penalty": float(gradient_penalty_value),
            "wasserstein": float(wasserstein_value),
        }

    @torch.no_grad()
    def generate(self, n_samples: int) -> Tensor:
        """Generate synthetic traffic sequences.

        Args:
            n_samples: Number of sequences to generate.

        Returns:
            Tensor of generated samples with shape ``(n_samples, seq_len, n_features)``.
        """
        self.generator.eval()
        noise = torch.randn(n_samples, int(self.config["latent_dim"]), device=self.device)
        samples = self.generator(noise)
        self.generator.train()
        return samples

    def save_weights(self, path: str | Path) -> None:
        """Save model and optimizer states.

        Args:
            path: Output checkpoint path.
        """
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "config": self.config,
            "generator_state_dict": self.generator.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_g_state_dict": self.optimizer_g.state_dict(),
            "optimizer_c_state_dict": self.optimizer_c.state_dict(),
        }
        torch.save(payload, checkpoint_path)
        logger.info("Saved WGAN-GP checkpoint to %s", checkpoint_path)

    @classmethod
    def from_weights(cls, path: str | Path) -> "WGANGP":
        """Load WGAN-GP model from checkpoint.

        Args:
            path: Checkpoint path.

        Returns:
            Initialized ``WGANGP`` instance with loaded weights.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        config = checkpoint.get("config", DEFAULT_CONFIG)

        model = cls(config=config)
        model.generator.load_state_dict(checkpoint["generator_state_dict"])
        model.critic.load_state_dict(checkpoint["critic_state_dict"])

        if "optimizer_g_state_dict" in checkpoint:
            model.optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        if "optimizer_c_state_dict" in checkpoint:
            model.optimizer_c.load_state_dict(checkpoint["optimizer_c_state_dict"])

        model.generator.to(model.device)
        model.critic.to(model.device)
        logger.info("Loaded WGAN-GP checkpoint from %s", checkpoint_path)
        return model
