"""Variational Autoencoder (VAE) for IoT traffic sequence anomaly detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CONFIG: dict[str, Any] = {
    "seq_len": 50,
    "n_features": 41,
    "encoder_hidden_dim": 128,
    "encoder_layers": 2,
    "encoder_bidirectional": True,
    "latent_dim": 64,
    "decoder_hidden_dim": 128,
    "decoder_layers": 2,
    "dropout": 0.1,
    "beta": 1.0,
}


class VAEEncoder(nn.Module):
    """BiLSTM encoder that maps an IoT sequence to latent distribution parameters.

    Args:
        config: Model hyperparameter dictionary.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the encoder network."""
        super().__init__()
        self.config = config
        self.seq_len = int(config["seq_len"])
        self.n_features = int(config["n_features"])
        self.hidden_dim = int(config["encoder_hidden_dim"])
        self.n_layers = int(config["encoder_layers"])
        self.bidirectional = bool(config["encoder_bidirectional"])
        self.latent_dim = int(config["latent_dim"])
        self.dropout = float(config["dropout"]) if self.n_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

        direction_factor = 2 if self.bidirectional else 1
        self.mu_head = nn.Linear(self.hidden_dim * direction_factor, self.latent_dim)
        self.log_var_head = nn.Linear(self.hidden_dim * direction_factor, self.latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode sequence into latent Gaussian parameters.

        Args:
            x: Input tensor of shape ``(batch, seq_len, n_features)``.

        Returns:
            Tuple ``(mu, log_var)`` where each has shape ``(batch, latent_dim)``.
        """
        _, (hidden_state, _) = self.lstm(x)

        if self.bidirectional:
            forward_hidden = hidden_state[-2, :, :]
            backward_hidden = hidden_state[-1, :, :]
            final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            final_hidden = hidden_state[-1, :, :]

        mu = self.mu_head(final_hidden)
        log_var = self.log_var_head(final_hidden)
        return mu, log_var


class VAEDecoder(nn.Module):
    """LSTM decoder that reconstructs IoT traffic sequences from latent vectors.

    Args:
        config: Model hyperparameter dictionary.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the decoder network."""
        super().__init__()
        self.config = config
        self.seq_len = int(config["seq_len"])
        self.n_features = int(config["n_features"])
        self.latent_dim = int(config["latent_dim"])
        self.hidden_dim = int(config["decoder_hidden_dim"])
        self.n_layers = int(config["decoder_layers"])
        self.dropout = float(config["dropout"]) if self.n_layers > 1 else 0.0

        self.latent_to_context = nn.Linear(self.latent_dim, self.hidden_dim)
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            dropout=self.dropout,
            batch_first=True,
        )
        self.output_layer = nn.Linear(self.hidden_dim, self.n_features)

    def forward(self, z: Tensor) -> Tensor:
        """Decode latent representation into reconstructed sequence.

        Args:
            z: Latent tensor of shape ``(batch, latent_dim)``.

        Returns:
            Reconstructed tensor with shape ``(batch, seq_len, n_features)``.
        """
        context = torch.relu(self.latent_to_context(z))
        repeated_context = context.unsqueeze(1).repeat(1, self.seq_len, 1)
        decoded_sequence, _ = self.lstm(repeated_context)
        x_recon = self.output_layer(decoded_sequence)
        return x_recon


class VAE(nn.Module):
    """Variational Autoencoder for IoT sequence reconstruction.

    Args:
        config: Optional override configuration. Missing keys are populated from
            ``DEFAULT_CONFIG``.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize VAE encoder and decoder modules."""
        super().__init__()
        merged_config = DEFAULT_CONFIG.copy()
        if config is not None:
            merged_config.update(config)

        self.config = merged_config
        self.encoder = VAEEncoder(config=self.config)
        self.decoder = VAEDecoder(config=self.config)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode input sequence into latent Gaussian parameters.

        Args:
            x: Input tensor of shape ``(batch, seq_len, n_features)``.

        Returns:
            Tuple of ``(mu, log_var)`` each with shape ``(batch, latent_dim)``.
        """
        return self.encoder(x)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Sample latent vector using reparameterization trick.

        Args:
            mu: Latent mean tensor of shape ``(batch, latent_dim)``.
            log_var: Latent log-variance tensor of shape ``(batch, latent_dim)``.

        Returns:
            Sampled latent tensor ``z`` with shape ``(batch, latent_dim)``.
        """
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent vectors into reconstructed sequence.

        Args:
            z: Latent tensor of shape ``(batch, latent_dim)``.

        Returns:
            Reconstructed tensor of shape ``(batch, seq_len, n_features)``.
        """
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run full VAE pass.

        Args:
            x: Input tensor of shape ``(batch, seq_len, n_features)``.

        Returns:
            Tuple ``(x_recon, mu, log_var)``.
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def save_weights(self, path: str | Path) -> None:
        """Save model checkpoint to disk.

        Args:
            path: Destination checkpoint path.
        """
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
        }
        torch.save(payload, checkpoint_path)
        logger.info("Saved VAE checkpoint to %s", checkpoint_path)

    @classmethod
    def from_weights(cls, path: str | Path) -> "VAE":
        """Load a VAE model instance from checkpoint.

        Args:
            path: Checkpoint file path.

        Returns:
            Initialized model with loaded weights in evaluation mode.

        Raises:
            FileNotFoundError: If checkpoint does not exist.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        config = checkpoint.get("config", DEFAULT_CONFIG)
        model = cls(config=config)

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        logger.info("Loaded VAE checkpoint from %s", checkpoint_path)
        return model


def vae_loss(
    x_recon: Tensor,
    x: Tensor,
    mu: Tensor,
    log_var: Tensor,
    beta: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute VAE loss components.

    Args:
        x_recon: Reconstructed input with shape ``(batch, seq_len, n_features)``.
        x: Original input with shape ``(batch, seq_len, n_features)``.
        mu: Latent mean tensor with shape ``(batch, latent_dim)``.
        log_var: Latent log-variance tensor with shape ``(batch, latent_dim)``.
        beta: Weight applied to KL divergence term.

    Returns:
        Tuple ``(total_loss, recon_loss, kl_loss)``.
    """
    recon_loss = torch.mean((x_recon - x) ** 2)
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def anomaly_score(
    model: VAE,
    x: Tensor | np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Compute per-sample reconstruction MSE anomaly scores.

    Args:
        model: Trained VAE model.
        x: Input batch tensor or array with shape ``(batch, seq_len, n_features)``.
        device: Target torch device.

    Returns:
        NumPy array of per-sample MSE anomaly scores with shape ``(batch,)``.
    """
    model = model.to(device)
    model.eval()

    input_tensor = x if isinstance(x, torch.Tensor) else torch.from_numpy(x)
    input_tensor = input_tensor.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        x_recon, _, _ = model(input_tensor)
        per_sample_mse = torch.mean((x_recon - input_tensor) ** 2, dim=(1, 2))

    return per_sample_mse.detach().cpu().numpy()
