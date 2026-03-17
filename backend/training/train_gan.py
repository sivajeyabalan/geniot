"""Training script for WGAN-GP IoT traffic generator."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from backend.data.datasets import IoTTrafficDataset
from backend.models.wgan_gp import DEFAULT_CONFIG, WGANGP


logger = logging.getLogger(__name__)

TRAINING_DEFAULTS: dict[str, Any] = {
    "epochs": 100,
    "batch_size": 64,
    "num_workers": 0,
    "pin_memory": True,
    "log_interval": 50,
}

WEIGHTS_DIR = Path("backend/models/weights")
CHECKPOINT_PATH = WEIGHTS_DIR / "gan.pt"
CURVES_PATH = WEIGHTS_DIR / "gan_training_curves.png"


def setup_logging(level: str = "INFO") -> None:
    """Configure logging.

    Args:
        level: Logging level name.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def resolve_processed_dir(explicit_dir: str | None = None) -> Path:
    """Resolve path to processed arrays.

    Args:
        explicit_dir: Optional explicit path.

    Returns:
        Existing processed directory.

    Raises:
        FileNotFoundError: If path cannot be found.
    """
    if explicit_dir is not None:
        candidate = Path(explicit_dir).expanduser().resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Processed directory not found: {candidate}")

    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root / "backend" / "data" / "processed",
        project_root / "data" / "processed",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Unable to locate processed data directory.")


def build_train_loader(processed_dir: Path, args: argparse.Namespace) -> DataLoader:
    """Create GAN training data loader.

    Args:
        processed_dir: Directory containing preprocessed arrays.
        args: Parsed CLI arguments.

    Returns:
        Training DataLoader for GAN.
    """
    train_dataset = IoTTrafficDataset.from_split(output_dir=str(processed_dir), split="train")
    return DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )


def save_training_curves(history: dict[str, list[float]], path: Path) -> None:
    """Save training loss curves.

    Args:
        history: Epoch-level average losses.
        path: Output path for curve image.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["critic_loss"]) + 1))

    plt.figure(figsize=(12, 5))
    plt.plot(epochs, history["critic_loss"], label="Critic Loss")
    plt.plot(epochs, history["generator_loss"], label="Generator Loss")
    plt.plot(epochs, history["gradient_penalty"], label="Gradient Penalty")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("WGAN-GP Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def train(args: argparse.Namespace) -> dict[str, float]:
    """Train WGAN-GP on IoT traffic windows.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Final epoch averaged losses.
    """
    processed_dir = resolve_processed_dir(args.processed_dir)
    logger.info("Using processed data directory: %s", processed_dir)

    train_loader = build_train_loader(processed_dir, args)
    logger.info("Train windows: %d", len(train_loader.dataset))

    # Infer sequence settings from first sample.
    sample = train_loader.dataset[0]
    if isinstance(sample, tuple):
        sample_x = sample[0]
    else:
        sample_x = sample

    config = DEFAULT_CONFIG.copy()
    config.update(
        {
            "latent_dim": args.latent_dim,
            "seq_len": int(sample_x.shape[0]),
            "n_features": int(sample_x.shape[1]),
            "hidden_dim": args.hidden_dim,
            "lstm_layers": args.lstm_layers,
            "critic_lr": args.learning_rate,
            "generator_lr": args.learning_rate,
            "betas": (0.0, 0.9),
            "lambda_gp": args.lambda_gp,
            "n_critic": args.n_critic,
        }
    )

    wgan = WGANGP(config=config)

    history: dict[str, list[float]] = {
        "critic_loss": [],
        "generator_loss": [],
        "gradient_penalty": [],
        "wasserstein": [],
    }

    output_dir = Path(args.weights_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / CHECKPOINT_PATH.name
    curves_path = output_dir / CURVES_PATH.name

    for epoch in range(1, args.epochs + 1):
        running = {
            "critic_loss": 0.0,
            "generator_loss": 0.0,
            "gradient_penalty": 0.0,
            "wasserstein": 0.0,
        }
        num_steps = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for step, batch in enumerate(progress, start=1):
            real_batch = batch[0] if isinstance(batch, (tuple, list)) else batch
            metrics = wgan.train_step(real_batch)

            for key in running:
                running[key] += float(metrics[key])
            num_steps += 1

            if step % args.log_interval == 0:
                logger.info(
                    "Epoch %d Step %d | critic=%.6f gen=%.6f gp=%.6f w=%.6f",
                    epoch,
                    step,
                    metrics["critic_loss"],
                    metrics["generator_loss"],
                    metrics["gradient_penalty"],
                    metrics["wasserstein"],
                )

        epoch_metrics = {k: (v / max(num_steps, 1)) for k, v in running.items()}
        for key, value in epoch_metrics.items():
            history[key].append(float(value))

        logger.info(
            "Epoch %d/%d | critic_loss=%.6f generator_loss=%.6f gradient_penalty=%.6f wasserstein=%.6f",
            epoch,
            args.epochs,
            epoch_metrics["critic_loss"],
            epoch_metrics["generator_loss"],
            epoch_metrics["gradient_penalty"],
            epoch_metrics["wasserstein"],
        )

    wgan.save_weights(checkpoint_path)
    save_training_curves(history, curves_path)
    logger.info("Saved checkpoint to %s", checkpoint_path)
    logger.info("Saved training curves to %s", curves_path)

    return {
        "critic_loss": history["critic_loss"][-1],
        "generator_loss": history["generator_loss"][-1],
        "gradient_penalty": history["gradient_penalty"][-1],
        "wasserstein": history["wasserstein"][-1],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for GAN training."""
    parser = argparse.ArgumentParser(description="Train WGAN-GP for IoT traffic synthesis")
    parser.add_argument("--processed-dir", type=str, default=None, help="Processed numpy directory")
    parser.add_argument("--weights-dir", type=str, default=str(WEIGHTS_DIR), help="Output directory for artifacts")

    parser.add_argument("--epochs", type=int, default=TRAINING_DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=TRAINING_DEFAULTS["batch_size"])
    parser.add_argument("--num-workers", type=int, default=TRAINING_DEFAULTS["num_workers"])
    parser.add_argument("--pin-memory", action="store_true", default=TRAINING_DEFAULTS["pin_memory"])
    parser.add_argument("--log-interval", type=int, default=TRAINING_DEFAULTS["log_interval"])

    parser.add_argument("--latent-dim", type=int, default=int(DEFAULT_CONFIG["latent_dim"]))
    parser.add_argument("--hidden-dim", type=int, default=int(DEFAULT_CONFIG["hidden_dim"]))
    parser.add_argument("--lstm-layers", type=int, default=int(DEFAULT_CONFIG["lstm_layers"]))
    parser.add_argument("--learning-rate", type=float, default=float(DEFAULT_CONFIG["generator_lr"]))
    parser.add_argument("--lambda-gp", type=float, default=float(DEFAULT_CONFIG["lambda_gp"]))
    parser.add_argument("--n-critic", type=int, default=int(DEFAULT_CONFIG["n_critic"]))

    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)
    train(args)


if __name__ == "__main__":
    main()
