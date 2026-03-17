"""Training script for VAE-based IoT anomaly detection."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from backend.data.datasets import IoTTrafficDataset
from backend.models.vae import DEFAULT_CONFIG, DEVICE, VAE, anomaly_score, vae_loss


logger = logging.getLogger(__name__)

TRAINING_DEFAULTS: dict[str, Any] = {
    "batch_size": 64,
    "epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-6,
    "beta": 1.0,
    "threshold_percentile": 95.0,
    "num_workers": 0,
    "pin_memory": True,
}

WEIGHTS_DIR = Path("backend/models/weights")
BEST_MODEL_PATH = WEIGHTS_DIR / "vae.pt"
THRESHOLD_PATH = WEIGHTS_DIR / "vae_threshold.txt"
TRAINING_CURVES_PATH = WEIGHTS_DIR / "vae_training_curves.png"
RECON_HIST_PATH = WEIGHTS_DIR / "vae_reconstruction_error_hist.png"


def setup_logging(level: str = "INFO") -> None:
    """Configure logging format and level.

    Args:
        level: Logging level name.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def resolve_processed_dir(explicit_dir: str | None = None) -> Path:
    """Resolve preprocessing output directory.

    Args:
        explicit_dir: Optional user-provided directory.

    Returns:
        Existing processed-data directory path.

    Raises:
        FileNotFoundError: If no candidate directory exists.
    """
    if explicit_dir is not None:
        candidate = Path(explicit_dir).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Processed directory not found: {candidate}")
        return candidate

    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root / "backend" / "data" / "processed",
        project_root / "data" / "processed",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not locate processed data directory. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def _normal_subset(dataset: IoTTrafficDataset) -> Subset:
    """Create subset of samples where label is normal (0).

    Args:
        dataset: Dataset containing labels.

    Returns:
        Subset containing indices for normal samples.

    Raises:
        ValueError: If labels are missing or no normal samples exist.
    """
    if dataset.y is None:
        raise ValueError("Labels are required to filter normal traffic samples (label==0).")

    normal_indices = np.where(dataset.y == 0)[0].tolist()
    if not normal_indices:
        raise ValueError("No normal samples found with label==0.")

    return Subset(dataset, normal_indices)


def build_dataloaders(processed_dir: Path, args: argparse.Namespace) -> tuple[DataLoader, DataLoader, IoTTrafficDataset]:
    """Build train/val dataloaders from dataset utilities.

    Args:
        processed_dir: Directory containing preprocessed arrays.
        args: Parsed CLI arguments.

    Returns:
        Tuple of train loader, validation loader, and test dataset.
    """
    train_dataset = IoTTrafficDataset.from_split(output_dir=str(processed_dir), split="train")
    val_dataset = IoTTrafficDataset.from_split(output_dir=str(processed_dir), split="val")
    test_dataset = IoTTrafficDataset.from_split(output_dir=str(processed_dir), split="test")

    train_subset = _normal_subset(train_dataset)
    val_subset = _normal_subset(val_dataset)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    return train_loader, val_loader, test_dataset


def run_epoch(
    model: VAE,
    loader: DataLoader,
    optimizer: Adam,
    beta: float,
    train: bool,
    device: torch.device,
) -> dict[str, float]:
    """Run one train or validation epoch.

    Args:
        model: VAE model instance.
        loader: DataLoader containing input batches.
        optimizer: Optimizer used in training mode.
        beta: KL divergence weight.
        train: If True, updates model weights.
        device: Torch device.

    Returns:
        Dictionary with average losses.
    """
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_samples = 0

    progress = tqdm(loader, desc="Train" if train else "Val", leave=False)
    for batch in progress:
        batch_x = batch[0].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            x_recon, mu, log_var = model(batch_x)
            loss, recon_loss, kl_loss = vae_loss(x_recon=x_recon, x=batch_x, mu=mu, log_var=log_var, beta=beta)

            if train:
                loss.backward()
                optimizer.step()

        batch_size = batch_x.size(0)
        n_samples += batch_size
        total_loss += loss.detach().item() * batch_size
        total_recon += recon_loss.detach().item() * batch_size
        total_kl += kl_loss.detach().item() * batch_size

    if n_samples == 0:
        return {"loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}

    return {
        "loss": total_loss / n_samples,
        "recon_loss": total_recon / n_samples,
        "kl_loss": total_kl / n_samples,
    }


def evaluate_on_test(
    model: VAE,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate anomaly detection performance on test split.

    Args:
        model: Trained VAE model.
        X_test: Test windows.
        y_test: Binary test labels.
        threshold: Decision threshold on reconstruction score.
        device: Torch device.

    Returns:
        Dictionary containing precision, recall, and F1.
    """
    scores = anomaly_score(model=model, x=X_test, device=device)
    preds = (scores >= threshold).astype(np.int64)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        preds,
        average="binary",
        zero_division=0,
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
    }


def select_threshold(
    val_scores: np.ndarray,
    y_val: np.ndarray,
    fallback_percentile: float,
) -> float:
    """Select anomaly threshold from validation scores.

    Uses validation F1 maximization when both classes are present; otherwise
    falls back to percentile thresholding on normal validation samples.

    Args:
        val_scores: Reconstruction scores for validation samples.
        y_val: Validation labels.
        fallback_percentile: Percentile for fallback thresholding.

    Returns:
        Selected scalar threshold.
    """
    y_val = y_val.astype(np.int64)
    normal_val_scores = val_scores[y_val == 0]
    if normal_val_scores.size == 0:
        raise ValueError("No normal samples in validation set for threshold selection.")

    unique_classes = np.unique(y_val)
    if unique_classes.size < 2:
        return float(np.percentile(normal_val_scores, fallback_percentile))

    candidate_thresholds = np.unique(
        np.quantile(val_scores, np.linspace(0.01, 0.99, 240))
    )

    best_threshold = float(np.percentile(normal_val_scores, fallback_percentile))
    best_f1 = -1.0
    for threshold in candidate_thresholds:
        preds = (val_scores >= threshold).astype(np.int64)
        _, _, f1, _ = precision_recall_fscore_support(
            y_val,
            preds,
            average="binary",
            zero_division=0,
        )
        if float(f1) > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)

    return best_threshold


def save_training_curves(history: dict[str, list[float]], output_path: Path) -> None:
    """Plot and save training/validation curves.

    Args:
        history: Epoch-wise loss history.
        output_path: File path for saved figure.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, history["train_loss"], label="Train Total Loss")
    plt.plot(epochs, history["val_loss"], label="Val Total Loss")
    plt.plot(epochs, history["train_recon_loss"], linestyle="--", label="Train Recon Loss")
    plt.plot(epochs, history["val_recon_loss"], linestyle="--", label="Val Recon Loss")
    plt.plot(epochs, history["train_kl_loss"], linestyle=":", label="Train KL Loss")
    plt.plot(epochs, history["val_kl_loss"], linestyle=":", label="Val KL Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_reconstruction_histogram(
    normal_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    """Plot and save reconstruction-error histogram.

    Args:
        normal_scores: Scores for normal samples.
        anomaly_scores: Scores for anomaly samples.
        threshold: Decision threshold.
        output_path: File path for saved figure.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(normal_scores, bins=50, alpha=0.6, label="Normal", density=True)
    plt.hist(anomaly_scores, bins=50, alpha=0.6, label="Anomaly", density=True)
    plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold={threshold:.6f}")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Density")
    plt.title("Reconstruction Error Histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train(args: argparse.Namespace) -> dict[str, float]:
    """Run end-to-end VAE training and test evaluation.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Final evaluation metrics.
    """
    processed_dir = resolve_processed_dir(args.processed_dir)
    logger.info("Using processed data directory: %s", processed_dir)

    train_loader, val_loader, test_dataset = build_dataloaders(processed_dir, args)
    train_subset_size = len(train_loader.dataset)
    val_subset_size = len(val_loader.dataset)
    test_size = len(test_dataset)
    logger.info("Normal-only train windows: %d", train_subset_size)
    logger.info("Normal-only val windows: %d", val_subset_size)
    logger.info("Test windows: %d", test_size)

    if test_dataset.y is None:
        raise ValueError("Test labels are required for anomaly evaluation.")

    val_all_dataset = IoTTrafficDataset.from_split(output_dir=str(processed_dir), split="val")
    if val_all_dataset.y is None:
        raise ValueError("Validation labels are required for threshold tuning.")

    X_test = np.asarray(test_dataset.X)
    y_test = np.asarray(test_dataset.y).astype(np.int64)
    X_val_all = np.asarray(val_all_dataset.X)
    y_val_all = np.asarray(val_all_dataset.y).astype(np.int64)

    model_config = DEFAULT_CONFIG.copy()
    model_config.update(
        {
            "seq_len": int(X_test.shape[1]),
            "n_features": int(X_test.shape[2]),
            "latent_dim": args.latent_dim,
            "encoder_hidden_dim": args.encoder_hidden_dim,
            "decoder_hidden_dim": args.decoder_hidden_dim,
            "beta": args.beta,
        }
    )

    model = VAE(config=model_config).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_epoch = -1
    weights_dir = Path(args.weights_dir).expanduser().resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)

    output_path = weights_dir / BEST_MODEL_PATH.name
    threshold_path = weights_dir / THRESHOLD_PATH.name
    curves_path = weights_dir / TRAINING_CURVES_PATH.name
    histogram_path = weights_dir / RECON_HIST_PATH.name

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_recon_loss": [],
        "val_recon_loss": [],
        "train_kl_loss": [],
        "val_kl_loss": [],
    }

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            beta=args.beta,
            train=True,
            device=DEVICE,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            beta=args.beta,
            train=False,
            device=DEVICE,
        )

        logger.info(
            "Epoch %d/%d | train_loss=%.6f val_loss=%.6f train_recon_loss=%.6f train_kl_loss=%.6f val_recon_loss=%.6f val_kl_loss=%.6f",
            epoch,
            args.epochs,
            train_metrics["loss"],
            val_metrics["loss"],
            train_metrics["recon_loss"],
            train_metrics["kl_loss"],
            val_metrics["recon_loss"],
            val_metrics["kl_loss"],
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_recon_loss"].append(train_metrics["recon_loss"])
        history["val_recon_loss"].append(val_metrics["recon_loss"])
        history["train_kl_loss"].append(train_metrics["kl_loss"])
        history["val_kl_loss"].append(val_metrics["kl_loss"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            model.save_weights(output_path)
            logger.info("Saved best checkpoint at epoch %d to %s", epoch, output_path)

    logger.info("Best val_loss=%.6f at epoch=%d", best_val_loss, best_epoch)

    best_model = VAE.from_weights(output_path)
    val_scores = anomaly_score(model=best_model, x=X_val_all, device=DEVICE)
    if args.threshold_strategy == "val_f1":
        threshold = select_threshold(
            val_scores=val_scores,
            y_val=y_val_all,
            fallback_percentile=args.threshold_percentile,
        )
        logger.info("Threshold strategy: validation F1 search")
    else:
        normal_val_scores = val_scores[y_val_all == 0]
        if normal_val_scores.size == 0:
            raise ValueError("No normal samples in validation set for percentile thresholding.")
        threshold = float(np.percentile(normal_val_scores, args.threshold_percentile))
        logger.info("Threshold strategy: %.1fth percentile on normal validation scores", args.threshold_percentile)

    threshold_path.write_text(f"{threshold:.10f}\n", encoding="utf-8")
    logger.info("Saved threshold to %s", threshold_path)

    test_scores = anomaly_score(model=best_model, x=X_test, device=DEVICE)
    test_metrics = evaluate_on_test(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        threshold=threshold,
        device=DEVICE,
    )

    save_training_curves(history=history, output_path=curves_path)
    normal_test_scores = test_scores[y_test == 0]
    anomaly_test_scores = test_scores[y_test == 1]
    save_reconstruction_histogram(
        normal_scores=normal_test_scores,
        anomaly_scores=anomaly_test_scores,
        threshold=threshold,
        output_path=histogram_path,
    )

    logger.info("Saved training curves to %s", curves_path)
    logger.info("Saved reconstruction histogram to %s", histogram_path)

    logger.info(
        "Final metrics | Precision=%.4f Recall=%.4f F1=%.4f",
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["f1"],
    )

    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"Selected Threshold: {threshold:.6f}")

    return test_metrics


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for VAE training."""
    parser = argparse.ArgumentParser(description="Train VAE for IoT anomaly detection")
    parser.add_argument("--processed-dir", type=str, default=None, help="Path to preprocessed numpy directory")
    parser.add_argument("--weights-dir", type=str, default=str(WEIGHTS_DIR), help="Directory to save model artifacts")
    parser.add_argument("--batch-size", type=int, default=TRAINING_DEFAULTS["batch_size"])
    parser.add_argument("--epochs", type=int, default=TRAINING_DEFAULTS["epochs"])
    parser.add_argument("--learning-rate", type=float, default=TRAINING_DEFAULTS["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=TRAINING_DEFAULTS["weight_decay"])
    parser.add_argument("--latent-dim", type=int, default=int(DEFAULT_CONFIG["latent_dim"]))
    parser.add_argument("--encoder-hidden-dim", type=int, default=int(DEFAULT_CONFIG["encoder_hidden_dim"]))
    parser.add_argument("--decoder-hidden-dim", type=int, default=int(DEFAULT_CONFIG["decoder_hidden_dim"]))
    parser.add_argument("--beta", type=float, default=TRAINING_DEFAULTS["beta"])
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=TRAINING_DEFAULTS["threshold_percentile"],
        help="Percentile used when threshold-strategy=percentile",
    )
    parser.add_argument(
        "--threshold-strategy",
        type=str,
        choices=["val_f1", "percentile"],
        default="val_f1",
        help="Threshold selection method",
    )
    parser.add_argument("--num-workers", type=int, default=TRAINING_DEFAULTS["num_workers"])
    parser.add_argument("--pin-memory", action="store_true", default=TRAINING_DEFAULTS["pin_memory"])
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main() -> None:
    """Entry point for command-line training."""
    parser = build_arg_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)

    train(args)


if __name__ == "__main__":
    main()
