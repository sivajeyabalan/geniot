"""FastAPI app entrypoint for GenIoT-Optimizer backend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.replay import ReplayRecorder
from backend.api.replay_routes import router as replay_router
from backend.api.routes import router as api_router
from backend.api.websocket import router as websocket_router
from backend.models.ppo_agent import IoTOptimizer
from backend.models.vae import VAE
from backend.models.wgan_gp import WGANGP


LOGGER = logging.getLogger(__name__)


def _load_vae(weights_dir: Path) -> tuple[VAE, float]:
    """Load VAE model and anomaly threshold from weights directory.

    Args:
        weights_dir: Directory that contains VAE checkpoint and threshold.

    Returns:
        Tuple of loaded VAE model and anomaly threshold.
    """
    vae_path = weights_dir / "vae.pt"
    threshold_path = weights_dir / "vae_threshold.txt"

    if vae_path.exists():
        vae_model = VAE.from_weights(vae_path)
    else:
        LOGGER.warning("VAE weights not found at %s. Using untrained VAE.", vae_path)
        vae_model = VAE()

    threshold = 0.1
    if threshold_path.exists():
        try:
            threshold = float(threshold_path.read_text(encoding="utf-8").strip())
        except ValueError:
            LOGGER.warning("Invalid threshold file at %s. Using default %.3f.", threshold_path, threshold)

    return vae_model, threshold


def _load_wgan(weights_dir: Path) -> WGANGP:
    """Load WGANGP model from disk if available.

    Args:
        weights_dir: Directory that may contain WGAN checkpoint.

    Returns:
        Initialized WGANGP model.
    """
    gan_path = weights_dir / "gan.pt"
    if gan_path.exists():
        return WGANGP.from_weights(gan_path)

    LOGGER.warning("WGAN-GP weights not found at %s. Using untrained WGANGP.", gan_path)
    return WGANGP()


def _load_optimizer(weights_dir: Path) -> IoTOptimizer:
    """Load PPO optimizer wrapper from disk.

    Args:
        weights_dir: Directory containing PPO checkpoint.

    Returns:
        IoTOptimizer instance.
    """
    ppo_path = weights_dir / "ppo_iot.zip"
    return IoTOptimizer(ppo_path)


def create_app() -> FastAPI:
    """Create and configure FastAPI app instance.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(title="GenIoT-Optimizer API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event() -> None:
        """Load model artifacts into app state at startup."""
        project_root = Path(__file__).resolve().parents[2]
        weights_dir = project_root / "backend" / "models" / "weights"
        replays_dir = project_root / "results" / "replays"

        app.state.models_loaded = False
        app.state.model_load_errors = []
        app.state.replay_recorder = ReplayRecorder(replays_dir=replays_dir)

        try:
            vae_model, threshold = _load_vae(weights_dir)
            app.state.vae = vae_model
            app.state.vae_threshold = threshold
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Failed to load VAE: %s", exc)
            app.state.model_load_errors.append(f"vae: {exc}")
            app.state.vae = VAE()
            app.state.vae_threshold = 0.1

        try:
            app.state.wgan = _load_wgan(weights_dir)
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Failed to load WGANGP: %s", exc)
            app.state.model_load_errors.append(f"wgan: {exc}")
            app.state.wgan = WGANGP()

        try:
            app.state.optimizer = _load_optimizer(weights_dir)
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Failed to load IoTOptimizer: %s", exc)
            app.state.model_load_errors.append(f"optimizer: {exc}")
            app.state.optimizer = None

        app.state.models_loaded = len(app.state.model_load_errors) == 0

    app.include_router(api_router, prefix="/api")
    app.include_router(replay_router, prefix="/api/replay")
    app.include_router(websocket_router)
    return app


app: FastAPI = create_app()
