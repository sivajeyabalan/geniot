"""REST routes for traffic generation, anomaly detection, and optimization."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from backend.environment.iot_network_env import IoTNetworkEnv, STATE_INDEX
from backend.models.vae import DEVICE as VAE_DEVICE
from backend.models.vae import anomaly_score


router = APIRouter(tags=["api"])

_METRICS_ENV = IoTNetworkEnv()
_METRICS_STATE, _ = _METRICS_ENV.reset(seed=42)


class GenerateTrafficRequest(BaseModel):
    """Request schema for synthetic traffic generation."""

    n_samples: int = Field(default=10, ge=1, le=2048)
    seq_len: int = Field(default=50, ge=1, le=500)


class GenerateTrafficResponse(BaseModel):
    """Response schema for synthetic traffic generation."""

    sequences: list[list[list[float]]]
    shape: list[int]


class DetectAnomalyRequest(BaseModel):
    """Request schema for anomaly scoring."""

    traffic_sequence: list[Any]


class DetectAnomalyResponse(BaseModel):
    """Response schema for anomaly detection endpoint."""

    score: float
    is_anomaly: bool
    threshold: float


class OptimizeRequest(BaseModel):
    """Request schema for RL-based optimization endpoint."""

    network_state: list[float]

    @field_validator("network_state")
    @classmethod
    def validate_network_state(cls, value: list[float]) -> list[float]:
        """Validate input network state length.

        Args:
            value: Incoming network state list.

        Returns:
            Validated list.

        Raises:
            ValueError: If length is not 12.
        """
        if len(value) != 12:
            raise ValueError("network_state must contain exactly 12 values")
        return value


class OptimizeResponse(BaseModel):
    """Response schema for optimization endpoint."""

    recommended_config: dict[str, float]
    expected_reward: float


class MetricsResponse(BaseModel):
    """Response schema for metrics snapshot endpoint."""

    timestamp: str
    latency: float
    throughput: float
    energy: float
    packet_loss: float
    qos_score: float
    congestion: float
    n_active_nodes: float
    collision_rate: float
    hop_count: float
    buffer_occupancy: float
    anomaly_score: float
    channel_quality: float


class HealthResponse(BaseModel):
    """Response schema for service health endpoint."""

    status: str
    models_loaded: bool


def _resize_sequence(sequence: np.ndarray, target_seq_len: int) -> np.ndarray:
    """Resize a generated sequence to target length by slicing or padding.

    Args:
        sequence: Input sequence with shape ``(seq_len, n_features)``.
        target_seq_len: Requested output sequence length.

    Returns:
        Sequence resized to target length.
    """
    current_len = sequence.shape[0]
    if current_len == target_seq_len:
        return sequence
    if current_len > target_seq_len:
        return sequence[:target_seq_len]

    pad_count = target_seq_len - current_len
    pad_frame = np.repeat(sequence[-1:, :], pad_count, axis=0)
    return np.concatenate([sequence, pad_frame], axis=0)


def _get_metrics_snapshot() -> MetricsResponse:
    """Generate one live metrics snapshot using IoTNetworkEnv.

    Returns:
        MetricsResponse object with current metrics.
    """
    global _METRICS_STATE

    action = _METRICS_ENV.action_space.sample()
    _METRICS_STATE, _, terminated, truncated, _ = _METRICS_ENV.step(action)
    if terminated or truncated:
        _METRICS_STATE, _ = _METRICS_ENV.reset()

    state = _METRICS_STATE
    return MetricsResponse(
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        latency=float(state[STATE_INDEX.latency]),
        throughput=float(state[STATE_INDEX.throughput]),
        energy=float(state[STATE_INDEX.energy]),
        packet_loss=float(state[STATE_INDEX.packet_loss]),
        qos_score=float(state[STATE_INDEX.qos_score]),
        congestion=float(state[STATE_INDEX.congestion]),
        n_active_nodes=float(state[STATE_INDEX.n_active_nodes]),
        collision_rate=float(state[STATE_INDEX.collision_rate]),
        hop_count=float(state[STATE_INDEX.hop_count]),
        buffer_occupancy=float(state[STATE_INDEX.buffer_occupancy]),
        anomaly_score=float(state[STATE_INDEX.anomaly_score]),
        channel_quality=float(state[STATE_INDEX.channel_quality]),
    )


@router.post("/generate-traffic", response_model=GenerateTrafficResponse)
async def generate_traffic(
    payload: GenerateTrafficRequest,
    request: Request,
) -> GenerateTrafficResponse:
    """Generate synthetic traffic windows using WGANGP.

    Args:
        payload: Generation request data.
        request: FastAPI request object.

    Returns:
        Generated traffic sequences and output shape.
    """
    wgan = getattr(request.app.state, "wgan", None)
    if wgan is None:
        raise HTTPException(status_code=503, detail="WGANGP model not available")

    with torch.no_grad():
        samples = wgan.generate(n_samples=payload.n_samples).detach().cpu().numpy()

    resized = np.stack([
        _resize_sequence(sequence=sample, target_seq_len=payload.seq_len)
        for sample in samples
    ])

    return GenerateTrafficResponse(
        sequences=resized.astype(np.float32).tolist(),
        shape=list(resized.shape),
    )


@router.post("/detect-anomaly", response_model=DetectAnomalyResponse)
async def detect_anomaly(
    payload: DetectAnomalyRequest,
    request: Request,
) -> DetectAnomalyResponse:
    """Compute anomaly score for an IoT traffic sequence.

    Args:
        payload: Input traffic sequence payload.
        request: FastAPI request object.

    Returns:
        Score, boolean anomaly flag, and threshold used.
    """
    vae_model = getattr(request.app.state, "vae", None)
    threshold = float(getattr(request.app.state, "vae_threshold", 0.1))

    if vae_model is None:
        raise HTTPException(status_code=503, detail="VAE model not available")

    sequence = np.asarray(payload.traffic_sequence, dtype=np.float32)

    if sequence.ndim == 2:
        sequence = np.expand_dims(sequence, axis=0)
    elif sequence.ndim != 3:
        raise HTTPException(
            status_code=422,
            detail="traffic_sequence must be 2D (seq_len, features) or 3D (batch, seq_len, features)",
        )

    scores = anomaly_score(model=vae_model, x=sequence, device=VAE_DEVICE)
    score = float(np.mean(scores))

    return DetectAnomalyResponse(
        score=score,
        is_anomaly=bool(score > threshold),
        threshold=threshold,
    )


@router.post("/optimize", response_model=OptimizeResponse)
async def optimize_network(
    payload: OptimizeRequest,
    request: Request,
) -> OptimizeResponse:
    """Recommend network configuration using trained PPO agent.

    Args:
        payload: Input network state payload.
        request: FastAPI request object.

    Returns:
        Recommended control config and expected reward.
    """
    optimizer = getattr(request.app.state, "optimizer", None)
    if optimizer is None:
        raise HTTPException(status_code=503, detail="IoTOptimizer model not available")

    state = np.asarray(payload.network_state, dtype=np.float32)
    recommendation = optimizer.recommend_config(state)

    expected_reward = float(recommendation.get("expected_reward", 0.0))
    recommended_config = {
        "routing": float(recommendation.get("routing", 0.0)),
        "sleep": float(recommendation.get("sleep", 0.0)),
        "power": float(recommendation.get("power", 0.0)),
        "buffer": float(recommendation.get("buffer", 0.0)),
    }

    return OptimizeResponse(
        recommended_config=recommended_config,
        expected_reward=expected_reward,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """Return current simulated network metrics snapshot.

    Returns:
        Metrics snapshot from IoTNetworkEnv dynamics.
    """
    return _get_metrics_snapshot()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Return service health and model loading status.

    Args:
        request: FastAPI request object.

    Returns:
        Health response with status and model load flag.
    """
    models_loaded = bool(getattr(request.app.state, "models_loaded", False))
    return HealthResponse(status="ok", models_loaded=models_loaded)
