"""Replay management routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field


router = APIRouter(tags=["replay"])


class ReplayStartRequest(BaseModel):
    """Request to start replay recording."""

    name: str | None = Field(default=None, max_length=128)
    metadata: dict[str, Any] | None = None


class ReplayStartResponse(BaseModel):
    """Start replay response."""

    recording: bool
    id: str
    name: str
    started_at: str


class ReplayStopResponse(BaseModel):
    """Stop replay response."""

    recording: bool
    file: str


class ReplayStatusResponse(BaseModel):
    """Replay status response."""

    recording: bool
    id: str | None = None
    name: str | None = None
    started_at: str | None = None
    event_count: int | None = None


@router.post("/start", response_model=ReplayStartResponse)
async def start_replay(payload: ReplayStartRequest, request: Request) -> ReplayStartResponse:
    recorder = getattr(request.app.state, "replay_recorder", None)
    if recorder is None:
        raise HTTPException(status_code=503, detail="Replay recorder unavailable")

    try:
        session = recorder.start(name=payload.name, metadata=payload.metadata)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return ReplayStartResponse(
        recording=True,
        id=session.replay_id,
        name=session.name,
        started_at=session.started_at,
    )


@router.post("/stop", response_model=ReplayStopResponse)
async def stop_replay(request: Request) -> ReplayStopResponse:
    recorder = getattr(request.app.state, "replay_recorder", None)
    if recorder is None:
        raise HTTPException(status_code=503, detail="Replay recorder unavailable")

    try:
        path = recorder.stop()
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return ReplayStopResponse(recording=False, file=str(path))


@router.get("/list")
async def list_replays(request: Request) -> dict[str, Any]:
    recorder = getattr(request.app.state, "replay_recorder", None)
    if recorder is None:
        raise HTTPException(status_code=503, detail="Replay recorder unavailable")

    return {"items": recorder.list_replays(), "status": recorder.status()}


@router.get("/status/current", response_model=ReplayStatusResponse)
async def replay_status(request: Request) -> ReplayStatusResponse:
    recorder = getattr(request.app.state, "replay_recorder", None)
    if recorder is None:
        raise HTTPException(status_code=503, detail="Replay recorder unavailable")

    status = recorder.status()
    return ReplayStatusResponse(**status)


@router.get("/{replay_id}")
async def get_replay(replay_id: str, request: Request) -> dict[str, Any]:
    recorder = getattr(request.app.state, "replay_recorder", None)
    if recorder is None:
        raise HTTPException(status_code=503, detail="Replay recorder unavailable")

    try:
        return recorder.get_replay(replay_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
