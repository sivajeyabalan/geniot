"""API routes for alert rule CRUD and history."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field


router = APIRouter(tags=["rules"])


class RuleUpsertRequest(BaseModel):
    """Request schema for create and update rule."""

    metric: str = Field(..., min_length=1, max_length=64)
    operator: str = Field(..., min_length=1, max_length=2)
    threshold: float
    duration_seconds: float = Field(default=0.0, ge=0.0)
    severity: str = Field(default="medium", min_length=1, max_length=16)
    enabled: bool = True


@router.get("/rules")
async def list_rules(request: Request) -> dict[str, Any]:
    """List all configured rules and recent alert history."""
    engine = getattr(request.app.state, "alert_rules_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Alert rules engine unavailable")

    return {
        "items": engine.list_rules(),
        "alert_history": engine.list_alert_history(limit=200),
    }


@router.post("/rules")
async def create_rule(payload: RuleUpsertRequest, request: Request) -> dict[str, Any]:
    """Create a new alert rule."""
    engine = getattr(request.app.state, "alert_rules_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Alert rules engine unavailable")

    try:
        item = engine.create_rule(
            metric=payload.metric,
            operator=payload.operator,
            threshold=payload.threshold,
            duration_seconds=payload.duration_seconds,
            severity=payload.severity.lower(),
            enabled=payload.enabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"item": item}


@router.put("/rules/{rule_id}")
async def update_rule(rule_id: str, payload: RuleUpsertRequest, request: Request) -> dict[str, Any]:
    """Update an existing alert rule."""
    engine = getattr(request.app.state, "alert_rules_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Alert rules engine unavailable")

    try:
        item = engine.update_rule(
            rule_id=rule_id,
            metric=payload.metric,
            operator=payload.operator,
            threshold=payload.threshold,
            duration_seconds=payload.duration_seconds,
            severity=payload.severity.lower(),
            enabled=payload.enabled,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"item": item}


@router.delete("/rules/{rule_id}")
async def delete_rule(rule_id: str, request: Request) -> dict[str, Any]:
    """Delete a rule by id."""
    engine = getattr(request.app.state, "alert_rules_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Alert rules engine unavailable")

    try:
        engine.delete_rule(rule_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {"deleted": True, "id": rule_id}
