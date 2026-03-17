"""Replay recording service for live metrics streams."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass
class ReplaySession:
    """In-memory replay session before persistence."""

    replay_id: str
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=utc_now_iso)
    ended_at: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)


class ReplayRecorder:
    """Manage replay session lifecycle and persistence."""

    def __init__(self, replays_dir: Path) -> None:
        self.replays_dir = replays_dir
        self.replays_dir.mkdir(parents=True, exist_ok=True)
        self.current: ReplaySession | None = None

    def start(self, name: str | None = None, metadata: dict[str, Any] | None = None) -> ReplaySession:
        if self.current is not None:
            raise ValueError("Replay recording already in progress")

        replay_id = uuid4().hex
        session_name = name or f"replay-{replay_id[:8]}"
        self.current = ReplaySession(
            replay_id=replay_id,
            name=session_name,
            metadata=metadata or {},
        )
        return self.current

    def record_event(self, event: dict[str, Any]) -> None:
        if self.current is None:
            return
        self.current.events.append(event)

    def stop(self) -> Path:
        if self.current is None:
            raise ValueError("No active replay recording")

        self.current.ended_at = utc_now_iso()
        payload = {
            "id": self.current.replay_id,
            "name": self.current.name,
            "started_at": self.current.started_at,
            "ended_at": self.current.ended_at,
            "metadata": self.current.metadata,
            "duration_seconds": self._duration_seconds(self.current.started_at, self.current.ended_at),
            "event_count": len(self.current.events),
            "events": self.current.events,
        }

        out_path = self.replays_dir / f"{self.current.replay_id}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.current = None
        return out_path

    def list_replays(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in sorted(self.replays_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            items.append(
                {
                    "id": payload.get("id", path.stem),
                    "name": payload.get("name", path.stem),
                    "started_at": payload.get("started_at"),
                    "ended_at": payload.get("ended_at"),
                    "duration_seconds": payload.get("duration_seconds"),
                    "event_count": payload.get("event_count", 0),
                    "file": str(path),
                }
            )
        return items

    def get_replay(self, replay_id: str) -> dict[str, Any]:
        path = self.replays_dir / f"{replay_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Replay not found: {replay_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def status(self) -> dict[str, Any]:
        if self.current is None:
            return {"recording": False}
        return {
            "recording": True,
            "id": self.current.replay_id,
            "name": self.current.name,
            "started_at": self.current.started_at,
            "event_count": len(self.current.events),
        }

    @staticmethod
    def _duration_seconds(started_at: str, ended_at: str | None) -> float | None:
        if ended_at is None:
            return None
        try:
            start_dt = datetime.fromisoformat(started_at)
            end_dt = datetime.fromisoformat(ended_at)
            return (end_dt - start_dt).total_seconds()
        except Exception:
            return None
