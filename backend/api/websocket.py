"""WebSocket streaming for live IoT network metrics."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.environment.iot_network_env import IoTNetworkEnv, STATE_INDEX


router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """Manage active WebSocket connections for live broadcasts."""

    def __init__(self) -> None:
        """Initialize connection manager state."""
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register an incoming WebSocket connection.

        Args:
            websocket: Incoming client WebSocket.
        """
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a disconnected WebSocket connection.

        Args:
            websocket: Client WebSocket to remove.
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast JSON message to all connected clients.

        Args:
            message: JSON-serializable payload.
        """
        disconnected: list[WebSocket] = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except RuntimeError:
                disconnected.append(connection)
            except Exception:
                disconnected.append(connection)

        for connection in disconnected:
            self.disconnect(connection)


manager = ConnectionManager()


@router.websocket("/ws/live-metrics")
async def live_metrics(websocket: WebSocket) -> None:
    """Stream simulated network metrics every 500ms over WebSocket.

    Args:
        websocket: WebSocket connection from client.
    """
    await manager.connect(websocket)

    env = IoTNetworkEnv()
    state, _ = env.reset()

    try:
        while True:
            action = env.action_space.sample()
            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                state, _ = env.reset()

            payload = {
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "latency_ms": float(1.0 + 99.0 * state[STATE_INDEX.latency]),
                "throughput_mbps": float(20.0 + 180.0 * state[STATE_INDEX.throughput]),
                "energy_nj_per_bit": float(5.0 + 15.0 * state[STATE_INDEX.energy]),
                "qos_score": float(state[STATE_INDEX.qos_score]),
                "anomaly_detected": bool(state[STATE_INDEX.anomaly_score] > 0.5),
                "anomaly_score": float(state[STATE_INDEX.anomaly_score]),
                "nodes_active": int(round(state[STATE_INDEX.n_active_nodes] * float(env.config["n_nodes"]))),
            }

            replay_recorder = getattr(websocket.app.state, "replay_recorder", None)
            if replay_recorder is not None:
                replay_recorder.record_event(
                    {
                        "timestamp": payload["timestamp"],
                        "type": "live-metric",
                        "action": {
                            "routing": float(action[0]),
                            "sleep": float(action[1]),
                            "power": float(action[2]),
                            "buffer": float(action[3]),
                        },
                        "metrics": payload,
                    }
                )

            await manager.broadcast(payload)
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
