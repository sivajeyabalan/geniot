# Backend API — FastAPI Instructions

> Copilot context file for `backend/api/` directory.

---

## main.py — App entry point

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router
from backend.api.websocket import ws_router

app = FastAPI(
    title="GenIoT-Optimizer API",
    description="Generative AI for IoT Network Optimization",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
app.include_router(ws_router)

# Load all models once at startup — NOT per request
@app.on_event("startup")
async def load_models():
    from backend.models.wgan_gp import WGANGP
    from backend.models.vae import VAE
    from backend.models.ppo_agent import IoTOptimizer
    app.state.gan = WGANGP.from_weights("backend/models/weights/gan.pt")
    app.state.vae = VAE.from_weights("backend/models/weights/vae.pt")
    app.state.optimizer = IoTOptimizer("backend/models/weights/ppo_iot.zip")
```

---

## routes.py — REST endpoints

### All endpoints

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| POST | `/api/generate-traffic` | Run GAN generator | `{sequences: [[...]], timestamps: [...]}` |
| POST | `/api/detect-anomaly` | Run VAE anomaly detection | `{score: float, is_anomaly: bool, threshold: float}` |
| POST | `/api/optimize` | Run PPO step | `{action: [...], recommended_config: {...}, expected_reward: float}` |
| GET | `/api/metrics` | Current network snapshot | `{latency, throughput, energy, qos, ...}` |
| GET | `/api/baselines` | Comparison table data | `{methods: [...], results: [...]}` |
| GET | `/api/health` | Health check | `{status: "ok"}` |

### Request/Response schemas (Pydantic)

```python
from pydantic import BaseModel

class GenerateTrafficRequest(BaseModel):
    n_samples: int = 10
    seq_len: int = 50

class AnomalyDetectRequest(BaseModel):
    traffic_sequence: list[list[float]]  # shape: (seq_len, n_features)

class OptimizeRequest(BaseModel):
    network_state: list[float]   # length 12 — matches observation_space

class MetricsResponse(BaseModel):
    timestamp: float
    latency_ms: float
    throughput_mbps: float
    energy_nj_per_bit: float
    packet_loss_pct: float
    qos_score: float
    anomaly_detected: bool
    anomaly_score: float
    n_active_nodes: int
```

---

## websocket.py — Live metrics stream

```python
from fastapi import WebSocket, WebSocketDisconnect
import asyncio, json, time

# WebSocket endpoint: ws://localhost:8000/ws/live-metrics
# Sends JSON every 500ms:
# {
#   "timestamp": 1700000000.0,
#   "latency": 32.4,
#   "throughput": 183.5,
#   "energy": 13.1,
#   "qos": 94.2,
#   "anomaly": false,
#   "anomaly_score": 0.12,
#   "nodes_active": 97
# }

# Manager handles multiple frontend clients simultaneously
class ConnectionManager:
    def __init__(self): ...
    async def connect(self, ws: WebSocket): ...
    def disconnect(self, ws: WebSocket): ...
    async def broadcast(self, data: dict): ...
```

---

## Data flow

```
Frontend (React)
    ↕ WebSocket (ws://localhost:8000/ws/live-metrics) — 500ms interval
    ↕ REST POST /api/generate-traffic  → GAN → synthetic sequences
    ↕ REST POST /api/detect-anomaly    → VAE → anomaly score
    ↕ REST POST /api/optimize          → PPO → config recommendation
FastAPI (port 8000)
    ↕ loads models from weights/ at startup
    ↕ uses NetworkSimulator for live metric generation
    ↕ stores metrics history in PostgreSQL
```

---

## Running locally

```bash
# Install deps
pip install -r requirements.txt

# Start API server (from project root)
uvicorn backend.api.main:app --reload --port 8000

# Or via Docker Compose
docker compose up
```