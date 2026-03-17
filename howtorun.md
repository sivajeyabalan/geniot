# How to Run GenIoT-Optimizer

This guide shows how to run backend, frontend, training, evaluation, and validation scripts.

## 1) Prerequisites
- Windows 10/11 (project currently used on Windows)
- Python 3.10+ (venv recommended)
- Node.js 18+ and npm
- Git (optional but recommended)

---

## 2) Project setup
From project root:

```bash
# Create and activate virtual environment (example)
python -m venv .venv
.venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

Frontend setup:

```bash
cd frontend
npm install
cd ..
```

---

## 3) Run backend API
From project root:

```bash
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Expected:
- API docs at: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/api/health

Quick health test:

```bash
curl http://127.0.0.1:8000/api/health
```

---

## 4) Run frontend dashboard
Open a second terminal:

```bash
cd frontend
npm run dev
```

Expected:
- Dashboard URL: http://localhost:5173
- WebSocket connection status should become Connected when backend is running.

---

## 5) Run full app (recommended order)
1. Start backend first (`uvicorn ...`)
2. Start frontend second (`npm run dev`)
3. Open browser at `http://localhost:5173`
4. Verify live cards/chart/anomaly panel update

---

## 6) Train models (optional)
From project root:

### Train VAE
```bash
python -m backend.training.train_vae
```

### Train WGAN-GP
```bash
python -m backend.training.train_gan
```

### Train PPO optimizer
```bash
python -m backend.training.train_rl
```

Weights are stored in:
- `backend/models/weights/`

---

## 7) Run baseline comparison
From project root:

```bash
python -m backend.training.baselines
```

Generated outputs:
- `results/comparison_table.csv`
- `results/comparison_chart.png`
- `results/comparison_summary.json` (if produced by script)

---

## 8) Run end-to-end stability check
From project root:

```bash
python -m backend.training.system_stability_check --minutes 5
```

Generated output:
- `results/system_stability_report.json`

Use this to verify backend + frontend + websocket stability over a fixed duration.

---

## 9) Test useful API endpoints

### Generate traffic
```bash
curl -X POST http://127.0.0.1:8000/api/generate-traffic \
  -H "Content-Type: application/json" \
  -d "{\"n_samples\": 2, \"seq_len\": 50}"
```

### Detect anomaly
```bash
curl -X POST http://127.0.0.1:8000/api/detect-anomaly \
  -H "Content-Type: application/json" \
  -d "{\"traffic_sequence\": [[0.1,0.2],[0.2,0.3]]}"
```

### Optimize state
```bash
curl -X POST http://127.0.0.1:8000/api/optimize \
  -H "Content-Type: application/json" \
  -d "{\"network_state\": [0.5,0.5,0.5,0.1,0.7,0.3,0.9,0.1,0.4,0.3,0.1,0.8]}"
```

---

## 10) Troubleshooting

### A) Port already in use
- Backend 8000 busy: stop old process or run another port.
- Frontend 5173 busy: Vite may auto-switch to 5174.

### B) PPO load error (`numpy._core.numeric`)
If this appears, use the latest code in `backend/models/ppo_agent.py` (compatibility alias already added).

### C) Missing model weights
If weights are missing:
- Run training scripts in Section 6
- Or allow fallback behavior where implemented

### D) WebSocket not connecting
- Ensure backend is running
- Ensure frontend URL uses correct ws endpoint: `ws://localhost:8000/ws/live-metrics`

---

## 11) Suggested demo run script (manual)
1. Start backend terminal
2. Start frontend terminal
3. Open dashboard and keep it running
4. Trigger anomalies/optimize via API or UI flows
5. Run baseline script and show generated table/chart
6. (Optional) run stability checker and show JSON report

---

## 12) Shutdown
- Stop backend with `Ctrl + C`
- Stop frontend with `Ctrl + C`
- Deactivate venv (if active):

```bash
deactivate
```