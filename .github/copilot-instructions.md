# GenIoT-Optimizer — GitHub Copilot Workspace Instructions

## Project Overview

This is **GenIoT-Optimizer**, a final year engineering project that simulates and optimizes IoT network performance using Generative AI. The system is based on the research paper:
> "Generative AI for Simulating and Optimizing IoT Network Performance" — Velammal College of Engineering and Technology, Madurai.

The project has **4 core modules**:
1. Synthetic Traffic Generator (WGAN-GP + VAE + Diffusion)
2. Anomaly Detection Engine (VAE reconstruction error)
3. RL Network Optimizer (PPO agent with custom Gym environment)
4. Digital Twin Dashboard (React + WebSocket)

---

## Repository Structure

```
geniot-optimizer/
├── .github/
│   └── copilot-instructions.md      ← you are here
├── backend/
│   ├── models/
│   │   ├── wgan_gp.py               ← WGAN-GP traffic generator
│   │   ├── vae.py                   ← VAE for anomaly detection
│   │   ├── diffusion.py             ← DDPM temporal model
│   │   └── ppo_agent.py             ← PPO reinforcement learning
│   ├── environment/
│   │   ├── iot_network_env.py       ← Custom Gymnasium environment
│   │   └── network_simulator.py     ← IoT network state simulator
│   ├── api/
│   │   ├── main.py                  ← FastAPI app entry point
│   │   ├── routes.py                ← REST endpoints
│   │   └── websocket.py             ← Live metrics WebSocket
│   ├── training/
│   │   ├── train_gan.py
│   │   ├── train_vae.py
│   │   └── train_rl.py
│   ├── data/
│   │   ├── preprocess.py
│   │   └── datasets.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── NetworkTopology.jsx
│   │   │   ├── TrafficChart.jsx
│   │   │   ├── AnomalyPanel.jsx
│   │   │   └── MetricsGrid.jsx
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── Simulation.jsx
│   │   │   └── Optimization.jsx
│   │   ├── store.js                 ← Zustand state
│   │   └── websocket.js
│   └── package.json
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_gan_training.ipynb
│   ├── 03_vae_anomaly.ipynb
│   └── 04_rl_training.ipynb
└── docker-compose.yml
```

---

## Tech Stack

### Backend (Python)
- **PyTorch 2.x** — all neural network models (GAN, VAE, Diffusion)
- **stable-baselines3** — PPO reinforcement learning agent
- **Gymnasium** — custom IoT network RL environment
- **FastAPI** — REST API + WebSocket server
- **scikit-learn** — metrics, baselines, preprocessing helpers
- **pandas, numpy** — data loading and preprocessing

### Frontend (React)
- **React 18 + Vite** — app framework
- **Recharts** — live line/area/bar charts
- **React Flow** — network topology visualization
- **Tailwind CSS** — styling
- **Zustand** — global state management
- **Native WebSocket API** — real-time metrics from backend

---

## Coding Conventions

### Python
- Python 3.10+
- Type hints on all function signatures
- Google-style docstrings on all classes and public methods
- All model hyperparameters defined in a `config: dict` argument or dataclass at the top of each file — never hardcoded inside methods
- Use `torch.nn.Module` base class for all neural network models
- Device-agnostic code: always use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Save/load model weights with `torch.save` / `torch.load` using state_dict pattern
- Logging via Python `logging` module — never bare `print()` in model or training code

### React / JavaScript
- Functional components only — no class components
- Custom hooks in `src/hooks/` for WebSocket, data fetching
- Zustand store slices in `src/store.js`
- All chart data formatted as `[{ timestamp, value, ... }]` arrays
- PropTypes or JSDoc comments on all components

---

## Key ML Concepts (for Copilot context)

### WGAN-GP (Wasserstein GAN with Gradient Penalty)
- Generator G maps noise z ~ N(0,I) → synthetic IoT traffic sequences
- Critic D (not discriminator) scores realism without sigmoid
- Loss: `L = E[D(x)] - E[D(G(z))] + λ * gradient_penalty`
- Gradient penalty enforces Lipschitz constraint on critic
- Train critic 5× per generator step (n_critic = 5)
- Use `torch.autograd.grad` for gradient penalty computation

### VAE (Variational Autoencoder)
- Encoder maps input x → mean μ and log variance log(σ²)
- Reparameterization: z = μ + σ * ε, ε ~ N(0,I)
- Decoder maps z → reconstructed x̂
- Loss: reconstruction loss + KL divergence
- Anomaly score = reconstruction error on test samples
- High reconstruction error → anomaly

### PPO (Proximal Policy Optimization)
- State: vector of network metrics (latency, throughput, packet loss, energy)
- Action: discrete or continuous network config changes
- Reward: `R = α1*R_latency + α2*R_throughput + α3*R_energy + α4*R_QoS`
- Use `stable_baselines3.PPO` with a custom `gymnasium.Env`
- Default weights: α1=0.3, α2=0.3, α3=0.2, α4=0.2

### IoT Network Environment (Gymnasium)
- `observation_space`: Box with shape (n_metrics,) — current network state
- `action_space`: Discrete or Box — configuration parameters to tune
- `step()`: simulate one timestep, return (obs, reward, done, truncated, info)
- `reset()`: return to initial network state, add slight random noise for diversity

---

## Datasets

| Dataset | Use Case | Download |
|---------|----------|----------|
| UNSW-NB15 | Anomaly detection training | research.unsw.edu.au/projects/unsw-nb15-dataset |
| CIC-IoT2023 | Traffic generation (GAN training) | ciciot.unb.ca |
| TON-IoT | RL environment state space | research.unsw.edu.au/projects/toniot-datasets |

**Preprocessing pipeline** (in `backend/data/preprocess.py`):
1. Drop NaN rows
2. Encode categorical features (label encoding)
3. Min-max normalize all numeric features to [0, 1]
4. Create sliding windows of length 50 for temporal models
5. Split 70/15/15 train/val/test

---

## API Endpoints (FastAPI)

```
POST /api/generate-traffic       → Run GAN generator, return synthetic sequence
POST /api/detect-anomaly         → Run VAE, return anomaly score + flag
POST /api/optimize               → Run PPO step, return recommended config
GET  /api/metrics                → Current network metrics snapshot
GET  /api/baselines              → Baseline comparison results
WS   /ws/live-metrics            → WebSocket: stream metrics every 500ms
```

---

## Performance Targets (from paper)

| Metric | Baseline best | Our target |
|--------|--------------|------------|
| Latency (ms) | 36.9 | ≤ 34 |
| Throughput (Mbps) | 151.8 | ≥ 175 |
| Energy (nJ/bit) | 14.6 | ≤ 13.5 |
| Anomaly F1 (%) | 86.4 | ≥ 89 |

---

## Important Notes for Copilot

- Never hardcode file paths — use `pathlib.Path` and relative paths
- All trained model weights go in `backend/models/weights/` (gitignored)
- The Gymnasium environment simulates a network, it does NOT connect to real devices
- WebSocket sends JSON: `{"timestamp": ..., "latency": ..., "throughput": ..., "anomaly": bool}`
- Frontend polls WebSocket at 500ms intervals and maintains a rolling 60-point window for charts
- Docker Compose runs: FastAPI on port 8000, React dev server on port 5173, PostgreSQL on 5432, Redis on 6379