# GenIoT-Optimizer Project Report

## 1. Introduction
GenIoT-Optimizer is a final-year engineering project that simulates and optimizes IoT network performance using generative AI and reinforcement learning. The objective is to reduce latency and energy while improving throughput, QoS, and anomaly response in a digital-twin style workflow.

### Problem Statement
IoT networks face dynamic traffic variation, limited energy budgets, and intermittent anomalies. Static optimization rules usually fail under changing load conditions. This project combines generative modeling, anomaly scoring, and RL control to create an adaptive optimizer.

### Objectives
- Build synthetic traffic generation with WGAN-GP.
- Detect anomalous traffic with VAE reconstruction error.
- Optimize network actions with PPO in a custom Gymnasium environment.
- Stream real-time metrics to a React dashboard.
- Compare GenIoT against classical and learning baselines.

## 2. Methodology

### 2.1 System Architecture
The platform contains four modules:
1. Synthetic Traffic Generator (WGAN-GP)
2. Anomaly Detection Engine (VAE)
3. Network Optimizer (PPO)
4. Digital Twin Dashboard (FastAPI + WebSocket + React)

### 2.2 IoT Environment Design
The custom environment uses a 12-dimensional normalized state:
- latency, throughput, energy, packet_loss, qos,
- congestion, active_nodes, collision_rate, hop_count,
- buffer_occupancy, anomaly_score, channel_quality

Action space is continuous with four controls:
- routing, sleep duty cycle, transmission power, buffer tuning

Reward used for optimization:
- R = 0.3*(1-latency) + 0.3*throughput + 0.2*(1-energy) + 0.2*qos

### 2.3 Model Pipelines

#### WGAN-GP (Traffic Generation)
- Generator: latent noise → sequence `(50, 41)`
- Critic: Wasserstein objective + gradient penalty
- Training style: multiple critic updates per generator update

#### VAE (Anomaly Detection)
- Encoder: BiLSTM to latent distribution `(mu, log_var)`
- Decoder: LSTM reconstruction of sequence
- Anomaly score: reconstruction MSE
- Threshold: loaded from saved calibration file

#### PPO (Optimization)
- Policy network trained in IoTNetworkEnv
- Learns action vector for routing/sleep/power/buffer
- Model artifact: `backend/models/weights/ppo_iot.zip`

### 2.4 Baseline Comparison Setup
Baselines evaluated over 100 episodes each:
1. RandomPolicy
2. DQNBaseline (discrete wrapper over same env)
3. GreedyHeuristic (fixed action)
4. LSTMPredictor (input=12, hidden=64)
5. GANOnlyHeuristic

GenIoT is evaluated as the final row in the same table.

Metrics:
- Latency (ms): `state[0] * 100`
- Throughput (Mbps): `state[1] * 250`
- Energy (nJ/bit): `state[2] * 25`
- Anomaly F1: VAE-based predictions vs environment anomaly labels

## 3. Results

### 3.1 Comparison Table
Results are exported to:
- `results/comparison_table.csv`
- `results/comparison_chart.png`

Table format follows paper-style “Table I” presentation.

### 3.2 Dashboard Validation
The dashboard consumes `/ws/live-metrics` updates and renders:
- Live metric cards
- Real-time traffic chart
- Anomaly event panel
- Connection status and reconnect behavior

### 3.3 End-to-End Stability
A stability checker script validates backend, frontend, and websocket continuity over a target duration:
- Script: `backend/training/system_stability_check.py`
- Output: `results/system_stability_report.json`

## 4. Conclusion
GenIoT-Optimizer demonstrates a practical AI-driven IoT optimization stack integrating generation, detection, and control. The architecture supports online visualization and quantitative baseline comparison. The system is extensible for larger state spaces, richer traffic traces, and more advanced RL algorithms.

## 5. Future Work
- Add multi-agent optimization across node clusters.
- Integrate uncertainty-aware anomaly calibration.
- Add scenario replay in dashboard for reproducible demos.
- Include hardware-in-the-loop interfaces for edge deployment.

## 6. Reproducibility Commands
From project root:

- Baseline comparison:
  - `python -m backend.training.baselines`

- Stability check (5 min):
  - `python -m backend.training.system_stability_check --minutes 5`

- Backend API:
  - `python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000`

- Frontend:
  - `cd frontend && npm run dev`
