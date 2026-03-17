# Copilot Prompts Cheatsheet — GenIoT-Optimizer

Quick reference for effective GitHub Copilot prompts while building this project.
Use these as inline comments above your code to get the best completions.

---

## Models

```python
# Implement WGAN-GP Generator using LSTM for IoT traffic sequence generation
# Input: (batch, latent_dim=128) noise → Output: (batch, 50, 41) traffic sequence
# Use 2-layer LSTM with hidden_size=256, followed by Linear + Tanh

# Implement the Wasserstein loss with gradient penalty for the Critic
# lambda_gp=10, n_critic=5 training steps per generator step

# Implement VAE encoder using BiLSTM that outputs mu and log_var
# Input: (batch, seq_len=50, n_features=41) → mu, log_var shape: (batch, 64)

# Implement reparameterization trick for VAE
# z = mu + std * epsilon, epsilon ~ N(0,I)

# Implement VAE loss: MSE reconstruction + beta-weighted KL divergence
# beta=1.0, reduction='sum', return total loss and both components separately

# Compute anomaly score as per-sample mean squared reconstruction error
# Return scores as numpy array, apply threshold to get binary predictions

# Create custom Gymnasium environment for IoT network optimization
# observation_space: Box(12,) normalized [0,1]
# action_space: Box(4,) for routing, sleep, power, buffer controls
# reward = 0.3*latency_reward + 0.3*throughput_reward + 0.2*energy_reward + 0.2*qos_reward
```

---

## Training

```python
# Write a training loop for WGAN-GP
# - Train critic n_critic=5 times per generator step
# - Log: critic_loss, gen_loss, gradient_penalty every 50 steps
# - Save checkpoint every 10 epochs to models/weights/

# Write an evaluation function for the VAE anomaly detector
# Returns: precision, recall, f1, roc_auc, optimal_threshold
# Plot ROC curve and reconstruction error histogram

# Write a PPO training script using stable_baselines3
# env=IoTNetworkEnv(), total_timesteps=500_000, tensorboard_log='./logs/'
# After training, evaluate and compare against random policy baseline
```

---

## API

```python
# FastAPI POST endpoint /api/generate-traffic
# Accepts GenerateTrafficRequest, uses app.state.gan to generate samples
# Returns list of sequences as JSON

# FastAPI WebSocket endpoint /ws/live-metrics
# Streams network metrics JSON every 500ms using asyncio.sleep
# Broadcasts to all connected clients using ConnectionManager

# FastAPI startup event that loads all three models into app.state
# Handle missing weights files gracefully with a warning
```

---

## Frontend

```jsx
// React component TrafficChart using Recharts AreaChart
// Props: data array of {timestamp, latency, throughput}
// Dual Y-axes: latency (left, ms) and throughput (right, Mbps)
// 60-point rolling window, updates every 500ms

// Custom React hook useWebSocket(url)
// Connects to WebSocket, returns { data, isConnected, error }
// Auto-reconnects with exponential backoff on disconnect
// Parses JSON messages and updates state

// Zustand store slice for metrics with addMetric action
// Keep only last 60 data points (rolling window)

// React Flow NetworkTopology component
// 50 sensor nodes arranged in a grid layout
// Color nodes red if their id is in anomalyNodes prop
// Animate edges with React Flow's animated prop when data is flowing
```

---

## Data

```python
# Load UNSW-NB15 CSV, drop metadata columns, encode categoricals with LabelEncoder
# Normalize with MinMaxScaler fit only on train split
# Save scaler to scaler.pkl for use during inference

# Create sliding windows of length 50 with stride 1 from a 2D numpy array
# Input shape: (n_timesteps, n_features) → Output: (n_windows, 50, n_features)

# Split dataset 70/15/15 train/val/test using stratified split on label column
# Ensure no data leakage: fit scaler only on train set
```

---

## Evaluation / Baselines

```python
# Run baseline comparison: evaluate DQN, LSTM-P, GAN-only, VAE-RL, and GenIoT
# on latency_ms, throughput_mbps, energy_nj_per_bit, f1_score
# Return results as a pandas DataFrame formatted like Table I in the paper

# Compute Maximum Mean Discrepancy (MMD) between real and synthetic traffic
# Use RBF kernel with sigma=1.0
```

---

## Tips for using Copilot effectively

1. **Always reference a context file** — before coding a new file, add `# See backend/MODELS.md for architecture` at the top
2. **Write the class skeleton first** — define `__init__`, method signatures with type hints, and docstrings. Copilot fills in implementations.
3. **Paste paper equations as comments** — e.g. `# L_GAN = E[D(x)] - E[D(G(z))] + lambda_gp * L_GP` directly above the loss function
4. **Use Copilot Chat for debugging** — select broken code, ask "Why does this WGAN-GP training diverge?"
5. **Open related files** — have `vae.py` open in another tab when writing `routes.py` so Copilot picks up the class interface