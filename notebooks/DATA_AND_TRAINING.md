# Data & Notebooks — Instructions

> Copilot context file for `backend/data/` and `notebooks/` directories.

---

## Data pipeline (backend/data/preprocess.py)

### Input
Raw CSV files from UNSW-NB15, CIC-IoT2023, or TON-IoT datasets.

### Output
Numpy arrays ready for model training:
- `X_train.npy` — shape (n_samples, seq_len=50, n_features=41)
- `X_val.npy`
- `X_test.npy`
- `y_test.npy` — binary anomaly labels (for evaluation only)
- `scaler.pkl` — fitted MinMaxScaler (save this! needed for inference)

### Steps in order

```python
def preprocess_pipeline(csv_path: str, output_dir: str, seq_len: int = 50):
    """
    Full preprocessing pipeline.
    1. Load CSV
    2. Drop columns: ['id', 'attack_cat'] or similar metadata
    3. Drop rows with NaN
    4. Encode: LabelEncoder on categorical columns
    5. Normalize: MinMaxScaler fit on train split only
    6. Create sliding windows of seq_len
    7. Split 70/15/15 train/val/test
    8. Save to output_dir as .npy files + scaler.pkl
    """
```

### UNSW-NB15 specific notes
- Target column: `label` (0=normal, 1=attack)
- Drop: `id`, `attack_cat`, `srcip`, `dstip` (identifiers, not features)
- Categorical cols: `proto`, `state`, `service` → LabelEncoder
- After dropping NaN, expect ~2.5M rows → sample 500k for faster training

### CIC-IoT2023 specific notes
- Multiple CSV files per attack type — concatenate all
- Remove rows where all values are 0 (dead sensors)
- Drop: `Timestamp`, `Label` before normalizing features

---

## Notebook guide

### 01_eda.ipynb
- Load dataset, show `.head()`, `.describe()`, `.info()`
- Plot feature distributions (histograms)
- Plot correlation heatmap
- Show class imbalance (normal vs anomaly ratio)
- Visualize a sample traffic sequence as a line chart

### 02_gan_training.ipynb
- Import `WGANGP` from `backend/models/wgan_gp`
- Load preprocessed `X_train.npy`
- Train for 100 epochs, log generator loss + critic loss
- Plot loss curves
- Generate 100 synthetic samples, compare statistics vs real
- Compute MMD between real and synthetic
- Save model weights

### 03_vae_anomaly.ipynb
- Import `VAE` from `backend/models/vae`
- Train on normal traffic only (`y_train == 0`)
- Plot reconstruction error distribution: normal vs anomaly
- Choose threshold at 95th percentile of normal reconstruction error
- Evaluate on test set: precision, recall, F1
- Plot ROC curve

### 04_rl_training.ipynb
- Import `IoTNetworkEnv` from `backend/environment/iot_network_env`
- Train PPO for 500,000 steps
- Plot episodic reward over training
- Evaluate: compare avg latency/throughput before vs after optimization
- Compare against DQN and random baselines

---

## Saving and loading models

Always use this pattern:

```python
# Saving
torch.save({
    'model_state_dict': model.state_dict(),
    'config': model.config,
    'epoch': epoch,
    'loss': loss,
}, 'backend/models/weights/vae.pt')

# Loading
checkpoint = torch.load('backend/models/weights/vae.pt', map_location=device)
model = VAE(config=checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## Baseline comparison (reproduce Table I from paper)

Implement these baselines in `backend/training/baselines.py`:

| Name | Description |
|------|-------------|
| `TraditionalOptimizer` | Genetic algorithm (use `scipy.optimize.differential_evolution`) |
| `DQNBaseline` | Use stable-baselines3 DQN with same env |
| `LSTMPredictor` | LSTM for traffic prediction + rule-based config selection |
| `GANOnly` | GAN for traffic generation + heuristic optimization |
| `VAERL` | VAE encoder as state + PPO (no GAN) |

Evaluate each on: latency_ms, throughput_mbps, energy_nj_per_bit, f1_score.
Report in a pandas DataFrame and export to `results/comparison_table.csv`.