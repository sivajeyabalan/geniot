# Backend — ML Models Instructions

> Copilot context file for `backend/models/` directory.
> Read this alongside `.github/copilot-instructions.md`.

---

## wgan_gp.py — Traffic Generator

### Class structure
```python
class Generator(nn.Module):
    # Input:  z (batch, latent_dim=128)
    # Output: fake_traffic (batch, seq_len=50, n_features=41)

class Critic(nn.Module):
    # Input:  x (batch, seq_len=50, n_features=41)
    # Output: score scalar (batch, 1) — no sigmoid

class WGANGP:
    # Orchestrates training loop
    # Methods: train_step(), compute_gradient_penalty(), generate()
```

### Gradient penalty
```python
def compute_gradient_penalty(critic, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1, device=device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)
    d_interpolated = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated, inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True, retain_graph=True
    )[0]
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp
```

### Architecture guidance
- Generator: Linear → reshape → LSTM (2 layers, hidden=256) → Linear → Tanh
- Critic: Linear → LSTM (2 layers, hidden=256) → Linear (no activation)
- Use `nn.utils.spectral_norm` on Critic linear layers as additional stabilization
- Batch size: 64, latent_dim: 128, n_critic: 5, lambda_gp: 10

---

## vae.py — Anomaly Detector

### Class structure
```python
class VAEEncoder(nn.Module):
    # Input:  x (batch, seq_len, n_features)
    # Output: mu, log_var  both shape (batch, latent_dim=64)

class VAEDecoder(nn.Module):
    # Input:  z (batch, latent_dim=64)
    # Output: x_recon (batch, seq_len, n_features)

class VAE(nn.Module):
    # Methods: encode(), reparameterize(), decode(), forward()
    # forward() returns: x_recon, mu, log_var

def vae_loss(x_recon, x, mu, log_var, beta=1.0):
    # reconstruction_loss = F.mse_loss(x_recon, x, reduction='sum')
    # kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # return reconstruction_loss + beta * kl_loss

def anomaly_score(model, x, device):
    # Returns per-sample reconstruction error (MSE)
    # High score = anomaly
```

### Architecture guidance
- Encoder: Conv1d layers (temporal) → flatten → Linear → (mu, log_var)
- Decoder: Linear → reshape → ConvTranspose1d layers
- Alternative simpler: BiLSTM encoder, LSTM decoder
- latent_dim: 64, beta: 1.0 (increase to 4.0 for disentangled representation)

---

## diffusion.py — Temporal Sequence Model

### Class structure
```python
class DDPM:
    # T=1000 timesteps, beta schedule linear from 1e-4 to 0.02
    # Methods: forward_diffusion(), reverse_step(), sample()

class UNet1D(nn.Module):
    # Denoising network
    # Input: x_t (noisy), t (timestep embedding)
    # Output: predicted noise epsilon

def get_beta_schedule(T=1000):
    return torch.linspace(1e-4, 0.02, T)
```

### Note
The diffusion model is the **optional bonus module**. Implement GAN + VAE first.
Only add this if time permits — it significantly improves temporal dependency modeling.

---

## ppo_agent.py — RL Optimizer

### Usage pattern (stable-baselines3)
```python
from stable_baselines3 import PPO
from backend.environment.iot_network_env import IoTNetworkEnv

env = IoTNetworkEnv(config={...})
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./logs/"
)
model.learn(total_timesteps=500_000)
model.save("weights/ppo_iot")
```

### Wrapper class
```python
class IoTOptimizer:
    def __init__(self, weights_path: str): ...
    def recommend_config(self, network_state: np.ndarray) -> dict: ...
    # Returns: {"routing_table": ..., "sleep_schedule": ..., "bandwidth_alloc": ...}
```

---

## Shared conventions for all models

- All `__init__` methods accept a `config: dict` parameter
- Default configs defined as module-level `DEFAULT_CONFIG` dict
- `save_weights(path)` and `load_weights(path)` methods on every model class
- Training metrics logged as dict returned from `train_step()`
- All models must work with `batch_size=1` for inference (no batch norm issues)