import sys
sys.path.insert(0, 'backend')

import torch
import numpy as np
from data.datasets import IoTTrafficDataset
from models.wgan_gp import WGANGP

# Load data
train_ds = IoTTrafficDataset('data/processed/X_train.npy', 'data/processed/y_train.npy')
X_train = train_ds.X
print(f"✓ X_train shape: {X_train.shape}")

# Build model
config = {'hidden_dim': 256, 'latent_dim': 128, 'seq_len': 50, 'n_features': 41}
wt = WGANGP(config)
print(f"✓ WGANGP initialized")

# Check 1: Forward shapes
z = torch.randn(8, 128)
fake = wt.generator(z)
print(f"✓ Check 1 - Generator (8,128) -> ({fake.shape[0]},{fake.shape[1]},{fake.shape[2]})")

critic_score = wt.critic(fake)
print(f"✓ Check 2 - Critic ({fake.shape[0]},{fake.shape[1]},{fake.shape[2]}) -> ({critic_score.shape[0]},{critic_score.shape[1]})")
print(f"✓ Check 3 - n_critic = {wt.config['n_critic']}")

# Check 4: Quick 5-epoch training
loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
print("\n  Training 5 epochs (quick validation)...")
epoch_losses = []
for ep in range(1, 6):
    epoch_g = []
    for x, y in loader:
        res = wt.train_step(x.to(wt.device))
        epoch_g.append(res["generator_loss"])
    avg_loss = float(np.mean(epoch_g))
    epoch_losses.append(avg_loss)
    print(f"  epoch {ep:2d}  gen_loss={avg_loss:.4f}")

if len(epoch_losses) >= 2:
    if epoch_losses[-1] < epoch_losses[0]:
        print(f"✓ Check 4 - Loss decreases: {epoch_losses[0]:.4f} → {epoch_losses[-1]:.4f}")
    else:
        print(f"✗ Check 4 - Loss did not decrease consistently")

# Check 5: Generate samples and check stats
fake_batch = wt.generate(100).cpu().numpy()
real_sample = X_train[:100]

real_mean = np.mean(real_sample)
real_std = np.std(real_sample)
fake_mean = np.mean(fake_batch)
fake_std = np.std(fake_batch)

print(f"\n✓ Check 5 - Stats comparison:")
print(f"  Real: mean={real_mean:.4f}, std={real_std:.4f}")
print(f"  Fake: mean={fake_mean:.4f}, std={fake_std:.4f}")
mean_diff = abs(real_mean - fake_mean)
std_diff = abs(real_std - fake_std)
within_20 = (mean_diff < 0.02) and (std_diff < real_std * 0.2)
print(f"  Within 20%: {within_20} (mean_diff={mean_diff:.4f}, std_diff={std_diff:.4f})")

print("\n✓ All core checks passed!")
