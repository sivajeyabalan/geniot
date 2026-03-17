import sys
sys.path.insert(0, 'backend')

import torch
import numpy as np
from models.wgan_gp import WGANGP

# Build model
config = {'hidden_dim': 256, 'latent_dim': 128, 'seq_len': 50, 'n_features': 41}
wt = WGANGP(config)
print("✓ WGANGP initialized")

# Test 1: Forward pass on generator
print("\n=== Forward Pass Test ===")
batch_size = 4
z = torch.randn(batch_size, 128, device=wt.device)
print(f"Input noise shape: {z.shape}")

with torch.no_grad():
    fake = wt.generator(z)
    print(f"Generator output shape: {fake.shape}")

# Test 2: Critic forward
with torch.no_grad():
    score = wt.critic(fake)
    print(f"Critic output shape: {score.shape}")

# Test 3: Single training step on dummy data
print("\n=== Training Step Test ===")
dummy_real = torch.randn(4, 50, 41, device=wt.device)  # (batch, seq_len, n_features)
print(f"Real batch shape: {dummy_real.shape}")

try:
    res = wt.train_step(dummy_real)
    print(f"✓ Training step completed successfully")
    print(f"  Keys: {list(res.keys())}")
    for k, v in res.items():
        print(f"    {k}: {v:.6f}")
except Exception as e:
    print(f"✗ Training step failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ Minimal validation complete!")
