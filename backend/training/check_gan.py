"""WGAN-GP success checklist."""
import sys, logging
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend.models.wgan_gp import WGANGP, DEFAULT_CONFIG
logging.basicConfig(level=logging.WARNING)

def mmd(x, y, bw=1.0):
    x = x.reshape(len(x), -1).astype("float32")
    y = y.reshape(len(y), -1).astype("float32")
    xx = np.mean(np.exp(-np.sum((x[:,None]-x[None,:])**2,axis=-1)/(2*bw**2)))
    yy = np.mean(np.exp(-np.sum((y[:,None]-y[None,:])**2,axis=-1)/(2*bw**2)))
    xy = np.mean(np.exp(-np.sum((x[:,None]-y[None,:])**2,axis=-1)/(2*bw**2)))
    return float(xx+yy-2*xy)

def check(label, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  |  {detail}" if detail else ""))

print("\n=== 1. Forward shapes ===")
w = WGANGP()
g_out = w.generator(torch.randn(8,DEFAULT_CONFIG["latent_dim"],device=w.device))
c_out = w.critic(torch.randn(8,50,41,device=w.device))
g_ok = tuple(g_out.shape)==(8,50,41)
d_ok = tuple(c_out.shape)==(8,1)
check("Generator (8,128)->(8,50,41)", g_ok, str(tuple(g_out.shape)))
check("Critic    (8,50,41)->(8,1)",   d_ok, str(tuple(c_out.shape)))

print("\n=== 2. Critic 5x ratio ===")
n_critic_cfg = int(WGANGP().config["n_critic"])
ratio_ok = n_critic_cfg == 5
check("n_critic == 5", ratio_ok, str(n_critic_cfg))
m_debug = WGANGP().train_step(torch.randn(32,50,41))
check("train_step keys ok", {"critic_loss","generator_loss"}.issubset(m_debug), str(list(m_debug)))

print("\n=== 3. Loss over 100 epochs ===")
processed_dir = Path("backend/data/processed")
X = np.load(processed_dir/"X_train.npy").astype("float32")
loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=64, shuffle=True, drop_last=True)
wt = WGANGP()
gl = []
for ep in range(1,101):
    eg = [wt.train_step(x.to(wt.device))["generator_loss"] for (x,) in loader]
    gl.append(float(np.mean(eg)))
    if ep % 20 == 0: print(f"  epoch {ep:3d}  gen_loss={gl[-1]:.4f}")
f10,l10 = float(np.mean(gl[:10])),float(np.mean(gl[-10:]))
loss_ok = l10 < f10
check("Generator loss decreases", loss_ok, f"first10={f10:.4f} last10={l10:.4f}")

print("\n=== 4. Stats ===")
n=min(100,len(X)); rs=X[:n]; fs=wt.generate(n).cpu().numpy()
rm,fm,rstd,fstd = rs.mean(),fs.mean(),rs.std(),fs.std()
merr=abs(fm-rm)/(abs(rm)+1e-8); sterr=abs(fstd-rstd)/(abs(rstd)+1e-8)
mean_ok=merr<=0.2; std_ok=sterr<=0.2
check("Mean within 20%", mean_ok, f"real={rm:.4f} fake={fm:.4f} err={merr*100:.1f}%")
check("Std  within 20%", std_ok,  f"real={rstd:.4f} fake={fstd:.4f} err={sterr*100:.1f}%")

print("\n=== 5. MMD ===")
mmd_val=mmd(rs,fs); mmd_ok=mmd_val<0.05
check("MMD < 0.05", mmd_ok, f"MMD={mmd_val:.6f}")

print("\n=== Summary ===")
results=[g_ok,d_ok,ratio_ok,loss_ok,mean_ok,std_ok,mmd_ok]
print(f"{sum(results)}/{len(results)} checks passed")

wd=Path("backend/models/weights"); wd.mkdir(parents=True,exist_ok=True)
wt.save_weights(wd/"gan.pt")
print(f"Weights saved -> {wd/'gan.pt'}")
