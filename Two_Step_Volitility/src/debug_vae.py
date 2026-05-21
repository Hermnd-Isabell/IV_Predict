# -*- coding: utf-8 -*-
"""Debug VAE step2 to find where it hangs"""
from pathlib import Path
import sys
import torch
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step2_dnn_surface import prepare_data, run_step2, DNN_Surface, day_loss

STEP0_GRID = Path("output/spx_step0/daily_grid_154.parquet")
STEP1_SAM = Path("output/spx_step1/sam_features.npz")
STEP1_PCA = Path("output/spx_step1/pca_features.npz")
STEP1_VAE = Path("output/spx_step1/vae_features.npz")
RAW_DATA = Path("data/raw/spx_options.csv")

print("[1] Loading data...")
t0 = time.time()
data_dict = prepare_data(STEP0_GRID, RAW_DATA, STEP1_SAM, STEP1_PCA, STEP1_VAE)
print(f"[1] Done in {time.time()-t0:.1f}s")

vae_data = data_dict["VAE"]
train_data = vae_data["train"]
val_data = vae_data["val"]
test_data = vae_data["test"]
n_grid = data_dict["n_grid"]

print(f"[2] Train days: {len(train_data)}, Val days: {len(val_data)}, Test days: {len(test_data)}")
print(f"[2] n_grid: {n_grid}")

# Test a single training step
print("[3] Testing single training step...")
model = DNN_Surface(input_dim=n_grid + 2, hidden_dim=50)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

item = train_data[0]
F_t = torch.tensor(item["F"], dtype=torch.float32).unsqueeze(0)
m_t = torch.tensor(item["m"], dtype=torch.float32)
tau_t = torch.tensor(item["tau"], dtype=torch.float32)
sigma_t = torch.tensor(item["sigma"], dtype=torch.float32)

print(f"[3] Day 0: F shape={F_t.shape}, n_obs={len(m_t)}")

t0 = time.time()
total, mse, p_cal, p_but, p_bound = day_loss(model, F_t, m_t, tau_t, sigma_t, lambda_penalty=1.0)
print(f"[3] day_loss computed in {time.time()-t0:.1f}s")
print(f"[3] total={total.item():.6f}, mse={mse.item():.6f}, cal={p_cal.item():.6f}, but={p_but.item():.6f}, bound={p_bound.item():.6f}")

t0 = time.time()
total.backward()
print(f"[3] backward in {time.time()-t0:.1f}s")

# Test a mini batch of 2 days
print("[4] Testing batch of 2 days...")
t0 = time.time()
optimizer.zero_grad()
for item in train_data[:2]:
    F_t = torch.tensor(item["F"], dtype=torch.float32).unsqueeze(0)
    m_t = torch.tensor(item["m"], dtype=torch.float32)
    tau_t = torch.tensor(item["tau"], dtype=torch.float32)
    sigma_t = torch.tensor(item["sigma"], dtype=torch.float32)
    total, _, _, _, _ = day_loss(model, F_t, m_t, tau_t, sigma_t, lambda_penalty=1.0)

total = total / 2
total.backward()
optimizer.step()
print(f"[4] Batch of 2 done in {time.time()-t0:.1f}s")

print("[5] All tests passed!")
