# -*- coding: utf-8 -*-
"""
Step 2 v3 KAN standalone (memory-efficient)
Uses day-level batching to avoid large flattened arrays.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.surfaces.kan import KANSurface
from step1_features_lstm import VAE

STEP0_GRID = PROJECT_ROOT / "output" / "spx_step0_v3" / "daily_grid_154_fixed.parquet"
STEP1_VAE_NPZ = PROJECT_ROOT / "output" / "spx_step1_v3" / "vae_features.npz"
STEP1_VAE_PT = PROJECT_ROOT / "output" / "spx_step1_v3" / "vae_model.pt"
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "spx_options_2009_2020_clean.csv"
RATE_DATA = PROJECT_ROOT / "data" / "raw" / "rate_cleaned_2009_2020.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "spx_step2_v3_kan_fast"

DAYS_PER_YEAR = 365.0
N_GRID = 154
EPOCHS = 50
LR = 0.001
BATCH_DAYS = 4
LAMBDA = 1.0
TRAIN_RATIO = 0.75
VAL_RATIO = 0.15


def load_vae_features():
    data = np.load(STEP1_VAE_NPZ)
    Z = data["Z"].astype(np.float64)
    Z_pred = data["Z_pred"].astype(np.float64)
    dates = data["dates"].astype(int)
    dates_test = data["dates_test"].astype(int)

    vae = VAE(input_dim=N_GRID, hidden_dim=128, latent_dim=10)
    vae.load_state_dict(torch.load(STEP1_VAE_PT, map_location="cpu"))
    vae.eval()

    with torch.no_grad():
        F_all = vae.decode(torch.tensor(Z, dtype=torch.float32)).numpy()
        F_test = vae.decode(torch.tensor(Z_pred, dtype=torch.float32)).numpy()

    return {"dates": dates, "dates_test": dates_test, "F_all": F_all, "F_test": F_test}


def prepare_data(vae_info):
    grid_df = pd.read_parquet(STEP0_GRID)
    grid_dates = np.array(sorted(grid_df["trade_date"].unique()))
    n_days = len(grid_dates)
    n_train = int(n_days * TRAIN_RATIO)
    n_val = int(n_days * VAL_RATIO)

    train_dates = grid_dates[:n_train]
    val_dates = grid_dates[n_train:n_train + n_val]
    test_dates = grid_dates[n_train + n_val:]

    df_raw = pd.read_csv(RAW_DATA)
    df_raw["tau"] = df_raw["remaining_time"] / DAYS_PER_YEAR

    rate_df = pd.read_csv(RATE_DATA)
    rate_df["trade_date"] = rate_df["trade_date"].astype(int)
    df_raw["trade_date"] = df_raw["trade_date"].astype(int)
    df_raw = df_raw.merge(rate_df, on="trade_date", how="left")
    df_raw["F_price"] = df_raw["fund_close"] * np.exp(df_raw["r"] * df_raw["tau"])
    df_raw["m"] = np.log(df_raw["exercise_price"] / df_raw["F_price"])
    df_raw["sigma"] = df_raw["implc_volatlty"]

    df_obs = df_raw[df_raw["trade_date"].isin(grid_dates)].copy()
    df_obs = df_obs[
        (df_obs["sigma"] > 0) & (df_obs["sigma"] <= 2.0) &
        (df_obs["tau"] > 0) & (np.isfinite(df_obs["m"]))
    ].reset_index(drop=True)

    date_to_f = dict(zip(vae_info["dates"], vae_info["F_all"]))
    date_to_f_test = dict(zip(vae_info["dates_test"], vae_info["F_test"]))

    groups = dict(list(df_obs.groupby("trade_date")))

    def _build(dates, fmap):
        out = []
        for d in dates:
            if d not in fmap or d not in groups:
                continue
            g = groups[d]
            out.append({
                "date": int(d), "F": fmap[d].astype(np.float64),
                "m": g["m"].values.astype(np.float64),
                "tau": g["tau"].values.astype(np.float64),
                "sigma": g["sigma"].values.astype(np.float64),
            })
        return out

    return _build(train_dates, date_to_f), _build(val_dates, date_to_f), _build(test_dates, date_to_f_test)


def evaluate(model, data, device="cpu"):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for item in data:
            F_t = torch.tensor(item["F"], dtype=torch.float32, device=device).unsqueeze(0)
            m_t = torch.tensor(item["m"], dtype=torch.float32, device=device)
            tau_t = torch.tensor(item["tau"], dtype=torch.float32, device=device)
            sigma_t = item["sigma"]
            n_obs = len(sigma_t)
            pred = model(F_t.expand(n_obs, -1), m_t.unsqueeze(1), tau_t.unsqueeze(1)).squeeze().cpu().numpy()
            preds.extend(pred)
            truths.extend(sigma_t)
    p, t = np.array(preds), np.array(truths)
    return np.sqrt(mean_squared_error(t, p)), np.mean(np.abs(p - t) / np.maximum(t, 1e-6))


def train_day_level(model, train_data, val_data, test_data, device="cpu"):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    n_train = len(train_data)
    best_val_rmse = float("inf")
    best_state = None
    best_epoch = 0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        indices = torch.randperm(n_train)
        train_loss_sum = 0.0
        n_batches = 0

        for i in range(0, n_train, BATCH_DAYS):
            idx = indices[i:i + BATCH_DAYS].tolist()
            batch = [train_data[j] for j in idx]

            optimizer.zero_grad()
            batch_loss = 0.0

            for item in batch:
                F_t = torch.tensor(item["F"], dtype=torch.float32, device=device).unsqueeze(0)
                m_t = torch.tensor(item["m"], dtype=torch.float32, device=device)
                tau_t = torch.tensor(item["tau"], dtype=torch.float32, device=device)
                sigma_t = torch.tensor(item["sigma"], dtype=torch.float32, device=device)
                n_obs = len(sigma_t)
                pred = model(F_t.expand(n_obs, -1), m_t.unsqueeze(1), tau_t.unsqueeze(1)).squeeze()
                batch_loss += nn.functional.mse_loss(pred, sigma_t)

            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            optimizer.step()

            train_loss_sum += batch_loss.item()
            n_batches += 1

        train_mse = train_loss_sum / n_batches

        val_rmse, val_mape = evaluate(model, val_data, device)
        test_rmse, test_mape = evaluate(model, test_data, device)

        history.append({
            "epoch": epoch, "train_mse": train_mse,
            "val_rmse": val_rmse, "val_mape": val_mape,
            "test_rmse": test_rmse, "test_mape": test_mape,
        })

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"  Ep {epoch:3d}: train_mse={train_mse:.6f} | Val RMSE={val_rmse:.6f} | Test RMSE={test_rmse:.6f} MAPE={test_mape:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  [Best] epoch {best_epoch}, Val RMSE={best_val_rmse:.6f}")

    return model, history, best_epoch, best_val_rmse


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    print("[Load] VAE features v3...")
    vae_info = load_vae_features()

    print("[Data] Preparing...")
    train_data, val_data, test_data = prepare_data(vae_info)
    n_train = sum(len(d["sigma"]) for d in train_data)
    n_val = sum(len(d["sigma"]) for d in val_data)
    n_test = sum(len(d["sigma"]) for d in test_data)
    print(f"  Train: {len(train_data)}d/{n_train}obs | Val: {len(val_data)}d/{n_val}obs | Test: {len(test_data)}d/{n_test}obs")

    print(f"\n{'='*60}")
    print("Step 2 v3 KAN (Day-level batching)")
    print(f"  hidden=8, kan_hidden=4, layers=2, epochs={EPOCHS}, batch_days={BATCH_DAYS}")
    print(f"{'='*60}")

    model = KANSurface(input_dim=N_GRID + 2, hidden_dim=8, kan_hidden=4, num_layers=2, output_activation="softplus").to(device)
    model, history, best_epoch, best_val_rmse = train_day_level(model, train_data, val_data, test_data, device)

    test_rmse, test_mape = evaluate(model, test_data, device)
    print(f"\n[Final] Test RMSE={test_rmse:.6f}, Test MAPE={test_mape:.6f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_log.csv", index=False)
    torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump({
            "model": "KANSurface", "hidden_dim": 8, "kan_hidden": 4, "num_layers": 2,
            "epochs": EPOCHS, "batch_days": BATCH_DAYS, "lr": LR,
            "test_rmse": float(test_rmse), "test_mape": float(test_mape),
            "best_epoch": best_epoch, "best_val_rmse": float(best_val_rmse),
        }, f, indent=2)

    print(f"[Saved] {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
