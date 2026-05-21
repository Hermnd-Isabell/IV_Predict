# -*- coding: utf-8 -*-
"""
Step 2 v3 双实验统一脚本
实验1: DNN 200 epochs
实验2: KAN 50 epochs
数据准备只执行一次，依次运行两个实验。
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

STEP0_GRID = PROJECT_ROOT / "output" / "spx_step0_v3" / "daily_grid_154_fixed.parquet"
STEP1_VAE_NPZ = PROJECT_ROOT / "output" / "spx_step1_v3" / "vae_features.npz"
STEP1_VAE_PT = PROJECT_ROOT / "output" / "spx_step1_v3" / "vae_model.pt"
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "spx_options_2009_2020_clean.csv"
RATE_DATA = PROJECT_ROOT / "data" / "raw" / "rate_cleaned_2009_2020.csv"

BANNER_WIDTH = 60
DAYS_PER_YEAR = 365.0
N_GRID = 154

# 共享训练配置
LR = 0.001
LAMBDA_PENALTY = 1.0
TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
N_PENALTY_DAYS_PER_EPOCH = 16

# DNN 配置
DNN_EPOCHS = 200
DNN_BATCH_SIZE = 1024
DNN_MAX_BATCHES = 500

# KAN 配置
KAN_EPOCHS = 50
KAN_BATCH_SIZE = 128
KAN_MAX_BATCHES = 200

# 无套利网格
GRID_C34_N = 40
GRID_C5_N = 4


def _build_arbitrage_grids():
    m_min, m_max = np.log(0.6), np.log(2.0)
    m_grid_c34 = np.linspace(-(2 * abs(m_min)) ** (1 / 3), (2 * m_max) ** (1 / 3), GRID_C34_N)
    tau_grid_c34 = np.exp(np.linspace(np.log(1 / DAYS_PER_YEAR), np.log(730 / DAYS_PER_YEAR + 1), GRID_C34_N))
    M_c34, Tau_c34 = np.meshgrid(m_grid_c34, tau_grid_c34, indexing="ij")

    m_grid_c5 = np.array([6 * m_min, 4 * m_min, 4 * m_max, 6 * m_max])
    tau_grid_c5 = tau_grid_c34.copy()
    M_c5, Tau_c5 = np.meshgrid(m_grid_c5, tau_grid_c5, indexing="ij")

    return (
        torch.tensor(M_c34.ravel(), dtype=torch.float32),
        torch.tensor(Tau_c34.ravel(), dtype=torch.float32),
        torch.tensor(M_c5.ravel(), dtype=torch.float32),
        torch.tensor(Tau_c5.ravel(), dtype=torch.float32),
    )


M_C34, TAU_C34, M_C5, TAU_C5 = _build_arbitrage_grids()
N_C34 = len(M_C34)
N_C5 = len(M_C5)


class MLPSurface(nn.Module):
    def __init__(self, input_dim, hidden_dim=50):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.hact = nn.Tanh()
        self.oact = nn.Softplus()

    def forward(self, F, tau, m):
        x = torch.cat([F, tau, m], dim=-1)
        x = self.hact(self.fc1(x))
        x = self.hact(self.fc2(x))
        x = self.hact(self.fc3(x))
        return self.oact(self.fc_out(x))


def compute_penalties(model, F_batch, device="cpu"):
    batch_days = F_batch.shape[0]
    m_c34 = M_C34.to(device).clone().requires_grad_(True)
    tau_c34 = TAU_C34.to(device).clone().requires_grad_(True)

    F_exp = F_batch.unsqueeze(1).expand(-1, N_C34, -1).reshape(-1, F_batch.shape[1])
    m_exp = m_c34.repeat(batch_days)
    tau_exp = tau_c34.repeat(batch_days)

    sigma = model(F_exp, tau_exp.unsqueeze(1), m_exp.unsqueeze(1)).squeeze()
    sigma_safe = torch.clamp(sigma, min=1e-6)

    grad_m = torch.autograd.grad(sigma.sum(), m_exp, create_graph=True, retain_graph=True)[0]
    grad_tau = torch.autograd.grad(sigma.sum(), tau_exp, create_graph=True, retain_graph=True)[0]
    grad_mm = torch.autograd.grad(grad_m.sum(), m_exp, create_graph=True, retain_graph=True)[0]

    l_cal = sigma + 2 * tau_exp * grad_tau
    p_cal = torch.clamp(-l_cal, min=0).mean()

    grad_m_safe = grad_m / sigma_safe
    term1 = (1 - m_exp * grad_m_safe) ** 2
    term2 = (sigma_safe * tau_exp * grad_m_safe) ** 2 / 4
    term3 = tau_exp * sigma_safe * grad_mm
    l_but = term1 - term2 + term3
    p_but = torch.clamp(-l_but, min=0).mean()

    m_c5 = M_C5.to(device).clone().requires_grad_(True)
    tau_c5 = TAU_C5.to(device).clone().requires_grad_(True)
    F_exp5 = F_batch.unsqueeze(1).expand(-1, N_C5, -1).reshape(-1, F_batch.shape[1])
    m_exp5 = m_c5.repeat(batch_days)
    tau_exp5 = tau_c5.repeat(batch_days)

    sigma5 = model(F_exp5, tau_exp5.unsqueeze(1), m_exp5.unsqueeze(1)).squeeze()
    grad_m5 = torch.autograd.grad(sigma5.sum(), m_exp5, create_graph=True, retain_graph=True)[0]
    grad_mm5 = torch.autograd.grad(grad_m5.sum(), m_exp5, create_graph=True, retain_graph=True)[0]
    p_bound = torch.abs(sigma5 * grad_mm5 + grad_m5 ** 2).mean()

    return p_cal, p_but, p_bound


def load_vae_features():
    from step1_features_lstm import VAE
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

    # 使用 groupby 避免 O(n*days) 的循环过滤
    date_to_f = dict(zip(vae_info["dates"], vae_info["F_all"]))
    date_to_f_test = dict(zip(vae_info["dates_test"], vae_info["F_test"]))

    groups = dict(list(df_obs.groupby("trade_date")))

    def _build_fast(dates, fmap):
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

    return _build_fast(train_dates, date_to_f), _build_fast(val_dates, date_to_f), _build_fast(test_dates, date_to_f_test)


def flatten_days(day_list):
    all_F, all_m, all_tau, all_sigma = [], [], [], []
    for item in day_list:
        n = len(item["sigma"])
        all_F.append(np.tile(item["F"], (n, 1)))
        all_m.append(item["m"])
        all_tau.append(item["tau"])
        all_sigma.append(item["sigma"])
    return {
        "F": np.vstack(all_F), "m": np.concatenate(all_m),
        "tau": np.concatenate(all_tau), "sigma": np.concatenate(all_sigma),
    }


def evaluate_model(model, data, device="cpu"):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for item in data:
            F_t = torch.tensor(item["F"], dtype=torch.float32, device=device).unsqueeze(0)
            m_t = torch.tensor(item["m"], dtype=torch.float32, device=device)
            tau_t = torch.tensor(item["tau"], dtype=torch.float32, device=device)
            sigma_t = item["sigma"]
            n_obs = len(sigma_t)
            F_exp = F_t.expand(n_obs, -1)
            pred = model(F_exp, tau_t.unsqueeze(1), m_t.unsqueeze(1)).squeeze().cpu().numpy()
            all_pred.extend(pred)
            all_true.extend(sigma_t)
    pred = np.array(all_pred)
    true = np.array(all_true)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mape = np.mean(np.abs(pred - true) / np.maximum(true, 1e-6))
    return rmse, mape


def train_experiment(model, train_data, val_data, test_data, epochs, batch_size, max_batches, device="cpu"):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_flat = flatten_days(train_data)
    n_train_obs = len(train_flat["sigma"])

    best_val_rmse = float("inf")
    best_state = None
    best_epoch = 0
    history = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train_obs)
        train_mse_sum = 0.0
        n_batches = 0

        for i in range(0, min(n_train_obs, max_batches * batch_size), batch_size):
            idx = perm[i:i + batch_size]
            F_b = torch.tensor(train_flat["F"][idx], dtype=torch.float32, device=device)
            m_b = torch.tensor(train_flat["m"][idx], dtype=torch.float32, device=device)
            tau_b = torch.tensor(train_flat["tau"][idx], dtype=torch.float32, device=device)
            sigma_b = torch.tensor(train_flat["sigma"][idx], dtype=torch.float32, device=device)

            pred = model(F_b, m_b.unsqueeze(1), tau_b.unsqueeze(1)).squeeze()
            mse_loss = nn.functional.mse_loss(pred, sigma_b)

            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()

            train_mse_sum += mse_loss.item()
            n_batches += 1
            if n_batches >= max_batches:
                break

        train_mse = train_mse_sum / n_batches

        # Penalty
        p_cal = p_but = p_bound = 0.0
        if LAMBDA_PENALTY > 0:
            n_train_days = len(train_data)
            sample_idx = np.random.choice(n_train_days, size=min(N_PENALTY_DAYS_PER_EPOCH, n_train_days), replace=False)
            F_penalty = torch.tensor(np.stack([train_data[i]["F"] for i in sample_idx]), dtype=torch.float32, device=device)
            with torch.enable_grad():
                p_cal, p_but, p_bound = compute_penalties(model, F_penalty, device)
            p_cal = p_cal.item()
            p_but = p_but.item()
            p_bound = p_bound.item()

        val_rmse, val_mape = evaluate_model(model, val_data, device)
        test_rmse, test_mape = evaluate_model(model, test_data, device)

        pen_sum = p_cal + p_but + p_bound
        total = train_mse + LAMBDA_PENALTY * pen_sum
        pen_ratio = LAMBDA_PENALTY * pen_sum / total if total > 0 else 0.0

        history.append({
            "epoch": epoch, "train_mse": train_mse,
            "l_c3": p_cal, "l_c4": p_but, "l_c5": p_bound, "pen_ratio": pen_ratio,
            "val_rmse": val_rmse, "val_mape": val_mape,
            "test_rmse": test_rmse, "test_mape": test_mape,
        })

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        mod = 20 if epochs >= 100 else 5
        if epoch % mod == 0 or epoch == epochs - 1:
            print(f"  Ep {epoch:3d}: train_mse={train_mse:.6f} pen_ratio={pen_ratio:.4f} | "
                  f"Val RMSE={val_rmse:.6f} | Test RMSE={test_rmse:.6f} MAPE={test_mape:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  [Best] epoch {best_epoch}, Val RMSE={best_val_rmse:.6f}")

    return model, history, best_epoch, best_val_rmse


def run_experiment(name, model, train_data, val_data, test_data, epochs, batch_size, max_batches, output_dir, device="cpu"):
    print(f"\n{'=' * BANNER_WIDTH}")
    print(f"Experiment: {name}")
    print(f"  Epochs={epochs}, Batch={batch_size}, MaxBatches={max_batches}")
    print(f"{'=' * BANNER_WIDTH}")

    model, history, best_epoch, best_val_rmse = train_experiment(
        model, train_data, val_data, test_data, epochs, batch_size, max_batches, device
    )

    test_rmse, test_mape = evaluate_model(model, test_data, device)
    print(f"\n[Final {name}] Test RMSE={test_rmse:.6f}, Test MAPE={test_mape:.6f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_dir / "training_log.csv", index=False)
    torch.save(model.state_dict(), output_dir / "model.pt")

    with open(output_dir / "summary.json", "w") as f:
        json.dump({
            "name": name, "epochs": epochs, "batch_size": batch_size, "max_batches": max_batches,
            "test_rmse": float(test_rmse), "test_mape": float(test_mape),
            "best_epoch": best_epoch, "best_val_rmse": float(best_val_rmse),
        }, f, indent=2)

    print(f"[Saved] {output_dir}")
    return test_rmse, test_mape


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

    # Experiment 1: DNN 200 epochs
    dnn_model = MLPSurface(input_dim=N_GRID + 2, hidden_dim=50).to(device)
    dnn_rmse, dnn_mape = run_experiment(
        "DNN 200 epochs", dnn_model, train_data, val_data, test_data,
        DNN_EPOCHS, DNN_BATCH_SIZE, DNN_MAX_BATCHES,
        PROJECT_ROOT / "output" / "spx_step2_v3_dnn200_fast", device
    )

    # Free memory before KAN
    import gc
    del dnn_model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Experiment 2: KAN 50 epochs
    kan_model = KANSurface(
        input_dim=N_GRID + 2, hidden_dim=8, kan_hidden=4, num_layers=2,
        output_activation="softplus",
    ).to(device)
    kan_rmse, kan_mape = run_experiment(
        "KAN 50 epochs", kan_model, train_data, val_data, test_data,
        KAN_EPOCHS, KAN_BATCH_SIZE, KAN_MAX_BATCHES,
        PROJECT_ROOT / "output" / "spx_step2_v3_kan_fast", device
    )

    print(f"\n{'=' * BANNER_WIDTH}")
    print("Summary")
    print(f"{'=' * BANNER_WIDTH}")
    print(f"  DNN 200ep: RMSE={dnn_rmse:.6f}, MAPE={dnn_mape:.6f}")
    print(f"  KAN  50ep: RMSE={kan_rmse:.6f}, MAPE={kan_mape:.6f}")
    print(f"{'=' * BANNER_WIDTH}")


if __name__ == "__main__":
    main()
