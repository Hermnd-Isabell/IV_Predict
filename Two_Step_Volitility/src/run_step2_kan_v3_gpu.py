# -*- coding: utf-8 -*-
"""
Step 2 KAN v3 GPU 版 (Observation-level batching)
- 使用优化后的 groupby 数据准备
- KAN 参数: hidden_dim=32, kan_hidden=16, num_layers=2
- Epochs=50, batch_size=256 obs (避免 KAN OOM)
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step1_features_lstm import VAE
from step2_dnn_surface_v3 import (
    evaluate_fixed_grid, map_features_to_f,
    DAYS_PER_YEAR, BANNER_WIDTH,
)
from models.surfaces.kan import KANSurface

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STEP0_GRID = PROJECT_ROOT / "output" / "spx_step0_v3" / "daily_grid_154_fixed.parquet"
STEP1_SAM = PROJECT_ROOT / "output" / "spx_step1_v3" / "sam_features.npz"
STEP1_PCA = PROJECT_ROOT / "output" / "spx_step1_v3" / "pca_features.npz"
STEP1_VAE = PROJECT_ROOT / "output" / "spx_step1_v3" / "vae_features.npz"
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "spx_options_2009_2020_clean.csv"
RATE_DATA = PROJECT_ROOT / "data" / "raw" / "rate_cleaned_2009_2020.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "spx_step2_v3_kan_gpu"

TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
N_GRID = 154

EPOCHS = 50
LR = 0.001
BATCH_SIZE = 256
LAMBDA_PENALTY = 1.0
MAX_BATCHES_PER_EPOCH = 500
N_PENALTY_DAYS_PER_EPOCH = 4

# 稀疏网格 (用于 penalty)
SPARSE_M_N = 20
SPARSE_TAU_N = 10


def build_sparse_grids(device="cpu"):
    m_min, m_max = np.log(0.6), np.log(2.0)
    m_grid = np.linspace(-(2 * abs(m_min)) ** (1 / 3), (2 * m_max) ** (1 / 3), SPARSE_M_N)
    tau_grid = np.exp(np.linspace(np.log(1 / DAYS_PER_YEAR), np.log(730 / DAYS_PER_YEAR + 1), SPARSE_TAU_N))
    M, Tau = np.meshgrid(m_grid, tau_grid, indexing="ij")
    return (
        torch.tensor(M.ravel(), dtype=torch.float32, device=device),
        torch.tensor(Tau.ravel(), dtype=torch.float32, device=device),
    )


def prepare_data_fast(grid_path, raw_path, rate_path, step1_sam, step1_pca, step1_vae,
                      train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    print("[Data] 读取网格数据...")
    grid_df = pd.read_parquet(grid_path)
    grid_dates = np.array(sorted(grid_df["trade_date"].unique()))
    n_days = len(grid_dates)
    n_train = int(n_days * train_ratio)
    n_val = int(n_days * val_ratio)

    train_dates = grid_dates[:n_train]
    val_dates = grid_dates[n_train:n_train + n_val]
    test_dates = grid_dates[n_train + n_val:]
    print(f"  总交易日: {n_days}, Train: {len(train_dates)}, Val: {len(val_dates)}, Test: {len(test_dates)}")

    df_grid_pivot = grid_df.pivot(index="trade_date", columns="grid_idx", values="iv_dfw")
    F_real = df_grid_pivot.values.astype(np.float64)
    date_to_f_real = dict(zip(df_grid_pivot.index.values, F_real))

    print("[Data] 读取原始期权数据...")
    df_raw = pd.read_csv(raw_path)
    df_raw["tau"] = df_raw["remaining_time"] / DAYS_PER_YEAR

    print("[Data] 读取无风险利率并计算 v2 moneyness...")
    rate_df = pd.read_csv(rate_path)
    rate_df["trade_date"] = rate_df["trade_date"].astype(int)
    df_raw["trade_date"] = df_raw["trade_date"].astype(int)
    df_raw = df_raw.merge(rate_df, on="trade_date", how="left")
    df_raw["F"] = df_raw["fund_close"] * np.exp(df_raw["r"] * df_raw["tau"])
    df_raw["m"] = np.log(df_raw["exercise_price"] / df_raw["F"])
    df_raw["sigma"] = df_raw["implc_volatlty"]

    df_obs = df_raw[df_raw["trade_date"].isin(grid_dates)].copy()
    df_obs = df_obs[
        (df_obs["sigma"] > 0) & (df_obs["sigma"] <= 2.0) &
        (df_obs["tau"] > 0) & (np.isfinite(df_obs["m"]))
    ].reset_index(drop=True)
    print(f"  观测点总数: {len(df_obs)}")

    groups = dict(list(df_obs.groupby("trade_date")))

    def _build_fast(dates, fmap):
        out = []
        for d in dates:
            if d not in fmap or d not in groups:
                continue
            g = groups[d]
            out.append({
                "date": int(d), "F": fmap[d],
                "m": g["m"].values.astype(np.float64),
                "tau": g["tau"].values.astype(np.float64),
                "sigma": g["sigma"].values.astype(np.float64),
            })
        return out

    train_base = _build_fast(train_dates, date_to_f_real)
    val_base = _build_fast(val_dates, date_to_f_real)
    test_base = _build_fast(test_dates, date_to_f_real)

    sam_data = np.load(step1_sam)
    Z_sam = sam_data["Z"].astype(np.float64)
    Z_pred_sam = sam_data["Z_pred"].astype(np.float64)
    F_sam_all = map_features_to_f(Z_sam, "SAM")
    F_sam_test = map_features_to_f(Z_pred_sam, "SAM")
    date_to_f_sam = dict(zip(grid_dates, F_sam_all))
    date_to_f_sam_test = dict(zip(test_dates, F_sam_test))
    sam = {
        "train": _build_fast(train_dates, date_to_f_sam),
        "val": _build_fast(val_dates, date_to_f_sam),
        "test": _build_fast(test_dates, date_to_f_sam_test),
    }

    pca_data = np.load(step1_pca)
    Z_pca = pca_data["Z"].astype(np.float64)
    Z_pred_pca = pca_data["Z_pred"].astype(np.float64)
    eigensurfaces = pca_data["eigensurfaces"].astype(np.float64)
    F_pca_all = map_features_to_f(Z_pca, "PCA", grid_df=grid_df, eigensurfaces=eigensurfaces)
    F_pca_test = map_features_to_f(Z_pred_pca, "PCA", grid_df=grid_df, eigensurfaces=eigensurfaces)
    date_to_f_pca = dict(zip(grid_dates, F_pca_all))
    date_to_f_pca_test = dict(zip(test_dates, F_pca_test))
    pca = {
        "train": _build_fast(train_dates, date_to_f_pca),
        "val": _build_fast(val_dates, date_to_f_pca),
        "test": _build_fast(test_dates, date_to_f_pca_test),
    }

    vae_data = np.load(step1_vae)
    Z_vae = vae_data["Z"].astype(np.float64)
    Z_pred_vae = vae_data["Z_pred"].astype(np.float64)
    vae_model = VAE(input_dim=F_real.shape[1], hidden_dim=128, latent_dim=10)
    vae_model.load_state_dict(torch.load(step1_vae.with_suffix("").parent / "vae_model.pt", map_location="cpu", weights_only=False))
    vae_model.eval()
    F_vae_all = map_features_to_f(Z_vae, "VAE", vae_model=vae_model)
    F_vae_test = map_features_to_f(Z_pred_vae, "VAE", vae_model=vae_model)
    date_to_f_vae = dict(zip(grid_dates, F_vae_all))
    date_to_f_vae_test = dict(zip(test_dates, F_vae_test))
    vae = {
        "train": _build_fast(train_dates, date_to_f_vae),
        "val": _build_fast(val_dates, date_to_f_vae),
        "test": _build_fast(test_dates, date_to_f_vae_test),
    }

    return {"SAM": sam, "PCA": pca, "VAE": vae, "n_grid": F_real.shape[1]}


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


def compute_penalties_sparse(model, F_batch, m_grid_base, tau_grid_base, device):
    batch_days = F_batch.shape[0]
    n_sparse = len(m_grid_base)
    F_exp = F_batch.unsqueeze(1).expand(-1, n_sparse, -1).reshape(-1, F_batch.shape[1])
    m_exp = m_grid_base.repeat(batch_days).clone().requires_grad_(True)
    tau_exp = tau_grid_base.repeat(batch_days).clone().requires_grad_(True)

    sigma = model(F_exp, tau_exp.unsqueeze(1), m_exp.unsqueeze(1)).squeeze()
    sigma_safe = torch.clamp(sigma, min=1e-6)

    grad_tau = torch.autograd.grad(sigma.sum(), tau_exp, create_graph=True, retain_graph=True)[0]
    l_cal = sigma + 2 * tau_exp * grad_tau
    p_cal = torch.clamp(-l_cal, min=0).mean()

    grad_m = torch.autograd.grad(sigma.sum(), m_exp, create_graph=True, retain_graph=True)[0]
    grad_mm = torch.autograd.grad(grad_m.sum(), m_exp, create_graph=True, retain_graph=True)[0]
    grad_m_safe = grad_m / sigma_safe
    term1 = (1 - m_exp * grad_m_safe) ** 2
    term2 = (sigma_safe * tau_exp * grad_m_safe) ** 2 / 4
    term3 = tau_exp * sigma_safe * grad_mm
    l_but = term1 - term2 + term3
    p_but = torch.clamp(-l_but, min=0).mean()

    return p_cal, p_but


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
            pred = model(F_t.expand(n_obs, -1), m_t.unsqueeze(1), tau_t.unsqueeze(1)).squeeze().cpu().numpy()
            all_pred.extend(pred)
            all_true.extend(sigma_t)
    p, t = np.array(all_pred), np.array(all_true)
    return np.sqrt(mean_squared_error(t, p)), np.mean(np.abs(p - t) / np.maximum(t, 1e-6))


def train_kan_obs_level(model, train_data, val_data, test_data, m_grid, tau_grid, device="cpu"):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_flat = flatten_days(train_data)
    n_train_obs = len(train_flat["sigma"])

    best_val_rmse = float("inf")
    best_state = None
    best_epoch = 0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_train_obs)
        train_mse_sum = 0.0
        n_batches = 0

        for i in range(0, min(n_train_obs, MAX_BATCHES_PER_EPOCH * BATCH_SIZE), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
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
            if n_batches >= MAX_BATCHES_PER_EPOCH:
                break

        train_mse = train_mse_sum / n_batches

        # Penalty
        p_cal = p_but = 0.0
        if LAMBDA_PENALTY > 0:
            n_train_days = len(train_data)
            sample_idx = np.random.choice(n_train_days, size=min(N_PENALTY_DAYS_PER_EPOCH, n_train_days), replace=False)
            F_penalty = torch.tensor(np.stack([train_data[i]["F"] for i in sample_idx]), dtype=torch.float32, device=device)
            with torch.enable_grad():
                pc, pb = compute_penalties_sparse(model, F_penalty, m_grid, tau_grid, device)
            p_cal = pc.item()
            p_but = pb.item()

        val_rmse, val_mape = evaluate_model(model, val_data, device)
        test_rmse, test_mape = evaluate_model(model, test_data, device)

        pen_sum = p_cal + p_but
        total = train_mse + LAMBDA_PENALTY * pen_sum
        pen_ratio = LAMBDA_PENALTY * pen_sum / total if total > 0 else 0.0

        history.append({
            "epoch": epoch, "train_mse": train_mse,
            "l_c3": p_cal, "l_c4": p_but, "pen_ratio": pen_ratio,
            "val_rmse": val_rmse, "val_mape": val_mape,
            "test_rmse": test_rmse, "test_mape": test_mape,
        })

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        mod = 5 if epoch < 10 else 10
        if epoch % mod == 0 or epoch == EPOCHS - 1:
            print(f"  Ep {epoch:3d}: train_mse={train_mse:.6f} l_c3={p_cal:.6f} l_c4={p_but:.6f} "
                  f"pen_ratio={pen_ratio:.4f} | Val RMSE={val_rmse:.6f} | Test RMSE={test_rmse:.6f} MAPE={test_mape:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  [Best] epoch {best_epoch}, Val RMSE={best_val_rmse:.6f}")

    return model, history, best_epoch, best_val_rmse


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    data_dict = prepare_data_fast(
        STEP0_GRID, RAW_DATA, RATE_DATA,
        STEP1_SAM, STEP1_PCA, STEP1_VAE,
    )

    m_grid, tau_grid = build_sparse_grids(device)
    results_summary = {}

    for feature_type in ["SAM", "VAE"]:
        print(f"\n{'=' * BANNER_WIDTH}")
        print(f"Step 2 KAN GPU: {feature_type}")
        print(f"  hidden=16, kan_hidden=8, layers=2, epochs={EPOCHS}, batch={BATCH_SIZE}")
        print(f"{'=' * BANNER_WIDTH}")

        model = KANSurface(
            input_dim=N_GRID + 2, hidden_dim=16, kan_hidden=8, num_layers=2,
            output_activation="softplus",
        ).to(device)

        model, history, best_epoch, best_val_rmse = train_kan_obs_level(
            model, data_dict[feature_type]["train"], data_dict[feature_type]["val"],
            data_dict[feature_type]["test"], m_grid, tau_grid, device,
        )

        test_rmse, test_mape = evaluate_model(model, data_dict[feature_type]["test"], device)
        print(f"\n[Final {feature_type}] Test RMSE={test_rmse:.6f}, Test MAPE={test_mape:.6f}")

        ft_lower = feature_type.lower()
        torch.save(model.state_dict(), OUTPUT_DIR / f"kan_{ft_lower}.pt")
        np.savez(
            OUTPUT_DIR / f"results_{ft_lower}.npz",
            rmse=test_rmse, mape=test_mape,
            hist_train_loss=np.array([h["train_mse"] for h in history]),
            hist_val_loss=np.array([h["val_rmse"] for h in history]),
        )

        results_summary[feature_type] = {"rmse": test_rmse, "mape": test_mape}

    print(f"\n{'=' * BANNER_WIDTH}")
    print("KAN GPU Summary")
    print(f"{'=' * BANNER_WIDTH}")
    for ft, res in results_summary.items():
        print(f"  {ft}: RMSE={res['rmse']:.6f}, MAPE={res['mape']:.6f}")

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n[Done] Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
