# -*- coding: utf-8 -*-
"""
Step 2 v3: DNN 无套利曲面重构 (修正版)
- 使用 Step 0 v2 网格 + Step 1 v2 特征
- moneyness 统一为 v2: F = S * exp(r*tau)
- 增加固定网格 Test RMSE 评估
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error

from models.surfaces.mlp import MLPSurface as DNN_Surface

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step1_features_lstm import VAE

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STEP0_GRID = PROJECT_ROOT / "output" / "spx_step0_v3" / "daily_grid_154_fixed.parquet"
STEP1_SAM = PROJECT_ROOT / "output" / "spx_step1_v3" / "sam_features.npz"
STEP1_PCA = PROJECT_ROOT / "output" / "spx_step1_v3" / "pca_features.npz"
STEP1_VAE = PROJECT_ROOT / "output" / "spx_step1_v3" / "vae_features.npz"
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "spx_options_2009_2020_clean.csv"
RATE_DATA = PROJECT_ROOT / "data" / "raw" / "rate_cleaned_2009_2020.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "spx_step2_v3"

BANNER_WIDTH = 50
DAYS_PER_YEAR = 365.0

DNN_HIDDEN = 50
DNN_EPOCHS = 20
DNN_BATCH_SIZE_DAYS = 32
DNN_LR = 0.001
DNN_LAMBDA = 1.0

GRID_C34_N = 40
GRID_C5_N = 4
TRAIN_RATIO = 0.75
VAL_RATIO = 0.15


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


def compute_arbitrage_penalties(model, F_batch, device="cpu"):
    batch_days = F_batch.shape[0]

    m_c34 = M_C34.to(device).clone().requires_grad_(True)
    tau_c34 = TAU_C34.to(device).clone().requires_grad_(True)

    F_expanded = F_batch.unsqueeze(1).expand(-1, N_C34, -1).reshape(-1, F_batch.shape[1])
    m_expanded = m_c34.repeat(batch_days)
    tau_expanded = tau_c34.repeat(batch_days)

    sigma = model(F_expanded, tau_expanded.unsqueeze(1), m_expanded.unsqueeze(1)).squeeze()
    sigma_safe = torch.clamp(sigma, min=1e-6)

    grad_m = torch.autograd.grad(sigma.sum(), m_expanded, create_graph=True, retain_graph=True)[0]
    grad_tau = torch.autograd.grad(sigma.sum(), tau_expanded, create_graph=True, retain_graph=True)[0]
    grad_mm = torch.autograd.grad(grad_m.sum(), m_expanded, create_graph=True)[0]

    l_cal = sigma + 2 * tau_expanded * grad_tau
    penalty_cal = torch.clamp(-l_cal, min=0).mean()

    grad_m_safe = grad_m / sigma_safe
    term1 = (1 - m_expanded * grad_m_safe) ** 2
    term2 = (sigma_safe * tau_expanded * grad_m_safe) ** 2 / 4
    term3 = tau_expanded * sigma_safe * grad_mm
    l_but = term1 - term2 + term3
    penalty_but = torch.clamp(-l_but, min=0).mean()

    m_c5 = M_C5.to(device).clone().requires_grad_(True)
    tau_c5 = TAU_C5.to(device).clone().requires_grad_(True)

    F_expanded5 = F_batch.unsqueeze(1).expand(-1, N_C5, -1).reshape(-1, F_batch.shape[1])
    m_expanded5 = m_c5.repeat(batch_days)
    tau_expanded5 = tau_c5.repeat(batch_days)

    sigma5 = model(F_expanded5, tau_expanded5.unsqueeze(1), m_expanded5.unsqueeze(1)).squeeze()
    grad_m5 = torch.autograd.grad(sigma5.sum(), m_expanded5, create_graph=True, retain_graph=True)[0]
    grad_mm5 = torch.autograd.grad(grad_m5.sum(), m_expanded5, create_graph=True)[0]

    boundary_val = torch.abs(sigma5 * grad_mm5 + grad_m5 ** 2)
    penalty_boundary = boundary_val.mean()

    return penalty_cal, penalty_but, penalty_boundary


def day_loss(model, F_day, m_obs, tau_obs, sigma_obs, lambda_penalty=DNN_LAMBDA, device="cpu"):
    n_obs = len(m_obs)
    F_exp = F_day.expand(n_obs, -1)

    sigma_pred = model(F_exp, tau_obs.unsqueeze(1), m_obs.unsqueeze(1)).squeeze()
    mse = nn.functional.mse_loss(sigma_pred, sigma_obs)

    if lambda_penalty > 0:
        p_cal, p_but, p_bound = compute_arbitrage_penalties(model, F_day, device)
        total = mse + lambda_penalty * (p_cal + p_but + p_bound)
    else:
        p_cal = p_but = p_bound = torch.tensor(0.0, device=device)
        total = mse

    return total, mse, p_cal, p_but, p_bound


def train_step2(train_data, val_data, n_grid, model_class=DNN_Surface,
                model_kwargs=None, output_activation="softplus",
                train_kwargs=None, device="cpu"):
    model_kwargs = model_kwargs or {}
    train_kwargs = train_kwargs or {
        "epochs": DNN_EPOCHS,
        "batch_size_days": DNN_BATCH_SIZE_DAYS,
        "lr": DNN_LR,
        "lambda_penalty": DNN_LAMBDA,
    }

    model = model_class(
        input_dim=n_grid + 2,
        output_activation=output_activation,
        **model_kwargs,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_kwargs["lr"])

    n_train = len(train_data)
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    history = {
        "train_loss": [], "val_loss": [], "mse": [],
        "pen_cal": [], "pen_but": [], "pen_bound": [],
    }

    epochs = train_kwargs["epochs"]
    batch_size_days = train_kwargs["batch_size_days"]
    lambda_penalty = train_kwargs["lambda_penalty"]

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(n_train)
        train_loss_epoch = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size_days):
            idx = indices[i : i + batch_size_days].tolist()
            batch = [train_data[j] for j in idx]

            optimizer.zero_grad()
            batch_total_loss = 0.0

            for item in batch:
                F_t = torch.tensor(item["F"], dtype=torch.float32, device=device).unsqueeze(0)
                m_t = torch.tensor(item["m"], dtype=torch.float32, device=device)
                tau_t = torch.tensor(item["tau"], dtype=torch.float32, device=device)
                sigma_t = torch.tensor(item["sigma"], dtype=torch.float32, device=device)

                total, _, _, _, _ = day_loss(model, F_t, m_t, tau_t, sigma_t, lambda_penalty, device)
                batch_total_loss += total

            batch_total_loss = batch_total_loss / len(batch)
            batch_total_loss.backward()
            optimizer.step()

            train_loss_epoch += batch_total_loss.item()
            n_batches += 1

        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for item in val_data:
                F_t = torch.tensor(item["F"], dtype=torch.float32, device=device).unsqueeze(0)
                m_t = torch.tensor(item["m"], dtype=torch.float32, device=device)
                tau_t = torch.tensor(item["tau"], dtype=torch.float32, device=device)
                sigma_t = torch.tensor(item["sigma"], dtype=torch.float32, device=device)
                n_obs = len(sigma_t)
                F_exp = F_t.expand(n_obs, -1)
                sigma_pred = model(F_exp, tau_t.unsqueeze(1), m_t.unsqueeze(1)).squeeze()
                val_mse += nn.functional.mse_loss(sigma_pred, sigma_t).item()

        val_pcal = val_pbut = val_pbound = 0.0
        if lambda_penalty > 0:
            for item in val_data:
                F_t = torch.tensor(item["F"], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.enable_grad():
                    p_cal, p_but, p_bound = compute_arbitrage_penalties(model, F_t, device)
                val_pcal += p_cal.item()
                val_pbut += p_but.item()
                val_pbound += p_bound.item()

        n_val = len(val_data)
        val_mse /= n_val
        if lambda_penalty > 0:
            val_pcal /= n_val
            val_pbut /= n_val
            val_pbound /= n_val
        val_total = val_mse + lambda_penalty * (val_pcal + val_pbut + val_pbound)

        history["train_loss"].append(train_loss_epoch / n_batches)
        history["val_loss"].append(val_total)
        history["mse"].append(val_mse)
        history["pen_cal"].append(val_pcal)
        history["pen_but"].append(val_pbut)
        history["pen_bound"].append(val_pbound)

        if val_total < best_val_loss:
            best_val_loss = val_total
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if epoch % 2 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:2d}: Train={train_loss_epoch/n_batches:.6f} | Val MSE={val_mse:.6f} Cal={val_pcal:.6f} But={val_pbut:.6f} Bound={val_pbound:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"  [DNN] 最优 epoch: {best_epoch}, best val loss: {best_val_loss:.6f}")
    return model, history


def evaluate_step2(model, test_data, device="cpu"):
    model.eval()
    rmse_list = []
    mape_list = []

    with torch.no_grad():
        for item in test_data:
            F_t = torch.tensor(item["F"], dtype=torch.float32, device=device).unsqueeze(0)
            m_t = torch.tensor(item["m"], dtype=torch.float32, device=device)
            tau_t = torch.tensor(item["tau"], dtype=torch.float32, device=device)
            sigma_t = item["sigma"]
            n_obs = len(sigma_t)

            F_exp = F_t.expand(n_obs, -1)
            sigma_pred = model(F_exp, tau_t.unsqueeze(1), m_t.unsqueeze(1)).squeeze().cpu().numpy()

            rmse = np.sqrt(mean_squared_error(sigma_t, sigma_pred))
            mape = np.mean(np.abs(sigma_pred - sigma_t) / np.maximum(sigma_t, 1e-6))
            rmse_list.append(rmse)
            mape_list.append(mape)

    return np.mean(rmse_list), np.mean(mape_list), rmse_list, mape_list


def evaluate_fixed_grid(model, test_data, grid_df, device="cpu"):
    """在固定 154 网格上评估，与 NW 插值作为 ground truth 对比。"""
    model.eval()
    errors = []

    grid_pivot = grid_df.pivot(index="trade_date", columns="grid_idx", values="iv_nw")
    date_to_nw = dict(zip(grid_pivot.index.values, grid_pivot.values.astype(np.float64)))

    m_grid = grid_df.sort_values("grid_idx")["m"].values[:154]
    tau_grid = grid_df.sort_values("grid_idx")["tau"].values[:154]

    with torch.no_grad():
        for item in test_data:
            d = item["date"]
            if d not in date_to_nw:
                continue
            F_t = torch.tensor(item["F"], dtype=torch.float32, device=device).unsqueeze(0)
            F_exp = F_t.expand(154, -1)
            m_t = torch.tensor(m_grid, dtype=torch.float32, device=device)
            tau_t = torch.tensor(tau_grid, dtype=torch.float32, device=device)
            sigma_pred = model(F_exp, tau_t.unsqueeze(1), m_t.unsqueeze(1)).squeeze().cpu().numpy()
            sigma_true = date_to_nw[d]
            errors.extend((sigma_pred - sigma_true) ** 2)

    return np.sqrt(np.mean(errors)) if errors else float("nan")


def check_arbitrage_violation(model, test_data, device="cpu"):
    model.eval()
    n_check = len(M_C34)
    violations_cal = []
    violations_but = []

    for item in test_data:
        m_check = M_C34.to(device).clone().requires_grad_(True)
        tau_check = TAU_C34.to(device).clone().requires_grad_(True)

        F_i = torch.tensor(item["F"], dtype=torch.float32, device=device).unsqueeze(0)
        F_exp = F_i.expand(n_check, -1)

        sigma = model(F_exp, tau_check.unsqueeze(1), m_check.unsqueeze(1)).squeeze()
        grad_m = torch.autograd.grad(sigma.sum(), m_check, create_graph=True, retain_graph=True)[0]
        grad_tau = torch.autograd.grad(sigma.sum(), tau_check, create_graph=True, retain_graph=True)[0]
        grad_mm = torch.autograd.grad(grad_m.sum(), m_check, create_graph=True)[0]

        sigma_safe = torch.clamp(sigma, min=1e-6)
        grad_m_safe = grad_m / sigma_safe

        l_cal = sigma + 2 * tau_check * grad_tau
        term1 = (1 - m_check * grad_m_safe) ** 2
        term2 = (sigma_safe * tau_check * grad_m_safe) ** 2 / 4
        term3 = tau_check * sigma_safe * grad_mm
        l_but = term1 - term2 + term3

        violations_cal.append(torch.clamp(-l_cal, min=0).sum().item())
        violations_but.append(torch.clamp(-l_but, min=0).sum().item())

    n_test = len(test_data)
    L_cal_neg = -np.sum(violations_cal) / (n_test * n_check)
    L_but_neg = -np.sum(violations_but) / (n_test * n_check)
    return L_cal_neg, L_but_neg


def map_features_to_f(Z, feature_type, grid_df=None, eigensurfaces=None, vae_model=None):
    if feature_type == "SAM":
        return Z
    elif feature_type == "PCA":
        if eigensurfaces is None:
            raise ValueError("PCA 需要 eigensurfaces")
        if grid_df is None:
            raise ValueError("PCA 需要 grid_df 以获取 sigma_0")
        first_date = grid_df["trade_date"].min()
        sigma_0 = grid_df[grid_df["trade_date"] == first_date].sort_values("grid_idx")["iv_dfw"].values.astype(np.float64)
        F = sigma_0 * np.exp(Z @ eigensurfaces)
        return F
    elif feature_type == "VAE":
        if vae_model is None:
            raise ValueError("VAE 需要 decoder 模型")
        vae_model.eval()
        with torch.no_grad():
            Z_t = torch.tensor(Z, dtype=torch.float32)
            F = vae_model.decode(Z_t).numpy()
        return F
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")


def prepare_data(grid_path, raw_path, rate_path, step1_sam, step1_pca, step1_vae,
                 train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    print("[Data] 读取网格数据...")
    grid_df = pd.read_parquet(grid_path)
    grid_dates = np.array(sorted(grid_df["trade_date"].unique()))
    n_days = len(grid_dates)
    n_train = int(n_days * train_ratio)
    n_val = int(n_days * val_ratio)

    train_dates = grid_dates[:n_train]
    val_dates = grid_dates[n_train : n_train + n_val]
    test_dates = grid_dates[n_train + n_val :]
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
        (df_obs["sigma"] > 0)
        & (df_obs["sigma"] <= 2.0)
        & (df_obs["tau"] > 0)
        & (np.isfinite(df_obs["m"]))
    ].reset_index(drop=True)
    print(f"  观测点总数: {len(df_obs)}")

    def _build_day_list(dates, date_to_f):
        out = []
        for d in dates:
            if d not in date_to_f:
                continue
            df_day = df_obs[df_obs["trade_date"] == d]
            if len(df_day) == 0:
                continue
            out.append({
                "date": int(d),
                "F": date_to_f[d],
                "m": df_day["m"].values.astype(np.float64),
                "tau": df_day["tau"].values.astype(np.float64),
                "sigma": df_day["sigma"].values.astype(np.float64),
            })
        return out

    train_base = _build_day_list(train_dates, date_to_f_real)
    val_base = _build_day_list(val_dates, date_to_f_real)
    test_base = _build_day_list(test_dates, date_to_f_real)

    sam_data = np.load(step1_sam)
    Z_sam = sam_data["Z"].astype(np.float64)
    Z_pred_sam = sam_data["Z_pred"].astype(np.float64)
    F_sam_all = map_features_to_f(Z_sam, "SAM")
    F_sam_test = map_features_to_f(Z_pred_sam, "SAM")
    date_to_f_sam = dict(zip(grid_dates, F_sam_all))
    date_to_f_sam_test = dict(zip(test_dates, F_sam_test))

    sam = {
        "train": _build_day_list(train_dates, date_to_f_sam),
        "val": _build_day_list(val_dates, date_to_f_sam),
        "test": _build_day_list(test_dates, date_to_f_sam_test),
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
        "train": _build_day_list(train_dates, date_to_f_pca),
        "val": _build_day_list(val_dates, date_to_f_pca),
        "test": _build_day_list(test_dates, date_to_f_pca_test),
    }

    print("[Data] 加载 VAE 模型...")
    vae_data = np.load(step1_vae)
    Z_vae = vae_data["Z"].astype(np.float64)
    Z_pred_vae = vae_data["Z_pred"].astype(np.float64)

    vae_model = VAE(input_dim=F_real.shape[1], hidden_dim=128, latent_dim=10)
    vae_model.load_state_dict(torch.load(step1_vae.with_suffix("").parent / "vae_model.pt", map_location="cpu"))
    vae_model.eval()

    F_vae_all = map_features_to_f(Z_vae, "VAE", vae_model=vae_model)
    F_vae_test = map_features_to_f(Z_pred_vae, "VAE", vae_model=vae_model)
    date_to_f_vae = dict(zip(grid_dates, F_vae_all))
    date_to_f_vae_test = dict(zip(test_dates, F_vae_test))

    vae = {
        "train": _build_day_list(train_dates, date_to_f_vae),
        "val": _build_day_list(val_dates, date_to_f_vae),
        "test": _build_day_list(test_dates, date_to_f_vae_test),
    }

    n_grid = F_real.shape[1]
    return {
        "SAM": sam,
        "PCA": pca,
        "VAE": vae,
        "n_grid": n_grid,
        "vae_model": vae_model,
    }, grid_df


def run_step2(feature_type, data_dict, grid_df, model_class=DNN_Surface,
              model_kwargs=None, output_activation="softplus",
              train_kwargs=None, device="cpu"):
    model_name = model_class.__name__
    print(f"\n{'=' * BANNER_WIDTH}")
    print(f"Step 2 v3: {model_name} ({feature_type})")
    print(f"{'=' * BANNER_WIDTH}")

    dataset = data_dict[feature_type]
    train_data = dataset["train"]
    val_data = dataset["val"]
    test_data = dataset["test"]
    n_grid = data_dict["n_grid"]

    n_train_obs = sum(len(d["sigma"]) for d in train_data)
    n_val_obs = sum(len(d["sigma"]) for d in val_data)
    n_test_obs = sum(len(d["sigma"]) for d in test_data)

    print(f"[Checkpoint 1] 特征映射")
    print(f"  {feature_type}: n_grid = {n_grid}")
    print(f"  训练观测点: {n_train_obs} (天数: {len(train_data)})")
    print(f"  验证观测点: {n_val_obs} (天数: {len(val_data)})")
    print(f"  测试观测点: {n_test_obs} (天数: {len(test_data)})")

    print(f"\n[Checkpoint 2] {model_name} 训练")
    model, history = train_step2(
        train_data, val_data, n_grid,
        model_class=model_class, model_kwargs=model_kwargs,
        output_activation=output_activation, train_kwargs=train_kwargs, device=device,
    )

    print(f"\n[Checkpoint 3] 测试评估")
    rmse, mape, rmse_daily, mape_daily = evaluate_step2(model, test_data, device)
    rmse_grid = evaluate_fixed_grid(model, test_data, grid_df, device)
    L_cal, L_but = check_arbitrage_violation(model, test_data, device)

    print(f"  Test RMSE (raw obs) = {rmse:.6f}")
    print(f"  Test RMSE (fixed grid) = {rmse_grid:.6f}")
    print(f"  Test MAPE  = {mape:.6f}")
    print(f"  L_cal = {L_cal:.8f}")
    print(f"  L_but = {L_but:.8f}")

    return model, history, rmse, mape, rmse_grid, L_cal, L_but


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    data_dict, grid_df = prepare_data(
        STEP0_GRID, RAW_DATA, RATE_DATA,
        STEP1_SAM, STEP1_PCA, STEP1_VAE,
    )

    results_summary = {}

    def _save_results(ft, model, hist, rmse, mape, rmse_grid, Lcal, Lbut):
        torch.save(model.state_dict(), OUTPUT_DIR / f"dnn_{ft.lower()}.pt")
        np.savez(
            OUTPUT_DIR / f"results_{ft.lower()}.npz",
            rmse=rmse, mape=mape, rmse_grid=rmse_grid,
            L_cal=Lcal, L_but=Lbut,
            hist_train_loss=np.array(hist["train_loss"]),
            hist_val_loss=np.array(hist["val_loss"]),
            hist_mse=np.array(hist["mse"]),
            hist_pen_cal=np.array(hist["pen_cal"]),
            hist_pen_but=np.array(hist["pen_but"]),
            hist_pen_bound=np.array(hist["pen_bound"]),
        )

    for ft in ["SAM", "PCA", "VAE"]:
        model, hist, rmse, mape, rmse_grid, Lcal, Lbut = run_step2(
            ft, data_dict, grid_df, device=device
        )
        _, _, rmse_d, mape_d = evaluate_step2(model, data_dict[ft]["test"], device)
        _save_results(ft, model, hist, rmse, mape, rmse_grid, Lcal, Lbut)
        results_summary[ft] = {
            "rmse": rmse, "mape": mape, "rmse_grid": rmse_grid,
            "L_cal": Lcal, "L_but": Lbut,
        }

    print(f"\n{'=' * BANNER_WIDTH}")
    print("[Checkpoint 4] 模型对比")
    print(f"{'=' * BANNER_WIDTH}")
    for ft, res in results_summary.items():
        print(f"  {ft:6s}: RMSE(raw)={res['rmse']:.6f}, RMSE(grid)={res['rmse_grid']:.6f}, "
              f"MAPE={res['mape']:.6f}, L_cal={res['L_cal']:.8f}, L_but={res['L_but']:.8f}")

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n[Done] Step 2 v3 完成，输出保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
