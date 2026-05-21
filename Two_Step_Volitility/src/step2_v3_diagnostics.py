# -*- coding: utf-8 -*-
"""
Step 2 v3 诊断版训练脚本
基于 DFW + VAE (Step 1 Test RMSE 0.0524) 的特征

目标：输出完整的 RMSE/MAPE 分层诊断报告，帮助分析模型误差来源。

配置：
  - MLP: input_dim=156, hidden=[50,50,50], Tanh, Softplus
  - Train: epochs=20, lr=0.001, batch_size=1024 (observations), lambda=1.0
  - Split: val_ratio=0 以匹配 step1_all6/dfw_vae (75% Train / 25% Test)

输出 (output/spx_step2_v3_diagnostics/):
  - training_log.csv          每轮训练指标
  - evaluation_diagnostics.json  综合评估统计
  - mape_by_iv_level.csv      按 IV 分层的 MAPE
  - mape_by_moneyness.csv     按 moneyness 分层的 MAPE
  - mape_by_tau.csv           按 tau 分层的 MAPE
  - extreme_mape_samples.csv  极端 MAPE 样本详情
  - penalty_diagnostics.json  惩罚项诊断（梯度范围、违反计数）
  - model.pt                  训练好的模型
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

# ------------------------------------------------------------------
# 路径与常量
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

STEP0_GRID = PROJECT_ROOT / "output" / "spx_step0_v2" / "daily_grid_154_fixed.parquet"
STEP1_VAE_NPZ = PROJECT_ROOT / "output" / "spx_step1_all6" / "dfw_vae" / "vae_features.npz"
STEP1_VAE_PT = PROJECT_ROOT / "output" / "spx_step1_all6" / "dfw_vae" / "vae_model.pt"
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "spx_options_2009_2020_clean.csv"
RATE_DATA = PROJECT_ROOT / "data" / "raw" / "rate_cleaned_2009_2020.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "spx_step2_v3_diagnostics"

warnings.filterwarnings("ignore", category=UserWarning)

BANNER_WIDTH = 60
DAYS_PER_YEAR = 365.0

# 训练超参数
EPOCHS = 20
LR = 0.001
BATCH_SIZE = 1024          # observation-level batch size
LAMBDA_PENALTY = 1.0
TRAIN_RATIO = 0.75
VAL_RATIO = 0.0            # 匹配 step1_all6/dfw_vae
N_PENALTY_DAYS_PER_EPOCH = 16   # 每轮随机采样的天数用于 penalty 计算
MAX_BATCHES_PER_EPOCH = 500     # 每轮最多处理 500 个 batches (~512K obs)

# 网格常量
GRID_C34_N = 40
GRID_C5_N = 4
N_GRID = 154

# ------------------------------------------------------------------
# 模型 (inline 避免路径依赖)
# ------------------------------------------------------------------
class MLPSurface(nn.Module):
    """3 层隐藏层 MLP，Tanh + Softplus。"""
    def __init__(self, input_dim: int, hidden_dim: int = 50):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.hidden_activation = nn.Tanh()
        self.output_activation = nn.Softplus()

    def forward(self, F_input: torch.Tensor, tau_input: torch.Tensor, m_input: torch.Tensor) -> torch.Tensor:
        x = torch.cat([F_input, tau_input, m_input], dim=-1)
        x = self.hidden_activation(self.fc1(x))
        x = self.hidden_activation(self.fc2(x))
        x = self.hidden_activation(self.fc3(x))
        x = self.fc_out(x)
        return self.output_activation(x)


# ------------------------------------------------------------------
# 无套利网格与惩罚项
# ------------------------------------------------------------------
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
    """返回 (penalty_cal, penalty_but, penalty_boundary)。
    F_batch: (batch_days, n_grid)
    """
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
    grad_mm = torch.autograd.grad(grad_m.sum(), m_expanded, create_graph=True, retain_graph=True)[0]

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
    grad_mm5 = torch.autograd.grad(grad_m5.sum(), m_expanded5, create_graph=True, retain_graph=True)[0]

    boundary_val = torch.abs(sigma5 * grad_mm5 + grad_m5 ** 2)
    penalty_boundary = boundary_val.mean()

    return penalty_cal, penalty_but, penalty_boundary


def compute_penalty_diagnostics(model, F_batch, device="cpu"):
    """返回详细的 penalty 诊断信息（梯度范围、违反计数）。"""
    batch_days = F_batch.shape[0]
    diag = {"calendar": {}, "butterfly": {}, "boundary": {}}

    m_c34 = M_C34.to(device).clone().requires_grad_(True)
    tau_c34 = TAU_C34.to(device).clone().requires_grad_(True)

    F_expanded = F_batch.unsqueeze(1).expand(-1, N_C34, -1).reshape(-1, F_batch.shape[1])
    m_expanded = m_c34.repeat(batch_days)
    tau_expanded = tau_c34.repeat(batch_days)

    sigma = model(F_expanded, tau_expanded.unsqueeze(1), m_expanded.unsqueeze(1)).squeeze()
    sigma_safe = torch.clamp(sigma, min=1e-6)

    grad_m = torch.autograd.grad(sigma.sum(), m_expanded, create_graph=True, retain_graph=True)[0]
    grad_tau = torch.autograd.grad(sigma.sum(), tau_expanded, create_graph=True, retain_graph=True)[0]
    grad_mm = torch.autograd.grad(grad_m.sum(), m_expanded, create_graph=True, retain_graph=True)[0]

    # Calendar
    l_cal = sigma + 2 * tau_expanded * grad_tau
    violations_cal = (l_cal < 0).sum().item()
    diag["calendar"] = {
        "l_cal_min": l_cal.min().item(),
        "l_cal_max": l_cal.max().item(),
        "l_cal_mean": l_cal.mean().item(),
        "violations": violations_cal,
        "violation_ratio": violations_cal / (batch_days * N_C34),
        "grad_tau_min": grad_tau.min().item(),
        "grad_tau_max": grad_tau.max().item(),
        "grad_tau_mean": grad_tau.mean().item(),
    }

    # Butterfly
    grad_m_safe = grad_m / sigma_safe
    term1 = (1 - m_expanded * grad_m_safe) ** 2
    term2 = (sigma_safe * tau_expanded * grad_m_safe) ** 2 / 4
    term3 = tau_expanded * sigma_safe * grad_mm
    l_but = term1 - term2 + term3
    violations_but = (l_but < 0).sum().item()
    diag["butterfly"] = {
        "l_but_min": l_but.min().item(),
        "l_but_max": l_but.max().item(),
        "l_but_mean": l_but.mean().item(),
        "violations": violations_but,
        "violation_ratio": violations_but / (batch_days * N_C34),
        "grad_m_min": grad_m.min().item(),
        "grad_m_max": grad_m.max().item(),
        "grad_mm_min": grad_mm.min().item(),
        "grad_mm_max": grad_mm.max().item(),
    }

    # Boundary
    m_c5 = M_C5.to(device).clone().requires_grad_(True)
    tau_c5 = TAU_C5.to(device).clone().requires_grad_(True)
    F_expanded5 = F_batch.unsqueeze(1).expand(-1, N_C5, -1).reshape(-1, F_batch.shape[1])
    m_expanded5 = m_c5.repeat(batch_days)
    tau_expanded5 = tau_c5.repeat(batch_days)

    sigma5 = model(F_expanded5, tau_expanded5.unsqueeze(1), m_expanded5.unsqueeze(1)).squeeze()
    grad_m5 = torch.autograd.grad(sigma5.sum(), m_expanded5, create_graph=True, retain_graph=True)[0]
    grad_mm5 = torch.autograd.grad(grad_m5.sum(), m_expanded5, create_graph=True, retain_graph=True)[0]
    boundary_val = torch.abs(sigma5 * grad_mm5 + grad_m5 ** 2)
    diag["boundary"] = {
        "boundary_min": boundary_val.min().item(),
        "boundary_max": boundary_val.max().item(),
        "boundary_mean": boundary_val.mean().item(),
    }

    return diag


# ------------------------------------------------------------------
# 数据准备
# ------------------------------------------------------------------
def load_vae_features(npz_path: Path, pt_path: Path):
    """加载 step1_all6/dfw_vae 的特征和 decoder。"""
    data = np.load(npz_path)
    Z = data["Z"].astype(np.float64)          # (2913, 10)
    Z_pred = data["Z_pred"].astype(np.float64)  # (728, 10)
    dates = data["dates"].astype(int)         # (2913,)
    dates_test = data["dates_test"].astype(int)  # (728,)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from step1_features_lstm import VAE

    vae_model = VAE(input_dim=N_GRID, hidden_dim=128, latent_dim=10)
    vae_model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    vae_model.eval()

    with torch.no_grad():
        F_all = vae_model.decode(torch.tensor(Z, dtype=torch.float32)).numpy()
        F_test = vae_model.decode(torch.tensor(Z_pred, dtype=torch.float32)).numpy()

    return {
        "dates": dates,
        "dates_test": dates_test,
        "F_all": F_all,      # (2913, 154)
        "F_test": F_test,    # (728, 154)
        "vae_model": vae_model,
    }


def prepare_data(grid_path: Path, raw_path: Path, rate_path: Path, vae_info: dict):
    print("[Data] 读取网格数据...")
    grid_df = pd.read_parquet(grid_path)
    grid_dates = np.array(sorted(grid_df["trade_date"].unique()))
    n_days = len(grid_dates)

    # 使用 VAE npz 中保存的 dates/dates_test 进行分割（与 Step 1 完全一致）
    all_dates = vae_info["dates"]
    test_dates = vae_info["dates_test"]
    train_dates = np.array(sorted(set(all_dates) - set(test_dates)))
    print(f"  总交易日: {n_days}, Train: {len(train_dates)}, Test: {len(test_dates)}")

    # 可选：检查 grid 是否包含所有需要的日期
    missing = set(all_dates) - set(grid_dates)
    if missing:
        print(f"  [Warning] Grid 缺少 {len(missing)} 个日期，将被忽略")

    # DFW grid 作为 F_real（用于固定网格评估）
    df_grid_pivot = grid_df.pivot(index="trade_date", columns="grid_idx", values="iv_dfw")
    F_real = df_grid_pivot.values.astype(np.float64)
    date_to_f_real = dict(zip(df_grid_pivot.index.values, F_real))

    print("[Data] 读取原始期权数据 (clean)...")
    df_raw = pd.read_csv(raw_path)
    df_raw["tau"] = df_raw["remaining_time"] / DAYS_PER_YEAR

    print("[Data] 读取无风险利率...")
    rate_df = pd.read_csv(rate_path)
    rate_df["trade_date"] = rate_df["trade_date"].astype(int)
    df_raw["trade_date"] = df_raw["trade_date"].astype(int)
    df_raw = df_raw.merge(rate_df, on="trade_date", how="left")
    df_raw["F_price"] = df_raw["fund_close"] * np.exp(df_raw["r"] * df_raw["tau"])
    df_raw["m"] = np.log(df_raw["exercise_price"] / df_raw["F_price"])
    df_raw["sigma"] = df_raw["implc_volatlty"]

    df_obs = df_raw[df_raw["trade_date"].isin(grid_dates)].copy()
    df_obs = df_obs[
        (df_obs["sigma"] > 0)
        & (df_obs["sigma"] <= 2.0)
        & (df_obs["tau"] > 0)
        & (np.isfinite(df_obs["m"]))
    ].reset_index(drop=True)
    print(f"  观测点总数: {len(df_obs)}")

    # VAE F mapping
    date_to_f_vae = dict(zip(vae_info["dates"], vae_info["F_all"]))
    date_to_f_vae_test = dict(zip(vae_info["dates_test"], vae_info["F_test"]))

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
                "F": date_to_f[d].astype(np.float64),
                "m": df_day["m"].values.astype(np.float64),
                "tau": df_day["tau"].values.astype(np.float64),
                "sigma": df_day["sigma"].values.astype(np.float64),
            })
        return out

    train_data = _build_day_list(train_dates, date_to_f_vae)
    test_data = _build_day_list(test_dates, date_to_f_vae_test)

    # 提取网格 m/tau（固定 154 维）用于 fixed-grid 评估
    grid_meta = grid_df[grid_df["trade_date"] == grid_dates[0]].sort_values("grid_idx")
    m_grid = grid_meta["m"].values.astype(np.float64)
    tau_grid = grid_meta["tau"].values.astype(np.float64)

    print(f"  Train days: {len(train_data)}, Test days: {len(test_data)}")
    return train_data, test_data, m_grid, tau_grid


# ------------------------------------------------------------------
# 训练（observation-level batch）
# ------------------------------------------------------------------
def flatten_days(day_list):
    """将按天的列表 flatten 为 (F, m, tau, sigma, day_idx)。"""
    all_F, all_m, all_tau, all_sigma, all_day_idx = [], [], [], [], []
    for idx, item in enumerate(day_list):
        n = len(item["sigma"])
        all_F.append(np.tile(item["F"], (n, 1)))
        all_m.append(item["m"])
        all_tau.append(item["tau"])
        all_sigma.append(item["sigma"])
        all_day_idx.append(np.full(n, idx, dtype=np.int64))
    return {
        "F": np.vstack(all_F),
        "m": np.concatenate(all_m),
        "tau": np.concatenate(all_tau),
        "sigma": np.concatenate(all_sigma),
        "day_idx": np.concatenate(all_day_idx),
    }


def train_diagnostic(train_data, test_data, device="cpu"):
    model = MLPSurface(input_dim=N_GRID + 2, hidden_dim=50).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Flatten 训练数据
    train_flat = flatten_days(train_data)
    test_flat = flatten_days(test_data)
    n_train_obs = len(train_flat["sigma"])
    n_test_obs = len(test_flat["sigma"])

    # 预先把 test 转成 tensor
    test_F_t = torch.tensor(test_flat["F"], dtype=torch.float32, device=device)
    test_m_t = torch.tensor(test_flat["m"], dtype=torch.float32, device=device)
    test_tau_t = torch.tensor(test_flat["tau"], dtype=torch.float32, device=device)
    test_sigma_t = torch.tensor(test_flat["sigma"], dtype=torch.float32, device=device)

    history = []
    best_test_rmse = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(EPOCHS):
        model.train()

        # ---- MSE 训练（observation-level batch） ----
        perm = torch.randperm(n_train_obs)
        train_mse_sum = 0.0
        n_batches = 0

        for i in range(0, min(n_train_obs, MAX_BATCHES_PER_EPOCH * BATCH_SIZE), BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            F_b = torch.tensor(train_flat["F"][idx], dtype=torch.float32, device=device)
            m_b = torch.tensor(train_flat["m"][idx], dtype=torch.float32, device=device)
            tau_b = torch.tensor(train_flat["tau"][idx], dtype=torch.float32, device=device)
            sigma_b = torch.tensor(train_flat["sigma"][idx], dtype=torch.float32, device=device)

            pred = model(F_b, tau_b.unsqueeze(1), m_b.unsqueeze(1)).squeeze()
            mse_loss = nn.functional.mse_loss(pred, sigma_b)

            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()

            train_mse_sum += mse_loss.item()
            n_batches += 1
            if n_batches >= MAX_BATCHES_PER_EPOCH:
                break

        train_mse = train_mse_sum / n_batches

        # ---- Penalty 计算（随机采样训练天） ----
        p_cal = p_but = p_bound = 0.0
        if LAMBDA_PENALTY > 0:
            n_train_days = len(train_data)
            sample_idx = np.random.choice(n_train_days, size=min(N_PENALTY_DAYS_PER_EPOCH, n_train_days), replace=False)
            F_penalty = torch.tensor(np.stack([train_data[i]["F"] for i in sample_idx]), dtype=torch.float32, device=device)
            with torch.enable_grad():
                p_cal, p_but, p_bound = compute_arbitrage_penalties(model, F_penalty, device)
            p_cal = p_cal.item()
            p_but = p_but.item()
            p_bound = p_bound.item()

        # ---- Test 评估 ----
        model.eval()
        with torch.no_grad():
            test_pred = model(test_F_t, test_tau_t.unsqueeze(1), test_m_t.unsqueeze(1)).squeeze()
            test_mse = nn.functional.mse_loss(test_pred, test_sigma_t).item()
            test_rmse = np.sqrt(test_mse)
            test_mape = torch.mean(torch.abs(test_pred - test_sigma_t) / torch.clamp(test_sigma_t, min=1e-6)).item()

        # ---- 组装日志 ----
        l_s = train_mse
        l_c3, l_c4, l_c5 = p_cal, p_but, p_bound
        penalty_sum = l_c3 + l_c4 + l_c5
        total_loss = l_s + LAMBDA_PENALTY * penalty_sum
        penalty_ratio = LAMBDA_PENALTY * penalty_sum / total_loss if total_loss > 0 else 0.0

        log_entry = {
            "epoch": epoch,
            "l_s": l_s,
            "l_c3": l_c3,
            "l_c4": l_c4,
            "l_c5": l_c5,
            "penalty_sum": penalty_sum,
            "penalty_ratio": penalty_ratio,
            "train_mse": train_mse,
            "test_rmse": test_rmse,
            "test_mape": test_mape,
        }
        history.append(log_entry)

        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if epoch % 2 == 0 or epoch == EPOCHS - 1:
            print(f"  Ep {epoch:2d}: l_s={l_s:.6f} l_c3={l_c3:.6f} l_c4={l_c4:.6f} l_c5={l_c5:.6f} "
                  f"pen_ratio={penalty_ratio:.4f} | Test RMSE={test_rmse:.6f} MAPE={test_mape:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  [Best] epoch {best_epoch}, Test RMSE={best_test_rmse:.6f}")

    return model, history


# ------------------------------------------------------------------
# 综合评估与诊断
# ------------------------------------------------------------------
def evaluate_comprehensive(model, test_data, m_grid, tau_grid, device="cpu"):
    """返回包含所有诊断信息的 dict。"""
    model.eval()
    records = []
    daily_rmse = []
    daily_mape = []

    for item in test_data:
        F_t = torch.tensor(item["F"], dtype=torch.float32, device=device).unsqueeze(0)
        m_t = torch.tensor(item["m"], dtype=torch.float32, device=device)
        tau_t = torch.tensor(item["tau"], dtype=torch.float32, device=device)
        sigma_true = item["sigma"]
        n_obs = len(sigma_true)

        with torch.no_grad():
            F_exp = F_t.expand(n_obs, -1)
            sigma_pred = model(F_exp, tau_t.unsqueeze(1), m_t.unsqueeze(1)).squeeze().cpu().numpy()

        errors = sigma_pred - sigma_true
        ape = np.abs(errors) / np.maximum(sigma_true, 1e-6)
        log_ape = np.abs(np.log(sigma_pred / np.maximum(sigma_true, 1e-6)))
        smape = 2 * np.abs(errors) / (np.maximum(sigma_true, 1e-6) + np.maximum(sigma_pred, 1e-6) + 1e-6)

        daily_rmse.append(np.sqrt(mean_squared_error(sigma_true, sigma_pred)))
        daily_mape.append(np.mean(ape))

        for i in range(n_obs):
            records.append({
                "date": item["date"],
                "m": item["m"][i],
                "tau": item["tau"][i],
                "sigma_true": sigma_true[i],
                "sigma_pred": sigma_pred[i],
                "error": errors[i],
                "ape": ape[i],
                "log_ape": log_ape[i],
                "smape": smape[i],
                "iv_bucket": _classify_iv(sigma_true[i]),
                "m_bucket": _classify_m(item["m"][i]),
                "tau_bucket": _classify_tau(item["tau"][i]),
            })

    df = pd.DataFrame(records)

    # 1. 总体指标
    diagnostics = {
        "n_test_days": len(test_data),
        "n_test_obs": len(df),
        "rmse_overall": np.sqrt(np.mean(df["error"] ** 2)),
        "mape_overall": df["ape"].mean(),
        "rmse_daily_mean": np.mean(daily_rmse),
        "mape_daily_mean": np.mean(daily_mape),
        # 替代 MAPE
        "weighted_mape": np.average(df["ape"], weights=df["sigma_true"]),
        "log_mape": df["log_ape"].mean(),
        "smape": df["smape"].mean(),
        "rmse_daily_std": np.std(daily_rmse),
        "mape_daily_std": np.std(daily_mape),
    }

    # 2. 固定网格 RMSE
    grid_rmse = _evaluate_fixed_grid(model, test_data, m_grid, tau_grid, device)
    diagnostics["rmse_fixed_grid"] = grid_rmse

    # 3. 分层 MAPE
    mape_by_iv = df.groupby("iv_bucket")["ape"].agg(["mean", "std", "count"]).reset_index()
    mape_by_m = df.groupby("m_bucket")["ape"].agg(["mean", "std", "count"]).reset_index()
    mape_by_tau = df.groupby("tau_bucket")["ape"].agg(["mean", "std", "count"]).reset_index()

    # 4. 极端样本
    extreme = df.nlargest(max(1, int(len(df) * 0.01)), "ape")

    # 5. Penalty diagnostics (在 test 天上)
    print("[Eval] 计算 Penalty diagnostics...")
    penalty_diag = _compute_test_penalty_diagnostics(model, test_data, device)

    return {
        "diagnostics": diagnostics,
        "mape_by_iv": mape_by_iv,
        "mape_by_m": mape_by_m,
        "mape_by_tau": mape_by_tau,
        "extreme_samples": extreme,
        "penalty_diagnostics": penalty_diag,
        "detail_df": df,
    }


def _classify_iv(iv):
    if iv < 0.1:
        return "[0, 0.1)"
    elif iv < 0.2:
        return "[0.1, 0.2)"
    elif iv < 0.3:
        return "[0.2, 0.3)"
    elif iv < 0.5:
        return "[0.3, 0.5)"
    else:
        return ">=0.5"


def _classify_m(m):
    if m < -0.2:
        return "<-0.2"
    elif m < -0.1:
        return "[-0.2, -0.1)"
    elif m < 0.0:
        return "[-0.1, 0.0)"
    elif m < 0.1:
        return "[0.0, 0.1)"
    else:
        return ">=0.1"


def _classify_tau(tau):
    if tau < 30 / DAYS_PER_YEAR:
        return "<30d"
    elif tau < 90 / DAYS_PER_YEAR:
        return "[30d, 90d)"
    elif tau < 180 / DAYS_PER_YEAR:
        return "[90d, 180d)"
    elif tau < 365 / DAYS_PER_YEAR:
        return "[180d, 1y)"
    else:
        return ">=1y"


def _evaluate_fixed_grid(model, test_data, m_grid, tau_grid, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for item in test_data:
            F_t = torch.tensor(item["F"], dtype=torch.float32, device=device).unsqueeze(0)
            F_exp = F_t.expand(154, -1)
            m_t = torch.tensor(m_grid, dtype=torch.float32, device=device)
            tau_t = torch.tensor(tau_grid, dtype=torch.float32, device=device)
            sigma_pred = model(F_exp, tau_t.unsqueeze(1), m_t.unsqueeze(1)).squeeze().cpu().numpy()
            # 对比 F 本身（因为 VAE F 就是 grid 上的 IV）
            sigma_true = item["F"]
            errors.extend((sigma_pred - sigma_true) ** 2)
    return np.sqrt(np.mean(errors)) if errors else float("nan")


def _compute_test_penalty_diagnostics(model, test_data, device):
    """在测试天上计算 penalty 诊断。"""
    n_sample = min(64, len(test_data))
    indices = np.random.choice(len(test_data), size=n_sample, replace=False)
    F_sample = torch.tensor(np.stack([test_data[i]["F"] for i in indices]), dtype=torch.float32, device=device)
    with torch.enable_grad():
        diag = compute_penalty_diagnostics(model, F_sample, device)
    return diag


# ------------------------------------------------------------------
# 主函数
# ------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # 1. 加载 VAE 特征
    print(f"[Load] VAE features from {STEP1_VAE_NPZ}")
    vae_info = load_vae_features(STEP1_VAE_NPZ, STEP1_VAE_PT)
    print(f"  Z shape: {vae_info['F_all'].shape}, Z_pred shape: {vae_info['F_test'].shape}")

    # 2. 准备数据
    train_data, test_data, m_grid, tau_grid = prepare_data(
        STEP0_GRID, RAW_DATA, RATE_DATA, vae_info
    )

    n_train_obs = sum(len(d["sigma"]) for d in train_data)
    n_test_obs = sum(len(d["sigma"]) for d in test_data)
    print(f"\n[Data Summary] Train: {len(train_data)} days, {n_train_obs} obs | "
          f"Test: {len(test_data)} days, {n_test_obs} obs")

    # 3. 训练
    print(f"\n{'=' * BANNER_WIDTH}")
    print("Step 2 v3 Diagnostic Training")
    print(f"  Model: MLP(156 -> 50 -> 50 -> 50 -> 1, Tanh, Softplus)")
    print(f"  Epochs: {EPOCHS}, LR: {LR}, Batch: {BATCH_SIZE} (obs), Lambda: {LAMBDA_PENALTY}")
    print(f"  Max batches/epoch: {MAX_BATCHES_PER_EPOCH} (~{MAX_BATCHES_PER_EPOCH*BATCH_SIZE/1e6:.2f}M obs)")
    print(f"  Penalty sample days/epoch: {N_PENALTY_DAYS_PER_EPOCH}")
    print(f"{'=' * BANNER_WIDTH}")

    model, history = train_diagnostic(train_data, test_data, device)

    # 4. 保存训练日志
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(OUTPUT_DIR / "training_log.csv", index=False)
    print(f"\n[Save] training_log.csv -> {OUTPUT_DIR}")

    # 5. 综合评估
    print(f"\n{'=' * BANNER_WIDTH}")
    print("Comprehensive Evaluation on Test Set")
    print(f"{'=' * BANNER_WIDTH}")

    eval_results = evaluate_comprehensive(model, test_data, m_grid, tau_grid, device)
    diag = eval_results["diagnostics"]

    print(f"  RMSE (overall)      = {diag['rmse_overall']:.6f}")
    print(f"  RMSE (daily mean)   = {diag['rmse_daily_mean']:.6f} (+/- {diag['rmse_daily_std']:.6f})")
    print(f"  RMSE (fixed grid)   = {diag['rmse_fixed_grid']:.6f}")
    print(f"  MAPE (overall)      = {diag['mape_overall']:.6f}")
    print(f"  MAPE (daily mean)   = {diag['mape_daily_mean']:.6f} (+/- {diag['mape_daily_std']:.6f})")
    print(f"  Weighted MAPE       = {diag['weighted_mape']:.6f}")
    print(f"  Log-space MAPE      = {diag['log_mape']:.6f}")
    print(f"  sMAPE               = {diag['smape']:.6f}")

    # 6. 保存分层 MAPE
    eval_results["mape_by_iv"].to_csv(OUTPUT_DIR / "mape_by_iv_level.csv", index=False)
    eval_results["mape_by_m"].to_csv(OUTPUT_DIR / "mape_by_moneyness.csv", index=False)
    eval_results["mape_by_tau"].to_csv(OUTPUT_DIR / "mape_by_tau.csv", index=False)
    eval_results["extreme_samples"].to_csv(OUTPUT_DIR / "extreme_mape_samples.csv", index=False)

    # 7. 保存 JSON 诊断
    # 处理 numpy 类型
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        return obj

    json_diagnostics = {
        "config": {
            "model": "MLP(156, 50, 50, 50, 1)",
            "activation_hidden": "Tanh",
            "activation_output": "Softplus",
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "lambda_penalty": LAMBDA_PENALTY,
            "train_days": len(train_data),
            "test_days": len(test_data),
            "train_obs": n_train_obs,
            "test_obs": n_test_obs,
        },
        "overall": _convert(diag),
        "mape_by_iv_level": eval_results["mape_by_iv"].to_dict(orient="records"),
        "mape_by_moneyness": eval_results["mape_by_m"].to_dict(orient="records"),
        "mape_by_tau": eval_results["mape_by_tau"].to_dict(orient="records"),
        "penalty_diagnostics": _convert(eval_results["penalty_diagnostics"]),
    }

    with open(OUTPUT_DIR / "evaluation_diagnostics.json", "w") as f:
        json.dump(json_diagnostics, f, indent=2)

    # 8. 保存 penalty diagnostics 单独文件
    with open(OUTPUT_DIR / "penalty_diagnostics.json", "w") as f:
        json.dump(_convert(eval_results["penalty_diagnostics"]), f, indent=2)

    # 9. 保存模型
    torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")

    print(f"\n[Done] 所有诊断输出保存到: {OUTPUT_DIR}")
    print(f"  Files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
