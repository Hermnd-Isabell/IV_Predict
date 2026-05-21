# -*- coding: utf-8 -*-
"""
Step 2: DNN 无套利曲面重构
复现 Zhang et al. (2023) Section 3.3 & 3.4

输入: Step 0 网格 + Step 1 特征 + 原始观测数据
输出: DNN 模型 + 测试评估 + 无套利违规检查
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from models.surfaces.mlp import MLPSurface as DNN_Surface

# 从 step1 导入 VAE 相关函数（用于 VAE 解码器重建）
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step1_features_lstm import VAE

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------
# 路径配置
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

STEP0_GRID = PROJECT_ROOT / "output" / "spx_step0" / "daily_grid_154.parquet"
STEP1_SAM = PROJECT_ROOT / "output" / "spx_step1" / "sam_features.npz"
STEP1_PCA = PROJECT_ROOT / "output" / "spx_step1" / "pca_features.npz"
STEP1_VAE = PROJECT_ROOT / "output" / "spx_step1" / "vae_features.npz"
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "spx_options_2009_2020.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "spx_step2"

BANNER_WIDTH = 50
DAYS_PER_YEAR = 365.0

# DNN 训练参数
DNN_HIDDEN = 50
DNN_EPOCHS = 20
DNN_BATCH_SIZE_DAYS = 32
DNN_LR = 0.001
DNN_LAMBDA = 1.0

# 无套利检查网格密度
GRID_C34_N = 40
GRID_C5_N = 4

# 划分比例（和 step1 一致）
TRAIN_RATIO = 0.75
VAL_RATIO = 0.15

# ------------------------------------------------------------------
# 全局套利检查网格
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


# ------------------------------------------------------------------
# 无套利惩罚计算
# ------------------------------------------------------------------
def compute_arbitrage_penalties(
    model: nn.Module,
    F_batch: torch.Tensor,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对一批 F_batch（每个交易日一个 F 向量），在密集网格上计算无套利惩罚。

    F_batch: (batch_days, F_dim)
    """
    batch_days = F_batch.shape[0]

    # ---- 条件 3 & 4 ----
    m_c34 = M_C34.to(device).clone().requires_grad_(True)
    tau_c34 = TAU_C34.to(device).clone().requires_grad_(True)

    F_expanded = F_batch.unsqueeze(1).expand(-1, N_C34, -1).reshape(-1, F_batch.shape[1])
    m_expanded = m_c34.repeat(batch_days)
    tau_expanded = tau_c34.repeat(batch_days)

    sigma = model(F_expanded, tau_expanded.unsqueeze(1), m_expanded.unsqueeze(1)).squeeze()
    sigma_safe = torch.clamp(sigma, min=1e-6)

    # 一阶导
    grad_m = torch.autograd.grad(sigma.sum(), m_expanded, create_graph=True, retain_graph=True)[0]
    grad_tau = torch.autograd.grad(sigma.sum(), tau_expanded, create_graph=True, retain_graph=True)[0]

    # 二阶导
    grad_mm = torch.autograd.grad(grad_m.sum(), m_expanded, create_graph=True)[0]

    # 条件 3: 日历套利
    l_cal = sigma + 2 * tau_expanded * grad_tau
    penalty_cal = torch.clamp(-l_cal, min=0).mean()

    # 条件 4: 蝶式套利
    grad_m_safe = grad_m / sigma_safe
    term1 = (1 - m_expanded * grad_m_safe) ** 2
    term2 = (sigma_safe * tau_expanded * grad_m_safe) ** 2 / 4
    term3 = tau_expanded * sigma_safe * grad_mm
    l_but = term1 - term2 + term3
    penalty_but = torch.clamp(-l_but, min=0).mean()

    # ---- 条件 5: 大 moneyness 行为 ----
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


# ------------------------------------------------------------------
# 单日损失
# ------------------------------------------------------------------
def day_loss(
    model: nn.Module,
    F_day: torch.Tensor,
    m_obs: torch.Tensor,
    tau_obs: torch.Tensor,
    sigma_obs: torch.Tensor,
    lambda_penalty: float = DNN_LAMBDA,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算单个交易日的损失。

    F_day: (1, F_dim)
    m_obs, tau_obs, sigma_obs: (n_obs,)
    """
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


# ------------------------------------------------------------------
# 训练 DNN
# ------------------------------------------------------------------
def train_step2(
    train_data: list[dict],
    val_data: list[dict],
    n_grid: int,
    model_class: type = DNN_Surface,
    model_kwargs: dict | None = None,
    output_activation: str = "softplus",
    train_kwargs: dict | None = None,
    device: str = "cpu",
) -> tuple[nn.Module, dict]:
    """
    按天 batch 训练 Step 2 曲面重构模型。
    train_data/val_data: list of {"F": (n_grid,), "m": (n_obs,), "tau": (n_obs,), "sigma": (n_obs,)}
    """
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
        "train_loss": [],
        "val_loss": [],
        "mse": [],
        "pen_cal": [],
        "pen_but": [],
        "pen_bound": [],
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

                total, _, _, _, _ = day_loss(
                    model, F_t, m_t, tau_t, sigma_t, lambda_penalty, device
                )
                batch_total_loss += total

            batch_total_loss = batch_total_loss / len(batch)
            batch_total_loss.backward()
            optimizer.step()

            train_loss_epoch += batch_total_loss.item()
            n_batches += 1

        # 验证（MSE 用 no_grad；套利惩罚需要梯度，单独计算）
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

        val_pcal = 0.0
        val_pbut = 0.0
        val_pbound = 0.0
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
            print(
                f"  Epoch {epoch:2d}: Train={train_loss_epoch/n_batches:.6f} | "
                f"Val MSE={val_mse:.6f} Cal={val_pcal:.6f} But={val_pbut:.6f} Bound={val_pbound:.6f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"  [DNN] 最优 epoch: {best_epoch}, best val loss: {best_val_loss:.6f}")
    return model, history


# ------------------------------------------------------------------
# 评估
# ------------------------------------------------------------------
def evaluate_step2(
    model: nn.Module,
    test_data: list[dict],
    device: str = "cpu",
) -> tuple[float, float, list[float], list[float]]:
    """按日评估 RMSE / MAPE。"""
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


# ------------------------------------------------------------------
# 无套利违规检查
# ------------------------------------------------------------------
def check_arbitrage_violation(
    model: nn.Module,
    test_data: list[dict],
    device: str = "cpu",
) -> tuple[float, float]:
    """
    在测试集上检查无套利违规。
    返回 L_cal_neg 和 L_but_neg（论文 Table 6 格式）。
    """
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


# ------------------------------------------------------------------
# 特征映射：从 Z 还原 F
# ------------------------------------------------------------------
def map_features_to_f(
    Z: np.ndarray,
    feature_type: str,
    grid_df: pd.DataFrame | None = None,
    eigensurfaces: np.ndarray | None = None,
    vae_model: VAE | None = None,
) -> np.ndarray:
    """
    将特征 Z 映射到离散 IV 点 F。

    Z: (n_days, feature_dim)
    返回 F: (n_days, n_grid)
    """
    if feature_type == "SAM":
        return Z

    elif feature_type == "PCA":
        if eigensurfaces is None:
            raise ValueError("PCA 需要 eigensurfaces")
        if grid_df is None:
            raise ValueError("PCA 需要 grid_df 以获取 sigma_0")
        first_date = grid_df["trade_date"].min()
        sigma_0 = (
            grid_df[grid_df["trade_date"] == first_date]
            .sort_values("grid_idx")["iv_dfw"]
            .values
            .astype(np.float64)
        )
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


# ------------------------------------------------------------------
# 数据准备
# ------------------------------------------------------------------
def prepare_data(
    grid_path: Path,
    raw_path: Path,
    step1_sam: Path,
    step1_pca: Path,
    step1_vae: Path,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> dict:
    """
    准备 DNN 训练所需的所有数据。
    返回 dict，包含三种特征类型的 train/val/test Data。
    """
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

    # 构造每日真实 F（来自 iv_dfw）
    df_grid_pivot = grid_df.pivot(index="trade_date", columns="grid_idx", values="iv_dfw")
    F_real = df_grid_pivot.values.astype(np.float64)  # (n_days, n_grid)
    date_to_f_real = dict(zip(df_grid_pivot.index.values, F_real))

    # 读取原始观测数据
    print("[Data] 读取原始期权数据...")
    df_raw = pd.read_csv(raw_path)
    df_raw["m"] = np.log(df_raw["exercise_price"] / df_raw["fund_close"])
    df_raw["tau"] = df_raw["remaining_time"] / DAYS_PER_YEAR
    df_raw["sigma"] = df_raw["implc_volatlty"]

    # 只保留 grid 中存在的日期，且过滤极端异常值
    df_obs = df_raw[df_raw["trade_date"].isin(grid_dates)].copy()
    df_obs = df_obs[
        (df_obs["sigma"] > 0)
        & (df_obs["sigma"] <= 2.0)
        & (df_obs["tau"] > 0)
        & (np.isfinite(df_obs["m"]))
    ].reset_index(drop=True)
    print(f"  观测点总数: {len(df_obs)}")

    # 按日期分组构造 list[dict]
    def _build_day_list(dates: np.ndarray, date_to_f: dict) -> list[dict]:
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

    # ---------- SAM ----------
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

    # ---------- PCA ----------
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

    # ---------- VAE ----------
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
    }


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def run_step2(
    feature_type: str,
    data_dict: dict,
    model_class: type = DNN_Surface,
    model_kwargs: dict | None = None,
    output_activation: str = "softplus",
    train_kwargs: dict | None = None,
    device: str = "cpu",
) -> tuple[nn.Module, dict, float, float, float, float]:
    model_name = model_class.__name__
    print(f"\n{'=' * BANNER_WIDTH}")
    print(f"Step 2: {model_name} Surface Reconstruction ({feature_type})")
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

    # 训练
    print(f"\n[Checkpoint 2] {model_class.__name__} 训练 (epochs={DNN_EPOCHS}, batch_days={DNN_BATCH_SIZE_DAYS})")
    model, history = train_step2(
        train_data, val_data, n_grid,
        model_class=model_class,
        model_kwargs=model_kwargs,
        output_activation=output_activation,
        train_kwargs=train_kwargs,
        device=device,
    )

    # 评估
    print(f"\n[Checkpoint 3] 测试评估")
    rmse, mape, rmse_daily, mape_daily = evaluate_step2(model, test_data, device)
    L_cal, L_but = check_arbitrage_violation(model, test_data, device)

    print(f"  Test RMSE  = {rmse:.6f}")
    print(f"  Test MAPE  = {mape:.6f}")
    print(f"  Calendar Arb Violation (L_cal) = {L_cal:.8f}")
    print(f"  Butterfly Arb Violation (L_but) = {L_but:.8f}")

    return model, history, rmse, mape, L_cal, L_but


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # 准备数据
    data_dict = prepare_data(
        STEP0_GRID, RAW_DATA,
        STEP1_SAM, STEP1_PCA, STEP1_VAE,
    )

    results_summary = {}

    def _save_results(ft, model, hist, rmse, mape, rmse_d, mape_d, Lcal, Lbut):
        torch.save(model.state_dict(), OUTPUT_DIR / f"dnn_{ft.lower()}.pt")
        np.savez(
            OUTPUT_DIR / f"results_{ft.lower()}.npz",
            rmse=rmse, mape=mape,
            rmse_daily=np.array(rmse_d),
            mape_daily=np.array(mape_d),
            L_cal=Lcal, L_but=Lbut,
            hist_train_loss=np.array(hist["train_loss"]),
            hist_val_loss=np.array(hist["val_loss"]),
            hist_mse=np.array(hist["mse"]),
            hist_pen_cal=np.array(hist["pen_cal"]),
            hist_pen_but=np.array(hist["pen_but"]),
            hist_pen_bound=np.array(hist["pen_bound"]),
        )

    # SAM
    model_sam, hist_sam, rmse_sam, mape_sam, Lcal_sam, Lbut_sam = run_step2("SAM", data_dict, device)
    _, _, rmse_d_sam, mape_d_sam = evaluate_step2(model_sam, data_dict["SAM"]["test"], device)
    _save_results("SAM", model_sam, hist_sam, rmse_sam, mape_sam, rmse_d_sam, mape_d_sam, Lcal_sam, Lbut_sam)
    results_summary["SAM"] = {"rmse": rmse_sam, "mape": mape_sam, "L_cal": Lcal_sam, "L_but": Lbut_sam}

    # PCA
    model_pca, hist_pca, rmse_pca, mape_pca, Lcal_pca, Lbut_pca = run_step2("PCA", data_dict, device)
    _, _, rmse_d_pca, mape_d_pca = evaluate_step2(model_pca, data_dict["PCA"]["test"], device)
    _save_results("PCA", model_pca, hist_pca, rmse_pca, mape_pca, rmse_d_pca, mape_d_pca, Lcal_pca, Lbut_pca)
    results_summary["PCA"] = {"rmse": rmse_pca, "mape": mape_pca, "L_cal": Lcal_pca, "L_but": Lbut_pca}

    # VAE
    model_vae, hist_vae, rmse_vae, mape_vae, Lcal_vae, Lbut_vae = run_step2("VAE", data_dict, device)
    _, _, rmse_d_vae, mape_d_vae = evaluate_step2(model_vae, data_dict["VAE"]["test"], device)
    _save_results("VAE", model_vae, hist_vae, rmse_vae, mape_vae, rmse_d_vae, mape_d_vae, Lcal_vae, Lbut_vae)
    results_summary["VAE"] = {"rmse": rmse_vae, "mape": mape_vae, "L_cal": Lcal_vae, "L_but": Lbut_vae}

    # 对比
    print(f"\n{'=' * BANNER_WIDTH}")
    print("[Checkpoint 4] 模型对比")
    print(f"{'=' * BANNER_WIDTH}")
    for ft, res in results_summary.items():
        print(f"  {ft:6s}: RMSE={res['rmse']:.6f}, MAPE={res['mape']:.6f}, "
              f"L_cal={res['L_cal']:.8f}, L_but={res['L_but']:.8f}")

    print(f"\n[Done] Step 2 完成，输出保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
