# -*- coding: utf-8 -*-
"""
Step 0: 单日 IV 曲面插值 —— DFW vs NW
复现 Zhang et al. (2023) Section 2.3

输入: 50etf_options.csv
输出:
  - output/step0/table1_nw_dfw.csv (每日 RMSE 对比)
  - output/step0/daily_grid_154.parquet (154 维 DFW/NW 网格)

优化说明：
  - NW 插值完全向量化（broadcasting），单日提速 50-100x
  - CV 内层循环复用 fold 的 pairwise 差分矩阵（仅缩放，不重算）
  - CV 仅在验证点上插值（不必跑到 154 维网格）
  - DFW 用 np.linalg.lstsq，避免 sklearn 单次 fit 的开销
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm

# ------------------------------------------------------------------
# 路径配置
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "spx_options_2009_2020.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "spx_step0"
TABLE1_FILENAME = "table1_nw_dfw.csv"
GRID_FILENAME = "daily_grid_154.parquet"

# ------------------------------------------------------------------
# 固定网格 I0（论文定义）
# ------------------------------------------------------------------
M_GRID = np.log(np.array([0.6, 0.8, 0.9, 0.95, 0.975, 1.0, 1.025, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0]))
TAU_DAYS_ALL = np.array([10, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730])

# ------------------------------------------------------------------
# 算法常量
# ------------------------------------------------------------------
DAYS_PER_YEAR = 365.0
IV_FLOOR = 0.01                       # DFW 预测下限
KERNEL_WEIGHT_EPS = 1e-10             # NW 核权重数值下限
MIN_OBS_PER_DAY = 6                   # 当日最少观测数（= DFW 系数个数）
CV_SEED = 42
NW_DEFAULT_BANDWIDTH = (0.1, 0.1)     # 观测过少时的兜底带宽
H_GRID_MIN, H_GRID_MAX, H_GRID_N = 0.02, 0.30, 10  # CV 带宽搜索网格
BANNER_WIDTH = 50

# 输入/输出列名
COL_DATE = "trade_date"
COL_IV = "implc_volatlty"
COL_K = "exercise_price"
COL_F = "fund_close"
COL_REM = "remaining_time"

GRID_COLUMNS = ["trade_date", "grid_idx", "m", "tau", "iv_dfw", "iv_nw"]


def get_tau_grid(df: pd.DataFrame) -> np.ndarray:
    """根据实际最大期限截断 tau 网格。"""
    max_days = df[COL_REM].max()
    tau_days = TAU_DAYS_ALL[TAU_DAYS_ALL <= max_days]
    return tau_days / DAYS_PER_YEAR


# ------------------------------------------------------------------
# 预处理
# ------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """计算 moneyness 和 tau。F 简化为 fund_close（忽略股息利率）。"""
    df = df.copy()
    df["F"] = df[COL_F]
    df["moneyness"] = np.log(df[COL_K] / df["F"])
    df["tau"] = df[COL_REM] / DAYS_PER_YEAR
    return df


# ------------------------------------------------------------------
# 方法 A: DFW 多项式插值
# ------------------------------------------------------------------
def _dfw_design(m: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """构造 DFW 二次多项式的设计矩阵：[1, m, tau, m^2, tau^2, m*tau]。"""
    return np.column_stack([np.ones(len(m)), m, tau, m**2, tau**2, m * tau])


def dfw_fit(m: np.ndarray, tau: np.ndarray, iv: np.ndarray) -> np.ndarray:
    """OLS 拟合 DFW 二次多项式，返回系数向量。"""
    X = _dfw_design(m, tau)
    coef, *_ = np.linalg.lstsq(X, iv, rcond=None)
    return coef


def dfw_predict(
    coef: np.ndarray,
    m_query: np.ndarray,
    tau_query: np.ndarray,
    floor: float = IV_FLOOR,
) -> np.ndarray:
    """用系数预测，并应用 floor。"""
    return np.maximum(_dfw_design(m_query, tau_query) @ coef, floor)


# ------------------------------------------------------------------
# 方法 B: NW 核回归（向量化版本）
# ------------------------------------------------------------------
def nw_interpolate_vec(
    m_obs: np.ndarray,
    tau_obs: np.ndarray,
    iv_obs: np.ndarray,
    m_query: np.ndarray,
    tau_query: np.ndarray,
    h1: float,
    h2: float,
) -> np.ndarray:
    """
    完全向量化的 NW 核回归。
    sigma_hat(m,tau) = sum_i exp(-0.5*((m-m_i)/h1)^2 - 0.5*((tau-tau_i)/h2)^2) * sigma_i
                       / sum_i exp(...)
    """
    u = (m_query[:, None] - m_obs[None, :]) / h1
    v = (tau_query[:, None] - tau_obs[None, :]) / h2
    w = np.exp(-0.5 * (u * u + v * v))
    s = w.sum(axis=1)
    s_safe = np.where(s > KERNEL_WEIGHT_EPS, s, 1.0)
    iv_pred = (w * iv_obs[None, :]).sum(axis=1) / s_safe
    mask = s <= KERNEL_WEIGHT_EPS
    if mask.any():
        iv_pred[mask] = iv_obs.mean()
    return iv_pred


def nw_cv_bandwidth(
    m: np.ndarray,
    tau: np.ndarray,
    iv: np.ndarray,
    h1_grid: np.ndarray,
    h2_grid: np.ndarray,
    n_splits: int = 5,
) -> tuple[float, float]:
    """
    五折交叉验证选最优 (h1, h2)。
    优化点：对每个 fold，先算一次 pairwise 差分矩阵 (du, dv)，
    然后在 h1×h2 上反复缩放求权重，避免 100x 重复差分计算。
    """
    if len(m) < n_splits:
        return NW_DEFAULT_BANDWIDTH

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=CV_SEED)

    n_h1, n_h2 = len(h1_grid), len(h2_grid)
    mse_acc = np.zeros((n_h1, n_h2))

    for train_idx, val_idx in kf.split(m):
        m_tr, tau_tr, iv_tr = m[train_idx], tau[train_idx], iv[train_idx]
        m_va, tau_va, iv_va = m[val_idx], tau[val_idx], iv[val_idx]

        # (n_val, n_train) 差分矩阵：每个 fold 只算一次
        du = m_va[:, None] - m_tr[None, :]
        dv = tau_va[:, None] - tau_tr[None, :]

        # 预先沿 h1 维度缓存 (du / h1)^2
        u2_cache = [(du / h1) ** 2 for h1 in h1_grid]

        for i in range(n_h1):
            u2 = u2_cache[i]
            for j in range(n_h2):
                v2 = (dv / h2_grid[j]) ** 2
                w = np.exp(-0.5 * (u2 + v2))
                s = w.sum(axis=1)
                s_safe = np.where(s > KERNEL_WEIGHT_EPS, s, 1.0)
                pred = (w * iv_tr[None, :]).sum(axis=1) / s_safe
                mask = s <= KERNEL_WEIGHT_EPS
                if mask.any():
                    pred[mask] = iv_tr.mean()
                mse_acc[i, j] += mean_squared_error(iv_va, pred)

    mse_avg = mse_acc / n_splits
    i_best, j_best = np.unravel_index(np.argmin(mse_avg), mse_avg.shape)
    return float(h1_grid[i_best]), float(h2_grid[j_best])


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def run_step0(df: pd.DataFrame) -> tuple[pd.DataFrame, list, np.ndarray]:
    """
    逐日插值 + RMSE 对比。

    入参 `df` 必须是 preprocess() 之后的 DataFrame。
    返回 (res_df, grid_data, tau_grid)。
    """
    tau_grid = get_tau_grid(df)

    # 构建完整网格坐标
    M_mesh, Tau_mesh = np.meshgrid(M_GRID, tau_grid, indexing="ij")
    m_flat = M_mesh.ravel()
    tau_flat = Tau_mesh.ravel()

    results: list[dict] = []
    grid_data: list[dict] = []

    h1_grid = np.linspace(H_GRID_MIN, H_GRID_MAX, H_GRID_N)
    h2_grid = np.linspace(H_GRID_MIN, H_GRID_MAX, H_GRID_N)

    grouped = df.groupby(COL_DATE, sort=True)

    for date, df_day in tqdm(grouped, desc="逐日插值", unit="天", total=grouped.ngroups):
        if len(df_day) < MIN_OBS_PER_DAY:
            continue

        m_obs = df_day["moneyness"].values
        tau_obs = df_day["tau"].values
        iv_obs = df_day[COL_IV].values

        # ---------- DFW ----------
        coef = dfw_fit(m_obs, tau_obs, iv_obs)
        iv_dfw_flat = dfw_predict(coef, m_flat, tau_flat)
        iv_dfw_obs = dfw_predict(coef, m_obs, tau_obs)
        rmse_dfw = np.sqrt(mean_squared_error(iv_obs, iv_dfw_obs))

        # ---------- NW（CV 选带宽）----------
        h1_best, h2_best = nw_cv_bandwidth(m_obs, tau_obs, iv_obs, h1_grid, h2_grid)
        iv_nw_flat = nw_interpolate_vec(
            m_obs, tau_obs, iv_obs, m_flat, tau_flat, h1_best, h2_best
        )
        iv_nw_obs = nw_interpolate_vec(
            m_obs, tau_obs, iv_obs, m_obs, tau_obs, h1_best, h2_best
        )
        rmse_nw = np.sqrt(mean_squared_error(iv_obs, iv_nw_obs))

        results.append({
            "date": date,
            "n": len(df_day),
            "dfw_rmse": rmse_dfw,
            "nw_rmse": rmse_nw,
            "h1": h1_best,
            "h2": h2_best,
        })

        grid_data.append({
            "date": date,
            "iv_dfw": iv_dfw_flat,
            "iv_nw": iv_nw_flat,
        })

    return pd.DataFrame(results), grid_data, tau_grid


def save_outputs(
    grid_data: list[dict],
    res_df: pd.DataFrame,
    tau_grid: np.ndarray,
) -> tuple[Path, Path]:
    """保存输出文件（输出目录在此处确保存在）。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 每日 RMSE 对比（Table 1）
    table1_path = OUTPUT_DIR / TABLE1_FILENAME
    res_df.to_csv(table1_path, index=False)

    # 2. 每天网格（parquet）—— 用 numpy 向量化构造
    grid_path = OUTPUT_DIR / GRID_FILENAME
    if not grid_data:
        pd.DataFrame(columns=GRID_COLUMNS).to_parquet(grid_path, index=False)
        return table1_path, grid_path

    M_mesh, T_mesh = np.meshgrid(M_GRID, tau_grid, indexing="ij")
    m_flat = M_mesh.ravel()
    tau_flat = T_mesh.ravel()
    n_grid = m_flat.size
    n_days = len(grid_data)

    grid_df = pd.DataFrame({
        "trade_date": np.concatenate([np.repeat(g["date"], n_grid) for g in grid_data]),
        "grid_idx": np.tile(np.arange(n_grid), n_days),
        "m": np.tile(m_flat, n_days),
        "tau": np.tile(tau_flat, n_days),
        "iv_dfw": np.concatenate([g["iv_dfw"] for g in grid_data]),
        "iv_nw": np.concatenate([g["iv_nw"] for g in grid_data]),
    })
    grid_df.to_parquet(grid_path, index=False)
    return table1_path, grid_path


def _section(title: str) -> None:
    print("=" * BANNER_WIDTH)
    print(title)
    print("=" * BANNER_WIDTH)


def print_checkpoints(
    df: pd.DataFrame,
    tau_grid: np.ndarray,
    res_df: pd.DataFrame,
    grid_data: list[dict],
) -> None:
    """打印检查点日志（df 必须是 preprocess 后的；不重复计算）。"""
    n_grid = len(M_GRID) * len(tau_grid)

    _section("[Checkpoint 1] 数据加载")
    print(f"  - 总交易日: {df[COL_DATE].nunique()}")
    print(f"  - 每日平均合约数: {df.groupby(COL_DATE).size().mean():.1f}")
    print(f"  - moneyness 范围: [{df['moneyness'].min():.3f}, {df['moneyness'].max():.3f}]")
    print(f"  - tau 范围: [{df['tau'].min():.3f}, {df['tau'].max():.3f}] 年")

    print()
    _section("[Checkpoint 2] 网格构建")
    print(f"  - m_grid 点数: {len(M_GRID)}")
    print(f"  - tau_grid 点数: {len(tau_grid)}")
    print(f"  - 总网格点: {n_grid} (论文 154)")

    print()
    _section("[Checkpoint 3] DFW 插值")
    print(f"  - 平均 RMSE: {res_df['dfw_rmse'].mean():.4f}")
    print(f"  - 中位数 RMSE: {res_df['dfw_rmse'].median():.4f}")

    print()
    _section("[Checkpoint 4] NW 插值（五折 CV）")
    print(f"  - 平均 RMSE: {res_df['nw_rmse'].mean():.4f}")
    print(f"  - 中位数 RMSE: {res_df['nw_rmse'].median():.4f}")
    print(f"  - 平均最优带宽 h1: {res_df['h1'].mean():.3f}")
    print(f"  - 平均最优带宽 h2: {res_df['h2'].mean():.3f}")

    print()
    _section("[Checkpoint 5] 输出文件")
    print(f"  - {TABLE1_FILENAME}: {len(res_df)} 行")
    print(f"  - {GRID_FILENAME}: {len(grid_data) * n_grid} 个网格点")

    print()
    _section("Table 1: Average RMSE for NW and DFW")
    print(f"DFW: {res_df['dfw_rmse'].mean():.4f}")
    print(f"NW:  {res_df['nw_rmse'].mean():.4f}")
    print("=" * BANNER_WIDTH)


def main() -> None:
    print(f"[Load] 加载数据: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"  - 原始记录: {len(df_raw)} 条")

    df = preprocess(df_raw)

    print("\n[Run] 开始 Step 0 插值...")
    res_df, grid_data, tau_grid = run_step0(df)

    print("\n[Save] 保存输出...")
    table1_path, grid_path = save_outputs(grid_data, res_df, tau_grid)
    print(f"  - {table1_path}")
    print(f"  - {grid_path}")

    print()
    print_checkpoints(df, tau_grid, res_df, grid_data)

    print("\n[Done] Step 0 完成。")


if __name__ == "__main__":
    main()
