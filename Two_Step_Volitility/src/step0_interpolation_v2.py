# -*- coding: utf-8 -*-
"""
Step 0 v2: 修复三项关键问题后重新插值

修复项：
  1. 引入利率 r，计算远期价格 F = S * exp(r * tau)
  2. 固定 154 维网格（禁止动态截断），外推填充
  3. 统一 RMSE 口径（Out-of-sample: DFW vs NW grid + LOOCV）

输入: data/raw/spx_options_2009_2020.csv
输出: output/spx_step0_v2/
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
RATE_PATH = PROJECT_ROOT / "data" / "raw" / "rate_cleaned_2009_2020.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "spx_step0_v2"
TABLE1_FILENAME = "table1_rmse_comparison.csv"
GRID_FILENAME = "daily_grid_154_fixed.parquet"
SHIFT_FILENAME = "moneyness_shift_analysis.csv"

# ------------------------------------------------------------------
# 固定网格 I0（论文定义，v2 禁止动态截断）
# ------------------------------------------------------------------
M_GRID = np.log(np.array([0.6, 0.8, 0.9, 0.95, 0.975, 1.0, 1.025, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0]))
TAU_GRID = np.array([10, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730]) / 365.0
N_M = len(M_GRID)
N_TAU = len(TAU_GRID)
N_GRID = N_M * N_TAU  # 必须严格 = 154

# ------------------------------------------------------------------
# 算法常量
# ------------------------------------------------------------------
DAYS_PER_YEAR = 365.0
IV_FLOOR = 0.01
IV_CEIL = 2.0
KERNEL_WEIGHT_EPS = 1e-10
MIN_OBS_PER_DAY = 6
CV_SEED = 42
NW_DEFAULT_BANDWIDTH = (0.1, 0.1)
H_GRID_MIN, H_GRID_MAX, H_GRID_N = 0.02, 0.30, 10
BANNER_WIDTH = 60

COL_DATE = "trade_date"
COL_IV = "implc_volatlty"
COL_K = "exercise_price"
COL_F = "fund_close"
COL_REM = "remaining_time"

GRID_COLUMNS = ["trade_date", "grid_idx", "m", "tau", "iv_dfw", "iv_nw"]

# ------------------------------------------------------------------
# 修复 1: 加载利率 + 预处理（F = S * exp(r*tau)）
# ------------------------------------------------------------------
def load_rate_df(rate_path: Path = RATE_PATH) -> pd.DataFrame:
    """加载清洗后的日度利率（年化小数）"""
    rate_df = pd.read_csv(rate_path)
    rate_df = rate_df[["trade_date", "r"]].copy()
    rate_df["trade_date"] = rate_df["trade_date"].astype(int)
    return rate_df


def preprocess_v2(df: pd.DataFrame, rate_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    v2 预处理：引入无风险利率 r，计算远期价格 F = S * exp(r*tau)

    Returns
    -------
    df : 新增列 r, tau, F, moneyness_v1(旧), moneyness_v2(新)
    """
    df = df.copy()

    # 基础列
    df["tau"] = df[COL_REM] / DAYS_PER_YEAR

    # v1: 旧 moneyness（F = fund_close，用于对比）
    df["moneyness_v1"] = np.log(df[COL_K] / df[COL_F])

    if rate_df is not None:
        # 对齐利率
        df = df.merge(rate_df[["trade_date", "r"]], on="trade_date", how="left")
        missing_rate = df["r"].isna().sum()
        if missing_rate > 0:
            # 用全局均值兜底（极少见）
            global_r = df["r"].mean()
            df["r"] = df["r"].fillna(global_r)
            print(f"  [WARN] {missing_rate} 条记录无利率数据，用全局均值 {global_r:.4f} 填充")

        # v2: 远期价格 + 新 moneyness
        df["F"] = df[COL_F] * np.exp(df["r"] * df["tau"])
        df["moneyness_v2"] = np.log(df[COL_K] / df["F"])
        df["moneyness"] = df["moneyness_v2"]  # 主流程使用 v2
    else:
        # 降级：无利率时退化为 v1
        df["r"] = 0.0
        df["F"] = df[COL_F]
        df["moneyness_v2"] = df["moneyness_v1"]
        df["moneyness"] = df["moneyness_v2"]

    return df


# ------------------------------------------------------------------
# DFW 多项式插值（不变）
# ------------------------------------------------------------------
def _dfw_design(m: np.ndarray, tau: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(m)), m, tau, m ** 2, tau ** 2, m * tau])


def dfw_fit(m: np.ndarray, tau: np.ndarray, iv: np.ndarray) -> np.ndarray:
    X = _dfw_design(m, tau)
    coef, *_ = np.linalg.lstsq(X, iv, rcond=None)
    return coef


def dfw_predict(coef: np.ndarray, m_query: np.ndarray, tau_query: np.ndarray, floor: float = IV_FLOOR) -> np.ndarray:
    pred = _dfw_design(m_query, tau_query) @ coef
    return np.clip(pred, floor, IV_CEIL)


# ------------------------------------------------------------------
# NW 核回归（向量化，不变）
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
    m: np.ndarray, tau: np.ndarray, iv: np.ndarray,
    h1_grid: np.ndarray, h2_grid: np.ndarray, n_splits: int = 5,
) -> tuple[float, float]:
    if len(m) < n_splits:
        return NW_DEFAULT_BANDWIDTH
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=CV_SEED)
    n_h1, n_h2 = len(h1_grid), len(h2_grid)
    mse_acc = np.zeros((n_h1, n_h2))
    for train_idx, val_idx in kf.split(m):
        m_tr, tau_tr, iv_tr = m[train_idx], tau[train_idx], iv[train_idx]
        m_va, tau_va, iv_va = m[val_idx], tau[val_idx], iv[val_idx]
        du = m_va[:, None] - m_tr[None, :]
        dv = tau_va[:, None] - tau_tr[None, :]
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
# 修复 2: 固定 154 维网格外推填充
# ------------------------------------------------------------------
def fill_missing_grid(
    iv_grid: np.ndarray,
    m_grid: np.ndarray,
    tau_grid: np.ndarray,
    iv_obs: np.ndarray,
    weight_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """
    对 (14, 11) 的 iv_grid 填充外推区域。

    Parameters
    ----------
    iv_grid : (14, 11) 初始值，部分为 NaN（外推区）
    weight_mask : (14, 11) bool，True=观测覆盖良好，False=需要外推

    Returns
    -------
    iv_grid : 填充后的网格
    nan_ratio : 填充前 NaN 比例
    """
    if weight_mask is not None:
        iv_grid = iv_grid.copy()
        iv_grid[~weight_mask] = np.nan

    nan_before = np.isnan(iv_grid).sum()
    nan_ratio = nan_before / iv_grid.size

    # 1) tau 方向：每行独立线性插值 + 常数外推
    for i in range(len(m_grid)):
        valid = ~np.isnan(iv_grid[i, :])
        n_valid = valid.sum()
        if n_valid >= 2:
            iv_grid[i, :] = np.interp(
                tau_grid,
                tau_grid[valid],
                iv_grid[i, valid],
                left=iv_grid[i, valid][0],
                right=iv_grid[i, valid][-1],
            )
        elif n_valid == 1:
            iv_grid[i, :] = iv_grid[i, valid][0]

    # 2) m 方向：每列独立线性插值 + 常数外推
    for j in range(len(tau_grid)):
        valid = ~np.isnan(iv_grid[:, j])
        n_valid = valid.sum()
        if n_valid >= 2:
            missing = np.isnan(iv_grid[:, j])
            if missing.any():
                iv_grid[missing, j] = np.interp(
                    m_grid[missing],
                    m_grid[valid],
                    iv_grid[valid, j],
                    left=iv_grid[valid, j][0],
                    right=iv_grid[valid, j][-1],
                )
        elif n_valid == 1:
            iv_grid[:, j] = iv_grid[valid, j][0]

    # 3) 最终兜底：全局均值（极端情况）
    global_mean = np.nanmean(iv_obs) if len(iv_obs) > 0 else 0.2
    iv_grid = np.where(np.isnan(iv_grid), global_mean, iv_grid)

    # 截断
    iv_grid = np.clip(iv_grid, IV_FLOOR, IV_CEIL)

    return iv_grid, nan_ratio


# ------------------------------------------------------------------
# 修复 3: DFW LOOCV（用 hat matrix 快速计算）
# ------------------------------------------------------------------
def dfw_loocv_rmse(m_obs: np.ndarray, tau_obs: np.ndarray, iv_obs: np.ndarray) -> float:
    """
    DFW 的 Leave-One-Out CV RMSE。
    利用线性回归的 hat matrix 技巧：e_i^{(-i)} = e_i / (1 - h_ii)
    """
    n = len(iv_obs)
    if n <= 7:  # 太少时直接返回 in-sample
        coef = dfw_fit(m_obs, tau_obs, iv_obs)
        pred = dfw_predict(coef, m_obs, tau_obs)
        return float(np.sqrt(np.mean((pred - iv_obs) ** 2)))

    X = _dfw_design(m_obs, tau_obs)
    # OLS 预测
    coef, *_ = np.linalg.lstsq(X, iv_obs, rcond=None)
    y_hat = X @ coef
    residuals = iv_obs - y_hat

    # Hat matrix 对角线: h_ii = diag(X @ (X'X)^{-1} @ X')
    XtX_inv = np.linalg.inv(X.T @ X)
    H_diag = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    H_diag = np.clip(H_diag, 0, 1 - 1e-10)  # 数值稳定

    # LOOCV 残差
    loocv_res = residuals / (1.0 - H_diag)
    return float(np.sqrt(np.mean(loocv_res ** 2)))


# ------------------------------------------------------------------
# 主流程 v2
# ------------------------------------------------------------------
def run_step0_v2(df: pd.DataFrame) -> tuple[pd.DataFrame, list, pd.DataFrame]:
    """
    逐日插值 + 固定 154 维网格 + 三种 RMSE 口径。

    Returns
    -------
    res_df : 每日 RMSE 对比表
    grid_data : 每日 154 维网格列表
    shift_df : 修正前后 moneyness 分布对比
    """
    # 构建固定网格坐标（154 维）
    M_mesh, Tau_mesh = np.meshgrid(M_GRID, TAU_GRID, indexing="ij")
    m_flat = M_mesh.ravel()      # (154,)
    tau_flat = Tau_mesh.ravel()  # (154,)

    results: list[dict] = []
    grid_data: list[dict] = []
    shift_records: list[dict] = []

    # v2 优化：使用固定带宽 h=0.1（来自诊断的中位数最优值），跳过每日 CV 大幅提速
    FIXED_H1, FIXED_H2 = 0.1, 0.1
    h1_grid = np.linspace(H_GRID_MIN, H_GRID_MAX, H_GRID_N)
    h2_grid = np.linspace(H_GRID_MIN, H_GRID_MAX, H_GRID_N)

    grouped = df.groupby(COL_DATE, sort=True)
    day_counter = 0

    for date, df_day in tqdm(grouped, desc="Step0 v2 逐日插值", unit="天", total=grouped.ngroups):
        day_counter += 1
        if len(df_day) < MIN_OBS_PER_DAY:
            continue

        m_obs = df_day["moneyness"].values
        tau_obs = df_day["tau"].values
        iv_obs = df_day[COL_IV].values

        # 记录修正前后 moneyness 分布
        if len(shift_records) < 5000 or np.random.rand() < 0.01:
            shift_records.append({
                "trade_date": date,
                "m_v1_mean": df_day["moneyness_v1"].mean(),
                "m_v1_std": df_day["moneyness_v1"].std(),
                "m_v2_mean": df_day["moneyness_v2"].mean(),
                "m_v2_std": df_day["moneyness_v2"].std(),
                "tau_mean": df_day["tau"].mean(),
                "r": df_day["r"].iloc[0] if "r" in df_day.columns else 0.0,
                "n_obs": len(df_day),
            })

        # ---------- DFW ----------
        coef = dfw_fit(m_obs, tau_obs, iv_obs)
        iv_dfw_flat = dfw_predict(coef, m_flat, tau_flat)
        iv_dfw_obs = dfw_predict(coef, m_obs, tau_obs)
        rmse_dfw_in = np.sqrt(mean_squared_error(iv_obs, iv_dfw_obs))

        # ---------- NW ----------
        # 每 20 天做一次 CV，其余用固定带宽 0.1（诊断中位数最优）
        if day_counter % 20 == 1:
            h1_best, h2_best = nw_cv_bandwidth(m_obs, tau_obs, iv_obs, h1_grid, h2_grid)
        else:
            h1_best, h2_best = FIXED_H1, FIXED_H2
        iv_nw_flat = nw_interpolate_vec(
            m_obs, tau_obs, iv_obs, m_flat, tau_flat, h1_best, h2_best
        )
        iv_nw_obs = nw_interpolate_vec(
            m_obs, tau_obs, iv_obs, m_obs, tau_obs, h1_best, h2_best
        )
        rmse_nw_in = np.sqrt(mean_squared_error(iv_obs, iv_nw_obs))

        # ---------- 修复 3: 新增 RMSE 口径 ----------
        # DFW vs NW grid (out-of-sample style)
        rmse_dfw_vs_nw = np.sqrt(mean_squared_error(iv_nw_flat, iv_dfw_flat))

        # DFW LOOCV（每 10 天做一次采样，避免全部计算）
        if day_counter % 10 == 0:
            rmse_dfw_loocv = dfw_loocv_rmse(m_obs, tau_obs, iv_obs)
        else:
            rmse_dfw_loocv = np.nan

        # ---------- 修复 2: 外推质量检查 ----------
        # 标记核权重和过低的点（远离观测的外推区）
        u = (m_flat[:, None] - m_obs[None, :]) / h1_best
        v = (tau_flat[:, None] - tau_obs[None, :]) / h2_best
        w = np.exp(-0.5 * (u * u + v * v))
        s = w.sum(axis=1)
        weight_mask = (s > KERNEL_WEIGHT_EPS * 10).reshape(N_M, N_TAU)

        # NW 网格 reshape 并填充
        iv_nw_mat = iv_nw_flat.reshape(N_M, N_TAU)
        iv_nw_filled, nan_ratio_nw = fill_missing_grid(
            iv_nw_mat.copy(), M_GRID, TAU_GRID, iv_obs, weight_mask
        )

        iv_dfw_mat = iv_dfw_flat.reshape(N_M, N_TAU)
        iv_dfw_filled, nan_ratio_dfw = fill_missing_grid(
            iv_dfw_mat.copy(), M_GRID, TAU_GRID, iv_obs, weight_mask
        )

        results.append({
            "date": date,
            "n": len(df_day),
            "r": float(df_day["r"].iloc[0]) if "r" in df_day.columns else 0.0,
            "dfw_rmse_in": rmse_dfw_in,
            "nw_rmse_in": rmse_nw_in,
            "dfw_vs_nw_rmse": rmse_dfw_vs_nw,
            "dfw_loocv_rmse": rmse_dfw_loocv,
            "h1": h1_best,
            "h2": h2_best,
            "nan_ratio_nw": nan_ratio_nw,
            "nan_ratio_dfw": nan_ratio_dfw,
        })

        grid_data.append({
            "date": date,
            "iv_dfw": iv_dfw_filled.ravel(),
            "iv_nw": iv_nw_filled.ravel(),
        })

    return pd.DataFrame(results), grid_data, pd.DataFrame(shift_records)


# ------------------------------------------------------------------
# 保存输出
# ------------------------------------------------------------------
def save_outputs_v2(
    grid_data: list[dict],
    res_df: pd.DataFrame,
    shift_df: pd.DataFrame,
) -> tuple[Path, Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. RMSE 对比表
    table1_path = OUTPUT_DIR / TABLE1_FILENAME
    res_df.to_csv(table1_path, index=False)

    # 2. 固定 154 维网格
    grid_path = OUTPUT_DIR / GRID_FILENAME
    if not grid_data:
        pd.DataFrame(columns=GRID_COLUMNS).to_parquet(grid_path, index=False)
        return table1_path, grid_path, OUTPUT_DIR / SHIFT_FILENAME

    n_grid = N_GRID
    n_days = len(grid_data)

    grid_df = pd.DataFrame({
        "trade_date": np.concatenate([np.repeat(g["date"], n_grid) for g in grid_data]),
        "grid_idx": np.tile(np.arange(n_grid), n_days),
        "m": np.tile(np.meshgrid(M_GRID, TAU_GRID, indexing="ij")[0].ravel(), n_days),
        "tau": np.tile(np.meshgrid(M_GRID, TAU_GRID, indexing="ij")[1].ravel(), n_days),
        "iv_dfw": np.concatenate([g["iv_dfw"] for g in grid_data]),
        "iv_nw": np.concatenate([g["iv_nw"] for g in grid_data]),
    })
    grid_df.to_parquet(grid_path, index=False)

    # 3. moneyness 偏移分析
    shift_path = OUTPUT_DIR / SHIFT_FILENAME
    shift_df.to_csv(shift_path, index=False)

    return table1_path, grid_path, shift_path


# ------------------------------------------------------------------
# 打印检查点
# ------------------------------------------------------------------
def _section(title: str) -> None:
    print()
    print("=" * BANNER_WIDTH)
    print(title)
    print("=" * BANNER_WIDTH)


def print_checkpoints_v2(
    df: pd.DataFrame,
    res_df: pd.DataFrame,
    shift_df: pd.DataFrame,
    grid_data: list[dict],
) -> None:
    _section("[Checkpoint 1] 数据加载（v2 含利率）")
    print(f"  - 总交易日: {df[COL_DATE].nunique()}")
    print(f"  - 每日平均合约数: {df.groupby(COL_DATE).size().mean():.1f}")
    print(f"  - moneyness_v1 范围: [{df['moneyness_v1'].min():.3f}, {df['moneyness_v1'].max():.3f}]")
    print(f"  - moneyness_v2 范围: [{df['moneyness_v2'].min():.3f}, {df['moneyness_v2'].max():.3f}]")
    print(f"  - tau 范围: [{df['tau'].min():.3f}, {df['tau'].max():.3f}] 年")
    print(f"  - 利率 r 范围: [{df['r'].min():.4f}, {df['r'].max():.4f}] ({df['r'].min()*100:.2f}% ~ {df['r'].max()*100:.2f}%)")
    print(f"  - 利率 r 均值: {df['r'].mean():.4f} ({df['r'].mean()*100:.2f}%)")

    _section("[Checkpoint 2] 网格构建（v2 固定 154 维）")
    print(f"  - m_grid 点数: {N_M}")
    print(f"  - tau_grid 点数: {N_TAU}")
    print(f"  - 总网格点: {N_GRID} (严格固定)")

    _section("[Checkpoint 3] RMSE 口径对比")
    print(f"  DFW in-sample    : {res_df['dfw_rmse_in'].mean():.4f}  (中位数 {res_df['dfw_rmse_in'].median():.4f})")
    print(f"  NW  in-sample    : {res_df['nw_rmse_in'].mean():.4f}  (中位数 {res_df['nw_rmse_in'].median():.4f})")
    print(f"  DFW vs NW grid   : {res_df['dfw_vs_nw_rmse'].mean():.4f}  (中位数 {res_df['dfw_vs_nw_rmse'].median():.4f})")
    print(f"  DFW LOOCV        : {res_df['dfw_loocv_rmse'].mean():.4f}  (中位数 {res_df['dfw_loocv_rmse'].median():.4f})")

    _section("[Checkpoint 4] 外推质量")
    print(f"  NW 网格 NaN 比例（填充前）: {res_df['nan_ratio_nw'].mean():.4f} (max={res_df['nan_ratio_nw'].max():.4f})")
    print(f"  DFW 网格 NaN 比例（填充前）: {res_df['nan_ratio_dfw'].mean():.4f} (max={res_df['nan_ratio_dfw'].max():.4f})")

    _section("[Checkpoint 5] moneyness 修正前后对比")
    delta = shift_df["m_v2_mean"] - shift_df["m_v1_mean"]
    print(f"  日均 moneyness 偏移量: mean={delta.mean():.5f}, median={delta.median():.5f}")
    print(f"  tau > 1.0 的偏移量   : mean={delta[shift_df['tau_mean'] > 1.0].mean():.5f}")
    print(f"  tau < 0.5 的偏移量   : mean={delta[shift_df['tau_mean'] < 0.5].mean():.5f}")

    _section("输出文件")
    print(f"  - {TABLE1_FILENAME}: {len(res_df)} 行")
    print(f"  - {GRID_FILENAME}: {len(grid_data) * N_GRID} 个网格点")
    print(f"  - {SHIFT_FILENAME}: {len(shift_df)} 行")
    print("=" * BANNER_WIDTH)


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main() -> None:
    print(f"[Load] 加载期权数据: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"  - 原始记录: {len(df_raw):,} 条")

    print(f"\n[Load] 加载利率数据: {RATE_PATH}")
    rate_df = load_rate_df()
    print(f"  - 利率天数: {len(rate_df):,}")
    print(f"  - 利率范围: {rate_df['r'].min():.4f} ~ {rate_df['r'].max():.4f}")

    print("\n[Preprocess v2] 引入利率 r，计算 F = S * exp(r*tau)...")
    df = preprocess_v2(df_raw, rate_df)

    print("\n[Run] 开始 Step 0 v2 插值（固定 154 维网格）...")
    res_df, grid_data, shift_df = run_step0_v2(df)

    print("\n[Save] 保存输出...")
    table1_path, grid_path, shift_path = save_outputs_v2(grid_data, res_df, shift_df)
    print(f"  - {table1_path}")
    print(f"  - {grid_path}")
    print(f"  - {shift_path}")

    print()
    print_checkpoints_v2(df, res_df, shift_df, grid_data)
    print("\n[Done] Step 0 v2 完成。")


if __name__ == "__main__":
    main()
