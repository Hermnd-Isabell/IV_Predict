# -*- coding: utf-8 -*-
"""
Step 0 插值质量分层诊断（2009-2020 数据集）

快速版：
  - DFW：逐日全局（极快，线性代数）
  - NW：仅采样 200 天（固定带宽 h=0.1），避免全部 2913 天重算
  - 分布统计：直接基于原始观测
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from step0_interpolation import (
    DAYS_PER_YEAR,
    MIN_OBS_PER_DAY,
    COL_DATE,
    COL_IV,
    COL_K,
    COL_F,
    COL_REM,
    dfw_fit,
    dfw_predict,
    nw_interpolate_vec,
    preprocess,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "spx_options_2009_2020.csv"
TABLE1_PATH = PROJECT_ROOT / "output" / "spx_step0" / "table1_nw_dfw.csv"
OUTPUT_PATH = PROJECT_ROOT / "output" / "spx_step0" / "step0_diagnosis_2009_2020.csv"

NW_H1 = 0.1
NW_H2 = 0.1
NW_SAMPLE_DAYS = 200

TAU_BINS = [0.0, 0.083, 0.25, 0.5, 1.0, 2.0, 3.0]
TAU_LABELS = ["<1m", "1-3m", "3-6m", "6-12m", "1-2y", "2-3y"]
M_BINS = [-np.inf, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, np.inf]
M_LABELS = ["<-0.2", "-0.2~-0.1", "-0.1~-0.05", "-0.05~0", "0~0.05", "0.05~0.1", "0.1~0.2", ">0.2"]

BANNER = "=" * 60


def section(title: str) -> None:
    print()
    print(BANNER)
    print(title)
    print(BANNER)


def per_obs_residuals_all_days(df: pd.DataFrame) -> pd.DataFrame:
    """全局逐日 DFW 残差（极快）。"""
    chunks: list[pd.DataFrame] = []
    grouped = df.groupby(COL_DATE, sort=True)
    for date, df_day in tqdm(grouped, total=grouped.ngroups, desc="DFW 逐日"):
        if len(df_day) < MIN_OBS_PER_DAY:
            continue
        m_obs = df_day["moneyness"].values
        tau_obs = df_day["tau"].values
        iv_obs = df_day[COL_IV].values

        coef = dfw_fit(m_obs, tau_obs, iv_obs)
        iv_dfw = dfw_predict(coef, m_obs, tau_obs)

        chunks.append(pd.DataFrame({
            "trade_date": date,
            "m": m_obs,
            "tau": tau_obs,
            "iv_obs": iv_obs,
            "res_dfw": iv_dfw - iv_obs,
        }))
    return pd.concat(chunks, ignore_index=True)


def per_obs_residuals_nw_sample(df: pd.DataFrame, n_days: int = NW_SAMPLE_DAYS) -> pd.DataFrame:
    """采样 n_days 计算 NW 残差（固定带宽 0.1）。"""
    dates = df[COL_DATE].unique()
    rng = np.random.default_rng(42)
    sample_dates = rng.choice(dates, size=min(n_days, len(dates)), replace=False)

    chunks: list[pd.DataFrame] = []
    for date in tqdm(sample_dates, desc=f"NW 采样({n_days}天)"):
        df_day = df[df[COL_DATE] == date]
        if len(df_day) < MIN_OBS_PER_DAY:
            continue
        m_obs = df_day["moneyness"].values
        tau_obs = df_day["tau"].values
        iv_obs = df_day[COL_IV].values

        iv_nw = nw_interpolate_vec(
            m_obs, tau_obs, iv_obs, m_obs, tau_obs, NW_H1, NW_H2
        )
        chunks.append(pd.DataFrame({
            "trade_date": date,
            "m": m_obs,
            "tau": tau_obs,
            "iv_obs": iv_obs,
            "res_nw": iv_nw - iv_obs,
        }))
    return pd.concat(chunks, ignore_index=True)


def stratified_rmse(residuals: pd.DataFrame, by: str, bins, labels) -> pd.DataFrame:
    residuals = residuals.copy()
    residuals["bin"] = pd.cut(residuals[by], bins=bins, labels=labels)

    def _agg(g):
        n = len(g)
        row: dict[str, object] = {
            "n_obs": n,
            "rmse_dfw": float(np.sqrt(np.mean(g["res_dfw"] ** 2))) if "res_dfw" in g.columns else np.nan,
            "mean_iv": float(g["iv_obs"].mean()),
        }
        if "res_nw" in g.columns:
            row["rmse_nw"] = float(np.sqrt(np.mean(g["res_nw"] ** 2)))
        return pd.Series(row)

    out = residuals.groupby("bin", observed=True).apply(_agg).reset_index()
    if "rmse_nw" in out.columns and "rmse_dfw" in out.columns:
        out["nw_minus_dfw"] = out["rmse_nw"] - out["rmse_dfw"]
    out["pct_obs"] = out["n_obs"] / len(residuals) * 100
    return out


def main() -> None:
    section("[Step 0 Diagnosis] 2009-2020 分层 RMSE 诊断（快速版）")
    print(f"  数据: {DATA_PATH}")
    print(f"  整体 RMSE 表: {TABLE1_PATH}")

    # 1) 整体 RMSE
    section("[1] 整体 RMSE（复现论文 Table 1）")
    daily = pd.read_csv(TABLE1_PATH)
    print(f"  交易日数: {len(daily):,}")
    print(f"  DFW mean RMSE   : {daily['dfw_rmse'].mean():.4f}   (论文 0.018)")
    print(f"  DFW median RMSE : {daily['dfw_rmse'].median():.4f}")
    print(f"  NW  mean RMSE   : {daily['nw_rmse'].mean():.4f}   (论文 0.026)")
    print(f"  NW  median RMSE : {daily['nw_rmse'].median():.4f}")
    nw_wins = (daily["nw_rmse"] < daily["dfw_rmse"]).mean() * 100
    print(f"  NW 胜出天数占比 : {nw_wins:.1f}%")

    # 2) 加载 + 预处理
    section("[2] 加载原始观测")
    df = pd.read_csv(DATA_PATH)
    df = preprocess(df)
    print(f"  原始观测点: {len(df):,}")
    print(f"  交易日数: {df[COL_DATE].nunique():,}")
    print(f"  moneyness 范围: [{df['moneyness'].min():.3f}, {df['moneyness'].max():.3f}]")
    print(f"  tau 范围: [{df['tau'].min():.3f}, {df['tau'].max():.3f}] 年")

    # 3) DFW 全局残差
    print("\n  计算全局 DFW 残差...")
    dfw_all = per_obs_residuals_all_days(df)
    print(f"  有效逐观测 DFW 残差: {len(dfw_all):,}")

    # 4) NW 采样残差
    print(f"\n  计算 NW 采样残差 ({NW_SAMPLE_DAYS} 天)...")
    nw_sample = per_obs_residuals_nw_sample(df, NW_SAMPLE_DAYS)
    print(f"  有效逐观测 NW 残差: {len(nw_sample):,}")

    # 合并（DFW 全部 + NW 采样）
    merged = dfw_all.merge(
        nw_sample[["trade_date", "m", "tau", "res_nw"]],
        on=["trade_date", "m", "tau"],
        how="left",
    )

    # 5) 按 tau 分层
    section("[3] 按 tau (到期月) 分层 RMSE")
    by_tau = stratified_rmse(merged, "tau", TAU_BINS, TAU_LABELS)
    by_tau.insert(0, "stratify", "tau")
    print(by_tau.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 6) 按 m 分层
    section("[4] 按 moneyness 分层 RMSE")
    by_m = stratified_rmse(merged, "m", M_BINS, M_LABELS)
    by_m.insert(0, "stratify", "m")
    print(by_m.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 7) 每日分布
    section("[5] 每日合约分布检查")
    daily_stats = (
        dfw_all.groupby("trade_date")
        .agg(
            n_obs=("m", "count"),
            m_min=("m", "min"),
            m_max=("m", "max"),
            m_std=("m", "std"),
            tau_min=("tau", "min"),
            tau_max=("tau", "max"),
            tau_nunique=("tau", "nunique"),
        )
        .reset_index()
    )
    print(f"  日均合约数      : {daily_stats['n_obs'].mean():.0f}")
    print(f"  日合约数 中位数 : {int(daily_stats['n_obs'].median())}")
    print(f"  日合约数 最大值 : {int(daily_stats['n_obs'].max())}")
    print(f"  m 标准差 均值   : {daily_stats['m_std'].mean():.3f}  (越大=分布越广)")
    print(f"  m 标准差 中位数 : {daily_stats['m_std'].median():.3f}")
    print(f"  m 范围 mean min : {daily_stats['m_min'].mean():.3f}")
    print(f"  m 范围 mean max : {daily_stats['m_max'].mean():.3f}")
    print(f"  tau unique 均值 : {daily_stats['tau_nunique'].mean():.1f}  (每日到期日数)")
    print(f"  tau unique 中位 : {int(daily_stats['tau_nunique'].median())}")

    # 8) 关键发现
    section("[6] 关键发现")
    worst_tau_dfw = by_tau.loc[by_tau["rmse_dfw"].idxmax()]
    densest_m = by_m.loc[by_m["n_obs"].idxmax()]
    worst_m_dfw = by_m.loc[by_m["rmse_dfw"].idxmax()]

    print(f"  - DFW RMSE 最高的 tau 区间: {worst_tau_dfw['bin']} "
          f"(RMSE={worst_tau_dfw['rmse_dfw']:.4f}, n={int(worst_tau_dfw['n_obs']):,})")
    print(f"  - 合约最密集的 m 区间    : {densest_m['bin']} "
          f"(n={int(densest_m['n_obs']):,}, 占比 {densest_m['pct_obs']:.1f}%)")
    print(f"  - DFW RMSE 最高的 m 区间 : {worst_m_dfw['bin']} "
          f"(RMSE={worst_m_dfw['rmse_dfw']:.4f}, n={int(worst_m_dfw['n_obs']):,})")

    if "rmse_nw" in by_tau.columns:
        worst_tau_nw = by_tau.loc[by_tau["rmse_nw"].idxmax()]
        max_gap_tau = by_tau.loc[by_tau["nw_minus_dfw"].abs().idxmax()]
        max_gap_m = by_m.loc[by_m["nw_minus_dfw"].abs().idxmax()]
        print(f"  - NW  RMSE 最高的 tau 区间: {worst_tau_nw['bin']} "
              f"(RMSE={worst_tau_nw['rmse_nw']:.4f}, n={int(worst_tau_nw['n_obs']):,})")
        print(f"  - DFW/NW 差距最大的 tau 区间: {max_gap_tau['bin']} "
              f"(gap={max_gap_tau['nw_minus_dfw']:+.4f})")
        print(f"  - DFW/NW 差距最大的 m 区间  : {max_gap_m['bin']} "
              f"(gap={max_gap_m['nw_minus_dfw']:+.4f})")
        print(f"  [注] NW 仅基于 {NW_SAMPLE_DAYS} 天采样，非全局统计")

    # 9) 保存
    section("[7] 保存结果")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([by_tau, by_m], ignore_index=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"  -> {OUTPUT_PATH}")
    print(f"  共 {len(combined)} 行")


if __name__ == "__main__":
    main()
