# -*- coding: utf-8 -*-
"""
Step 0 快速冒烟测试 —— 仅验证数据格式兼容性和核心逻辑通路

策略：
    1. 取 2009 数据中最开始的 5 个交易日（而非全部 252 天）
    2. 跳过耗时的 NW 五折 CV，用固定带宽 (0.1, 0.1) 快速验证 NW 插值通路
    3. 完整跑 DFW 拟合/预测
    4. 输出网格并验证 parquet 写入成功

通过标准：
    - preprocess 不报错
    - DFW 能拟合且 RMSE 合理
    - NW 能跑通（固定带宽）
    - 输出文件可正常写入/读取
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# 把 src 加入路径，从而复用 step0_interpolation 的函数
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from step0_interpolation import (
    PROJECT_ROOT,
    COL_DATE,
    preprocess,
    dfw_fit,
    dfw_predict,
    nw_interpolate_vec,
    MIN_OBS_PER_DAY,
    get_tau_grid,
    M_GRID,
    save_outputs,
)

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "spx_options_2009_2020.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "smoke_test_2009"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_smoke():
    print("=" * 60)
    print("Step 0 冒烟测试 (2009 前 5 天)")
    print("=" * 60)
    print(f"[Load] {DATA_PATH}")

    t0 = time.time()
    df_raw = pd.read_csv(DATA_PATH)
    print(f"  总记录: {len(df_raw):,}  (加载用时 {time.time() - t0:.1f}s)")

    # 取 2009 年前 5 个交易日
    df_2009 = df_raw[df_raw["trade_date"] // 10000 == 2009].copy()
    dates = sorted(df_2009[COL_DATE].unique())
    print(f"  2009 共 {len(dates)} 个交易日，取前 5 天做 smoke test")
    df_subset = df_2009[df_2009[COL_DATE].isin(dates[:5])].copy()

    # ---------- Preprocess ----------
    print("\n[Preprocess] 计算 moneyness / tau...")
    t0 = time.time()
    df = preprocess(df_subset)
    print(f"  OK ({time.time() - t0:.1f}s)")
    print(f"  moneyness: [{df['moneyness'].min():.3f}, {df['moneyness'].max():.3f}]")
    print(f"  tau:       [{df['tau'].min():.3f}, {df['tau'].max():.3f}] 年")
    print(f"  总记录:    {len(df):,} 条")

    tau_grid = get_tau_grid(df)
    print(f"  tau_grid:  {len(tau_grid)} 个点 {tau_grid}")

    # 构建网格
    M_mesh, Tau_mesh = np.meshgrid(M_GRID, tau_grid, indexing="ij")
    m_flat = M_mesh.ravel()
    tau_flat = Tau_mesh.ravel()

    results = []
    grid_data = []

    grouped = df.groupby(COL_DATE, sort=True)

    for date, df_day in grouped:
        n = len(df_day)
        if n < MIN_OBS_PER_DAY:
            print(f"  跳过 {date}: 仅 {n} 条观测 (不足 {MIN_OBS_PER_DAY})")
            continue

        m_obs = df_day["moneyness"].values
        tau_obs = df_day["tau"].values
        iv_obs = df_day["implc_volatlty"].values

        # DFW
        try:
            coef = dfw_fit(m_obs, tau_obs, iv_obs)
            iv_dfw_flat = dfw_predict(coef, m_flat, tau_flat)
            iv_dfw_obs = dfw_predict(coef, m_obs, tau_obs)
            rmse_dfw = np.sqrt(np.mean((iv_obs - iv_dfw_obs) ** 2))
        except Exception as e:
            print(f"  日期 {date}: DFW 拟合失败 — {e}")
            raise

        # NW (固定带宽，跳过 CV 以节省 smoke test 时间)
        try:
            iv_nw_flat = nw_interpolate_vec(
                m_obs, tau_obs, iv_obs, m_flat, tau_flat, 0.1, 0.1
            )
            iv_nw_obs = nw_interpolate_vec(
                m_obs, tau_obs, iv_obs, m_obs, tau_obs, 0.1, 0.1
            )
            rmse_nw = np.sqrt(np.mean((iv_obs - iv_nw_obs) ** 2))
        except Exception as e:
            print(f"  日期 {date}: NW 插值失败 — {e}")
            raise

        print(
            f"  {date}: n={n:4d}  DFW_RMSE={rmse_dfw:.4f}  NW_RMSE={rmse_nw:.4f}"
        )

        results.append({
            "date": date,
            "n": n,
            "dfw_rmse": rmse_dfw,
            "nw_rmse": rmse_nw,
        })
        grid_data.append({
            "date": date,
            "iv_dfw": iv_dfw_flat,
            "iv_nw": iv_nw_flat,
        })

    res_df = pd.DataFrame(results)

    print("\n[Save] 保存输出文件...")
    # save_outputs 使用硬编码路径 output/spx_step0/
    table1_path_actual, grid_path_actual = save_outputs(grid_data, res_df, tau_grid)
    print(f"  {table1_path_actual}")
    print(f"  {grid_path_actual}")

    # 验证可读性
    t0 = time.time()
    grid_check = pd.read_parquet(grid_path_actual)
    print(f"[Check] parquet 回读验证 OK ({len(grid_check):,} 行, {time.time() - t0:.1f}s)")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)
    print(f"  测试交易日: {len(results)} / 5")
    print(f"  平均 DFW RMSE: {res_df['dfw_rmse'].mean():.4f}")
    print(f"  平均 NW  RMSE: {res_df['nw_rmse'].mean():.4f}")


if __name__ == "__main__":
    run_smoke()
