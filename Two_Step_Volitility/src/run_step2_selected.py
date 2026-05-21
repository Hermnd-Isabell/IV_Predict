# -*- coding: utf-8 -*-
"""
Step 2: 用 DFW+VAE 和 NW+SAM 两种组合跑 DNN 曲面重构
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

from models.surfaces.mlp import MLPSurface as DNN_Surface

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step1_features_lstm import VAE
from step2_dnn_surface_v3 import (
    compute_arbitrage_penalties,
    check_arbitrage_violation,
    evaluate_step2,
    evaluate_fixed_grid,
    map_features_to_f,
    train_step2,
    run_step2,
    DAYS_PER_YEAR,
    BANNER_WIDTH,
    TRAIN_RATIO,
    VAL_RATIO,
)

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STEP0_GRID = PROJECT_ROOT / "output" / "spx_step0_v2" / "daily_grid_154_fixed.parquet"
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "spx_options_2009_2020.csv"
RATE_DATA = PROJECT_ROOT / "data" / "raw" / "rate_cleaned_2009_2020.csv"
STEP1_DIR = PROJECT_ROOT / "output" / "spx_step1_all6"
OUTPUT_DIR = PROJECT_ROOT / "output" / "spx_step2_selected"

DNN_LAMBDA = 1.0


def prepare_data_selected(
    grid_path,
    raw_path,
    rate_path,
    step1_dir,
    feature_type,      # "SAM" | "VAE"
    step0_method,      # "dfw" | "nw"
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
):
    """
    加载指定组合的 Step 0 网格和 Step 1 特征。
    step0_method: 决定用 iv_dfw 还是 iv_nw 作为 ground truth
    """
    gt_col = "iv_dfw" if step0_method == "dfw" else "iv_nw"
    print(f"[Data] 读取网格数据 (ground truth: {gt_col})...")
    grid_df = pd.read_parquet(grid_path)
    grid_dates = np.array(sorted(grid_df["trade_date"].unique()))
    n_days = len(grid_dates)
    n_train = int(n_days * train_ratio)
    n_val = int(n_days * val_ratio)

    train_dates = grid_dates[:n_train]
    val_dates = grid_dates[n_train : n_train + n_val]
    test_dates = grid_dates[n_train + n_val :]
    print(f"  总交易日: {n_days}, Train: {len(train_dates)}, Val: {len(val_dates)}, Test: {len(test_dates)}")

    df_grid_pivot = grid_df.pivot(index="trade_date", columns="grid_idx", values=gt_col)
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

    # 加载 Step 1 特征
    ft_lower = feature_type.lower()
    step1_path = step1_dir / f"{step0_method}_{ft_lower}" / f"{ft_lower}_features.npz"
    step1_data = np.load(step1_path)
    Z_all = step1_data["Z"].astype(np.float64)
    Z_pred = step1_data["Z_pred"].astype(np.float64)

    if feature_type == "SAM":
        F_all = map_features_to_f(Z_all, "SAM")
        F_test = map_features_to_f(Z_pred, "SAM")
        date_to_f_all = dict(zip(grid_dates, F_all))
        date_to_f_test = dict(zip(test_dates, F_test))
        dataset = {
            "train": _build_day_list(train_dates, date_to_f_all),
            "val": _build_day_list(val_dates, date_to_f_all),
            "test": _build_day_list(test_dates, date_to_f_test),
        }
        vae_model = None
    elif feature_type == "VAE":
        vae_model = VAE(input_dim=F_real.shape[1], hidden_dim=128, latent_dim=10)
        vae_pt_path = step1_dir / f"{step0_method}_{ft_lower}" / f"{ft_lower}_model.pt"
        vae_model.load_state_dict(torch.load(vae_pt_path, map_location="cpu"))
        vae_model.eval()
        F_all = map_features_to_f(Z_all, "VAE", vae_model=vae_model)
        F_test = map_features_to_f(Z_pred, "VAE", vae_model=vae_model)
        date_to_f_all = dict(zip(grid_dates, F_all))
        date_to_f_test = dict(zip(test_dates, F_test))
        dataset = {
            "train": _build_day_list(train_dates, date_to_f_all),
            "val": _build_day_list(val_dates, date_to_f_all),
            "test": _build_day_list(test_dates, date_to_f_test),
        }
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    return {
        "dataset": dataset,
        "n_grid": F_real.shape[1],
        "vae_model": vae_model,
    }, grid_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    combos = [
        ("DFW", "VAE", "dfw"),
        ("NW", "SAM", "nw"),
    ]

    results_summary = {}

    for label, feature_type, step0_method in combos:
        print(f"\n{'=' * 60}")
        print(f"Running Step 2: {label} + {feature_type}")
        print(f"{'=' * 60}")

        data_dict, grid_df = prepare_data_selected(
            STEP0_GRID, RAW_DATA, RATE_DATA, STEP1_DIR,
            feature_type, step0_method,
        )

        model, hist, rmse, mape, rmse_grid, Lcal, Lbut = run_step2(
            feature_type,
            {"SAM": data_dict["dataset"], "PCA": data_dict["dataset"], "VAE": data_dict["dataset"],
             "n_grid": data_dict["n_grid"]},
            grid_df,
            device=device,
        )

        # 保存结果
        ft_lower = feature_type.lower()
        label_lower = f"{step0_method}_{ft_lower}"
        torch.save(model.state_dict(), OUTPUT_DIR / f"dnn_{label_lower}.pt")
        np.savez(
            OUTPUT_DIR / f"results_{label_lower}.npz",
            rmse=rmse, mape=mape, rmse_grid=rmse_grid,
            L_cal=Lcal, L_but=Lbut,
            hist_train_loss=np.array(hist["train_loss"]),
            hist_val_loss=np.array(hist["val_loss"]),
            hist_mse=np.array(hist["mse"]),
            hist_pen_cal=np.array(hist["pen_cal"]),
            hist_pen_but=np.array(hist["pen_but"]),
            hist_pen_bound=np.array(hist["pen_bound"]),
        )

        results_summary[label] = {
            "feature_type": feature_type,
            "step0_method": step0_method,
            "rmse": rmse,
            "mape": mape,
            "rmse_grid": rmse_grid,
            "L_cal": Lcal,
            "L_but": Lbut,
        }

        print(f"\n  [{label}] Done: RMSE={rmse:.6f}, MAPE={mape:.6f}, Grid={rmse_grid:.6f}, Lcal={Lcal:.8f}, Lbut={Lbut:.8f}")

    # 汇总
    print(f"\n{'=' * 60}")
    print("Step 2 汇总")
    print(f"{'=' * 60}")
    print(f"{'组合':<12} {'RMSE(raw)':>10} {'RMSE(grid)':>11} {'MAPE':>8} {'L_cal':>10} {'L_but':>10}")
    print("-" * 60)
    for label, res in results_summary.items():
        print(f"{label:<12} {res['rmse']:>10.6f} {res['rmse_grid']:>11.6f} {res['mape']:>8.6f} {res['L_cal']:>10.6f} {res['L_but']:>10.6f}")
    print("=" * 60)

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n[Done] 输出保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
