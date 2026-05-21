# -*- coding: utf-8 -*-
"""
Step 2 KAN: KAN + λ=1 无套利约束，跑 SAM 和 VAE
输出 RMSE + MAPE
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step1_features_lstm import VAE
from step2_dnn_surface_v3 import (
    prepare_data,
    evaluate_step2,
    evaluate_fixed_grid,
    check_arbitrage_violation,
    DAYS_PER_YEAR,
    BANNER_WIDTH,
)
from step2_kan_arbitrage import run_step2_kan_arbitrage
from models.surfaces.kan import KANSurface

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STEP0_GRID = PROJECT_ROOT / "output" / "spx_step0_v3" / "daily_grid_154_fixed.parquet"
STEP1_SAM = PROJECT_ROOT / "output" / "spx_step1_v3" / "sam_features.npz"
STEP1_PCA = PROJECT_ROOT / "output" / "spx_step1_v3" / "pca_features.npz"
STEP1_VAE = PROJECT_ROOT / "output" / "spx_step1_v3" / "vae_features.npz"
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "spx_options_2009_2020_clean.csv"
RATE_DATA = PROJECT_ROOT / "data" / "raw" / "rate_cleaned_2009_2020.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "spx_step2_kan"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    data_dict, grid_df = prepare_data(
        STEP0_GRID, RAW_DATA, RATE_DATA,
        STEP1_SAM, STEP1_PCA, STEP1_VAE,
    )

    results_summary = {}

    for feature_type in ["SAM", "VAE"]:
        print(f"\n{'=' * BANNER_WIDTH}")
        print(f"Step 2 KAN: {feature_type}")
        print(f"{'=' * BANNER_WIDTH}")

        model, hist, rmse, mape, Lcal, Lbut = run_step2_kan_arbitrage(
            feature_type,
            data_dict,
            model_class=KANSurface,
            model_kwargs={"hidden_dim": 16, "kan_hidden": 8, "num_layers": 2},
            output_activation="softplus",
            train_kwargs={
                "epochs": 20,
                "batch_size_days": 2,
                "lr": 0.001,
                "lambda_penalty": 1.0,
                "warmup_epochs": 10,
                "penalty_interval": 5,
            },
            device=device,
        )

        # 额外评估 fixed grid RMSE
        rmse_grid = evaluate_fixed_grid(model, data_dict[feature_type]["test"], grid_df, device)

        print(f"  Test RMSE (fixed grid) = {rmse_grid:.6f}")

        ft_lower = feature_type.lower()
        torch.save(model.state_dict(), OUTPUT_DIR / f"kan_{ft_lower}.pt")
        np.savez(
            OUTPUT_DIR / f"results_{ft_lower}.npz",
            rmse=rmse, mape=mape, rmse_grid=rmse_grid,
            L_cal=Lcal, L_but=Lbut,
            hist_train_loss=np.array(hist["train_loss"]),
            hist_val_loss=np.array(hist["val_loss"]),
            hist_mse=np.array(hist["mse"]),
            hist_pen_cal=np.array(hist["pen_cal"]),
            hist_pen_but=np.array(hist["pen_but"]),
            hist_pen_bound=np.array(hist["pen_bound"]),
            hist_lambda=np.array(hist["lambda"]),
        )

        results_summary[feature_type] = {
            "rmse": rmse, "mape": mape, "rmse_grid": rmse_grid,
            "L_cal": Lcal, "L_but": Lbut,
        }

    print(f"\n{'=' * BANNER_WIDTH}")
    print("[Checkpoint 4] KAN 模型对比")
    print(f"{'=' * BANNER_WIDTH}")
    print(f"{'Feature':<8} {'RMSE(raw)':>10} {'RMSE(grid)':>11} {'MAPE':>8} {'L_cal':>10} {'L_but':>10}")
    print("-" * 60)
    for ft, res in results_summary.items():
        print(f"{ft:<8} {res['rmse']:>10.6f} {res['rmse_grid']:>11.6f} {res['mape']:>8.6f} {res['L_cal']:>10.6f} {res['L_but']:>10.6f}")
    print("=" * 60)

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n[Done] KAN Step 2 完成，输出保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
