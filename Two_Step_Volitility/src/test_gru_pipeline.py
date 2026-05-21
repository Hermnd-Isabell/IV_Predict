# -*- coding: utf-8 -*-
"""GRU 验证快速测试：仅 SAM 特征，验证架构切换机制"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import get_step1_model, get_step2_model
from step1_features_lstm import run_step1
from step2_dnn_surface import prepare_data, run_step2, evaluate_step2

GRID_PATH = PROJECT_ROOT / "output" / "spx_step0" / "daily_grid_154.parquet"
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "spx_options_2009_2020.csv"
STEP1_DIR = PROJECT_ROOT / "output" / "spx_step1_gru"
STEP2_DIR = PROJECT_ROOT / "output" / "spx_step2_gru"

import pandas as pd
import numpy as np


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    print("=" * 50)
    print("GRU 验证快速测试: SAM + GRU + MLP")
    print("=" * 50)

    # Load grid
    print("\n[Load] Reading grid data...")
    df_grid = pd.read_parquet(GRID_PATH)
    print(f"  Records: {len(df_grid)}")

    # Step 1: SAM + GRU
    print("\n[Step 1] SAM + GRU")
    gru_class = get_step1_model("gru")
    results_sam = run_step1(
        df_grid,
        feature_type="SAM",
        model_class=gru_class,
        model_kwargs={"hidden_dim": 12, "num_layers": 1, "dropout": 0.0},
        train_kwargs={"epochs": 200, "batch_size": 128, "lr": 0.01},
        device=device,
    )

    # Save Step 1
    STEP1_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        STEP1_DIR / "sam_features.npz",
        Z=results_sam["Z"],
        Z_pred=results_sam["Z_pred_test"],
        dates=results_sam["dates"],
        dates_test=results_sam["dates_test"],
        rmse_train=results_sam["rmse_train"],
        rmse_val=results_sam["rmse_val"],
        rmse_test=results_sam["rmse_test"],
    )
    print(f"  Saved: {STEP1_DIR / 'sam_features.npz'}")

    # Step 2: MLP
    print("\n[Step 2] MLP Surface Reconstruction")
    step1_sam = STEP1_DIR / "sam_features.npz"
    step1_pca = PROJECT_ROOT / "output" / "spx_step1" / "pca_features.npz"
    step1_vae = PROJECT_ROOT / "output" / "spx_step1" / "vae_features.npz"

    data_dict = prepare_data(GRID_PATH, RAW_PATH, step1_sam, step1_pca, step1_vae)

    mlp_class = get_step2_model("mlp")
    model, history, rmse, mape, Lcal, Lbut = run_step2(
        "SAM",
        data_dict,
        model_class=mlp_class,
        model_kwargs={"hidden_dim": 50},
        output_activation="softplus",
        train_kwargs={"epochs": 20, "batch_size_days": 32, "lr": 0.001, "lambda_penalty": 1.0},
        device=device,
    )

    # Save Step 2
    STEP2_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), STEP2_DIR / "mlp_sam.pt")

    _, _, rmse_d, mape_d = evaluate_step2(model, data_dict["SAM"]["test"], device)
    np.savez(
        STEP2_DIR / "results_mlp_sam.npz",
        rmse=rmse,
        mape=mape,
        rmse_daily=np.array(rmse_d),
        mape_daily=np.array(mape_d),
        L_cal=Lcal,
        L_but=Lbut,
    )
    print(f"  Saved: {STEP2_DIR / 'results_mlp_sam.npz'}")

    # Summary
    print("\n" + "=" * 50)
    print("GRU 验证结果")
    print("=" * 50)
    print(f"  Step 1 (GRU) Test RMSE: {results_sam['rmse_test']:.6f}")
    print(f"  Step 2 (MLP) Test RMSE: {rmse:.6f}")
    print(f"  Step 2 (MLP) Test MAPE: {mape:.6f}")
    print(f"  L_cal: {Lcal:.8f}, L_but: {Lbut:.8f}")
    print("=" * 50)

    # Compare with LSTM baseline
    print("\n[对比] LSTM 基准 (2002-2007)")
    print(f"  LSTM Step 1 Test RMSE: 0.1641")
    print(f"  LSTM Step 2 Test RMSE: 0.1104")
    print(f"  GRU  Step 1 Test RMSE: {results_sam['rmse_test']:.4f}")
    print(f"  GRU  Step 2 Test RMSE: {rmse:.4f}")
    print(f"  Step 2 差异: {rmse - 0.1104:+.4f} ({(rmse/0.1104 - 1)*100:+.1f}%)")


if __name__ == "__main__":
    main()
