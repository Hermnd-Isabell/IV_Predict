# -*- coding: utf-8 -*-
"""KAN 长时训练验证：单层 h=16，跑 50 epochs，验证收敛极限"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import get_step2_model
from step2_dnn_surface import prepare_data, run_step2, evaluate_step2

GRID_PATH = PROJECT_ROOT / "output" / "spx_step0" / "daily_grid_154.parquet"
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "spx_options.csv"
STEP1_GRU_DIR = PROJECT_ROOT / "output" / "spx_step1_gru"
STEP2_KAN_DIR = PROJECT_ROOT / "output" / "spx_step2_gru_kan_long"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    print("=" * 60)
    print("KAN 长时训练: hidden=16, kan_hidden=8, L=1, 50 epochs")
    print("=" * 60)

    step1_sam = STEP1_GRU_DIR / "sam_features.npz"
    step1_pca = PROJECT_ROOT / "output" / "spx_step1" / "pca_features.npz"
    step1_vae = PROJECT_ROOT / "output" / "spx_step1" / "vae_features.npz"

    print(f"\n[Load] GRU Step 1 features: {step1_sam}")
    data_dict = prepare_data(GRID_PATH, RAW_PATH, step1_sam, step1_pca, step1_vae)

    print("\n" + "=" * 60)
    print("Step 2: SAM + KAN (h=16, k=8, L=1, 50ep)")
    print("=" * 60)
    kan_class = get_step2_model("kan")
    model, history, rmse, mape, Lcal, Lbut = run_step2(
        "SAM",
        data_dict,
        model_class=kan_class,
        model_kwargs={"hidden_dim": 16, "kan_hidden": 8, "num_layers": 1},
        output_activation="softplus",
        train_kwargs={"epochs": 50, "batch_size_days": 8, "lr": 0.001, "lambda_penalty": 0.0},
        device=device,
    )

    STEP2_KAN_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), STEP2_KAN_DIR / "kan_long_sam.pt")

    _, _, rmse_d, mape_d = evaluate_step2(model, data_dict["SAM"]["test"], device)
    np.savez(
        STEP2_KAN_DIR / "results_kan_long_sam.npz",
        rmse=rmse, mape=mape,
        rmse_daily=np.array(rmse_d), mape_daily=np.array(mape_d),
        L_cal=Lcal, L_but=Lbut,
    )

    print("\n" + "=" * 60)
    print("KAN 全规模对比")
    print("=" * 60)
    print(f"{'配置':<30} {'Test RMSE':<12} {'Test MAPE':<12}")
    print("-" * 60)
    print(f"{'LSTM + MLP (基准)':<30} {'0.1104':<12} {'27.65%':<12}")
    print(f"{'GRU  + MLP':<30} {'0.1102':<12} {'27.72%':<12}")
    print(f"{'GRU  + ResNet':<30} {'0.1130':<12} {'27.32%':<12}")
    print(f"{'GRU  + KAN (2ep, h=8)':<30} {'0.0978':<12} {'24.91%':<12}")
    print(f"{'GRU  + KAN (20ep, h=16)':<30} {'0.1041':<12} {'25.83%':<12}")
    print(f"{'GRU  + KAN (50ep, h=16)':<30} {rmse:<12.4f} {mape:<12.2%}")
    print("=" * 60)

    print(f"\n[Done] 输出保存到: {STEP2_KAN_DIR}")


if __name__ == "__main__":
    main()
