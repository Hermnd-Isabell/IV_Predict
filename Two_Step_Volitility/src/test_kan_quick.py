# -*- coding: utf-8 -*-
"""KAN 快速验证：5 epochs，确认训练可行并获取初步 RMSE"""
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
STEP2_KAN_DIR = PROJECT_ROOT / "output" / "spx_step2_gru_kan"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    print("=" * 60)
    print("KAN 快速验证 (2 epochs, batch_days=8, 超小模型)")
    print("=" * 60)

    step1_sam = STEP1_GRU_DIR / "sam_features.npz"
    step1_pca = PROJECT_ROOT / "output" / "spx_step1" / "pca_features.npz"
    step1_vae = PROJECT_ROOT / "output" / "spx_step1" / "vae_features.npz"

    print(f"\n[Load] GRU Step 1 features: {step1_sam}")
    data_dict = prepare_data(GRID_PATH, RAW_PATH, step1_sam, step1_pca, step1_vae)

    print("\n" + "=" * 60)
    print("Step 2: SAM + KAN (2ep, h=8/k=4/L=1, no-arb OFF)")
    print("=" * 60)
    kan_class = get_step2_model("kan")
    model, history, rmse, mape, Lcal, Lbut = run_step2(
        "SAM",
        data_dict,
        model_class=kan_class,
        model_kwargs={"hidden_dim": 8, "kan_hidden": 4, "num_layers": 1},
        output_activation="softplus",
        train_kwargs={"epochs": 2, "batch_size_days": 8, "lr": 0.001, "lambda_penalty": 0.0},
        device=device,
    )

    STEP2_KAN_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), STEP2_KAN_DIR / "kan_sam_quick.pt")

    _, _, rmse_d, mape_d = evaluate_step2(model, data_dict["SAM"]["test"], device)
    np.savez(
        STEP2_KAN_DIR / "results_kan_sam_quick.npz",
        rmse=rmse, mape=mape,
        rmse_daily=np.array(rmse_d), mape_daily=np.array(mape_d),
        L_cal=Lcal, L_but=Lbut,
    )

    print("\n" + "=" * 60)
    print("四种配置 Step 2 对比")
    print("=" * 60)
    print(f"{'配置':<25} {'Test RMSE':<12} {'Test MAPE':<12} {'L_cal':<8} {'L_but':<8}")
    print("-" * 60)
    print(f"{'LSTM + MLP (基准)':<25} {'0.1104':<12} {'27.65%':<12} {'0':<8} {'0':<8}")
    print(f"{'GRU  + MLP':<25} {'0.1102':<12} {'27.72%':<12} {'0':<8} {'0':<8}")
    print(f"{'GRU  + ResNet':<25} {'0.1130':<12} {'27.32%':<12} {'~0':<8} {'~0':<8}")
    print(f"{'GRU  + KAN (2ep)':<25} {rmse:<12.4f} {mape:<12.2%} {Lcal:<8.6f} {Lbut:<8.6f}")
    print("=" * 60)

    print(f"\n[Done] 输出保存到: {STEP2_KAN_DIR}")


if __name__ == "__main__":
    main()
