# -*- coding: utf-8 -*-
"""KAN + λ=1 无套利约束重新训练验证"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import get_step2_model
from step2_dnn_surface import prepare_data, evaluate_step2
from step2_kan_arbitrage import run_step2_kan_arbitrage

GRID_PATH = PROJECT_ROOT / "output" / "spx_step0" / "daily_grid_154.parquet"
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "spx_options_2009_2020.csv"
STEP1_DIR = PROJECT_ROOT / "output" / "spx_step1"
STEP2_DIR = PROJECT_ROOT / "output" / "spx_step2_gru_kan_lambda1"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    print("=" * 60)
    print("KAN + λ=1 无套利约束重新训练")
    print("=" * 60)

    step1_sam = STEP1_DIR / "sam_features.npz"
    step1_pca = PROJECT_ROOT / "output" / "spx_step1" / "pca_features.npz"
    step1_vae = PROJECT_ROOT / "output" / "spx_step1" / "vae_features.npz"

    print(f"\n[Load] GRU Step 1 features: {step1_sam}")
    data_dict = prepare_data(GRID_PATH, RAW_PATH, step1_sam, step1_pca, step1_vae)

    print("\n" + "=" * 60)
    print("Step 2: SAM + KAN (h=16, k=8, L=1, λ=1, 50ep)")
    print("=" * 60)

    kan_class = get_step2_model("kan")
    model, history, rmse, mape, L_cal, L_but = run_step2_kan_arbitrage(
        "SAM",
        data_dict,
        model_class=kan_class,
        model_kwargs={"hidden_dim": 16, "kan_hidden": 8, "num_layers": 1},
        output_activation="softplus",
        train_kwargs={
            "epochs": 50,
            "batch_size_days": 8,
            "lr": 0.001,
            "lambda_penalty": 1.0,
            "warmup_epochs": 10,
            "penalty_interval": 5,
        },
        device=device,
    )

    STEP2_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), STEP2_DIR / "kan_lambda1_sam.pt")

    _, _, rmse_d, mape_d = evaluate_step2(model, data_dict["SAM"]["test"], device)
    np.savez(
        STEP2_DIR / "results_kan_lambda1_sam.npz",
        rmse=rmse,
        mape=mape,
        rmse_daily=np.array(rmse_d),
        mape_daily=np.array(mape_d),
        L_cal=L_cal,
        L_but=L_but,
        hist_train_loss=np.array(history["train_loss"]),
        hist_val_loss=np.array(history["val_loss"]),
        hist_mse=np.array(history["mse"]),
        hist_pen_cal=np.array(history["pen_cal"]),
        hist_pen_but=np.array(history["pen_but"]),
        hist_lambda=np.array(history["lambda"]),
    )

    print("\n" + "=" * 60)
    print("KAN 无套利约束对比")
    print("=" * 60)
    print(f"{'配置':<35} {'Test RMSE':<12} {'L_cal':<12} {'L_but':<12}")
    print("-" * 60)
    print(
        f"{'KAN (λ=0, 50ep)':<35} "
        f"{'0.0898':<12} {'-0.00230':<12} {'-0.00070':<12}"
    )
    print(
        f"{'KAN (λ=1, 50ep, sparse)':<35} "
        f"{rmse:<12.4f} {L_cal:<12.6f} {L_but:<12.6f}"
    )
    print("=" * 60)

    print(f"\n[Done] 输出保存到: {STEP2_DIR}")


if __name__ == "__main__":
    main()
