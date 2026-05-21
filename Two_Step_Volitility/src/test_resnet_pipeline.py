# -*- coding: utf-8 -*-
"""ResNet 验证快速测试：复用 GRU Step 1，只跑 Step 2 ResNet"""
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
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "spx_options_2009_2020.csv"
STEP1_GRU_DIR = PROJECT_ROOT / "output" / "spx_step1_gru"
STEP2_RESNET_DIR = PROJECT_ROOT / "output" / "spx_step2_gru_resnet"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    print("=" * 60)
    print("ResNet 验证: 复用 GRU Step 1, 只跑 Step 2 ResNet")
    print("=" * 60)

    # 复用已有的 GRU Step 1 特征
    step1_sam = STEP1_GRU_DIR / "sam_features.npz"
    step1_pca = PROJECT_ROOT / "output" / "spx_step1" / "pca_features.npz"
    step1_vae = PROJECT_ROOT / "output" / "spx_step1" / "vae_features.npz"

    print(f"\n[Load] GRU Step 1 features: {step1_sam}")

    # 准备数据
    data_dict = prepare_data(GRID_PATH, RAW_PATH, step1_sam, step1_pca, step1_vae)

    # 跑 SAM + ResNet
    print("\n" + "=" * 60)
    print("Step 2: SAM + ResNet")
    print("=" * 60)
    resnet_class = get_step2_model("resnet")
    model, history, rmse, mape, Lcal, Lbut = run_step2(
        "SAM",
        data_dict,
        model_class=resnet_class,
        model_kwargs={"hidden_dims": [64, 64, 64], "num_blocks": 3, "activation": "tanh"},
        output_activation="softplus",
        train_kwargs={"epochs": 20, "batch_size_days": 32, "lr": 0.001, "lambda_penalty": 1.0},
        device=device,
    )

    # 保存
    STEP2_RESNET_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), STEP2_RESNET_DIR / "resnet_sam.pt")

    _, _, rmse_d, mape_d = evaluate_step2(model, data_dict["SAM"]["test"], device)
    np.savez(
        STEP2_RESNET_DIR / "results_resnet_sam.npz",
        rmse=rmse, mape=mape,
        rmse_daily=np.array(rmse_d), mape_daily=np.array(mape_d),
        L_cal=Lcal, L_but=Lbut,
    )

    # 汇总对比
    print("\n" + "=" * 60)
    print("三种配置 Step 2 对比")
    print("=" * 60)
    print(f"{'配置':<25} {'Test RMSE':<12} {'Test MAPE':<12} {'L_cal':<8} {'L_but':<8}")
    print("-" * 60)
    print(f"{'LSTM + MLP (基准)':<25} {'0.1104':<12} {'27.65%':<12} {'0':<8} {'0':<8}")
    print(f"{'GRU  + MLP':<25} {'0.1102':<12} {'27.72%':<12} {'0':<8} {'0':<8}")
    print(f"{'GRU  + ResNet':<25} {rmse:<12.4f} {mape:<12.2%} {Lcal:<8.6f} {Lbut:<8.6f}")
    print("=" * 60)

    # 改进幅度
    print("\n改进幅度:")
    print(f"  GRU  + ResNet vs GRU  + MLP:  RMSE {(rmse/0.1102 - 1)*100:+.1f}%")
    print(f"  GRU  + ResNet vs LSTM + MLP:  RMSE {(rmse/0.1104 - 1)*100:+.1f}%")

    if rmse < 0.10:
        print("\n[结论] ResNet 显著改善 (< 0.10) - 情景 A")
    elif rmse < 0.110:
        print("\n[结论] ResNet 轻微改善 (0.105-0.110) - 情景 B")
    else:
        print("\n[结论] ResNet 无改善 (>= 0.110) - 情景 C")

    print(f"\n[Done] 输出保存到: {STEP2_RESNET_DIR}")


if __name__ == "__main__":
    main()
