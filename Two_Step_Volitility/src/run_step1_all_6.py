# -*- coding: utf-8 -*-
"""
Step 1: 跑全 6 种组合
  Step0: DFW / NW  ×  Step1: SAM / PCA / VAE
只划分训练集和测试集 (无验证集)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from step1_features_lstm import run_step1, save_results

GRID_PATH = PROJECT_ROOT / "output" / "spx_step0_v2" / "daily_grid_154_fixed.parquet"
OUTPUT_DIR = PROJECT_ROOT / "output" / "spx_step1_all6"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    print(f"[Load] Grid v2: {GRID_PATH}")
    df_grid = pd.read_parquet(GRID_PATH)
    print(f"  Rows: {len(df_grid):,}, Days: {df_grid['trade_date'].nunique()}")

    combos = [
        ("DFW", "SAM", "iv_dfw"),
        ("DFW", "PCA", "iv_dfw"),
        ("DFW", "VAE", "iv_dfw"),
        ("NW",  "SAM", "iv_nw"),
        ("NW",  "PCA", "iv_nw"),
        ("NW",  "VAE", "iv_nw"),
    ]

    summary = []
    for step0_name, feat_name, iv_col in combos:
        results = run_step1(
            df_grid,
            feature_type=feat_name,
            device=device,
            val_ratio=0.0,
            iv_col=iv_col,
        )
        # 保存到子目录
        sub_dir = OUTPUT_DIR / f"{step0_name.lower()}_{feat_name.lower()}"
        save_results(results, sub_dir)

        summary.append({
            "step0": step0_name,
            "step1": feat_name,
            "train_rmse": results["rmse_train"],
            "test_rmse": results["rmse_test"],
        })
        print()

    # 汇总表
    print("=" * 60)
    print("6 种组合汇总 (无验证集, 75% Train / 25% Test)")
    print("=" * 60)
    print(f"{'组合':<12} {'Train RMSE':>12} {'Test RMSE':>12}")
    print("-" * 60)
    for row in summary:
        label = f"{row['step0']}+{row['step1']}"
        print(f"{label:<12} {row['train_rmse']:>12.6f} {row['test_rmse']:>12.6f}")
    print("=" * 60)

    # 保存 CSV
    import csv
    csv_path = OUTPUT_DIR / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step0", "step1", "train_rmse", "test_rmse"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"\n[Done] 汇总保存到: {csv_path}")


if __name__ == "__main__":
    main()
