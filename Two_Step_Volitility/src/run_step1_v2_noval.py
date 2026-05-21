# -*- coding: utf-8 -*-
"""
Step 1 v2 (no val): 使用 Step 0 v2 网格重新生成 SAM/PCA/VAE 特征
只划分训练集和测试集，不保留验证集
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
OUTPUT_DIR = PROJECT_ROOT / "output" / "spx_step1_v2_noval"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    print(f"[Load] Grid v2: {GRID_PATH}")
    df_grid = pd.read_parquet(GRID_PATH)
    print(f"  Rows: {len(df_grid):,}, Days: {df_grid['trade_date'].nunique()}")

    # SAM — 无验证集
    results_sam = run_step1(df_grid, feature_type="SAM", device=device, val_ratio=0.0)
    save_results(results_sam, OUTPUT_DIR)

    # PCA — 无验证集
    results_pca = run_step1(df_grid, feature_type="PCA", device=device, val_ratio=0.0)
    save_results(results_pca, OUTPUT_DIR)

    # VAE — 无验证集
    results_vae = run_step1(df_grid, feature_type="VAE", device=device, val_ratio=0.0)
    save_results(results_vae, OUTPUT_DIR)

    print(f"\n[Done] Step 1 v2 (no val) 输出保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
