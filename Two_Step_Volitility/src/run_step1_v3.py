# -*- coding: utf-8 -*-
"""
Step 1 v3: 使用 Step 0 v3 网格重新生成 SAM/PCA/VAE 特征
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from step1_features_lstm import run_step1, save_results

GRID_PATH = PROJECT_ROOT / "output" / "spx_step0_v3" / "daily_grid_154_fixed.parquet"
OUTPUT_DIR = PROJECT_ROOT / "output" / "spx_step1_v3"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    print(f"[Load] Grid v3: {GRID_PATH}")
    df_grid = pd.read_parquet(GRID_PATH)
    print(f"  Rows: {len(df_grid):,}, Days: {df_grid['trade_date'].nunique()}")

    # SAM
    results_sam = run_step1(df_grid, feature_type="SAM", device=device)
    save_results(results_sam, OUTPUT_DIR)

    # PCA
    results_pca = run_step1(df_grid, feature_type="PCA", device=device)
    save_results(results_pca, OUTPUT_DIR)

    # VAE
    results_vae = run_step1(df_grid, feature_type="VAE", device=device)
    save_results(results_vae, OUTPUT_DIR)

    print(f"\n[Done] Step 1 v3 输出保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
