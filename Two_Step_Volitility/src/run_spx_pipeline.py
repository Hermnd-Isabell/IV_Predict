# -*- coding: utf-8 -*-
"""
SPX 数据全链路运行脚本
自动执行 Step 0 (插值) + Step 1 (特征提取+LSTM)
"""

from pathlib import Path

import numpy as np
import pandas as pd

# 修改 step0 和 step1 的导入路径
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from step0_interpolation import (
    preprocess,
    run_step0,
    save_outputs as save_step0,
    print_checkpoints as print_step0,
    DATA_PATH as STEP0_DATA,
    OUTPUT_DIR as STEP0_OUT,
)
from step1_features_lstm import (
    run_step1,
    save_results as save_step1,
    INPUT_GRID as STEP1_INPUT,
    OUTPUT_DIR as STEP1_OUT,
)


def main() -> None:
    spx_data = Path(__file__).resolve().parent.parent / "data" / "raw" / "spx_options.csv"
    spx_out0 = Path(__file__).resolve().parent.parent / "output" / "spx_step0"
    spx_out1 = Path(__file__).resolve().parent.parent / "output" / "spx_step1"

    # Monkey-patch 数据路径
    import step0_interpolation
    import step1_features_lstm

    step0_interpolation.DATA_PATH = spx_data
    step0_interpolation.OUTPUT_DIR = spx_out0
    step1_features_lstm.INPUT_GRID = spx_out0 / "daily_grid_154.parquet"
    step1_features_lstm.OUTPUT_DIR = spx_out1

    # ====== Step 0 ======
    print(f"[SPX Pipeline] 加载数据: {spx_data}")
    df_raw = pd.read_csv(spx_data)
    print(f"  原始记录: {len(df_raw)} 条")

    df = preprocess(df_raw)
    print("\n[SPX Pipeline] 开始 Step 0 插值...")
    res_df, grid_data, tau_grid = run_step0(df)

    print("\n[SPX Pipeline] 保存 Step 0 输出...")
    table1_path, grid_path = save_step0(grid_data, res_df, tau_grid)
    print(f"  - {table1_path}")
    print(f"  - {grid_path}")

    print()
    print_step0(df, tau_grid, res_df, grid_data)

    # ====== Step 1 ======
    print("\n[SPX Pipeline] 开始 Step 1 特征提取...")
    df_grid = pd.read_parquet(grid_path)
    n_grid = df_grid["grid_idx"].nunique()
    print(f"  网格维度: {n_grid}")

    results_sam = run_step1(df_grid, "SAM")
    save_step1(results_sam, spx_out1)

    results_pca = run_step1(df_grid, "PCA")
    save_step1(results_pca, spx_out1)

    results_vae = run_step1(df_grid, "VAE", vae_latent_dim=10)
    save_step1(results_vae, spx_out1)

    print("\n[SPX Pipeline] 全部完成！")


if __name__ == "__main__":
    main()
