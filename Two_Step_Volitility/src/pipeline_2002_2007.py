# -*- coding: utf-8 -*-
"""流水线：2002-2007 SPX 数据三步一键跑"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent

def run(script_name: str) -> None:
    script = SRC_DIR / script_name
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, str(script)], cwd=str(SRC_DIR.parent))
    if result.returncode != 0:
        print(f"FAILED: {script_name} exited with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"DONE: {script_name}")

def main() -> None:
    # Step 0: 插值
    run("step0_interpolation.py")
    # Step 1: 特征提取 + LSTM
    run("step1_features_lstm.py")
    # Step 2: DNN 曲面重构
    run("step2_dnn_surface.py")
    # Figure 7 分析
    run("analyze_daily_rmse_figure7.py")
    print("\n" + "="*60)
    print("ALL STEPS COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()
