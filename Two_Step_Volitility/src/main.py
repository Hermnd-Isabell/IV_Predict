# -*- coding: utf-8 -*-
"""
配置文件驱动的完整流水线入口
支持通过 JSON 配置切换 Step 1 / Step 2 的网络架构。

用法:
    python main.py                    # 使用默认配置 config.json
    python main.py --config config_transformer.json
    python main.py --config my_config.json --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# ------------------------------------------------------------------
# 导入模型注册表
# ------------------------------------------------------------------
from models import get_step1_model, get_step2_model

# ------------------------------------------------------------------
# 导入各 Step 模块（需确保路径正确）
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from step1_features_lstm import run_step1, save_results as save_step1
from step2_dnn_surface import prepare_data, run_step2

BANNER_WIDTH = 50


def load_config(config_path: str) -> dict:
    """加载 JSON 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(f"[Config] 加载配置: {config_path}")
    print(f"  描述: {config.get('description', 'N/A')}")
    return config


def resolve_paths(config: dict, project_root: Path) -> dict:
    """将配置中的相对路径解析为绝对路径。"""
    data_cfg = config.get("data", {})
    resolved = {}
    for key in ["grid_path", "raw_path", "output_step1", "output_step2"]:
        if key in data_cfg:
            p = Path(data_cfg[key])
            if not p.is_absolute():
                p = project_root / p
            resolved[key] = p
    return resolved


def run_pipeline(config: dict, device: str) -> None:
    """运行完整流水线。"""
    paths = resolve_paths(config, PROJECT_ROOT)

    grid_path = paths.get("grid_path")
    raw_path = paths.get("raw_path")
    output_step1 = paths.get("output_step1")
    output_step2 = paths.get("output_step2")

    if grid_path is None or not grid_path.exists():
        raise FileNotFoundError(
            f"Step 0 网格数据不存在: {grid_path}\n"
            f"请先运行: python src/step0_interpolation.py"
        )

    # ------------------------------------------------------------------
    # Step 1 配置解析
    # ------------------------------------------------------------------
    step1_cfg = config["step1"]
    step1_model_type = step1_cfg["model_type"]
    step1_model_class = get_step1_model(step1_model_type)
    step1_model_kwargs = step1_cfg.get("model_kwargs", {})
    step1_output_activation = step1_cfg.get("output_activation", "relu")
    step1_train_kwargs = step1_cfg.get("train_kwargs", {})

    print(f"\n{'=' * BANNER_WIDTH}")
    print(f"[Pipeline] Step 1 模型: {step1_model_type} ({step1_model_class.__name__})")
    print(f"{'=' * BANNER_WIDTH}")

    # ------------------------------------------------------------------
    # Step 2 配置解析
    # ------------------------------------------------------------------
    step2_cfg = config["step2"]
    step2_model_type = step2_cfg["model_type"]
    step2_model_class = get_step2_model(step2_model_type)
    step2_model_kwargs = step2_cfg.get("model_kwargs", {})
    step2_output_activation = step2_cfg.get("output_activation", "softplus")
    step2_train_kwargs = step2_cfg.get("train_kwargs", {})

    print(f"\n{'=' * BANNER_WIDTH}")
    print(f"[Pipeline] Step 2 模型: {step2_model_type} ({step2_model_class.__name__})")
    print(f"{'=' * BANNER_WIDTH}")

    # ------------------------------------------------------------------
    # Step 1: 特征提取 + 预测
    # ------------------------------------------------------------------
    import pandas as pd

    print(f"\n[Load] 读取网格数据: {grid_path}")
    df_grid = pd.read_parquet(grid_path)
    print(f"  记录数: {len(df_grid)}")

    step1_results = {}
    for feature_type in ["SAM", "PCA", "VAE"]:
        results = run_step1(
            df_grid,
            feature_type=feature_type,
            model_class=step1_model_class,
            model_kwargs=step1_model_kwargs,
            train_kwargs=step1_train_kwargs,
            device=device,
        )
        # 保存时覆盖 output_activation 为正确的值
        if feature_type in ("PCA", "VAE"):
            results["output_activation"] = "identity"
        else:
            results["output_activation"] = step1_output_activation
        save_step1(results, output_step1)
        step1_results[feature_type] = results

    # ------------------------------------------------------------------
    # Step 2: 曲面重构
    # ------------------------------------------------------------------
    output_step2.mkdir(parents=True, exist_ok=True)

    step1_sam = output_step1 / "sam_features.npz"
    step1_pca = output_step1 / "pca_features.npz"
    step1_vae = output_step1 / "vae_features.npz"

    data_dict = prepare_data(grid_path, raw_path, step1_sam, step1_pca, step1_vae)

    results_summary = {}
    for feature_type in ["SAM", "PCA", "VAE"]:
        model, history, rmse, mape, Lcal, Lbut = run_step2(
            feature_type,
            data_dict,
            model_class=step2_model_class,
            model_kwargs=step2_model_kwargs,
            output_activation=step2_output_activation,
            train_kwargs=step2_train_kwargs,
            device=device,
        )

        # 保存模型和结果
        ft_lower = feature_type.lower()
        torch.save(model.state_dict(), output_step2 / f"{step2_model_type}_{ft_lower}.pt")

        # 重新评估获取每日数据
        from step2_dnn_surface import evaluate_step2

        _, _, rmse_d, mape_d = evaluate_step2(model, data_dict[feature_type]["test"], device)
        import numpy as np

        np.savez(
            output_step2 / f"results_{step2_model_type}_{ft_lower}.npz",
            rmse=rmse,
            mape=mape,
            rmse_daily=np.array(rmse_d),
            mape_daily=np.array(mape_d),
            L_cal=Lcal,
            L_but=Lbut,
            model_type=step2_model_type,
            hist_train_loss=np.array(history["train_loss"]),
            hist_val_loss=np.array(history["val_loss"]),
            hist_mse=np.array(history["mse"]),
            hist_pen_cal=np.array(history["pen_cal"]),
            hist_pen_but=np.array(history["pen_but"]),
            hist_pen_bound=np.array(history["pen_bound"]),
        )

        results_summary[feature_type] = {
            "rmse": rmse,
            "mape": mape,
            "L_cal": Lcal,
            "L_but": Lbut,
        }

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    print(f"\n{'=' * BANNER_WIDTH}")
    print("[Pipeline] 最终结果汇总")
    print(f"{'=' * BANNER_WIDTH}")
    print(f"  Step 1: {step1_model_type} | Step 2: {step2_model_type}")
    print(f"{'-' * BANNER_WIDTH}")
    for ft, res in results_summary.items():
        print(
            f"  {ft:6s}: RMSE={res['rmse']:.6f}, MAPE={res['mape']:.6f}, "
            f"L_cal={res['L_cal']:.8f}, L_but={res['L_but']:.8f}"
        )
    print(f"{'=' * BANNER_WIDTH}")
    print(f"\n[Done] 输出保存到:\n  Step 1: {output_step1}\n  Step 2: {output_step2}")


def main():
    parser = argparse.ArgumentParser(
        description="Two-Step Volatility Surface Modeling — 配置驱动流水线"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="配置文件路径 (默认: config.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="计算设备 (默认: auto)",
    )
    args = parser.parse_args()

    # 设备选择
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[Device] {device}")

    # 加载配置
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    config = load_config(str(config_path))

    # 运行流水线
    run_pipeline(config, device)


if __name__ == "__main__":
    main()
