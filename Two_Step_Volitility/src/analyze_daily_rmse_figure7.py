# -*- coding: utf-8 -*-
"""
复现论文 Figure 7 —— 每日 RMSE/MAPE 时间序列分析
基于 Zhang et al. (2023) Section 4.3 的评估方法
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ------------------------------------------------------------------
# 路径配置
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEP1_DIR = PROJECT_ROOT / "output" / "spx_step1"
STEP2_DIR = PROJECT_ROOT / "output" / "spx_step2"
OUTPUT_DIR = STEP2_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BANNER_WIDTH = 50

# 历史事件表
EVENTS = {
    "2002-07-10": "WorldCom 破产（美国最大企业破产案）",
    "2002-08-05": "VIX 飙升至 45+",
    "2002-10-09": "SPX 跌至 776（熊市低点）",
    "2003-03-20": "伊拉克战争爆发",
    "2003-05-01": "布什'任务完成'演讲",
    "2003-07-15": "伊拉克战争后 VIX 持续回落",
    "2003-10-27": "SPX 突破 1050",
    "2004-03-20": "马德里爆炸案",
    "2004-05-01": "Abu Ghraib 丑闻曝光",
    "2004-06-30": "伊拉克主权移交",
    "2004-07-30": "9/11 委员会报告发布",
    "2004-10-08": "SPX 突破 1120",
    "2004-11-02": "美国总统大选（布什 vs 克里）",
}

COLORS = {"SAM": "#d62728", "PCA": "#2ca02c", "VAE": "#1f77b4"}


# ------------------------------------------------------------------
# Step 1: 加载数据
# ------------------------------------------------------------------
def load_results(feature_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step2_path = STEP2_DIR / f"results_{feature_type.lower()}.npz"
    step1_path = STEP1_DIR / f"{feature_type.lower()}_features.npz"

    d2 = np.load(step2_path)
    rmse = d2["rmse_daily"]
    mape = d2["mape_daily"]

    d1 = np.load(step1_path)
    dates = d1["dates_test"]

    dates = pd.to_datetime([str(int(d)) for d in dates])
    return dates, rmse, mape


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def main() -> None:
    print("=" * BANNER_WIDTH)
    print("Figure 7 Analysis: Daily RMSE/MAPE Time Series")
    print("=" * BANNER_WIDTH)

    # Step 1: 加载
    data_list = []
    for ft in ["SAM", "PCA", "VAE"]:
        dates, rmse, mape = load_results(ft)
        data_list.append({
            "trade_date": dates,
            "model": [ft] * len(dates),
            "rmse": rmse,
            "mape": mape,
        })

    df = pd.concat([pd.DataFrame(d) for d in data_list], ignore_index=True)
    df.sort_values(["trade_date", "model"], inplace=True)

    n_sam = len(df[df["model"] == "SAM"])
    n_pca = len(df[df["model"] == "PCA"])
    n_vae = len(df[df["model"] == "VAE"])
    start_date = df["trade_date"].min().strftime("%Y-%m-%d")
    end_date = df["trade_date"].max().strftime("%Y-%m-%d")

    print(f"\n[Checkpoint 1] 数据加载")
    print(f"  SAM: {n_sam} 天, PCA: {n_pca} 天, VAE: {n_vae} 天")
    print(f"  测试期范围: {start_date} ~ {end_date}")

    # Step 2: 绘制 Figure 7
    print(f"\n[Checkpoint 2] Figure 7 绘制")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=300, sharex=True)

    for ft in ["SAM", "PCA", "VAE"]:
        sub = df[df["model"] == ft].sort_values("trade_date")
        axes[0].plot(sub["trade_date"], sub["rmse"], label=f"{ft}-DNN", color=COLORS[ft], linewidth=1.2)
        axes[1].plot(sub["trade_date"], sub["mape"], label=f"{ft}-DNN", color=COLORS[ft], linewidth=1.2)

    axes[0].set_ylabel("RMSE", fontsize=12)
    axes[0].set_title("(a) Daily RMSE in the Test Period", fontsize=14, fontweight="bold")
    axes[0].legend(loc="upper left", fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=df["rmse"].mean(), color="gray", linestyle="--", alpha=0.5, label="Mean")

    axes[1].set_ylabel("MAPE", fontsize=12)
    axes[1].set_title("(b) Daily MAPE in the Test Period", fontsize=14, fontweight="bold")
    axes[1].legend(loc="upper left", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=df["mape"].mean(), color="gray", linestyle="--", alpha=0.5, label="Mean")

    # X 轴格式
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Date", fontsize=12)
    plt.tight_layout()

    fig_path = OUTPUT_DIR / "figure7_daily_rmse_mape.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存路径: {fig_path}")

    # Step 3: 极端误差分析
    print(f"\n[Checkpoint 3] 极端误差分析 (Top 20)")
    extreme_rows = []
    for ft in ["SAM", "PCA", "VAE"]:
        sub = df[df["model"] == ft]
        top_rmse = sub.nlargest(20, "rmse")[["trade_date", "rmse", "mape"]].copy()
        top_rmse["model"] = ft
        top_rmse["metric"] = "rmse"
        top_rmse["rank"] = range(1, 21)

        top_mape = sub.nlargest(20, "mape")[["trade_date", "rmse", "mape"]].copy()
        top_mape["model"] = ft
        top_mape["metric"] = "mape"
        top_mape["rank"] = range(1, 21)

        extreme_rows.extend([top_rmse, top_mape])

        # 打印 Top 1
        top1 = top_rmse.iloc[0]
        print(f"  {ft} Top 1 RMSE: {top1['trade_date'].strftime('%Y-%m-%d')} = {top1['rmse']:.4f}")

    extreme_df = pd.concat(extreme_rows, ignore_index=True)
    extreme_path = OUTPUT_DIR / "extreme_error_dates.csv"
    extreme_df.to_csv(extreme_path, index=False)
    print(f"  保存路径: {extreme_path}")

    # Step 4: 历史事件对照
    print(f"\n[Checkpoint 4] 历史事件对照")
    event_rows = []
    n_covered = 0
    n_extreme = 0
    typical_examples = []

    for ft in ["SAM", "PCA", "VAE"]:
        sub = df[df["model"] == ft]
        mean_rmse = sub["rmse"].mean()

        for event_date_str, event_name in EVENTS.items():
            ed = pd.to_datetime(event_date_str)
            mask = (sub["trade_date"] >= ed - pd.Timedelta(days=3)) & (sub["trade_date"] <= ed + pd.Timedelta(days=3))
            window = sub[mask]

            if len(window) > 0:
                n_covered += len(window)
                n_extreme += (window["rmse"] > 2 * mean_rmse).sum()

                max_row = window.loc[window["rmse"].idxmax()]
                event_rows.append({
                    "event_date": event_date_str,
                    "event_name": event_name,
                    "model": ft,
                    "window_start": (ed - pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
                    "window_end": (ed + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
                    "max_rmse": max_row["rmse"],
                    "max_mape": max_row["mape"],
                    "max_date": max_row["trade_date"].strftime("%Y-%m-%d"),
                    "mean_window_rmse": window["rmse"].mean(),
                    "n_days": len(window),
                    "n_extreme": (window["rmse"] > 2 * mean_rmse).sum(),
                })

    event_df = pd.DataFrame(event_rows)
    event_path = OUTPUT_DIR / "error_event_correlation.csv"
    event_df.to_csv(event_path, index=False)
    print(f"  事件窗口覆盖天数: {n_covered}")
    print(f"  事件窗口内 RMSE > 2×均值的天数: {n_extreme}")

    # 打印典型关联
    if len(event_df) > 0:
        # 找出每个事件在所有模型中的最高 RMSE
        best_per_event = event_df.loc[event_df.groupby("event_name")["max_rmse"].idxmax()]
        for _, row in best_per_event.head(5).iterrows():
            print(f"    - {row['event_name']} ({row['event_date']}): 窗口内最高 RMSE = {row['max_rmse']:.4f} ({row['model']})")

    # Step 5: 统计摘要
    print(f"\n[Checkpoint 5] 统计摘要")
    summary_rows = []
    for ft in ["SAM", "PCA", "VAE"]:
        sub = df[df["model"] == ft].sort_values("trade_date")
        rmse_vals = sub["rmse"]
        mape_vals = sub["mape"]
        mean_rmse = rmse_vals.mean()

        stats = {
            "model": ft,
            "rmse_mean": mean_rmse,
            "rmse_median": rmse_vals.median(),
            "rmse_std": rmse_vals.std(),
            "rmse_min": rmse_vals.min(),
            "rmse_min_date": sub.loc[rmse_vals.idxmin(), "trade_date"].strftime("%Y-%m-%d"),
            "rmse_max": rmse_vals.max(),
            "rmse_max_date": sub.loc[rmse_vals.idxmax(), "trade_date"].strftime("%Y-%m-%d"),
            "rmse_p95": rmse_vals.quantile(0.95),
            "rmse_above_2x": (rmse_vals > 2 * mean_rmse).sum(),
            "rmse_above_2x_pct": (rmse_vals > 2 * mean_rmse).mean() * 100,
            "mape_mean": mape_vals.mean(),
            "mape_median": mape_vals.median(),
            "mape_std": mape_vals.std(),
            "mape_max": mape_vals.max(),
            "mape_max_date": sub.loc[mape_vals.idxmax(), "trade_date"].strftime("%Y-%m-%d"),
        }
        summary_rows.append(stats)

        print(f"\n  {ft}-DNN RMSE:")
        print(f"    均值: {stats['rmse_mean']:.4f}")
        print(f"    中位数: {stats['rmse_median']:.4f}")
        print(f"    标准差: {stats['rmse_std']:.4f}")
        print(f"    最小值: {stats['rmse_min']:.4f} (日期: {stats['rmse_min_date']})")
        print(f"    最大值: {stats['rmse_max']:.4f} (日期: {stats['rmse_max_date']})")
        print(f"    95% 分位数: {stats['rmse_p95']:.4f}")
        print(f"    超过 2×均值的天数: {stats['rmse_above_2x']} / {len(sub)} ({stats['rmse_above_2x_pct']:.1f}%)")

        # Top 5 极端日期
        top5 = sub.nlargest(5, "rmse")[["trade_date", "rmse", "mape"]]
        print(f"\n    [Top 5 极端日期]")
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            print(f"    {i}. {row['trade_date'].strftime('%Y-%m-%d')}: RMSE={row['rmse']:.4f}, MAPE={row['mape']:.4f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "summary_statistics.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'=' * BANNER_WIDTH}")
    print("[Done] Figure 7 分析完成")
    print(f"  输出文件:")
    print(f"    - {OUTPUT_DIR / 'figure7_daily_rmse_mape.png'}")
    print(f"    - {OUTPUT_DIR / 'extreme_error_dates.csv'}")
    print(f"    - {OUTPUT_DIR / 'error_event_correlation.csv'}")
    print(f"    - {OUTPUT_DIR / 'summary_statistics.csv'}")
    print(f"{'=' * BANNER_WIDTH}")


if __name__ == "__main__":
    main()
