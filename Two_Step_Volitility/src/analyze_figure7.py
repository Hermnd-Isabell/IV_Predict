# -*- coding: utf-8 -*-
"""Figure 7 每日 RMSE/MAPE 时间序列分析"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUTPUT_DIR = PROJECT_ROOT / "output" / "step2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 历史事件表
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# 加载数据
# ------------------------------------------------------------------
def load_results(feature_type: str) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    path = PROJECT_ROOT / "output" / "spx_step2" / f"results_{feature_type.lower()}.npz"
    data = np.load(path, allow_pickle=True)
    rmse = data["rmse_daily"]
    mape = data["mape_daily"]

    # 从 Step 1 补日期
    d1 = np.load(PROJECT_ROOT / "output" / "spx_step1" / "sam_features.npz", allow_pickle=True)
    dates_int = d1["dates_test"]
    dates = pd.to_datetime(dates_int.astype(str), format="%Y%m%d")
    return dates, rmse, mape


def main():
    print("=" * 60)
    print("Figure 7 分析: 每日 RMSE/MAPE 时间序列")
    print("=" * 60)

    # --- Step 1: 加载三种模型 ---
    models = {}
    for ft in ["SAM", "PCA", "VAE"]:
        dates, rmse, mape = load_results(ft)
        models[ft] = {"dates": dates, "rmse": rmse, "mape": mape}
        print(f"[Load] {ft}: {len(dates)} 天, 范围 {dates[0].date()} ~ {dates[-1].date()}")

    # 构建统一 DataFrame
    rows = []
    for ft in ["SAM", "PCA", "VAE"]:
        for i in range(len(models[ft]["dates"])):
            rows.append({
                "trade_date": models[ft]["dates"][i],
                "model": ft,
                "rmse": models[ft]["rmse"][i],
                "mape": models[ft]["mape"][i],
            })
    df = pd.DataFrame(rows)

    print(f"\n[Checkpoint 1] 数据加载")
    print(f"  SAM: {len(models['SAM']['dates'])} 天")
    print(f"  PCA: {len(models['PCA']['dates'])} 天")
    print(f"  VAE: {len(models['VAE']['dates'])} 天")
    print(f"  测试期范围: {df['trade_date'].min().date()} ~ {df['trade_date'].max().date()}")

    # --- Step 2: 绘制 Figure 7 ---
    colors = {"SAM": "#d62728", "PCA": "#2ca02c", "VAE": "#1f77b4"}

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=300)

    # 子图 (a): 每日 RMSE
    ax = axes[0]
    for ft in ["SAM", "PCA", "VAE"]:
        sub = df[df["model"] == ft].sort_values("trade_date")
        ax.plot(sub["trade_date"], sub["rmse"], label=f"{ft}-DNN", color=colors[ft], linewidth=0.8)
    ax.set_title("(a) Daily RMSE in the Test Period", fontsize=14)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # 子图 (b): 每日 MAPE
    ax = axes[1]
    for ft in ["SAM", "PCA", "VAE"]:
        sub = df[df["model"] == ft].sort_values("trade_date")
        ax.plot(sub["trade_date"], sub["mape"], label=f"{ft}-DNN", color=colors[ft], linewidth=0.8)
    ax.set_title("(b) Daily MAPE in the Test Period", fontsize=14)
    ax.set_ylabel("MAPE", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    fig_path = OUTPUT_DIR / "figure7_daily_rmse_mape.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Checkpoint 2] Figure 7 绘制完成")
    print(f"  保存路径: {fig_path}")

    # --- Step 3: 极端误差分析 (Top 20) ---
    extreme_rows = []
    for model in ["SAM", "PCA", "VAE"]:
        sub = df[df["model"] == model]
        top_rmse = sub.nlargest(20, "rmse")[["trade_date", "rmse", "mape"]].copy()
        top_rmse["metric"] = "rmse"
        top_rmse["rank"] = range(1, 21)
        top_rmse["model"] = model

        top_mape = sub.nlargest(20, "mape")[["trade_date", "rmse", "mape"]].copy()
        top_mape["metric"] = "mape"
        top_mape["rank"] = range(1, 21)
        top_mape["model"] = model

        extreme_rows.extend(top_rmse.to_dict("records"))
        extreme_rows.extend(top_mape.to_dict("records"))

    extreme_df = pd.DataFrame(extreme_rows)
    extreme_path = OUTPUT_DIR / "extreme_error_dates.csv"
    extreme_df.to_csv(extreme_path, index=False)

    print(f"\n[Checkpoint 3] 极端误差分析")
    for model in ["SAM", "PCA", "VAE"]:
        sub = df[df["model"] == model]
        top1 = sub.nlargest(1, "rmse").iloc[0]
        print(f"  {model} Top 1 RMSE: {top1['trade_date'].date()} = {top1['rmse']:.4f}")

    # --- Step 4: 历史事件对照 ---
    event_rows = []
    n_covered = 0
    n_extreme = 0
    for event_date_str, event_name in EVENTS.items():
        ed = pd.to_datetime(event_date_str)
        mask = (df["trade_date"] >= ed - pd.Timedelta(days=3)) & (df["trade_date"] <= ed + pd.Timedelta(days=3))
        window = df[mask].copy()
        if len(window) > 0:
            n_covered += 1
            # 计算窗口内每种模型的均值 RMSE
            for model in ["SAM", "PCA", "VAE"]:
                sub = window[window["model"] == model]
                if len(sub) > 0:
                    max_row = sub.loc[sub["rmse"].idxmax()]
                    mean_rmse = sub["rmse"].mean()
                    event_rows.append({
                        "event_date": event_date_str,
                        "event_name": event_name,
                        "model": model,
                        "window_start": sub["trade_date"].min().date(),
                        "window_end": sub["trade_date"].max().date(),
                        "max_rmse": max_row["rmse"],
                        "max_rmse_date": max_row["trade_date"].date(),
                        "mean_rmse": mean_rmse,
                        "is_extreme": mean_rmse > df[df["model"] == model]["rmse"].mean() * 2,
                    })
                    if mean_rmse > df[df["model"] == model]["rmse"].mean() * 2:
                        n_extreme += 1

    event_df = pd.DataFrame(event_rows)
    event_path = OUTPUT_DIR / "error_event_correlation.csv"
    event_df.to_csv(event_path, index=False)

    print(f"\n[Checkpoint 4] 历史事件对照")
    print(f"  事件窗口覆盖天数: {n_covered}")
    print(f"  事件窗口内 RMSE > 2×均值的天数: {n_extreme}")
    # 典型关联
    if len(event_df) > 0:
        top_event = event_df.loc[event_df["max_rmse"].idxmax()]
        print(f"  典型事件-误差关联:")
        print(f"    - {top_event['event_name']} ({top_event['event_date']}): 窗口内最高 RMSE = {top_event['max_rmse']:.4f} ({top_event['model']})")

    # --- Step 5: 统计摘要 ---
    print(f"\n[Checkpoint 5] 统计摘要")
    for model in ["SAM", "PCA", "VAE"]:
        sub = df[df["model"] == model]["rmse"]
        mean_val = sub.mean()
        median_val = sub.median()
        std_val = sub.std()
        min_val = sub.min()
        max_val = sub.max()
        min_date = df[(df["model"] == model) & (df["rmse"] == min_val)]["trade_date"].iloc[0]
        max_date = df[(df["model"] == model) & (df["rmse"] == max_val)]["trade_date"].iloc[0]
        p95 = sub.quantile(0.95)
        over2x = (sub > 2 * mean_val).sum()
        total = len(sub)

        print(f"\n{model}-DNN RMSE:")
        print(f"  均值: {mean_val:.4f}")
        print(f"  中位数: {median_val:.4f}")
        print(f"  标准差: {std_val:.4f}")
        print(f"  最小值: {min_val:.4f} (日期: {min_date.date()})")
        print(f"  最大值: {max_val:.4f} (日期: {max_date.date()})")
        print(f"  95% 分位数: {p95:.4f}")
        print(f"  超过 2×均值的天数: {over2x} / {total} ({over2x/total*100:.1f}%)")

        # Top 5 极端日期
        top5 = df[df["model"] == model].nlargest(5, "rmse")[["trade_date", "rmse", "mape"]]
        print(f"  [Top 5 极端日期]")
        for idx, row in top5.iterrows():
            print(f"    {row['trade_date'].date()}: RMSE={row['rmse']:.4f}, MAPE={row['mape']:.2%}")

    print(f"\n[Done] 所有输出保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
