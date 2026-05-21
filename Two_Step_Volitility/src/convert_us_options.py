# -*- coding: utf-8 -*-
"""
SPX 期权数据 ETL —— 2009-2020
将 C:\\Users\\yuzih\\Desktop\\USOptions\\{year}\\SPX_Extracted\\ 下的日级 CSV
合并为兼容 50ETF_options.csv 的 6 列格式：

    trade_date, call_put, exercise_price, remaining_time, implc_volatlty, fund_close

设计要点（用户决策）：
    1. fund_close 估算采用方案 A：每日取最短到期日的所有 paired strikes 计算
       F = K + (C_mid - P_mid)，取中位数（鲁棒、稳定，已实测同日跨 K 标准差 < 0.1）。
    2. 异常 IV 过滤：implied_volatility ∈ (0, 2.0]。
    3. bid=ask=0 过滤（无报价合约）。
    4. 2014/2016/2018 等缺失日期容忍跳过。
    5. 保留原始 spx_options.csv（2002-2007）作为备份，本脚本输出到独立文件
       data/raw/spx_options_2009_2020.csv。

输出：
    - data/raw/spx_options_2009_2020.csv （主数据）
    - data/raw/spx_options_2009_2020_quality.csv （每日 F 估计质量报告）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ------------------------------------------------------------------
# 配置
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(r"C:\Users\yuzih\Desktop\USOptions")
YEARS = list(range(2009, 2021))  # 2009 ~ 2020 共 12 年

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = OUTPUT_DIR / "spx_options_2009_2020.csv"
QUALITY_PATH = OUTPUT_DIR / "spx_options_2009_2020_quality.csv"

# 数据过滤参数
IV_MIN = 0.0          # 严格 > 0
IV_MAX = 2.0          # 200% 以上视为异常
MIN_PAIRS_PER_EXPIRY = 3  # 单一到期日至少 3 个 paired strikes 才用于 F 估计

# 输出列顺序（与旧数据完全一致）
OUTPUT_COLUMNS = [
    "trade_date",
    "call_put",
    "exercise_price",
    "remaining_time",
    "implc_volatlty",
    "fund_close",
]

QUALITY_COLUMNS = [
    "quote_date",
    "n_records",
    "S_est",
    "F_cross_expiry_std",
    "n_pairs_nearest_expiry",
    "n_valid_expiries",
    "S_estimate_ok",
]


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------
def parse_date_int(date_str: str) -> int:
    """'2010-01-04' -> 20100104"""
    cleaned = date_str.strip().split()[0].replace("-", "")
    if len(cleaned) != 8 or not cleaned.isdigit():
        raise ValueError(f"Unexpected date format: {date_str!r}")
    return int(cleaned)


def estimate_fund_close_method_a(df_day: pd.DataFrame) -> tuple[float | None, float, int, int]:
    """
    方案 A：取最短到期日 paired strikes 的 F = K + C - P 中位数作为 fund_close。

    Returns
    -------
    S_est : float | None
        估计的标的收盘价，失败返回 None。
    F_cross_expiry_std : float
        同日不同到期日的 F 估计中位数的标准差（跨期稳定性，越小越好；NaN -> 0）。
    n_pairs_used : int
        最短到期日参与中位数计算的 call/put 配对数。
    n_expiries : int
        当日有效（≥ MIN_PAIRS_PER_EXPIRY 配对）的到期日总数。
    """
    # groupby 切片已是新 DataFrame，无需 copy
    df = df_day
    df["mid"] = (df["bid"] + df["ask"]) / 2.0

    # call / put 配对
    calls = (
        df[df["type"] == "call"][["strike", "expiration", "mid"]]
        .rename(columns={"mid": "C"})
    )
    puts = (
        df[df["type"] == "put"][["strike", "expiration", "mid"]]
        .rename(columns={"mid": "P"})
    )
    paired = calls.merge(puts, on=["strike", "expiration"])
    if paired.empty:
        return None, 0.0, 0, 0

    # 计算 T(天) 和 F
    quote_dt = pd.to_datetime(df["quote_date"].iloc[0])
    paired["T_days"] = (pd.to_datetime(paired["expiration"]) - quote_dt).dt.days
    paired["F_est"] = paired["strike"] + paired["C"] - paired["P"]

    # 仅保留 T > 0 且 C/P 均有报价
    paired = paired[(paired["T_days"] > 0) & (paired["C"] > 0) & (paired["P"] > 0)]
    if paired.empty:
        return None, 0.0, 0, 0

    # 每个到期日的样本量与 F 中位数
    by_expiry = (
        paired.groupby("T_days")
        .agg(n=("F_est", "size"), F_med=("F_est", "median"))
        .reset_index()
    )
    valid = by_expiry[by_expiry["n"] >= MIN_PAIRS_PER_EXPIRY]
    if valid.empty:
        # 配对数太少的兜底：仍用整体最短到期日
        valid = by_expiry

    # 方案 A 核心：取最短到期日的 F 中位数
    nearest = valid.sort_values("T_days").iloc[0]
    min_T = int(nearest["T_days"])
    S_est = float(nearest["F_med"])

    # 跨期稳定性：所有 valid 到期日 F 中位数的标准差
    if len(valid) >= 2:
        F_cross_expiry_std = float(valid["F_med"].std())
    else:
        F_cross_expiry_std = 0.0

    n_pairs_used = int(paired[paired["T_days"] == min_T].shape[0])
    n_expiries = int(len(valid))

    return S_est, F_cross_expiry_std, n_pairs_used, n_expiries


# ------------------------------------------------------------------
# 单年处理
# ------------------------------------------------------------------
def process_year(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    处理单一年份 -> (主数据 6 列 DataFrame, 质量报告 DataFrame)。
    """
    year_dir = DATA_ROOT / str(year) / "SPX_Extracted"
    if not year_dir.exists():
        print(f"  [WARN] {year} 目录不存在: {year_dir}")
        return pd.DataFrame(columns=OUTPUT_COLUMNS), pd.DataFrame(columns=QUALITY_COLUMNS)

    csv_files = sorted(year_dir.glob("SPX_*.csv"))
    if not csv_files:
        print(f"  [WARN] {year} 未发现任何 CSV")
        return pd.DataFrame(columns=OUTPUT_COLUMNS), pd.DataFrame(columns=QUALITY_COLUMNS)

    # 一次性读全年（每个文件约 2000-5000 行，单年 < 80MB）
    daily_frames: list[pd.DataFrame] = []
    for f in tqdm(csv_files, desc=f"  {year} 读取", leave=False):
        try:
            df = pd.read_csv(f)
            if not df.empty:
                daily_frames.append(df)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as e:
            print(f"    跳过 {f.name}: {e}")

    if not daily_frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), pd.DataFrame(columns=QUALITY_COLUMNS)

    raw = pd.concat(daily_frames, ignore_index=True)

    # 类型清洗 + 过滤
    raw["quote_date"] = raw["quote_date"].astype(str)
    raw["expiration"] = raw["expiration"].astype(str)
    raw["type"] = raw["type"].astype(str).str.lower().str.strip()

    raw = raw[raw["type"].isin(["call", "put"])]
    raw["remaining_time"] = (
        pd.to_datetime(raw["expiration"]) - pd.to_datetime(raw["quote_date"])
    ).dt.days

    n_before = len(raw)
    raw = raw[raw["remaining_time"] > 0]
    raw = raw[(raw["implied_volatility"] > IV_MIN) & (raw["implied_volatility"] <= IV_MAX)]
    raw = raw[(raw["bid"] > 0) | (raw["ask"] > 0)]
    n_filtered = n_before - len(raw)

    # 去重
    raw = raw.drop_duplicates(
        subset=["contract", "quote_date", "strike", "expiration", "type"]
    )

    # 估算每日 fund_close
    quality_rows: list[dict] = []
    daily_S: dict[str, float] = {}

    for qd, g in raw.groupby("quote_date", sort=True):
        S, F_std, n_pairs, n_expiries = estimate_fund_close_method_a(g)
        if S is not None:
            daily_S[qd] = S
        quality_rows.append({
            "quote_date": qd,
            "n_records": len(g),
            "S_est": S if S is not None else np.nan,
            "F_cross_expiry_std": F_std,
            "n_pairs_nearest_expiry": n_pairs,
            "n_valid_expiries": n_expiries,
            "S_estimate_ok": S is not None,
        })

    quality_df = pd.DataFrame(quality_rows)

    # 构造主输出（仅保留成功估算 S 的日期）
    raw["fund_close"] = raw["quote_date"].map(daily_S)
    raw = raw.dropna(subset=["fund_close"])

    out = pd.DataFrame({
        "trade_date": raw["quote_date"].str.replace("-", "").astype(int),
        "call_put": raw["type"].str[0].str.upper(),    # call -> C, put -> P
        "exercise_price": raw["strike"].astype(float),
        "remaining_time": raw["remaining_time"].astype(int),
        "implc_volatlty": raw["implied_volatility"].astype(float),
        "fund_close": raw["fund_close"].astype(float),
    })

    out = out[OUTPUT_COLUMNS]

    print(
        f"  {year}: files={len(csv_files):>3}  raw_rows={n_before:>8,}  "
        f"filtered={n_filtered:>7,}  out_rows={len(out):>8,}  "
        f"days={out['trade_date'].nunique():>3}  "
        f"S_ok={int(quality_df['S_estimate_ok'].sum())}/{len(quality_df)}"
    )

    return out, quality_df


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"SPX 期权 ETL: {YEARS[0]} ~ {YEARS[-1]} ({len(YEARS)} 年)")
    print(f"输入根目录: {DATA_ROOT}")
    print(f"输出文件: {OUTPUT_PATH}")
    print("=" * 60)

    main_parts: list[pd.DataFrame] = []
    quality_parts: list[pd.DataFrame] = []

    for year in tqdm(YEARS, desc="按年处理"):
        out_year, q_year = process_year(year)
        if not out_year.empty:
            main_parts.append(out_year)
        if not q_year.empty:
            quality_parts.append(q_year)

    if not main_parts:
        raise RuntimeError("没有任何年份产生有效数据，ETL 失败")

    final = pd.concat(main_parts, ignore_index=True)
    final = final.sort_values(
        ["trade_date", "remaining_time", "exercise_price", "call_put"]
    ).reset_index(drop=True)

    quality_df = pd.concat(quality_parts, ignore_index=True)

    # 保存
    final.to_csv(OUTPUT_PATH, index=False)
    quality_df.to_csv(QUALITY_PATH, index=False)

    # 汇总
    print()
    print("=" * 60)
    print("ETL 完成")
    print("=" * 60)
    print(f"  最终记录数: {len(final):,}")
    print(f"  交易日数: {final['trade_date'].nunique():,}")
    print(
        f"  日期范围: {final['trade_date'].min()} ~ {final['trade_date'].max()}"
    )
    print(f"  call/put 占比: {(final['call_put'] == 'C').mean():.2%} / {(final['call_put'] == 'P').mean():.2%}")
    print(
        f"  fund_close 范围: [{final['fund_close'].min():.2f}, {final['fund_close'].max():.2f}]"
    )
    print(
        f"  exercise_price 范围: [{final['exercise_price'].min():.2f}, {final['exercise_price'].max():.2f}]"
    )
    print(
        f"  remaining_time 范围: [{final['remaining_time'].min()}, {final['remaining_time'].max()}] 天"
    )
    print(
        f"  implc_volatlty 范围: [{final['implc_volatlty'].min():.4f}, {final['implc_volatlty'].max():.4f}]"
    )

    print()
    print("--- F 估计质量报告 ---")
    print(f"  总质量行数: {len(quality_df):,}")
    print(f"  S 估计成功: {int(quality_df['S_estimate_ok'].sum()):,}/{len(quality_df):,}")
    ok = quality_df[quality_df["S_estimate_ok"]]
    if not ok.empty:
        print(f"  跨期 F 标准差 - 中位数: {ok['F_cross_expiry_std'].median():.3f}")
        print(f"  跨期 F 标准差 - 95% 分位: {ok['F_cross_expiry_std'].quantile(0.95):.3f}")
        print(f"  跨期 F 标准差 - 最大值: {ok['F_cross_expiry_std'].max():.3f}")
        high_std = ok[ok["F_cross_expiry_std"] > 50]
        if len(high_std) > 0:
            print(f"  [WARN] 跨期 F 标准差 > 50 的日期: {len(high_std)} 天（数据质量差，需关注）")
            print(f"     示例: {high_std['quote_date'].head(5).tolist()}")

    print()
    print(f"主数据保存至: {OUTPUT_PATH}")
    print(f"质量报告保存至: {QUALITY_PATH}")


if __name__ == "__main__":
    main()
