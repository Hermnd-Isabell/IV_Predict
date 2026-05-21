# -*- coding: utf-8 -*-
"""
清洗后数据 ETL 合并脚本
将 SPX_Extracted_Processed/*.csv 合并为统一格式，供 Step 0 使用。

处理逻辑：
  1. 遍历 2009-2020 年 SPX_Extracted_Processed 目录下所有 *_clean.csv
  2. 计算 mid price、remaining_time
  3. 用 Put-Call Parity 估算 fund_close（近月配对中位数）
  4. 过滤 is_reliable=False（如有）
  5. 输出兼容格式的合并 CSV
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ------------------------------------------------------------------
# 配置
# ------------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\yuzih\Desktop\USOptions")
YEARS = list(range(2009, 2021))

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "spx_options_2009_2020_clean.csv"
DAYS_PER_YEAR = 365.0


def parse_date_int(date_str: str) -> int:
    return int(date_str.replace("-", ""))


def estimate_fund_close(df_day: pd.DataFrame) -> float | None:
    """
    对单日清洗数据，用 Put-Call Parity 估算标的价格。
    方法：F = K + C_mid - P_mid，取近月配对的中位数。
    """
    df = df_day.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2.0

    # 过滤无报价
    df = df[(df["bid"] > 0) | (df["ask"] > 0)]
    df = df[df["implied_volatility"] > 0]
    if len(df) == 0:
        return None

    calls = df[df["type"] == "call"][["strike", "expiration", "mid"]].rename(columns={"mid": "C"})
    puts = df[df["type"] == "put"][["strike", "expiration", "mid"]].rename(columns={"mid": "P"})
    paired = calls.merge(puts, on=["strike", "expiration"])
    if len(paired) == 0:
        return None

    paired["F_est"] = paired["strike"] + paired["C"] - paired["P"]

    # 取最短到期日的配对
    paired["quote_date"] = pd.to_datetime(df["quote_date"].iloc[0])
    paired["exp_date"] = pd.to_datetime(paired["expiration"])
    paired["T_days"] = (paired["exp_date"] - paired["quote_date"]).dt.days
    min_tau = paired["T_days"].min()
    near = paired[paired["T_days"] == min_tau]

    if len(near) == 0:
        near = paired

    F_median = near["F_est"].median()
    if pd.isna(F_median):
        return None

    return float(F_median)


def process_year(year: int) -> pd.DataFrame | None:
    """处理单年的所有清洗文件。"""
    src_dir = BASE_DIR / str(year) / "SPX_Extracted_Processed"
    pattern = str(src_dir / "SPX_*options_clean.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"  [{year}] 未找到清洗文件")
        return None

    dfs: list[pd.DataFrame] = []
    daily_fund_close: dict[str, float] = {}

    for f in tqdm(files, desc=f"  {year}", leave=False):
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue

            # 过滤不可靠数据（如有）
            if "is_reliable" in df.columns:
                df = df[df["is_reliable"] == True]

            if len(df) == 0:
                continue

            qd = df["quote_date"].iloc[0]
            F = estimate_fund_close(df)
            if F is not None:
                daily_fund_close[qd] = F

            dfs.append(df)
        except Exception as e:
            print(f"    跳过 {f}: {e}")

    if not dfs:
        return None

    raw = pd.concat(dfs, ignore_index=True)

    # 去重
    raw = raw.drop_duplicates(
        subset=["contract", "quote_date", "strike", "expiration", "type"]
    )

    # 计算 remaining_time
    raw["quote_date_dt"] = pd.to_datetime(raw["quote_date"])
    raw["expiration_dt"] = pd.to_datetime(raw["expiration"])
    raw["remaining_time"] = (raw["expiration_dt"] - raw["quote_date_dt"]).dt.days

    # 过滤
    raw = raw[raw["remaining_time"] > 0]
    raw = raw[raw["implied_volatility"] > 0]
    raw = raw[raw["bid"] >= 0]
    raw = raw[raw["ask"] >= 0]
    raw = raw[raw["implied_volatility"] <= 2.0]

    # 构造输出
    out = pd.DataFrame()
    out["trade_date"] = raw["quote_date"].apply(parse_date_int)
    out["call_put"] = raw["type"].str.upper().str[0]
    out["exercise_price"] = raw["strike"]
    out["remaining_time"] = raw["remaining_time"].astype(int)
    out["implc_volatlty"] = raw["implied_volatility"]
    out["fund_close"] = out["trade_date"].astype(str).apply(lambda d: daily_fund_close.get(
        pd.to_datetime(d, format="%Y%m%d").strftime("%Y-%m-%d"), np.nan
    ))

    # 对于无法估算 fund_close 的日期，用同月最近有值的日期填充
    out["fund_close"] = out["fund_close"].ffill().bfill()

    n_before = len(out)
    out = out.dropna(subset=["fund_close"])
    n_after = len(out)
    if n_before != n_after:
        print(f"  [{year}] 丢弃无 fund_close 的行: {n_before - n_after}")

    out = out[["trade_date", "call_put", "exercise_price", "remaining_time", "implc_volatlty", "fund_close"]]
    out = out.sort_values(["trade_date", "remaining_time", "exercise_price", "call_put"]).reset_index(drop=True)

    return out


def main() -> None:
    print("=" * 60)
    print("清洗数据 ETL 合并")
    print(f"输出: {OUTPUT_PATH}")
    print("=" * 60)

    all_years_df: list[pd.DataFrame] = []
    total_files = 0

    for year in YEARS:
        print(f"\n[{year}] 处理中...")
        df_year = process_year(year)
        if df_year is not None and len(df_year) > 0:
            all_years_df.append(df_year)
            print(f"  记录数: {len(df_year)}, 交易日: {df_year['trade_date'].nunique()}")
            total_files += 1

    if not all_years_df:
        raise RuntimeError("没有成功处理任何年份")

    final = pd.concat(all_years_df, ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'=' * 60}")
    print("合并完成")
    print(f"{'=' * 60}")
    print(f"总记录数: {len(final)}")
    print(f"总交易日: {final['trade_date'].nunique()}")
    print(f"年份范围: {final['trade_date'].min()} ~ {final['trade_date'].max()}")
    print(f"fund_close 范围: [{final['fund_close'].min():.2f}, {final['fund_close'].max():.2f}]")
    print(f"IV 范围: [{final['implc_volatlty'].min():.4f}, {final['implc_volatlty'].max():.4f}]")
    print(f"保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
