# -*- coding: utf-8 -*-
"""
SPX 期权数据格式转换器
将 C:\\Users\\yuzih\\Desktop\\USOptions\\ 下的多 CSV 合并为统一的格式，
兼容 50ETF_options.csv 的列结构，供 Step 0 使用。

关键处理：
  1. 合并 2002/2003/2004 三年的 SPX 期权 CSV
  2. 对每个交易日，用 Put-Call Parity 回归估算 SPX 收盘价 (fund_close)
  3. 转换列名和格式
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
DATA_DIRS = [
    r"C:\Users\yuzih\Desktop\USOptions\2002\SPX_Extracted",
    r"C:\Users\yuzih\Desktop\USOptions\2003\SPX_Extracted",
    r"C:\Users\yuzih\Desktop\USOptions\2004\SPX_Extracted",
    r"C:\Users\yuzih\Desktop\USOptions\2005\SPX_Extracted",
    r"C:\Users\yuzih\Desktop\USOptions\2006\SPX_Extracted",
    r"C:\Users\yuzih\Desktop\USOptions\2007\SPX_Extracted",
]

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "spx_options.csv"

# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------
def parse_date_int(date_str: str) -> int:
    """将 '2002-05-01' 转为 20020501。"""
    return int(date_str.replace("-", ""))


def estimate_underlying_price(df_day: pd.DataFrame, max_tau_years: float = 0.5) -> float | None:
    """
    对单个交易日的期权数据，用 Put-Call Parity 回归估算标的价格。

    方法：C - P = S - K * e^(-rT) ≈ S - K （短期限近似）
    先过滤异常 strike（过低/过高的非 SPX 合约），再用回归。
    """
    df = df_day.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2.0

    # 过滤无效合约：IV=0 或 bid=ask=0
    df = df[df["implied_volatility"] > 0]
    df = df[(df["bid"] > 0) | (df["ask"] > 0)]
    if len(df) == 0:
        return None

    calls = (
        df[df["type"] == "call"][["strike", "expiration", "mid"]]
        .rename(columns={"mid": "C"})
    )
    puts = (
        df[df["type"] == "put"][["strike", "expiration", "mid"]]
        .rename(columns={"mid": "P"})
    )
    paired = calls.merge(puts, on=["strike", "expiration"])
    paired["diff"] = paired["C"] - paired["P"]

    # 解析日期算 T（年）
    paired["quote_date"] = pd.to_datetime(df["quote_date"].iloc[0])
    paired["exp_date"] = pd.to_datetime(paired["expiration"])
    paired["T"] = (paired["exp_date"] - paired["quote_date"]).dt.days / 365.0

    # 筛选近月、 nonzero 价格
    near = paired[
        (paired["T"] < max_tau_years)
        & (paired["C"] > 0)
        & (paired["P"] > 0)
    ].copy()

    if len(near) < 5:
        # 近月不够，放宽到全部期限
        near = paired[(paired["C"] > 0) & (paired["P"] > 0)].copy()
        if len(near) < 5:
            return None

    # 计算每个配对的 S 估计
    near["S_est"] = near["strike"] + near["diff"]

    # 用 S_est 的中位数做 sanity check，过滤异常 strike
    # （例如 strike 120 的 OEX 伪合约会被排除）
    S_median_all = near["S_est"].median()
    if pd.isna(S_median_all):
        return None

    # 保留 S_est 在 [median*0.6, median*1.4] 内的配对
    near_clean = near[
        (near["S_est"] >= S_median_all * 0.6)
        & (near["S_est"] <= S_median_all * 1.4)
    ].copy()

    if len(near_clean) < 5:
        near_clean = near.copy()

    strike_min, strike_max = near_clean["strike"].min(), near_clean["strike"].max()

    # ---- 方法 1：线性回归 ----
    X = np.column_stack([np.ones(len(near_clean)), near_clean["strike"].values])
    y = near_clean["diff"].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    S_est = coef[0]
    slope = coef[1]

    # 严格的合理性检查
    reg_ok = (
        strike_min * 0.7 <= S_est <= strike_max * 1.3
        and -1.15 <= slope <= -0.85
    )

    if reg_ok:
        return float(S_est)

    # ---- 方法 2：加权平均（对 |diff| 小的 strike 加权） ----
    near_clean["S_temp"] = near_clean["strike"] + near_clean["diff"]
    weights = 1.0 / (near_clean["diff"].abs() + 1.0)
    S_wa = np.average(near_clean["S_temp"], weights=weights)

    if strike_min * 0.5 <= S_wa <= strike_max * 1.5:
        return float(S_wa)

    # ---- 方法 3：中位数 ----
    S_median = near_clean["S_est"].median()
    if pd.notna(S_median) and strike_min * 0.5 <= S_median <= strike_max * 1.5:
        return float(S_median)

    return None


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def main() -> None:
    # 收集所有文件
    all_files: list[str] = []
    for d in DATA_DIRS:
        pattern = os.path.join(d, "SPX_*.csv")
        files = sorted(glob.glob(pattern))
        all_files.extend(files)

    print(f"发现 {len(all_files)} 个 CSV 文件")
    if not all_files:
        raise FileNotFoundError("未找到任何 SPX CSV 文件")

    # 逐个读取并合并
    dfs: list[pd.DataFrame] = []
    for f in tqdm(all_files, desc="读取 CSV"):
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            dfs.append(df)
        except Exception as e:
            print(f"跳过 {f}: {e}")

    raw = pd.concat(dfs, ignore_index=True)
    print(f"合并后总行数: {len(raw)}")
    print(f"原始列: {list(raw.columns)}")

    # 去重（同一合约同日可能出现多次）
    raw = raw.drop_duplicates(
        subset=["contract", "quote_date", "strike", "expiration", "type"]
    )
    print(f"去重后行数: {len(raw)}")

    # 解析日期
    raw["quote_date_dt"] = pd.to_datetime(raw["quote_date"])
    raw["expiration_dt"] = pd.to_datetime(raw["expiration"])
    raw["remaining_time"] = (raw["expiration_dt"] - raw["quote_date_dt"]).dt.days

    # 过滤异常值
    raw = raw[raw["remaining_time"] > 0]
    raw = raw[raw["implied_volatility"] > 0]
    raw = raw[raw["bid"] >= 0]
    raw = raw[raw["ask"] >= 0]
    # IV > 2.0 (200%) 视为异常值
    n_iv_outlier = (raw["implied_volatility"] > 2.0).sum()
    if n_iv_outlier > 0:
        print(f"  过滤 IV > 2.0 的异常值: {n_iv_outlier} 条")
        raw = raw[raw["implied_volatility"] <= 2.0]
    # 过滤 bid=ask=0（无报价）
    n_zero_quote = ((raw["bid"] == 0) & (raw["ask"] == 0)).sum()
    if n_zero_quote > 0:
        print(f"  过滤 bid=ask=0 的无报价合约: {n_zero_quote} 条")
        raw = raw[(raw["bid"] > 0) | (raw["ask"] > 0)]
    print(f"过滤后行数: {len(raw)}")

    # 估算每日标的价格 fund_close
    print("估算每日 SPX 标的价格...")
    daily_prices: dict[int, float] = {}
    grouped = raw.groupby("quote_date", sort=True)
    last_valid_S: float | None = None
    skipped_dates: list[str] = []

    for qd, g in tqdm(grouped, total=grouped.ngroups, desc="标的价格回归"):
        date_int = parse_date_int(qd)
        S = estimate_underlying_price(g)

        if S is not None:
            daily_prices[date_int] = S
            last_valid_S = S
        elif last_valid_S is not None:
            daily_prices[date_int] = last_valid_S
            skipped_dates.append(qd)
        else:
            skipped_dates.append(qd)

    if skipped_dates:
        print(f"  警告: {len(skipped_dates)} 天使用 fallback（前一日或无数据）")
        print(f"    示例: {skipped_dates[:5]}")

    print(f"成功估算 {len(daily_prices)} / {grouped.ngroups} 个交易日的标的价格")

    # 构造输出 DataFrame
    out = pd.DataFrame()
    out["trade_date"] = raw["quote_date"].apply(parse_date_int)
    out["call_put"] = raw["type"].str.upper().str[0]  # call -> C, put -> P
    out["exercise_price"] = raw["strike"]
    out["remaining_time"] = raw["remaining_time"].astype(int)
    out["implc_volatlty"] = raw["implied_volatility"]

    # 映射 fund_close
    out["fund_close"] = out["trade_date"].map(daily_prices)

    # 丢弃无法映射到标的价格的行
    n_before = len(out)
    out = out.dropna(subset=["fund_close"])
    n_after = len(out)
    print(f"丢弃无标的价格的行: {n_before - n_after} 行")

    # 最终需要的列，和 50ETF 兼容
    out = out[["trade_date", "call_put", "exercise_price", "remaining_time", "implc_volatlty", "fund_close"]]

    # 排序
    out = out.sort_values(["trade_date", "remaining_time", "exercise_price", "call_put"]).reset_index(drop=True)

    # 保存
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\n保存到: {OUTPUT_PATH}")
    print(f"最终记录数: {len(out)}")
    print(f"交易日数: {out['trade_date'].nunique()}")
    print(f"标的价格范围: [{out['fund_close'].min():.2f}, {out['fund_close'].max():.2f}]")
    print(f"行权价范围: [{out['exercise_price'].min():.2f}, {out['exercise_price'].max():.2f}]")
    print(f"剩余期限范围: [{out['remaining_time'].min()}, {out['remaining_time'].max()}] 天")
    print(f"IV 范围: [{out['implc_volatlty'].min():.4f}, {out['implc_volatlty'].max():.4f}]")


if __name__ == "__main__":
    main()
