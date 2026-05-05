# -*- coding: utf-8 -*-
"""
策略B: B-Spline截面相对定价偏差驱动 (Delta-Neutral Long-Short)
核心信号: residual_zscore = (residual - 到期月均值) / 到期月标准差

信号含义（截面逻辑）:
  - residual > 0 (zscore > 0): 相对高估 -> 做空(Short)
  - residual < 0 (zscore < 0): 相对低估 -> 做多(Long)

关键认知: residual本质是市场微观结构噪声，截面分布可能包含可套利信息。

使用方式:
  from src.strategies.strategy_b_section import generate_signals
  signal_df, long_df, short_df = generate_signals(df_day)

隔夜衰减测试:
  from src.strategies.strategy_b_section import analyze_overnight_decay
  decay_stats = analyze_overnight_decay(df_history, signal_col='residual_iv')
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

from src.core.signal_utils import (
    compute_residual_zscore,
    apply_liquidity_filter,
    rank_and_select,
    analyze_signal_decay,
)


# =============================================================================
# 核心信号构造: 截面残差偏差
# =============================================================================
def compute_section_signal(
    df: pd.DataFrame,
    by_maturity: bool = True,
) -> pd.DataFrame:
    """
    构造策略B核心信号:
      residual        = market_IV - baseline_IV（已由上游计算为 residual_iv）
      residual_zscore = (residual - 组均值) / 组标准差

    参数:
        df: DataFrame, 当日所有合约数据（必须含 residual_iv 列）
        by_maturity: True按到期月标准化（消除不同到期月尺度差异）,
                     False按交易日全局标准化

    返回:
        df: 增加列 ['residual_zscore', 'residual_zscore_global']
    """
    df = df.copy()

    # 确保residual存在
    if 'residual_iv' not in df.columns:
        raise ValueError("df must contain 'residual_iv' column")

    # 按到期月标准化（默认）
    if by_maturity:
        df = compute_residual_zscore(df, by_maturity=True)
        df = df.rename(columns={'residual_zscore': 'residual_zscore_maturity'})
    else:
        df['residual_zscore_maturity'] = np.nan

    # 全局标准化（备用）
    df = compute_residual_zscore(df, by_maturity=False)
    df = df.rename(columns={'residual_zscore': 'residual_zscore_global'})

    # 默认使用到期月标准化
    df['residual_zscore'] = df['residual_zscore_maturity']

    return df


# =============================================================================
# 组合构建: 截面排序 + 多空筛选
# =============================================================================
def build_portfolio(
    df: pd.DataFrame,
    signal_col: str = 'residual',
    zscore_threshold: float = 1.0,
    long_pct: float = 0.10,
    short_pct: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    基于截面residual构建Delta-Neutral Long-Short组合。

    筛选规则:
      - |residual_zscore| > zscore_threshold 才进入排序池
      - 多头: residual最负的 Bottom long_pct（相对低估 → 做多）
      - 空头: residual最正的 Top short_pct（相对高估 → 做空）

    注意: 与策略A排序方向相反！
      - 策略B: residual大 = 高估 = 做空
      - 策略B: residual小 = 低估 = 做多
    """
    df = df.copy()

    # |zscore|阈值过滤
    if 'residual_zscore' in df.columns:
        active = df[df['residual_zscore'].abs() > zscore_threshold].copy()
    else:
        active = df.copy()

    if len(active) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # 排序分档:
    #   Long: residual最小（最负）→ 最被低估
    #   Short: residual最大（最正）→ 最被高估
    long_df, short_df = rank_and_select(
        active,
        signal_col=signal_col,
        long_pct=long_pct,
        short_pct=short_pct,
        long_ascending=True,    # residual小 → Long
        short_ascending=False,  # residual大 → Short
    )

    return long_df, short_df


# =============================================================================
# 隔夜信号衰减分析（策略B特有诊断）
# =============================================================================
def analyze_overnight_decay(
    df: pd.DataFrame,
    signal_col: str = 'residual_iv',
) -> Dict:
    """
    分析T日收盘residual信号到T+1日的衰减情况。

    诊断指标:
      - decay_corr: residual_t 与 T+1日收益的相关性
                    （期望为负或接近0，说明噪声隔夜均值回归）
      - autocorr_lag1: residual一阶自相关
                       （期望接近0，说明噪声无记忆）
      - decay_rate_mean: (residual_t - residual_{t+1}) / residual_t 均值
                         （期望 > 0.5，说明隔夜衰减显著）

    返回:
        dict: 衰减统计
    """
    return analyze_signal_decay(df, signal_col=signal_col)


def analyze_overnight_return_decay(
    df: pd.DataFrame,
    signal_col: str = 'residual_iv',
) -> Dict:
    """
    若数据含开盘价，分析residual_t与overnight_return的关系。
    overnight_return = (open_{t+1} - close_t) / close_t
    """
    df = df.copy().sort_values(['security_id', 'trade_date'])

    # 计算overnight return（若open存在）
    if 'open' in df.columns:
        df['next_open'] = df.groupby('security_id')['open'].shift(-1)
        df['overnight_return'] = (df['next_open'] - df['close']) / df['close'].clip(lower=1e-6)
    else:
        # 无开盘价：用close_t+1近似
        df['next_close'] = df.groupby('security_id')['close'].shift(-1)
        df['overnight_return'] = (df['next_close'] - df['close']) / df['close'].clip(lower=1e-6)

    valid = df.dropna(subset=['overnight_return', signal_col])

    if len(valid) == 0:
        return {
            'overnight_corr': np.nan,
            'overnight_corr_pvalue': np.nan,
        }

    # residual_t 与 overnight_return 的相关性
    # 期望为负：高估合约(residual>0)隔夜下跌
    corr = valid[signal_col].corr(valid['overnight_return'])

    return {
        'overnight_corr': float(corr) if not pd.isna(corr) else np.nan,
        'n_samples': len(valid),
    }


# =============================================================================
# 主入口: 统一接口
# =============================================================================
def generate_signals(
    df_day: pd.DataFrame,
    zscore_threshold: float = 1.0,
    long_pct: float = 0.10,
    short_pct: float = 0.10,
    liquidity_params: Dict = None,
    by_maturity: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    策略B主入口: 生成当日全量信号 + 多空持仓。

    参数:
        df_day: 当日所有合约数据（含 residual_iv, baseline_iv 列）
        zscore_threshold: |residual_zscore|最小阈值（默认1.0）
        long_pct: 多头分位比例（默认10%，取最被低估）
        short_pct: 空头分位比例（默认10%，取最被高估）
        liquidity_params: 流动性过滤参数字典
        by_maturity: True按到期月标准化, False全局标准化

    返回:
        signal_df: 全量合约信号（含 residual, residual_zscore）
        long_df:   多头持仓合约（position_type='Long'）
        short_df:  空头持仓合约（position_type='Short'）
    """
    # Step 1: 流动性过滤
    liq_params = liquidity_params or {
        'volume_pct': 0.3,
        'min_oi': 100,
        'min_remaining_time': 3,
    }
    df_filt = apply_liquidity_filter(df_day, **liq_params)

    if len(df_filt) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Step 2: 构造截面残差信号
    signal_df = compute_section_signal(df_filt, by_maturity=by_maturity)

    # Step 3: 构建多空组合
    # 用原始residual排序（zscore仅用于阈值过滤）
    long_df, short_df = build_portfolio(
        signal_df,
        signal_col='residual_iv',
        zscore_threshold=zscore_threshold,
        long_pct=long_pct,
        short_pct=short_pct,
    )

    if len(long_df) > 0:
        long_df['position_type'] = 'Long'
    if len(short_df) > 0:
        short_df['position_type'] = 'Short'

    return signal_df, long_df, short_df


# =============================================================================
# 便捷函数: 批量处理多日的信号
# =============================================================================
def generate_signals_batch(
    df: pd.DataFrame,
    zscore_threshold: float = 1.0,
    long_pct: float = 0.10,
    short_pct: float = 0.10,
    liquidity_params: Dict = None,
    by_maturity: bool = True,
) -> pd.DataFrame:
    """
    批量生成多日的策略B信号，返回每日多空持仓的拼接DataFrame。
    """
    all_positions = []

    for date, day_df in df.groupby('trade_date'):
        signal_df, long_df, short_df = generate_signals(
            day_df,
            zscore_threshold=zscore_threshold,
            long_pct=long_pct,
            short_pct=short_pct,
            liquidity_params=liquidity_params,
            by_maturity=by_maturity,
        )

        for pos_df in [long_df, short_df]:
            if len(pos_df) > 0:
                pos_df = pos_df.copy()
                pos_df['weight'] = 1.0 / len(pos_df) if len(pos_df) > 0 else 0.0
                all_positions.append(pos_df)

    if len(all_positions) == 0:
        return pd.DataFrame()

    return pd.concat(all_positions, ignore_index=True)
