# -*- coding: utf-8 -*-
"""
策略C: Double-Sort 双信号组合策略（条件筛选）

修复后调整:
  - 残差不可预测 (Signal-to-Noise < 1)
  - 废弃 XGBoost 方向一致性过滤（残差是纯噪声，方向过滤无意义）
  - 退化为: B-Spline 显著偏离池 + 纯截面排序
  - 本质上是策略B的加强版（|zscore|阈值更严格: 1.5 vs 1.0）

使用方式:
  from src.strategies.strategy_c_doublesort import generate_signals
  signal_df, long_df, short_df, filter_stats = generate_signals(df_day)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict

from src.core.signal_utils import (
    compute_residual_zscore,
    apply_liquidity_filter,
)


# =============================================================================
# Step 1: B-Spline 显著偏离池
# =============================================================================
def build_deviation_pool(
    df: pd.DataFrame,
    zscore_threshold: float = 1.5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    构建显著定价偏差池（仅B-Spline截面信号，无XGBoost时序过滤）。

    输入:
        df: 已过滤后的当日合约数据（含 residual_iv, baseline_iv）
        zscore_threshold: |residual_zscore| 阈值（默认1.5）

    返回:
        df_active:  通过阈值的全量合约（含 residual_zscore）
        overvalued: 高估池（residual_zscore > threshold）候选空头
        undervalued: 低估池（residual_zscore < -threshold）候选多头
    """
    df = df.copy()

    # 计算截面Z-Score（按到期月标准化）
    df = compute_residual_zscore(df, by_maturity=True)

    # 阈值过滤
    mask = df['residual_zscore'].abs() > zscore_threshold
    df_active = df[mask].copy()

    # 分池
    overvalued = df_active[df_active['residual_zscore'] > zscore_threshold].copy()
    undervalued = df_active[df_active['residual_zscore'] < -zscore_threshold].copy()

    return df_active, overvalued, undervalued


# =============================================================================
# Step 2: 分档与等权配置（废弃方向一致性过滤）
# =============================================================================
def build_portfolio(
    long_candidates: pd.DataFrame,
    short_candidates: pd.DataFrame,
    long_pct: float = 0.10,
    short_pct: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    在显著偏离池中分档。

    分档规则:
      - 多头: 低估池中按 residual_zscore 升序取 Top long_pct（最被低估）
      - 空头: 高估池中按 residual_zscore 降序取 Top short_pct（最被高估）
    """
    long_df = pd.DataFrame()
    short_df = pd.DataFrame()

    # 多头：按 residual_zscore 升序（最负=最低估）
    if len(long_candidates) > 0:
        long_sorted = long_candidates.sort_values('residual_zscore', ascending=True)
        n_long = max(1, int(len(long_sorted) * long_pct))
        long_df = long_sorted.head(n_long).copy()
        long_df['position_type'] = 'Long'

    # 空头：按 residual_zscore 降序（最正=最高估）
    if len(short_candidates) > 0:
        short_sorted = short_candidates.sort_values('residual_zscore', ascending=False)
        n_short = max(1, int(len(short_sorted) * short_pct))
        short_df = short_sorted.head(n_short).copy()
        short_df['position_type'] = 'Short'

    return long_df, short_df


# =============================================================================
# 主入口: 统一接口
# =============================================================================
def generate_signals(
    df_day: pd.DataFrame,
    zscore_threshold: float = 1.5,
    long_pct: float = 0.10,
    short_pct: float = 0.10,
    liquidity_params: Dict = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    策略C主入口: 纯截面Double-Sort生成当日全量信号 + 多空持仓。

    参数:
        df_day: 当日所有合约数据（含 baseline_iv, residual_iv）
        zscore_threshold: |residual_zscore|阈值（默认1.5）
        long_pct: 多头分位比例（默认10%）
        short_pct: 空头分位比例（默认10%）
        liquidity_params: 流动性过滤参数字典

    返回:
        signal_df:   全量合约信号（含 residual_zscore）
        long_df:     多头持仓合约（position_type='Long'）
        short_df:    空头持仓合约（position_type='Short'）
        filter_stats: 过滤统计（n_long_passed, n_short_passed等）
    """
    # Step 0: 流动性过滤
    liq_params = liquidity_params or {
        'volume_pct': 0.3,
        'min_oi': 100,
        'min_remaining_time': 5,
        'abs_residual_threshold': 0.003,
    }
    df_filt = apply_liquidity_filter(df_day, **liq_params)

    if len(df_filt) == 0:
        empty_stats = {
            'n_overvalued_total': 0,
            'n_undervalued_total': 0,
            'n_short_passed': 0,
            'n_long_passed': 0,
            'direction_agreement_rate': 0.0,
        }
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), empty_stats

    # Step 1: B-Spline显著偏离池
    df_active, overvalued, undervalued = build_deviation_pool(df_filt, zscore_threshold)

    # Step 2: 分档（废弃方向一致性过滤）
    long_df, short_df = build_portfolio(
        undervalued, overvalued, long_pct, short_pct
    )

    stats = {
        'n_overvalued_total': len(overvalued),
        'n_undervalued_total': len(undervalued),
        'n_short_passed': len(short_df),
        'n_long_passed': len(long_df),
        'direction_agreement_rate': 1.0,  # 不再过滤，全部通过
    }

    return df_active, long_df, short_df, stats


# =============================================================================
# 便捷函数: 批量处理多日的信号
# =============================================================================
def generate_signals_batch(
    df: pd.DataFrame,
    zscore_threshold: float = 1.5,
    long_pct: float = 0.10,
    short_pct: float = 0.10,
    liquidity_params: Dict = None,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    批量生成多日的策略C信号，返回每日多空持仓 + 过滤统计。
    """
    all_positions = []
    daily_stats = []

    for date, day_df in df.groupby('trade_date'):
        signal_df, long_df, short_df, stats = generate_signals(
            day_df,
            zscore_threshold=zscore_threshold,
            long_pct=long_pct,
            short_pct=short_pct,
            liquidity_params=liquidity_params,
        )

        stats['trade_date'] = date
        daily_stats.append(stats)

        for pos_df in [long_df, short_df]:
            if len(pos_df) > 0:
                pos_df = pos_df.copy()
                pos_df['weight'] = 1.0 / len(pos_df) if len(pos_df) > 0 else 0.0
                all_positions.append(pos_df)

    if len(all_positions) == 0:
        return pd.DataFrame(), daily_stats

    return pd.concat(all_positions, ignore_index=True), daily_stats
