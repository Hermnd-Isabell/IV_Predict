# -*- coding: utf-8 -*-
"""
策略A: XGBoost残差预测偏差驱动 (Delta-Neutral Long-Short)
核心信号: signal = pred_residual = pred_IV_{t+1} - baseline_IV_{t+1}

修复后调整:
  - 残差不可预测 (Signal-to-Noise < 1)
  - 信号从 delta_residual (pred_res_t1 - res_t) 简化为 pred_residual
  - 信号含义: 预测T+1日定价偏差方向
    - pred_residual > 0: 预测T+1日定价偏高 -> 做多(Long)
    - pred_residual < 0: 预测T+1日定价偏低 -> 做空(Short)

使用方式:
  from src.strategies.strategy_a_residual import generate_signals
  signal_df, long_df, short_df = generate_signals(
      df_day, xgb_model, feature_cols, daily_mw_data, forward_table
  )
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

from src.core.signal_utils import (
    predict_iv_with_model,
    batch_interpolate_t1_baseline,
    apply_liquidity_filter,
    rank_and_select,
)


# =============================================================================
# 核心信号构造
# =============================================================================
def compute_residual_signal(
    df: pd.DataFrame,
    xgb_model,
    feature_cols: List[str],
    daily_mw_data: Dict,
    forward_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    构造策略A核心信号:
      pred_IV_t1       = XGBoost预测T+1日IV
      baseline_IV_t1   = B-Spline T日曲面插值到T+1位置
      signal           = pred_residual = pred_IV_t1 - baseline_IV_t1

    参数:
        df: DataFrame, 当日所有合约数据（必须含 baseline_iv, residual_iv 列）
        xgb_model: 已加载的XGBoost模型
        feature_cols: XGBoost特征列名列表
        daily_mw_data: {(date, cp, mat): (Mu, Wu, cs, F, tau)} M-W spline字典
        forward_table: DataFrame, [trade_date, last_edate, F, tau]

    返回:
        df: 增加列 ['pred_iv', 'baseline_iv_t1', 'pred_residual', 'signal']
    """
    df = df.copy()

    # 1. XGBoost预测T+1日绝对IV
    df['pred_iv'] = predict_iv_with_model(df, xgb_model, feature_cols)

    # 2. B-Spline T日曲面插值到T+1日合约位置（baseline代理）
    df['baseline_iv_t1'] = batch_interpolate_t1_baseline(df, daily_mw_data, forward_table)

    # 3. 核心信号: 预测T+1日残差 = pred_IV - baseline_IV_t1
    df['pred_residual'] = df['pred_iv'] - df['baseline_iv_t1']
    df['signal'] = df['pred_residual']

    return df


# =============================================================================
# 组合构建: 信号分档 + 多空筛选
# =============================================================================
def build_portfolio(
    df: pd.DataFrame,
    signal_col: str = 'signal',
    signal_threshold: float = 0.003,
    long_pct: float = 0.10,
    short_pct: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    基于signal构建Delta-Neutral Long-Short组合。

    筛选规则:
      - |signal| > signal_threshold 才进入排序池
      - 多头: signal Top long_pct（预测残差为正 -> 做多）
      - 空头: signal Bottom short_pct（预测残差为负 -> 做空）
    """
    df = df.copy()

    active = df[df[signal_col].abs() > signal_threshold].copy()

    if len(active) == 0:
        return pd.DataFrame(), pd.DataFrame()

    long_df, short_df = rank_and_select(
        active,
        signal_col=signal_col,
        long_pct=long_pct,
        short_pct=short_pct,
        long_ascending=False,   # signal大 -> Long
        short_ascending=True,   # signal小 -> Short
    )

    return long_df, short_df


# =============================================================================
# 主入口: 统一接口
# =============================================================================
def generate_signals(
    df_day: pd.DataFrame,
    xgb_model,
    feature_cols: List[str],
    daily_mw_data: Dict,
    forward_table: pd.DataFrame,
    signal_threshold: float = 0.003,
    long_pct: float = 0.10,
    short_pct: float = 0.10,
    liquidity_params: Dict = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    策略A主入口: 生成当日全量信号 + 多空持仓。

    参数:
        df_day: 当日所有合约数据（含 baseline_iv, residual_iv, M, 特征列等）
        xgb_model: XGBoost模型
        feature_cols: 模型特征列名
        daily_mw_data: M-W spline字典
        forward_table: 远期价格表
        signal_threshold: |signal|最小阈值（默认0.003）
        long_pct: 多头分位比例（默认10%）
        short_pct: 空头分位比例（默认10%）
        liquidity_params: 流动性过滤参数字典

    返回:
        signal_df: 全量合约信号（含 signal, pred_residual, baseline_iv_t1）
        long_df:   多头持仓合约（含 position_type='Long'）
        short_df:  空头持仓合约（含 position_type='Short'）
    """
    liq_params = liquidity_params or {
        'volume_pct': 0.3,
        'min_oi': 100,
        'min_remaining_time': 3,
    }
    df_filt = apply_liquidity_filter(df_day, **liq_params)

    if len(df_filt) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    signal_df = compute_residual_signal(
        df_filt, xgb_model, feature_cols, daily_mw_data, forward_table
    )

    long_df, short_df = build_portfolio(
        signal_df,
        signal_col='signal',
        signal_threshold=signal_threshold,
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
    xgb_model,
    feature_cols: List[str],
    daily_mw_data: Dict,
    forward_table: pd.DataFrame,
    signal_threshold: float = 0.003,
    long_pct: float = 0.10,
    short_pct: float = 0.10,
    liquidity_params: Dict = None,
) -> pd.DataFrame:
    """
    批量生成多日的策略A信号，返回每日多空持仓的拼接DataFrame。
    """
    all_positions = []

    for date, day_df in df.groupby('trade_date'):
        signal_df, long_df, short_df = generate_signals(
            day_df, xgb_model, feature_cols, daily_mw_data, forward_table,
            signal_threshold, long_pct, short_pct, liquidity_params
        )

        for pos_df in [long_df, short_df]:
            if len(pos_df) > 0:
                pos_df = pos_df.copy()
                pos_df['weight'] = 1.0 / len(pos_df) if len(pos_df) > 0 else 0.0
                all_positions.append(pos_df)

    if len(all_positions) == 0:
        return pd.DataFrame()

    return pd.concat(all_positions, ignore_index=True)
