# -*- coding: utf-8 -*-
"""
策略共享工具模块
提供：流动性过滤、信号分档、B-Spline基线插值、模型加载等纯信号层函数
（不包含回测、成本、对冲逻辑）
"""

import os
import pickle
import warnings
from typing import Tuple, Dict, Optional, List

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

warnings.filterwarnings('ignore')


# =============================================================================
# 1. 模型加载
# =============================================================================
def load_xgb_model(model_path: str):
    """加载XGBoost模型（支持.pkl格式）"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


# =============================================================================
# 2. 流动性过滤
# =============================================================================
def apply_liquidity_filter(
    df: pd.DataFrame,
    volume_pct: float = 0.3,
    min_oi: int = 100,
    min_remaining_time: int = 3,
    abs_residual_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    流动性过滤 + 基础筛选
    - volume > 该到期月当日中位数 * volume_pct
    - open_interest > min_oi
    - remaining_time > min_remaining_time
    - |residual| > abs_residual_threshold（可选，策略C使用）
    """
    df = df.copy()

    # 基础筛选
    mask = (
        (df['remaining_time'] > min_remaining_time) &
        (df['implc_volatlty'] > 0) &
        (~df['baseline_iv'].isna()) &
        (~df['residual_iv'].isna())
    )

    if 'open_interest' in df.columns:
        mask = mask & (df['open_interest'] >= min_oi)

    # 按到期月计算volume中位数并过滤
    if 'volume' in df.columns:
        vol_median = df.groupby(['trade_date', 'last_edate'])['volume'].transform('median')
        mask = mask & (df['volume'] >= vol_median * volume_pct)

    # residual阈值（策略C使用）
    if abs_residual_threshold > 0 and 'residual_iv' in df.columns:
        mask = mask & (df['residual_iv'].abs() >= abs_residual_threshold)

    return df[mask].copy()


# =============================================================================
# 3. 信号标准化与分档
# =============================================================================
def compute_residual_zscore(df: pd.DataFrame, by_maturity: bool = True) -> pd.DataFrame:
    """
    计算residual的Z-Score标准化
    by_maturity=True: 按(trade_date, last_edate)分组标准化
    by_maturity=False: 按trade_date全局标准化
    """
    df = df.copy()
    if by_maturity:
        grouped = df.groupby(['trade_date', 'last_edate'])['residual_iv']
    else:
        grouped = df.groupby('trade_date')['residual_iv']

    df['residual_mean'] = grouped.transform('mean')
    df['residual_std'] = grouped.transform('std').clip(lower=1e-6)
    df['residual_zscore'] = (df['residual_iv'] - df['residual_mean']) / df['residual_std']
    df = df.drop(columns=['residual_mean', 'residual_std'])
    return df


def rank_and_select(
    df: pd.DataFrame,
    signal_col: str,
    long_pct: float = 0.10,
    short_pct: float = 0.10,
    long_ascending: bool = False,
    short_ascending: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按信号排序，取Top long_pct做多，Bottom short_pct做空

    long_ascending: False表示信号越大越做多（默认，用于策略A）
    short_ascending: True表示信号越小越做空（默认，用于策略A）

    策略B需要反向：residual越大=越高估=做空，此时 long_ascending=True, short_ascending=False
    """
    df = df.copy()
    df['rank'] = df.groupby('trade_date')[signal_col].rank(pct=True)

    long_mask = df['rank'] >= (1 - long_pct) if not long_ascending else df['rank'] <= long_pct
    short_mask = df['rank'] <= short_pct if short_ascending else df['rank'] >= (1 - short_pct)

    long_df = df[long_mask].copy()
    short_df = df[short_mask].copy()

    return long_df, short_df


# =============================================================================
# 4. T+1日基线代理插值（从T日M-W曲面插值到T+1日合约位置）
# =============================================================================
def interpolate_t1_baseline(
    row: pd.Series,
    daily_mw_data: Dict,
    fwd_lookup: Dict,
) -> float:
    """
    对单合约，用T日M-W spline插值T+1日baseline_IV。

    公式: baseline_pred = sqrt(W_interp / tau_{t+1})
    其中 W_interp 是在 M_{t+1} = ln(K_{t+1}/F_{t+1}) 处查询T日spline的值。

    Returns:
        baseline_pred: float, 若无法插值则返回np.nan
    """
    date_t = row['trade_date']
    cp = row['call_put']
    mat = row['last_edate']
    date_t1 = row.get('next_trade_date')
    mat_t1 = row.get('next_last_edate', mat)
    K_t1 = row.get('next_exercise_price', row['exercise_price'])
    tau_t1 = row.get('next_remaining_time', row['remaining_time']) / 365.0

    if pd.isna(tau_t1) or tau_t1 <= 0:
        return np.nan

    # 查找F_{t+1}
    fwd_t1 = fwd_lookup.get((date_t1, mat_t1))
    if fwd_t1 is not None:
        F_t1 = fwd_t1[0]
    else:
        # fallback: 用T日F
        fwd_t = fwd_lookup.get((date_t, mat))
        F_t1 = fwd_t[0] if fwd_t is not None else row.get('F', row.get('F_implied', row['fund_close']))

    # 查找T日spline
    key = (date_t, cp, mat)
    if key in daily_mw_data:
        Mu, Wu, cs, F_t, tau_t = daily_mw_data[key]
        M_t1 = np.log(K_t1 / F_t1)

        if cs is not None:
            if M_t1 < Mu[0]:
                W_pred = Wu[0]
            elif M_t1 > Mu[-1]:
                W_pred = Wu[-1]
            else:
                W_pred = float(cs(M_t1))
        else:
            W_pred = np.interp(M_t1, Mu, Wu, left=Wu[0], right=Wu[-1])

        return np.sqrt(max(W_pred, 1e-6) / tau_t1)
    else:
        # fallback: 找最近到期月的spline
        same_day_cp = [(d, c, m) for (d, c, m) in daily_mw_data.keys()
                       if d == date_t and c == cp]
        if len(same_day_cp) > 0:
            best_key = min(same_day_cp,
                           key=lambda k: abs(daily_mw_data[k][4] - tau_t1))
            Mu, Wu, cs, F_t, tau_t = daily_mw_data[best_key]
            M_t1 = np.log(K_t1 / F_t1)
            if cs is not None:
                if M_t1 < Mu[0]:
                    W_pred = Wu[0]
                elif M_t1 > Mu[-1]:
                    W_pred = Wu[-1]
                else:
                    W_pred = float(cs(M_t1))
            else:
                W_pred = np.interp(M_t1, Mu, Wu, left=Wu[0], right=Wu[-1])
            return np.sqrt(max(W_pred, 1e-6) / tau_t1)

    return np.nan


def batch_interpolate_t1_baseline(
    df: pd.DataFrame,
    daily_mw_data: Dict,
    forward_table: pd.DataFrame,
) -> pd.Series:
    """
    批量计算T+1日baseline代理，返回Series（与df.index对齐）
    """
    fwd_lookup = {}
    for _, row in forward_table.iterrows():
        fwd_lookup[(row['trade_date'], row['last_edate'])] = (row['F'], row['tau'])

    results = []
    for _, row in df.iterrows():
        val = interpolate_t1_baseline(row, daily_mw_data, fwd_lookup)
        results.append(val)

    return pd.Series(results, index=df.index)


# =============================================================================
# 5. XGBoost预测辅助
# =============================================================================
def predict_iv_with_model(
    df: pd.DataFrame,
    model,
    feature_cols: List[str],
) -> pd.Series:
    """
    用XGBoost模型预测IV，返回pred_IV Series
    """
    X = df[feature_cols].fillna(0)
    pred = model.predict(X)
    return pd.Series(pred, index=df.index)


# =============================================================================
# 6. 隔夜信号衰减分析（策略B/C使用）
# =============================================================================
def analyze_signal_decay(df: pd.DataFrame, signal_col: str = 'residual_iv') -> Dict:
    """
    分析信号隔夜衰减：signal_t 与 (signal_{t+1} - signal_t) 的相关性

    Returns:
        dict: 衰减统计
    """
    df = df.copy().sort_values(['security_id', 'trade_date'])
    df['signal_next'] = df.groupby('security_id')[signal_col].shift(-1)
    df['signal_change'] = df['signal_next'] - df[signal_col]

    # 仅统计有次日数据的
    valid = df.dropna(subset=['signal_next'])

    if len(valid) == 0:
        return {
            'decay_corr': np.nan,
            'decay_rate_mean': np.nan,
            'autocorr_lag1': np.nan,
        }

    # 衰减相关系数: signal_t 与 signal_change（期望为负，即信号越大越衰减）
    decay_corr = valid[signal_col].corr(valid['signal_change'])

    # 衰减率均值
    decay_rate = valid['signal_change'] / valid[signal_col].clip(lower=1e-6)
    decay_rate_mean = decay_rate.replace([np.inf, -np.inf], np.nan).mean()

    # 一阶自相关
    autocorr = valid[signal_col].corr(valid['signal_next'])

    return {
        'decay_corr': float(decay_corr) if not pd.isna(decay_corr) else np.nan,
        'decay_rate_mean': float(decay_rate_mean) if not pd.isna(decay_rate_mean) else np.nan,
        'autocorr_lag1': float(autocorr) if not pd.isna(autocorr) else np.nan,
    }


# =============================================================================
# 7. 分档辅助
# =============================================================================
def bin_by_moneyness_m(df: pd.DataFrame, M_col: str = 'M') -> pd.Series:
    """按log-moneyness M分箱: ITM/ATM/OTM"""
    def _bin(M):
        if pd.isna(M):
            return 'Unknown'
        if M < -0.1:
            return 'ITM'
        elif M <= 0.1:
            return 'ATM'
        else:
            return 'OTM'
    return df[M_col].apply(_bin)


def bin_by_time(df: pd.DataFrame, t_col: str = 'remaining_time') -> pd.Series:
    """按剩余期限分箱: near/mid/far"""
    def _bin(t):
        if pd.isna(t):
            return 'Unknown'
        if t <= 30:
            return 'near'
        elif t <= 90:
            return 'mid'
        else:
            return 'far'
    return df[t_col].apply(_bin)
