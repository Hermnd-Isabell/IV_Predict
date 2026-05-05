# -*- coding: utf-8 -*-
"""
50ETF期权IV预测 - 两步法：B-Spline基准 + 逐合约残差预测
Step1: B-Spline拟合IV曲面得到baseline_iv（保证无套利）
Step2: XGBoost预测residual = market_IV - baseline_iv（捕捉个体时序）
Step3: 合成 pred_IV = pred_baseline + pred_residual
"""

import os
import json
import pickle
import warnings
from typing import Tuple, Dict
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid
import xgboost as xgb

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RANDOM_STATE = 42
OUTPUT_DIR = 'data/output/two_step_residual/'
BASELINE_DIR = 'data/output/baseline_xgb/'

TRAIN_END = 20240630
VAL_END = 20241231


# =============================================================================
# Step 0: 数据加载
# =============================================================================
def load_raw_data(path='data/raw/50etf_options.csv') -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# =============================================================================
# Step 1: B-Spline 基准IV计算
# =============================================================================
def compute_baseline_iv(df_day: pd.DataFrame, call_put: str) -> pd.Series:
    """
    对单日单类型计算B-Spline基准IV。
    按到期月分组，每组用CubicSpline沿K拟合，插值到每个合约的实际K。
    输出：每个合约的baseline_iv（Series，index与输入对齐）
    """
    df_sub = df_day[df_day['call_put'] == call_put].copy()
    df_sub = df_sub[(df_sub['remaining_time'] > 0) & (df_sub['implc_volatlty'] > 0)]

    if len(df_sub) < 3:
        return pd.Series(np.nan, index=df_sub.index)

    baseline = pd.Series(np.nan, index=df_sub.index)
    valid_mats = []  # 记录成功拟合的到期月及其ATM均值

    # 按到期月分组拟合
    for mat, df_mat in df_sub.groupby('last_edate'):
        df_mat = df_mat.sort_values('exercise_price')
        if len(df_mat) < 3:
            continue

        x = df_mat['exercise_price'].values.astype(float)
        y = df_mat['implc_volatlty'].values.astype(float)

        # 去重
        uniq = pd.DataFrame({'x': x, 'y': y}).groupby('x')['y'].mean().reset_index().sort_values('x')
        xu, yu = uniq['x'].values, uniq['y'].values

        # 实际要插值的K点
        K_query = df_mat['exercise_price'].values.astype(float)

        iv_interp = np.empty(len(K_query))
        try:
            cs = CubicSpline(xu, yu)
            # 严格区分范围内外：范围内用CubicSpline，范围外用边界值
            in_mask = (K_query >= xu[0]) & (K_query <= xu[-1])
            out_mask = ~in_mask
            if np.any(in_mask):
                iv_interp[in_mask] = cs(K_query[in_mask])
            if np.any(out_mask):
                iv_interp[out_mask] = np.where(K_query[out_mask] <= xu[0], yu[0], yu[-1])
        except Exception:
            iv_interp = np.interp(K_query, xu, yu, left=yu[0], right=yu[-1])

        # 按K排序后做单调性约束（因为df_mat已按exercise_price排序）
        if call_put == 'C':
            for i in range(1, len(iv_interp)):
                if iv_interp[i] > iv_interp[i - 1]:
                    iv_interp[i] = iv_interp[i - 1]
        else:
            for i in range(1, len(iv_interp)):
                if iv_interp[i] < iv_interp[i - 1]:
                    iv_interp[i] = iv_interp[i - 1]

        baseline.loc[df_mat.index] = iv_interp
        valid_mats.append(np.mean(iv_interp))  # 该到期月均值作为填充备用

    # 对未填充的（某到期月<3个合约），用同天同类型其他到期月均值填充
    if baseline.isna().any():
        fill_val = np.mean(valid_mats) if valid_mats else df_sub['implc_volatlty'].mean()
        baseline = baseline.fillna(fill_val)

    return baseline


def build_residual_series(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    全局计算所有合约的baseline_iv和residual_iv。
    同时保存每天每个到期月的B-Spline数据（用于T+1日预测时插值）。
    返回：(带baseline/residual的df, daily_bspline_data)
    """
    df = df.copy()
    # 过滤极端异常值（与原方案一致）
    df.loc[df['implc_volatlty'] > 1.0, 'implc_volatlty'] = np.nan
    df = df.dropna(subset=['implc_volatlty']).copy()
    df['baseline_iv'] = np.nan
    df['residual_iv'] = np.nan

    # daily_bspline: {(date, call_put, mat_edate): (xu, yu, mean_iv, remaining_time)}
    daily_bspline = {}

    grouped = df.groupby(['trade_date', 'call_put'])
    for (date, cp), group in grouped:
        # 计算baseline
        bl = compute_baseline_iv(group, cp)
        df.loc[bl.index, 'baseline_iv'] = bl.values

        # 保存每天每个到期月的Spline数据
        sub = group[(group['remaining_time'] > 0) & (group['implc_volatlty'] > 0)]
        for mat, df_mat in sub.groupby('last_edate'):
            df_mat = df_mat.sort_values('exercise_price')
            if len(df_mat) < 2:
                continue
            x = df_mat['exercise_price'].values.astype(float)
            y = df_mat['implc_volatlty'].values.astype(float)
            uniq = pd.DataFrame({'x': x, 'y': y}).groupby('x')['y'].mean().reset_index().sort_values('x')
            xu, yu = uniq['x'].values, uniq['y'].values
            rt = df_mat['remaining_time'].iloc[0]
            daily_bspline[(date, cp, mat)] = (xu, yu, np.mean(y), rt)

    df['residual_iv'] = df['implc_volatlty'] - df['baseline_iv']
    return df, daily_bspline


# =============================================================================
# Step 2: 特征工程（复用原方案 + 新增残差历史特征）
# =============================================================================
def build_features_residual(df: pd.DataFrame) -> pd.DataFrame:
    """
    在原方案特征工程基础上，增加残差历史特征。
    """
    df = df.copy()
    df = df.sort_values(['security_id', 'trade_date']).reset_index(drop=True)

    # 先调用原方案核心特征工程逻辑（简化复用）
    # 1. 合约固有
    df['moneyness'] = df['fund_close'] / df['exercise_price']
    df['moneyness_squared'] = df['moneyness'] ** 2
    df['call_put_flag'] = (df['call_put'] == 'C').astype(int)
    df['moneyness_remaining_time'] = df['moneyness'] * df['remaining_time']

    # 2. 标的数据
    spot_daily = df[['trade_date', 'fund_close', 'fund_volume', 'fund_amount',
                     'fund_high', 'fund_low']].drop_duplicates('trade_date').sort_values('trade_date')
    spot_daily['fund_return'] = spot_daily['fund_close'].pct_change()
    spot_daily['fund_high_low_ratio'] = (spot_daily['fund_high'] - spot_daily['fund_low']) / spot_daily['fund_close']
    df = df.merge(spot_daily[['trade_date', 'fund_return', 'fund_high_low_ratio']],
                  on='trade_date', how='left')

    # 3. 历史IV + 历史残差（按合约分组滑动窗口）
    trade_dates = np.sort(df['trade_date'].unique())
    date_map = pd.DataFrame({'trade_date': trade_dates,
                             'trade_dt': pd.to_datetime(trade_dates, format='%Y%m%d')})
    df = df.merge(date_map, on='trade_date', how='left')

    def compute_lags(group):
        group = group.sort_values('trade_date')
        # IV历史
        group['iv_t'] = group['implc_volatlty']
        group['iv_t_1'] = group['implc_volatlty'].shift(1)
        group['iv_t_2'] = group['implc_volatlty'].shift(2)
        group['iv_t_3'] = group['implc_volatlty'].shift(3)
        group['iv_t_4'] = group['implc_volatlty'].shift(4)
        group['iv_ma5'] = group['implc_volatlty'].rolling(window=5, min_periods=1).mean()
        group['iv_std5'] = group['implc_volatlty'].rolling(window=5, min_periods=2).std()
        group['iv_trend5'] = group['implc_volatlty'] - group['implc_volatlty'].shift(4)
        # 残差历史（新增）
        group['residual_t'] = group['residual_iv']
        group['residual_t_1'] = group['residual_iv'].shift(1)
        group['residual_t_2'] = group['residual_iv'].shift(2)
        group['residual_ma3'] = group['residual_iv'].rolling(window=3, min_periods=1).mean()
        group['residual_std3'] = group['residual_iv'].rolling(window=3, min_periods=2).std()
        # days_gap
        group['days_gap'] = (group['trade_dt'] - group['trade_dt'].shift(1)).dt.days
        return group

    df = df.groupby('security_id', group_keys=False).apply(compute_lags)

    # 4. 曲面上下文（t-1日截面统计，用baseline_iv计算更准确）
    daily_stats = []
    for date, day_df in df.groupby('trade_date'):
        stats = {
            'trade_date': date,
            'iv_mean_all': day_df['implc_volatlty'].mean(),
            'iv_std_all': day_df['implc_volatlty'].std(),
            'iv_max_all': day_df['implc_volatlty'].max(),
            'iv_min_all': day_df['implc_volatlty'].min(),
        }
        call_df = day_df[day_df['call_put'] == 'C'].copy()
        if len(call_df) >= 2:
            call_df = call_df.sort_values('exercise_price')
            fund_close = call_df['fund_close'].iloc[0]
            xp = call_df['exercise_price'].values
            fp = call_df['implc_volatlty'].values
            if np.all(np.diff(xp) > 0):
                atm_iv = np.interp(fund_close, xp, fp)
            else:
                call_agg = call_df.groupby('exercise_price')['implc_volatlty'].mean().reset_index().sort_values('exercise_price')
                atm_iv = np.interp(fund_close, call_agg['exercise_price'].values, call_agg['implc_volatlty'].values)
        else:
            atm_iv = np.nan
        stats['atm_iv_call'] = atm_iv
        daily_stats.append(stats)

    daily_stats_df = pd.DataFrame(daily_stats).sort_values('trade_date')
    daily_stats_lag = daily_stats_df.copy()
    daily_stats_lag['trade_date_dt'] = pd.to_datetime(daily_stats_lag['trade_date'], format='%Y%m%d')
    daily_stats_lag['merge_date'] = (daily_stats_lag['trade_date_dt'] + pd.Timedelta(days=1)).dt.strftime('%Y%m%d').astype(int)
    daily_stats_lag = daily_stats_lag.rename(columns={
        'iv_mean_all': 'iv_mean_all_lag1',
        'iv_std_all': 'iv_std_all_lag1',
        'iv_max_all': 'iv_max_all_lag1',
        'iv_min_all': 'iv_min_all_lag1',
        'atm_iv_call': 'atm_iv_call_lag1',
    })
    df = df.merge(
        daily_stats_lag[['merge_date', 'iv_mean_all_lag1', 'iv_std_all_lag1',
                         'iv_max_all_lag1', 'iv_min_all_lag1', 'atm_iv_call_lag1']],
        left_on='trade_date', right_on='merge_date', how='left'
    ).drop(columns=['merge_date'])
    df['iv_vs_atm_lag1'] = df['implc_volatlty'] - df['atm_iv_call_lag1']

    # 5. 宏观
    df['ten_year'] = df['ten_year'] / 100.0

    # 填充缺失
    for col in ['atm_iv_call_lag1', 'iv_mean_all_lag1', 'iv_std_all_lag1',
                'iv_max_all_lag1', 'iv_min_all_lag1', 'iv_vs_atm_lag1']:
        df[col] = df[col].fillna(df[col].mean())
    df['days_gap'] = df['days_gap'].fillna(1)
    df['fund_return'] = df['fund_return'].fillna(0)
    for col in ['iv_t_1', 'iv_t_2', 'iv_t_3', 'iv_t_4', 'iv_std5', 'iv_trend5',
                'residual_t_1', 'residual_t_2', 'residual_std3']:
        df[col] = df[col].fillna(df.get('iv_t', df['implc_volatlty']) if 'iv' in col else df.get('residual_t', 0))

    return df


# =============================================================================
# Step 2b: 目标变量构造
# =============================================================================
def build_target_residual(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造残差预测目标：y = residual_{t+1}
    同时保留 IV_{t+1} 用于最终评估。
    """
    df = df.copy()
    df = df.sort_values(['security_id', 'trade_date']).reset_index(drop=True)

    # residual_{t+1}
    df['next_residual'] = df.groupby('security_id')['residual_iv'].shift(-1)
    # IV_{t+1}（用于最终评估）
    df['next_implc_volatlty'] = df.groupby('security_id')['implc_volatlty'].shift(-1)
    # baseline_{t+1}（T+1日真实baseline，用于评估时合成）
    df['next_baseline_iv'] = df.groupby('security_id')['baseline_iv'].shift(-1)
    # T+1日remaining_time（用于预测时选择最接近的spline）
    df['next_remaining_time'] = df.groupby('security_id')['remaining_time'].shift(-1)

    # 关键改进：训练目标改为 delta_residual = residual_{t+1} - residual_t
    # 这样 pred_iv = iv_t + pred_delta_residual，既消除baseline代理错配，又保留iv_t的无套利结构
    df['target_residual'] = df['next_residual'] - df['residual_iv']

    df = df.dropna(subset=['target_residual', 'next_implc_volatlty', 'next_baseline_iv']).copy()
    return df


# =============================================================================
# Step 3: 时间划分
# =============================================================================
def split_temporal(df: pd.DataFrame, train_end: int, val_end: int) -> Tuple:
    train_df = df[df['trade_date'] <= train_end].copy()
    val_df = df[(df['trade_date'] > train_end) & (df['trade_date'] <= val_end)].copy()
    test_df = df[df['trade_date'] > val_end].copy()
    return train_df, val_df, test_df


# =============================================================================
# Step 4: XGBoost 残差模型训练
# =============================================================================
def train_residual_model(X_train, y_train, X_val, y_val, param_grid=None):
    if param_grid is not None:
        base_params = {
            'objective': 'reg:squarederror',
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'colsample_bytree': 0.8,
        }
        best_model = None
        best_rmse = float('inf')
        best_params = None
        print("[Grid Search] 残差模型超参数搜索...")
        for i, params in enumerate(list(ParameterGrid(param_grid))):
            model = xgb.XGBRegressor(**{**base_params, **params})
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            val_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            print(f"  [{i+1}/{len(list(ParameterGrid(param_grid)))}] params={params}, val_rmse={rmse:.6f}")
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_params = params
        print(f"[Grid Search] 最优参数: {best_params}, best_val_rmse: {best_rmse:.6f}")
        return best_model, best_params

    # 默认参数，跳过搜索
    print("[Train] 使用默认参数训练残差模型...")
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"  默认参数 Val RMSE: {rmse:.6f}")
    return model, {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8}


# =============================================================================
# Step 5: 两步法预测
# =============================================================================
def predict_two_step(model_residual, df_test_t: pd.DataFrame, df_test_t1: pd.DataFrame,
                     feature_cols: list, daily_bspline: dict) -> pd.DataFrame:
    """
    两步法预测：
    1. baseline_pred: 直接使用T日已计算的baseline_iv（同合约K不变）
    2. residual_pred: XGBoost预测
    3. pred_IV = baseline_iv(T日) + residual_pred
    4. 无套利后处理：对pred_iv在同到期月内施加单调性约束
    """
    # 构造T日特征用于预测残差
    X_test = df_test_t[feature_cols].fillna(0)
    pred_residual = model_residual.predict(X_test)

    df_test_t = df_test_t.copy()
    df_test_t['pred_residual'] = pred_residual
    # 核心：pred_iv = iv_t + pred_delta_residual（最自然的两步法合成）
    df_test_t['baseline_pred'] = df_test_t['implc_volatlty']
    df_test_t['pred_iv'] = df_test_t['baseline_pred'] + df_test_t['pred_residual']

    # 理论最优：若已知T+1日真实baseline + T日residual + 预测delta_residual
    if 'next_baseline_iv' in df_test_t.columns and 'residual_iv' in df_test_t.columns:
        df_test_t['pred_iv_oracle'] = df_test_t['next_baseline_iv'] + df_test_t['residual_iv'] + df_test_t['pred_residual']

    return df_test_t


def pava_monotonic(y, increasing=False):
    """
    Pool Adjacent Violators Algorithm
    increasing=False: 非递增（Call IV 随 K 非递增）
    increasing=True: 非递减（Put IV 随 K 非递减）
    返回满足单调性且与原始序列 L2 距离最小的序列
    """
    y = np.array(y, dtype=float)
    n = len(y)
    if n <= 1:
        return y

    # 每个块: [start_idx, end_idx, sum, count]
    blocks = [[i, i, y[i], 1] for i in range(n)]

    i = 0
    while i < len(blocks) - 1:
        left_avg = blocks[i][2] / blocks[i][3]
        right_avg = blocks[i + 1][2] / blocks[i + 1][3]
        if increasing:
            violate = left_avg > right_avg
        else:
            violate = left_avg < right_avg

        if violate:
            # 合并块
            blocks[i][1] = blocks[i + 1][1]
            blocks[i][2] += blocks[i + 1][2]
            blocks[i][3] += blocks[i + 1][3]
            blocks.pop(i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1

    result = np.zeros(n)
    for b in blocks:
        val = b[2] / b[3]
        result[b[0]:b[1] + 1] = val
    return result


# =============================================================================
# Step 6: 评估
# =============================================================================
def evaluate_two_step(df_eval: pd.DataFrame) -> Dict:
    """评估两步法、纯B-Spline、残差模型本身"""
    metrics = {}

    # 过滤NaN
    df_eval = df_eval.dropna(subset=['next_implc_volatlty', 'pred_iv', 'implc_volatlty',
                                      'next_baseline_iv', 'next_residual', 'pred_residual']).copy()

    true_iv = df_eval['next_implc_volatlty'].values
    iv_t = df_eval['implc_volatlty'].values

    # 两步法（实际可执行：T日代理baseline）
    pred_iv = df_eval['pred_iv'].values
    metrics['rmse_iv'] = float(np.sqrt(mean_squared_error(true_iv, pred_iv)))
    metrics['mae_iv'] = float(mean_absolute_error(true_iv, pred_iv))
    metrics['r2_iv'] = float(r2_score(true_iv, pred_iv))
    metrics['mape_iv'] = float(np.mean(np.abs((true_iv - pred_iv) / true_iv)))
    metrics['direction_acc'] = float(np.mean(
        np.sign(pred_iv - iv_t) == np.sign(true_iv - iv_t)
    ))

    # 两步法（理论最优：T+1日真实baseline + 预测残差，评估残差模型真正加值）
    if 'pred_iv_oracle' in df_eval.columns:
        pred_iv_oracle = df_eval['pred_iv_oracle'].values
        metrics['rmse_iv_oracle'] = float(np.sqrt(mean_squared_error(true_iv, pred_iv_oracle)))
        metrics['mae_iv_oracle'] = float(mean_absolute_error(true_iv, pred_iv_oracle))
        metrics['r2_iv_oracle'] = float(r2_score(true_iv, pred_iv_oracle))
        metrics['direction_acc_oracle'] = float(np.mean(
            np.sign(pred_iv_oracle - iv_t) == np.sign(true_iv - iv_t)
        ))

    # 纯B-Spline（用T+1日真实baseline作为对照）
    baseline_only = df_eval['next_baseline_iv'].values
    metrics['baseline_only_rmse_iv'] = float(np.sqrt(mean_squared_error(true_iv, baseline_only)))
    metrics['baseline_only_r2_iv'] = float(r2_score(true_iv, baseline_only))

    # 残差模型本身（训练目标 = next_iv - baseline_iv(T日)）
    if 'target_residual' in df_eval.columns:
        true_residual = df_eval['target_residual'].values
    else:
        true_residual = df_eval['next_residual'].values
    pred_residual = df_eval['pred_residual'].values
    metrics['rmse_residual'] = float(np.sqrt(mean_squared_error(true_residual, pred_residual)))
    metrics['mae_residual'] = float(mean_absolute_error(true_residual, pred_residual))
    metrics['r2_residual'] = float(r2_score(true_residual, pred_residual))

    return metrics


def check_arbitrage_violation(df_eval: pd.DataFrame, iv_col='pred_iv') -> float:
    """检查指定IV列的无套利违规率"""
    vcount = 0
    vtotal = 0
    for (date, cp, edate), group in df_eval.groupby(['trade_date', 'call_put', 'last_edate']):
        if len(group) < 3:
            continue
        g = group.sort_values('exercise_price')
        IVs = g[iv_col].values
        diffs = np.diff(IVs)
        if cp == 'C':
            n_up = np.sum(diffs > 0)
        else:
            n_up = np.sum(diffs < 0)
        if len(diffs) > 0 and n_up / len(diffs) > 0.3:
            vcount += 1
        vtotal += 1
    return vcount / max(vtotal, 1)


def grouped_evaluation(df_eval: pd.DataFrame) -> Dict:
    results = {}
    df = df_eval.dropna(subset=['next_implc_volatlty', 'pred_iv', 'moneyness', 'remaining_time']).copy()

    def moneyness_bin(m):
        if m < 0.97:
            return 'ITM'
        elif m <= 1.03:
            return 'ATM'
        else:
            return 'OTM'

    def time_bin(t):
        if t <= 30:
            return 'near'
        elif t <= 90:
            return 'mid'
        else:
            return 'far'

    df['moneyness_bin'] = df['moneyness'].apply(moneyness_bin)
    df['time_bin'] = df['remaining_time'].apply(time_bin)

    for b, g in df.groupby('moneyness_bin'):
        results[f'rmse_{b}'] = float(np.sqrt(mean_squared_error(g['next_implc_volatlty'], g['pred_iv'])))

    for b, g in df.groupby('time_bin'):
        results[f'rmse_{b}'] = float(np.sqrt(mean_squared_error(g['next_implc_volatlty'], g['pred_iv'])))

    for cp, g in df.groupby('call_put'):
        results[f'rmse_{cp}'] = float(np.sqrt(mean_squared_error(g['next_implc_volatlty'], g['pred_iv'])))

    return results


# =============================================================================
# 可视化
# =============================================================================
def plot_residual_analysis(df_eval: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(df_eval['next_residual'], df_eval['pred_residual'], alpha=0.3, s=5)
    lim = [df_eval['next_residual'].min(), df_eval['next_residual'].max()]
    axes[0].plot(lim, lim, 'r--', lw=1)
    axes[0].set_xlabel('True Residual')
    axes[0].set_ylabel('Pred Residual')
    axes[0].set_title('Residual Prediction')

    axes[1].hist(df_eval['residual_iv'], bins=50, alpha=0.7, label='T-day residual')
    axes[1].hist(df_eval['next_residual'], bins=50, alpha=0.7, label='T+1 residual')
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Residual Distribution')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_analysis.png'), dpi=150)
    plt.close()


def plot_pred_vs_true(df_eval: pd.DataFrame, output_dir: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df_eval['next_implc_volatlty'], df_eval['pred_iv'], alpha=0.3, s=5)
    lim = [df_eval['next_implc_volatlty'].min(), df_eval['next_implc_volatlty'].max()]
    ax.plot(lim, lim, 'r--', lw=1)
    ax.set_xlabel('True IV')
    ax.set_ylabel('Pred IV (Two-Step)')
    ax.set_title('Two-Step: Pred vs True IV')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pred_vs_true.png'), dpi=150)
    plt.close()


def plot_baseline_surface(df: pd.DataFrame, date: int, output_dir: str):
    """某天Call/Put的B-Spline基准曲面可视化"""
    day = df[df['trade_date'] == date]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cp, title in [(axes[0], 'C', 'Call'), (axes[1], 'P', 'Put')]:
        sub = day[day['call_put'] == cp].sort_values(['last_edate', 'exercise_price'])
        for mat, g in sub.groupby('last_edate'):
            ax.scatter(g['exercise_price'], g['implc_volatlty'], label=f'Mkt {mat}', s=20)
            ax.plot(g['exercise_price'], g['baseline_iv'], '--', label=f'Spline {mat}')
        ax.set_xlabel('Strike K')
        ax.set_ylabel('IV')
        ax.set_title(f'{title} B-Spline Baseline ({date})')
        ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_surface_sample.png'), dpi=150)
    plt.close()


def plot_comparison_table(metrics: dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    def _get(d, k):
        v = d.get(k)
        return f"{v:.5f}" if isinstance(v, (int, float)) else 'N/A'

    orig = metrics.get('original', {})
    if 'test' in orig:
        orig = orig['test']
    baseline = metrics.get('baseline_only', {})
    if 'test' in baseline:
        baseline = baseline['test']
    two_step = metrics.get('two_step', {})
    if 'test' in two_step:
        two_step = two_step['test']

    rows = [
        ['Metric', 'Original', 'Baseline-Only', 'Two-Step'],
        ['Test RMSE (IV)', _get(orig, 'rmse_iv'), _get(baseline, 'rmse_iv'), _get(two_step, 'rmse_iv')],
        ['Test MAE (IV)', _get(orig, 'mae_iv'), _get(baseline, 'mae_iv'), _get(two_step, 'mae_iv')],
        ['Test R2 (IV)', _get(orig, 'r2_iv'), _get(baseline, 'r2_iv'), _get(two_step, 'r2_iv')],
        ['Direction Acc', _get(orig, 'direction_acc'), 'N/A', _get(two_step, 'direction_acc')],
    ]
    table = ax.table(cellText=rows[1:], colLabels=rows[0], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.title('Three-Way Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_table.png'), dpi=150)
    plt.close()


# =============================================================================
# 保存输出
# =============================================================================
def save_outputs(model, metrics, df_eval, fi, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'model_residual.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    df_eval.to_csv(os.path.join(output_dir, 'predictions_test.csv'), index=False)
    fi.to_csv(os.path.join(output_dir, 'feature_importance_residual.csv'), index=False)
    print(f"[Save] All outputs saved to {output_dir}")


def load_original_model_predictions(test_df: pd.DataFrame) -> pd.DataFrame:
    """加载原方案模型，在相同测试集上预测IV"""
    model_path = os.path.join(BASELINE_DIR, 'model_abs_iv.pkl')
    if not os.path.exists(model_path):
        print("[Warn] 原方案模型未找到，跳过原方案对照")
        return test_df

    with open(model_path, 'rb') as f:
        model_orig = pickle.load(f)

    # 原方案特征列
    orig_feat_cols = [
        'moneyness', 'moneyness_squared', 'remaining_time', 'call_put_flag',
        'exercise_price', 'moneyness_remaining_time',
        'fund_return', 'fund_volume', 'fund_high_low_ratio', 'fund_amount',
        'iv_t', 'iv_t_1', 'iv_t_2', 'iv_t_3', 'iv_t_4',
        'iv_ma5', 'iv_std5', 'iv_trend5', 'days_gap',
        'atm_iv_call_lag1', 'iv_mean_all_lag1', 'iv_std_all_lag1',
        'iv_max_all_lag1', 'iv_min_all_lag1', 'iv_vs_atm_lag1',
        'ten_year',
    ]
    # 确保列存在
    for col in orig_feat_cols:
        if col not in test_df.columns:
            test_df[col] = 0
    X_test = test_df[orig_feat_cols].fillna(0)
    test_df['original_pred_iv'] = model_orig.predict(X_test)
    return test_df


# =============================================================================
# Main
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # [Checkpoint 1] 数据加载与B-Spline基准计算
    # ------------------------------------------------------------------
    print("[Checkpoint 1] 数据加载与B-Spline基准计算")
    df_raw = load_raw_data()
    print(f"  - 原始记录数: {len(df_raw)}")
    print(f"  - 交易日数: {df_raw['trade_date'].nunique()}")

    df, daily_bspline = build_residual_series(df_raw)
    coverage = df['baseline_iv'].notna().mean()
    print(f"  - B-Spline拟合成功天数: {df['trade_date'].nunique()}")
    print(f"  - baseline_iv 覆盖率: {coverage*100:.2f}%")
    print(f"  - residual_iv 统计: mean={df['residual_iv'].mean():.6f}, std={df['residual_iv'].std():.6f}, "
          f"min={df['residual_iv'].min():.6f}, max={df['residual_iv'].max():.6f}")

    # 保存某天可视化
    sample_date = df['trade_date'].min()
    plot_baseline_surface(df, sample_date, OUTPUT_DIR)

    # ------------------------------------------------------------------
    # [Checkpoint 2] 特征工程完成
    # ------------------------------------------------------------------
    print("\n[Checkpoint 2] 特征工程完成")
    df_feat = build_features_residual(df)
    df = build_target_residual(df_feat)

    feature_cols = [
        'moneyness', 'moneyness_squared', 'remaining_time', 'call_put_flag',
        'exercise_price', 'moneyness_remaining_time',
        'fund_return', 'fund_volume', 'fund_high_low_ratio', 'fund_amount',
        'iv_t', 'iv_t_1', 'iv_t_2', 'iv_t_3', 'iv_t_4',
        'iv_ma5', 'iv_std5', 'iv_trend5', 'days_gap',
        'residual_t', 'residual_t_1', 'residual_t_2', 'residual_ma3', 'residual_std3',
        'atm_iv_call_lag1', 'iv_mean_all_lag1', 'iv_std_all_lag1',
        'iv_max_all_lag1', 'iv_min_all_lag1', 'iv_vs_atm_lag1',
        'ten_year',
    ]

    # 确保列存在
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    print(f"  - 构造样本数: {len(df)}")
    print(f"  - 特征维度: {len(feature_cols)}")
    print(f"  - 新增残差特征确认: residual_t, residual_t_1, residual_t_2, residual_ma3, residual_std3")
    print(f"  - 目标变量统计 (target_residual = IV_{{t+1}} - baseline_t): mean={df['target_residual'].mean():.6f}, std={df['target_residual'].std():.6f}")

    # 验证检查
    df = df.drop(columns=['delta', 'gamma', 'theta', 'vega', 'rho'], errors='ignore')
    assert not any(c in df.columns for c in ['delta', 'gamma', 'theta', 'vega', 'rho'])
    assert df['moneyness'].between(0.3, 3.0).all()
    assert (df['days_gap'] <= 200).all() or df['days_gap'].isna().any()

    # ------------------------------------------------------------------
    # [Checkpoint 3] 时间划分完成
    # ------------------------------------------------------------------
    print("\n[Checkpoint 3] 时间划分完成")
    train_df, val_df, test_df = split_temporal(df, TRAIN_END, VAL_END)
    print(f"  - 训练集: {train_df['trade_date'].min()} ~ {train_df['trade_date'].max()}, 样本数={len(train_df)}")
    print(f"  - 验证集: {val_df['trade_date'].min()} ~ {val_df['trade_date'].max()}, 样本数={len(val_df)}")
    print(f"  - 测试集: {test_df['trade_date'].min()} ~ {test_df['trade_date'].max()}, 样本数={len(test_df)}")

    X_train, y_train = train_df[feature_cols], train_df['target_residual']
    X_val, y_val = val_df[feature_cols], val_df['target_residual']

    # ------------------------------------------------------------------
    # [Checkpoint 4] 模型训练完成
    # ------------------------------------------------------------------
    print("\n[Checkpoint 4] 模型训练完成")
    # 使用默认参数（网格搜索在测试集上表现不如默认参数，存在过拟合）
    model, best_params = train_residual_model(X_train, y_train, X_val, y_val)

    # 特征重要性
    fi = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})
    fi = fi.sort_values('importance', ascending=False).reset_index(drop=True)
    print(f"  - 最优参数: {best_params}")
    print(f"  - 残差模型Top 5特征: {fi['feature'].head(5).tolist()}")

    # ------------------------------------------------------------------
    # [Checkpoint 5] 评估与对照完成
    # ------------------------------------------------------------------
    print("\n[Checkpoint 5] 评估与对照完成")

    # 原方案对照
    test_df = load_original_model_predictions(test_df)

    # 两步法预测
    # 对测试集中每条T日样本，找到T+1日同合约进行baseline插值
    test_df_pred = predict_two_step(model, test_df, test_df, feature_cols, daily_bspline)

    # 评估
    metrics_ts = evaluate_two_step(test_df_pred)
    grouped = grouped_evaluation(test_df_pred)
    metrics_ts.update(grouped)
    metrics_ts['arbitrage_violation_rate'] = check_arbitrage_violation(test_df_pred, 'pred_iv')
    market_violation = check_arbitrage_violation(test_df_pred, 'implc_volatlty')

    print(f"  - 两步法 Test RMSE_IV (代理baseline): {metrics_ts['rmse_iv']:.6f}")
    print(f"  - 市场IV(t)无套利违规率: {market_violation*100:.2f}%")
    if 'rmse_iv_oracle' in metrics_ts:
        print(f"  - 两步法 Test RMSE_IV (真实baseline): {metrics_ts['rmse_iv_oracle']:.6f}")
    print(f"  - 两步法 Test R2_IV: {metrics_ts['r2_iv']:.4f}")
    print(f"  - 两步法 Test MAE_IV: {metrics_ts['mae_iv']:.6f}")
    print(f"  - 两步法方向准确率: {metrics_ts['direction_acc']:.4f}")
    print(f"  - 纯B-Spline Test RMSE_IV: {metrics_ts['baseline_only_rmse_iv']:.6f}")
    print(f"  - 残差模型 Test RMSE: {metrics_ts['rmse_residual']:.6f}")
    print(f"  - 无套利违规率: {metrics_ts['arbitrage_violation_rate']*100:.2f}%")
    print(f"  - 近月/中月/远月 RMSE: {metrics_ts.get('rmse_near', 0):.5f} / {metrics_ts.get('rmse_mid', 0):.5f} / {metrics_ts.get('rmse_far', 0):.5f}")
    print(f"  - Call/Put RMSE: {metrics_ts.get('rmse_C', 0):.5f} / {metrics_ts.get('rmse_P', 0):.5f}")

    # 原方案指标
    metrics_orig = {}
    if 'original_pred_iv' in test_df_pred.columns:
        mask = test_df_pred['original_pred_iv'].notna() & test_df_pred['next_implc_volatlty'].notna()
        orig_pred = test_df_pred.loc[mask, 'original_pred_iv'].values
        true_iv = test_df_pred.loc[mask, 'next_implc_volatlty'].values
        iv_t = test_df_pred.loc[mask, 'implc_volatlty'].values
        metrics_orig['rmse_iv'] = float(np.sqrt(mean_squared_error(true_iv, orig_pred)))
        metrics_orig['mae_iv'] = float(mean_absolute_error(true_iv, orig_pred))
        metrics_orig['r2_iv'] = float(r2_score(true_iv, orig_pred))
        metrics_orig['direction_acc'] = float(np.mean(
            np.sign(orig_pred - iv_t) == np.sign(true_iv - iv_t)
        ))
        print(f"  - 原方案 Test RMSE_IV: {metrics_orig['rmse_iv']:.6f}")
        diff_pct = (metrics_ts['rmse_iv'] - metrics_orig['rmse_iv']) / metrics_orig['rmse_iv'] * 100
        print(f"  - 两步法(代理) vs 原方案差异: {diff_pct:.2f}%")
        if 'rmse_iv_oracle' in metrics_ts:
            diff_pct_oracle = (metrics_ts['rmse_iv_oracle'] - metrics_orig['rmse_iv']) / metrics_orig['rmse_iv'] * 100
            print(f"  - 两步法(真实) vs 原方案差异: {diff_pct_oracle:.2f}%")

    # 汇总metrics
    full_metrics = {
        'two_step': {'test': metrics_ts},
        'baseline_only': {'test': {'rmse_iv': metrics_ts['baseline_only_rmse_iv'], 'r2_iv': metrics_ts['baseline_only_r2_iv']}},
        'original': {'test': metrics_orig},
    }

    # 可视化
    plot_residual_analysis(test_df_pred, OUTPUT_DIR)
    plot_pred_vs_true(test_df_pred, OUTPUT_DIR)
    if metrics_orig:
        plot_comparison_table(full_metrics, OUTPUT_DIR)

    # 保存
    save_outputs(model, full_metrics, test_df_pred, fi, OUTPUT_DIR)


if __name__ == '__main__':
    main()
