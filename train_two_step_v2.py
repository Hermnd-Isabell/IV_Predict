# -*- coding: utf-8 -*-
"""
50ETF期权IV预测 - 路线B: M-W空间B-Spline + 残差预测(v2)
核心改进:
  1. K空间 -> Log-moneyness M = ln(K/F_implied)
  2. 理论远期 -> Put-Call Parity隐含远期
  3. IV空间 -> 总方差 W = IV^2 * tau 空间插值
  4. 预测时T日曲面插值到T+1日位置
合成: pred_IV = interp_baseline(M_{t+1}, tau_{t+1}) + residual_t + pred_delta_residual
"""

import os
import json
import pickle
import warnings
from typing import Tuple, Dict, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid
import xgboost as xgb

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RANDOM_STATE = 42
OUTPUT_DIR = 'data/output/two_step_v2/'
BASELINE_DIR = 'data/output/baseline_xgb/'
OLD_TWOSTEP_DIR = 'data/output/two_step_residual/'

TRAIN_END = 20240630
VAL_END = 20241231

FEATURE_COLS = [
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


# =============================================================================
# Step 0: 数据加载
# =============================================================================
def load_raw_data(path='data/raw/50etf_options.csv') -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# =============================================================================
# Step 1: 隐含远期 F_implied 计算
# =============================================================================
def compute_implied_forward(df_day: pd.DataFrame, maturity_mask: pd.Series,
                            r: float, tau: float) -> Optional[float]:
    """
    对单日单到期月用Put-Call Parity反推隐含远期。
    F = K + e^(r*tau) * (C - P)
    取中位数(防异常值)。
    """
    calls = df_day[maturity_mask & (df_day['call_put'] == 'C')]
    puts = df_day[maturity_mask & (df_day['call_put'] == 'P')]

    common_strikes = sorted(set(calls['exercise_price']) & set(puts['exercise_price']))
    if len(common_strikes) == 0:
        return None

    F_list = []
    cp_diffs = []
    for K in common_strikes:
        c_rows = calls[calls['exercise_price'] == K]
        p_rows = puts[puts['exercise_price'] == K]
        if len(c_rows) == 0 or len(p_rows) == 0:
            continue
        C = c_rows['close'].values[0]
        P_ = p_rows['close'].values[0]
        cp_diffs.append(abs(C - P_))
        F = K + np.exp(r * tau) * (C - P_)
        F_list.append(F)

    if len(F_list) == 0:
        return None

    # 异常值过滤: |C-P| > 3 * median(|C-P|)
    if len(cp_diffs) >= 3:
        med_diff = np.median(cp_diffs)
        valid = [F_list[i] for i in range(len(F_list)) if cp_diffs[i] <= 3 * med_diff]
        if len(valid) >= 1:
            F_list = valid

    if len(F_list) >= 2:
        return float(np.median(F_list))
    return float(F_list[0])


def build_forward_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    全局计算每天每到期月的F_implied（向量化版本，0.3秒完成）。
    输出: forward_table [trade_date, last_edate, F_implied, F_theory, r, tau]
    """
    t0 = pd.Timestamp.now()

    # Step 1: pivot call/put close prices
    sub = df[['trade_date', 'last_edate', 'exercise_price', 'call_put', 'close']].copy()
    pivoted = sub.pivot_table(index=['trade_date', 'last_edate', 'exercise_price'],
                              columns='call_put', values='close').reset_index()
    pivoted = pivoted.dropna(subset=['C', 'P'])

    # Step 2: group meta
    group_meta = df.groupby(['trade_date', 'last_edate']).first()[['ten_year', 'remaining_time', 'fund_close']]
    group_meta['r'] = group_meta['ten_year'] / 100.0
    group_meta['tau'] = group_meta['remaining_time'] / 365.0
    group_meta['F_theory'] = group_meta['fund_close'] * np.exp(group_meta['r'] * group_meta['tau'])

    # Step 3: compute F per strike pair
    merged = pivoted.merge(group_meta[['r', 'tau']].reset_index(), on=['trade_date', 'last_edate'])
    merged['F'] = merged['exercise_price'] + np.exp(merged['r'] * merged['tau']) * (merged['C'] - merged['P'])

    # Step 4: median F per group
    F_implied = merged.groupby(['trade_date', 'last_edate'])['F'].median().reset_index()
    F_implied.columns = ['trade_date', 'last_edate', 'F_implied']

    # Step 5: combine
    forward_table = group_meta[['F_theory']].reset_index().merge(
        F_implied, on=['trade_date', 'last_edate'], how='left')
    forward_table['F_implied'] = forward_table['F_implied'].fillna(forward_table['F_theory'])
    forward_table['F'] = forward_table['F_implied']  # alias for fit_mw_surface
    forward_table = forward_table.merge(
        group_meta[['r', 'tau']].reset_index(), on=['trade_date', 'last_edate'])

    print(f"    [build_forward_table] elapsed: {(pd.Timestamp.now() - t0).total_seconds():.2f}s")
    return forward_table


# =============================================================================
# Step 2: M-W空间 B-Spline 曲面拟合
# =============================================================================
def fit_mw_surface(df_day: pd.DataFrame, call_put: str,
                   forward_table_day: pd.DataFrame) -> Tuple[Dict, pd.Series]:
    """
    对单日单类型(C/P)在M-W空间拟合B-Spline。
    按last_edate分组，每组沿M方向拟合LSQUnivariateSpline(M, W)。
    通过减少内部节点数强制平滑，避免精确插值导致residual恒为0。

    返回:
      - spline_dict: {last_edate: (M_nodes, W_nodes, cs_object, F_implied, tau_mat)}
      - baseline_series: Series of baseline_IV, indexed by contract index in df_day
    """
    df_sub = df_day[df_day['call_put'] == call_put].copy()
    # Filter extreme outlier IVs before B-Spline fitting
    df_sub = df_sub[(df_sub['remaining_time'] > 0) & (df_sub['implc_volatlty'] > 0)]
    df_sub = df_sub[df_sub['implc_volatlty'] <= 1.0]

    if len(df_sub) == 0:
        return {}, pd.Series(dtype=float)

    baseline = pd.Series(np.nan, index=df_sub.index)
    spline_dict = {}
    valid_mats = []

    for mat, df_mat in df_sub.groupby('last_edate'):
        df_mat = df_mat.sort_values('exercise_price')
        if len(df_mat) < 2:
            continue

        # Lookup F from forward_table (F = F_implied with F_theory fallback)
        fwd = forward_table_day[forward_table_day['last_edate'] == mat]
        if len(fwd) == 0:
            continue
        F = fwd['F'].values[0]
        tau_mat_val = fwd['tau'].values[0]

        K = df_mat['exercise_price'].values.astype(float)
        IV = df_mat['implc_volatlty'].values.astype(float)

        # M = ln(K/F), W = IV^2 * tau
        M = np.log(K / F)
        W = IV ** 2 * tau_mat_val

        # 去重M并排序
        uniq = pd.DataFrame({'M': M, 'W': W}).groupby('M')['W'].mean().reset_index().sort_values('M')
        Mu, Wu = uniq['M'].values, uniq['W'].values

        n = len(Mu)
        if n < 4:
            # 合约数不足，跳过该到期月（不拟合，baseline设为NaN后续填充）
            continue

        # 动态节点数：合约越多节点越多，但始终少于数据点
        n_knots = max(3, min(n - 2, int(n * 0.4)))

        # 在 Mu 的分布范围内均匀放置内部节点（避开边界 10%，防止外推压力）
        t = np.quantile(Mu, np.linspace(0.10, 0.90, n_knots))
        t = np.unique(t)

        # 确保至少 3 个内部节点
        if len(t) < 3:
            m_min, m_max = Mu.min(), Mu.max()
            t = np.linspace(m_min + 0.01, m_max - 0.01, 3)

        # 拟合 LSQUnivariateSpline（k=3 为三次样条）
        try:
            cs = LSQUnivariateSpline(Mu, Wu, t=t, k=3)
        except Exception as e:
            print(f"[WARN] LSQUnivariateSpline failed for mat={mat}, n={n}, fallback to UnivariateSpline: {e}")
            cs = UnivariateSpline(Mu, Wu, s=1e-5)

        # 查询: 每个合约的M位置
        M_query = np.log(df_mat['exercise_price'].values.astype(float) / F)
        W_interp = cs(M_query)

        # 平坦外推：M在范围外时用边界值
        W_interp = np.where(M_query < Mu[0], Wu[0], W_interp)
        W_interp = np.where(M_query > Mu[-1], Wu[-1], W_interp)

        # W > 0 保护
        W_interp = np.maximum(W_interp, 1e-6)

        # 还原 baseline_IV = sqrt(W / tau)
        baseline_IV = np.sqrt(W_interp / tau_mat_val)
        baseline.loc[df_mat.index] = baseline_IV
        valid_mats.append(np.mean(baseline_IV))

        spline_dict[mat] = (Mu, Wu, cs, F, tau_mat_val)

    # Fill unfilled with mean of valid mats
    if baseline.isna().any():
        fill_val = np.mean(valid_mats) if valid_mats else df_sub['implc_volatlty'].mean()
        baseline = baseline.fillna(fill_val)

    return spline_dict, baseline


def build_residual_series(df: pd.DataFrame, forward_table: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    全局计算baseline_IV和residual。
    保存 daily_mw_data = {(date, call_put, mat): (M_nodes, W_nodes, cs, F, tau_mat)}
    优化: forward_table转为dict O(1)查询, M向量化计算.
    """
    import time
    t0 = time.time()
    df = df.copy()
    # Exclude expiration days and zero IV
    df = df[(df['remaining_time'] > 0) & (df['implc_volatlty'] > 0)].copy()
    # Filter extreme outliers
    df.loc[df['implc_volatlty'] > 1.0, 'implc_volatlty'] = np.nan
    df = df.dropna(subset=['implc_volatlty']).copy()
    df['baseline_iv'] = np.nan
    df['residual_iv'] = np.nan
    df['M'] = np.nan  # log-moneyness

    # Pre-index forward_table for O(1) lookup: {(date, mat): {F, tau}}
    fwd_dict = {}
    for _, row in forward_table.iterrows():
        fwd_dict[(row['trade_date'], row['last_edate'])] = {
            'F': row['F'], 'tau': row['tau']
        }

    daily_mw_data = {}
    fit_days = 0
    total_contracts_fitted = 0
    M_min_all, M_max_all = float('inf'), float('-inf')
    W_min_all, W_max_all = float('inf'), float('-inf')

    groups = list(df.groupby(['trade_date', 'call_put']))
    n_groups = len(groups)
    for i, ((date, cp), group) in enumerate(groups):
        if i % 500 == 0 and i > 0:
            print(f"    ... processed {i}/{n_groups} groups ({time.time()-t0:.1f}s)")
        # Build per-day forward_table subset from dict
        mats_in_group = group['last_edate'].unique()
        fwd_rows = []
        for mat in mats_in_group:
            key = (date, mat)
            if key in fwd_dict:
                d = fwd_dict[key]
                fwd_rows.append({'last_edate': mat, 'F': d['F'], 'tau': d['tau']})
        if len(fwd_rows) == 0:
            continue
        fwd_day = pd.DataFrame(fwd_rows)
        spline_dict, baseline = fit_mw_surface(group, cp, fwd_day)

        if len(spline_dict) > 0:
            fit_days += 1
            total_contracts_fitted += baseline.notna().sum()
            for mat, (Mu, Wu, cs, F, tau_mat) in spline_dict.items():
                daily_mw_data[(date, cp, mat)] = (Mu, Wu, cs, F, tau_mat)
                M_min_all = min(M_min_all, Mu.min())
                M_max_all = max(M_max_all, Mu.max())
                W_min_all = min(W_min_all, Wu.min())
                W_max_all = max(W_max_all, Wu.max())

        # Vectorized M computation for this group
        if len(spline_dict) > 0:
            for mat, (Mu, Wu, cs, F, tau_mat) in spline_dict.items():
                mask = (df['trade_date'] == date) & (df['call_put'] == cp) & (df['last_edate'] == mat)
                if mask.any():
                    df.loc[mask, 'M'] = np.log(df.loc[mask, 'exercise_price'].values / F)

        df.loc[baseline.index, 'baseline_iv'] = baseline.values

    df['residual_iv'] = df['implc_volatlty'] - df['baseline_iv']

    n_trade_days = df['trade_date'].nunique()
    print(f"  [MW Spline] elapsed: {time.time()-t0:.1f}s")
    print(f"  [MW Spline] Fit success days: {fit_days}/{n_trade_days}")
    print(f"  [MW Spline] Avg contracts/day/type: {total_contracts_fitted/max(fit_days,1):.1f}")
    print(f"  [MW Spline] M range: [{M_min_all:.4f}, {M_max_all:.4f}]")
    print(f"  [MW Spline] W range: [{W_min_all:.6f}, {W_max_all:.6f}]")
    print(f"  [MW Spline] Residual stats: mean={df['residual_iv'].mean():.6f}, std={df['residual_iv'].std():.6f}, "
          f"min={df['residual_iv'].min():.6f}, max={df['residual_iv'].max():.6f}")

    return df, daily_mw_data


# =============================================================================
# Step 3: 特征工程（复用原方案 + 残差历史）
# =============================================================================
def build_features_delta_residual(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(['security_id', 'trade_date']).reset_index(drop=True)

    # 合约固有
    df['moneyness'] = df['fund_close'] / df['exercise_price']
    df['moneyness_squared'] = df['moneyness'] ** 2
    df['call_put_flag'] = (df['call_put'] == 'C').astype(int)
    df['moneyness_remaining_time'] = df['moneyness'] * df['remaining_time']

    # 标的数据
    spot_daily = df[['trade_date', 'fund_close', 'fund_volume', 'fund_amount',
                     'fund_high', 'fund_low']].drop_duplicates('trade_date').sort_values('trade_date')
    spot_daily['fund_return'] = spot_daily['fund_close'].pct_change()
    spot_daily['fund_high_low_ratio'] = (spot_daily['fund_high'] - spot_daily['fund_low']) / spot_daily['fund_close']
    df = df.merge(spot_daily[['trade_date', 'fund_return', 'fund_high_low_ratio']],
                  on='trade_date', how='left')

    # 日期映射
    trade_dates = np.sort(df['trade_date'].unique())
    date_map = pd.DataFrame({'trade_date': trade_dates,
                             'trade_dt': pd.to_datetime(trade_dates, format='%Y%m%d')})
    df = df.merge(date_map, on='trade_date', how='left')

    # 历史IV + 历史残差（按合约分组）
    def compute_lags(group):
        group = group.sort_values('trade_date')
        group['iv_t'] = group['implc_volatlty']
        group['iv_t_1'] = group['implc_volatlty'].shift(1)
        group['iv_t_2'] = group['implc_volatlty'].shift(2)
        group['iv_t_3'] = group['implc_volatlty'].shift(3)
        group['iv_t_4'] = group['implc_volatlty'].shift(4)
        group['iv_ma5'] = group['implc_volatlty'].rolling(window=5, min_periods=1).mean()
        group['iv_std5'] = group['implc_volatlty'].rolling(window=5, min_periods=2).std()
        group['iv_trend5'] = group['implc_volatlty'] - group['implc_volatlty'].shift(4)
        # 残差历史
        group['residual_t'] = group['residual_iv']
        group['residual_t_1'] = group['residual_iv'].shift(1)
        group['residual_t_2'] = group['residual_iv'].shift(2)
        group['residual_ma3'] = group['residual_iv'].rolling(window=3, min_periods=1).mean()
        group['residual_std3'] = group['residual_iv'].rolling(window=3, min_periods=2).std()
        group['days_gap'] = (group['trade_dt'] - group['trade_dt'].shift(1)).dt.days
        return group

    df = df.groupby('security_id', group_keys=False).apply(compute_lags)

    # 曲面上下文(t-1日)
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

    # 宏观
    df['ten_year'] = df['ten_year'] / 100.0

    # 填充缺失
    for col in ['atm_iv_call_lag1', 'iv_mean_all_lag1', 'iv_std_all_lag1',
                'iv_max_all_lag1', 'iv_min_all_lag1', 'iv_vs_atm_lag1']:
        df[col] = df[col].fillna(df[col].mean())
    df['days_gap'] = df['days_gap'].fillna(1)
    df['fund_return'] = df['fund_return'].fillna(0)
    for col in ['iv_t_1', 'iv_t_2', 'iv_t_3', 'iv_t_4', 'iv_std5', 'iv_trend5',
                'residual_t_1', 'residual_t_2', 'residual_std3']:
        if 'iv' in col:
            df[col] = df[col].fillna(df['iv_t'])
        else:
            df[col] = df[col].fillna(df['residual_t'])

    return df


# =============================================================================
# Step 3b: 目标变量构造
# =============================================================================
def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造目标变量：
    - target_residual = residual_{t+1} - residual_t (delta_residual)
    - next_implc_volatlty = IV_{t+1} (用于最终评估)
    - next_baseline_iv = baseline_{t+1} (用于评估baseline proxy质量)
    - next_exercise_price, next_remaining_time, next_M (用于预测时插值)
    """
    df = df.copy().sort_values(['security_id', 'trade_date']).reset_index(drop=True)
    df['next_trade_date'] = df.groupby('security_id')['trade_date'].shift(-1)
    df['next_last_edate'] = df.groupby('security_id')['last_edate'].shift(-1)
    df['next_residual'] = df.groupby('security_id')['residual_iv'].shift(-1)
    df['next_implc_volatlty'] = df.groupby('security_id')['implc_volatlty'].shift(-1)
    df['next_baseline_iv'] = df.groupby('security_id')['baseline_iv'].shift(-1)
    df['next_remaining_time'] = df.groupby('security_id')['remaining_time'].shift(-1)
    df['next_exercise_price'] = df.groupby('security_id')['exercise_price'].shift(-1)
    df['next_M'] = df.groupby('security_id')['M'].shift(-1)

    # 方案A: 预测残差绝对值 residual_{t+1}（prompt推荐优先尝试）
    df['target_residual'] = df['next_residual']

    df = df.dropna(subset=['target_residual', 'next_implc_volatlty', 'next_baseline_iv']).copy()
    return df


# =============================================================================
# Step 4: 时间划分
# =============================================================================
def split_temporal(df: pd.DataFrame, train_end: int, val_end: int) -> Tuple:
    train_df = df[df['trade_date'] <= train_end].copy()
    val_df = df[(df['trade_date'] > train_end) & (df['trade_date'] <= val_end)].copy()
    test_df = df[df['trade_date'] > val_end].copy()
    return train_df, val_df, test_df


# =============================================================================
# Step 5: 模型训练
# =============================================================================
def train_delta_residual_model(X_train, y_train, X_val, y_val, param_grid=None):
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
        print("[Grid Search] delta_residual模型超参数搜索...")
        combos = list(ParameterGrid(param_grid))
        for i, params in enumerate(combos):
            model = xgb.XGBRegressor(**{**base_params, **params})
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            val_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            print(f"  [{i+1}/{len(combos)}] params={params}, val_rmse={rmse:.6f}")
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_params = params
        print(f"[Grid Search] 最优参数: {best_params}, best_val_rmse: {best_rmse:.6f}")
        return best_model, best_params

    print("[Train] 使用默认参数训练delta_residual模型...")
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"  默认参数 Val RMSE(delta_residual): {rmse:.6f}")
    return model, {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8}


# =============================================================================
# Step 6: 两步法预测 (核心: T日曲面插值到T+1日)
# =============================================================================
def predict_two_step_v2(model, df_test_t: pd.DataFrame, feature_cols: list,
                        daily_mw_data: dict, forward_table: pd.DataFrame) -> pd.DataFrame:
    """
    Two-step prediction v2:
    pred_IV = interp_baseline(M_{t+1}, tau_{t+1}) + residual_t + pred_delta_residual

    For each T-day contract:
    1. Find T+1 day same contract (by security_id)
    2. Get T-day spline for same (date, call_put, mat)
    3. Compute M_{t+1} = ln(K_{t+1} / F_t)
    4. Query W at M_{t+1} on T-day spline (flat extrapolation outside range)
    5. baseline_pred = sqrt(max(W, 1e-6) / tau_{t+1})
    6. XGBoost predicts delta_residual
    7. pred_IV = baseline_pred + residual_t + pred_delta_residual
    """
    df = df_test_t.copy()
    X_test = df[feature_cols].fillna(0)
    pred_delta_residual = model.predict(X_test)
    df['pred_delta_residual'] = pred_delta_residual

    # Pre-compute forward lookup for both T and T+1
    fwd_lookup = {}
    for _, row in forward_table.iterrows():
        fwd_lookup[(row['trade_date'], row['last_edate'])] = (row['F'], row['tau'])

    baseline_pred_list = []
    baseline_t1_true_list = df['next_baseline_iv'].values.tolist()

    for idx, row in df.iterrows():
        date_t = row['trade_date']
        date_t1 = row['next_trade_date']
        cp = row['call_put']
        mat = row['last_edate']
        mat_t1 = row['next_last_edate']
        K_t1 = row['next_exercise_price']
        tau_t1 = row['next_remaining_time'] / 365.0

        # Look up F_{t+1} for computing M_{t+1} = ln(K_{t+1} / F_{t+1})
        fwd_t1 = fwd_lookup.get((date_t1, mat_t1))
        if fwd_t1 is not None:
            F_t1 = fwd_t1[0]
        else:
            # Fallback: use T-day F if T+1 not available
            fwd_t = fwd_lookup.get((date_t, mat))
            F_t1 = fwd_t[0] if fwd_t is not None else row['F']

        # Look up T-day spline for same maturity
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

            baseline_pred = np.sqrt(max(W_pred, 1e-6) / tau_t1)
        else:
            # Fallback: find closest tau maturity and use its spline
            same_day_cp = [(d, c, m) for (d, c, m) in daily_mw_data.keys()
                           if d == date_t and c == cp]
            if len(same_day_cp) > 0:
                # Find closest by remaining_time difference
                best_key = None
                best_diff = float('inf')
                for k in same_day_cp:
                    _, _, _, _, tau_k = daily_mw_data[k]
                    diff = abs(tau_k - tau_t1)
                    if diff < best_diff:
                        best_diff = diff
                        best_key = k
                if best_key is not None:
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
                    baseline_pred = np.sqrt(max(W_pred, 1e-6) / tau_t1)
                else:
                    baseline_pred = row['baseline_iv']  # fallback to T-day baseline
            else:
                baseline_pred = row['baseline_iv']  # fallback to T-day baseline

        baseline_pred_list.append(baseline_pred)

    df['baseline_iv_t1_interp'] = baseline_pred_list
    df['baseline_iv_t1_true'] = baseline_t1_true_list
    df['baseline_iv_t'] = df['baseline_iv']

    # pred_residual_t1 = residual_t + pred_delta_residual
    df['pred_residual_t1'] = df['residual_iv'] + df['pred_delta_residual']
    # pred_iv = baseline_pred + residual_t + pred_delta_residual
    df['pred_iv'] = df['baseline_iv_t1_interp'] + df['residual_iv'] + df['pred_delta_residual']

    # True values
    df['true_iv'] = df['next_implc_volatlty']
    df['true_residual_t1'] = df['next_residual']

    # Residuals for evaluation
    df['iv_residual'] = df['pred_iv'] - df['true_iv']
    df['baseline_proxy_residual'] = df['baseline_iv_t1_interp'] - df['baseline_iv_t1_true']

    return df


# =============================================================================
# Step 7: 评估
# =============================================================================
def evaluate_two_step_v2(df_eval: pd.DataFrame) -> Dict:
    """评估两步法v2的预测质量"""
    metrics = {}
    df = df_eval.dropna(subset=['true_iv', 'pred_iv', 'baseline_iv_t1_true',
                                'baseline_iv_t1_interp', 'implc_volatlty']).copy()

    true_iv = df['true_iv'].values
    pred_iv = df['pred_iv'].values
    iv_t = df['implc_volatlty'].values
    baseline_true = df['baseline_iv_t1_true'].values
    baseline_interp = df['baseline_iv_t1_interp'].values

    # Primary IV metrics
    metrics['rmse_iv'] = float(np.sqrt(mean_squared_error(true_iv, pred_iv)))
    metrics['mae_iv'] = float(mean_absolute_error(true_iv, pred_iv))
    metrics['r2_iv'] = float(r2_score(true_iv, pred_iv))
    metrics['mape_iv'] = float(np.mean(np.abs((true_iv - pred_iv) / true_iv)))
    metrics['direction_acc'] = float(np.mean(
        np.sign(pred_iv - iv_t) == np.sign(true_iv - iv_t)
    ))

    # Baseline proxy quality
    metrics['baseline_proxy_corr'] = float(np.corrcoef(baseline_interp, baseline_true)[0, 1])
    metrics['baseline_proxy_rmse'] = float(np.sqrt(mean_squared_error(baseline_true, baseline_interp)))

    # Residual prediction quality
    true_res = df['true_residual_t1'].values
    pred_res = df['pred_residual_t1'].values
    metrics['rmse_residual'] = float(np.sqrt(mean_squared_error(true_res, pred_res)))
    metrics['mae_residual'] = float(mean_absolute_error(true_res, pred_res))
    metrics['r2_residual'] = float(r2_score(true_res, pred_res))

    # Delta residual prediction quality
    true_delta_res = df['target_residual'].values
    pred_delta_res = df['pred_delta_residual'].values
    metrics['rmse_delta_residual'] = float(np.sqrt(mean_squared_error(true_delta_res, pred_delta_res)))
    metrics['mae_delta_residual'] = float(mean_absolute_error(true_delta_res, pred_delta_res))

    return metrics


def grouped_evaluation_v2(df_eval: pd.DataFrame) -> Dict:
    """分组评估：按remaining_time、moneyness(M)、call_put"""
    results = {}
    df = df_eval.dropna(subset=['true_iv', 'pred_iv', 'M', 'remaining_time']).copy()

    def time_bin(t):
        if t <= 30:
            return 'near'
        elif t <= 90:
            return 'mid'
        else:
            return 'far'

    def moneyness_bin_m(M):
        # Use log-moneyness M, not fund_close/exercise_price
        if M < -0.1:
            return 'ITM'
        elif M <= 0.1:
            return 'ATM'
        else:
            return 'OTM'

    df['time_bin'] = df['remaining_time'].apply(time_bin)
    df['moneyness_bin'] = df['M'].apply(moneyness_bin_m)

    for b, g in df.groupby('time_bin'):
        results[f'rmse_{b}'] = float(np.sqrt(mean_squared_error(g['true_iv'], g['pred_iv'])))
        results[f'mae_{b}'] = float(mean_absolute_error(g['true_iv'], g['pred_iv']))
        results[f'n_{b}'] = len(g)

    for b, g in df.groupby('moneyness_bin'):
        results[f'rmse_{b}'] = float(np.sqrt(mean_squared_error(g['true_iv'], g['pred_iv'])))
        results[f'mae_{b}'] = float(mean_absolute_error(g['true_iv'], g['pred_iv']))
        results[f'n_{b}'] = len(g)

    for cp, g in df.groupby('call_put'):
        results[f'rmse_{cp}'] = float(np.sqrt(mean_squared_error(g['true_iv'], g['pred_iv'])))
        results[f'mae_{cp}'] = float(mean_absolute_error(g['true_iv'], g['pred_iv']))
        results[f'n_{cp}'] = len(g)

    return results


# =============================================================================
# Step 8: 可视化
# =============================================================================
def plot_residual_analysis(df_eval: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(df_eval['target_residual'], df_eval['pred_delta_residual'], alpha=0.3, s=5)
    lim = [df_eval['target_residual'].min(), df_eval['target_residual'].max()]
    axes[0].plot(lim, lim, 'r--', lw=1)
    axes[0].set_xlabel('True Delta Residual')
    axes[0].set_ylabel('Pred Delta Residual')
    axes[0].set_title('Delta Residual Prediction')

    axes[1].hist(df_eval['residual_iv'], bins=50, alpha=0.7, label='T-day residual')
    axes[1].hist(df_eval['true_residual_t1'], bins=50, alpha=0.7, label='T+1 residual')
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Residual Distribution')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_analysis.png'), dpi=150)
    plt.close()


def plot_pred_vs_true(df_eval: pd.DataFrame, output_dir: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df_eval['true_iv'], df_eval['pred_iv'], alpha=0.3, s=5)
    lim = [df_eval['true_iv'].min(), df_eval['true_iv'].max()]
    ax.plot(lim, lim, 'r--', lw=1)
    ax.set_xlabel('True IV')
    ax.set_ylabel('Pred IV (Two-Step v2)')
    ax.set_title('Two-Step v2: Pred vs True IV')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pred_vs_true.png'), dpi=150)
    plt.close()


def plot_baseline_proxy(df_eval: pd.DataFrame, output_dir: str):
    """可视化baseline proxy质量：interp vs true"""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df_eval['baseline_iv_t1_true'], df_eval['baseline_iv_t1_interp'], alpha=0.3, s=5)
    lim = [df_eval['baseline_iv_t1_true'].min(), df_eval['baseline_iv_t1_true'].max()]
    ax.plot(lim, lim, 'r--', lw=1)
    ax.set_xlabel('True Baseline IV (t+1)')
    ax.set_ylabel('Interpolated Baseline IV (t+1)')
    ax.set_title('Baseline Proxy Quality')
    corr = df_eval['baseline_iv_t1_interp'].corr(df_eval['baseline_iv_t1_true'])
    ax.text(0.05, 0.95, f'Corr={corr:.4f}', transform=ax.transAxes, verticalalignment='top')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_proxy.png'), dpi=150)
    plt.close()


def plot_timeseries_sample(df_eval: pd.DataFrame, output_dir: str):
    top_contracts = df_eval['security_id'].value_counts().head(2).index.tolist()
    fig, axes = plt.subplots(len(top_contracts), 1, figsize=(14, 4 * len(top_contracts)), sharex=True)
    if len(top_contracts) == 1:
        axes = [axes]

    for ax, sid in zip(axes, top_contracts):
        sub = df_eval[df_eval['security_id'] == sid].sort_values('trade_date')
        ax.plot(sub['trade_date'], sub['true_iv'], label='True IV', marker='o', markersize=2)
        ax.plot(sub['trade_date'], sub['pred_iv'], label='Pred IV', marker='x', markersize=2)
        ax.plot(sub['trade_date'], sub['baseline_iv_t1_interp'], label='Baseline (interp)', linestyle='--', alpha=0.7)
        ax.set_title(f'Time Series: {sid}')
        ax.set_ylabel('Implied Volatility')
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Trade Date')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timeseries_sample.png'), dpi=150)
    plt.close()


def plot_mw_surface_sample(df: pd.DataFrame, forward_table: pd.DataFrame,
                           date: int, output_dir: str):
    day = df[df['trade_date'] == date].copy()
    # Merge with explicit column selection and rename to avoid conflicts
    fwd_col = 'F' if 'F' in forward_table.columns else 'F_implied'
    fwd_sub = forward_table[['trade_date', 'last_edate', fwd_col]].copy()
    fwd_sub.columns = ['trade_date', 'last_edate', 'F_merged']
    day = day.merge(fwd_sub, on=['trade_date', 'last_edate'], how='left')
    day['M'] = np.log(day['exercise_price'] / day['F_merged'])
    day['W'] = day['implc_volatlty'] ** 2 * (day['remaining_time'] / 365.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cp, title in [(axes[0], 'C', 'Call'), (axes[1], 'P', 'Put')]:
        sub = day[day['call_put'] == cp].sort_values(['last_edate', 'M'])
        for mat, g in sub.groupby('last_edate'):
            ax.scatter(g['M'], g['W'], label=f'Mkt {mat}', s=20)
        ax.set_xlabel('M = ln(K/F)')
        ax.set_ylabel('W = IV^2 * tau')
        ax.set_title(f'{title} M-W Surface ({date})')
        ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'm_w_surface_sample.png'), dpi=150)
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
    old = metrics.get('old_two_step', {})
    if 'test' in old:
        old = old['test']
    new_ = metrics.get('new_two_step_v2', {})
    if 'test' in new_:
        new_ = new_['test']

    rows = [
        ['Metric', 'Original', 'Old Two-Step', 'New Two-Step v2'],
        ['Test RMSE (IV)', _get(orig, 'rmse_iv'), _get(old, 'rmse_iv'), _get(new_, 'rmse_iv')],
        ['Test MAE (IV)', _get(orig, 'mae_iv'), _get(old, 'mae_iv'), _get(new_, 'mae_iv')],
        ['Test R2 (IV)', _get(orig, 'r2_iv'), _get(old, 'r2_iv'), _get(new_, 'r2_iv')],
        ['Direction Acc', _get(orig, 'direction_acc'), _get(old, 'direction_acc'), _get(new_, 'direction_acc')],
        ['Baseline Proxy Corr', 'N/A', 'N/A', _get(new_, 'baseline_proxy_corr')],
        ['Baseline Proxy RMSE', 'N/A', 'N/A', _get(new_, 'baseline_proxy_rmse')],
    ]
    table = ax.table(cellText=rows[1:], colLabels=rows[0], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.title('Three-Way Comparison: Original vs Old Two-Step vs New Two-Step v2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_table.png'), dpi=150)
    plt.close()


# =============================================================================
# Step 9: 保存输出
# =============================================================================
def save_outputs(model, metrics, df_eval, fi, forward_table, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'model_delta_residual.pkl'), 'wb') as f:
        pickle.dump(model, f)

    forward_table.to_csv(os.path.join(output_dir, 'forward_table.csv'), index=False)

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Ensure output columns match spec
    required_cols = [
        'security_id', 'trade_date', 'call_put', 'exercise_price',
        'remaining_time', 'moneyness', 'implc_volatlty',
        'true_iv', 'F_implied', 'baseline_iv_t', 'baseline_iv_t1_true',
        'baseline_iv_t1_interp', 'residual_iv', 'true_residual_t1',
        'pred_delta_residual', 'pred_residual_t1', 'pred_iv',
        'iv_residual', 'baseline_proxy_residual',
    ]
    # Add any missing columns with NaN
    for col in required_cols:
        if col not in df_eval.columns:
            df_eval[col] = np.nan

    df_eval.to_csv(os.path.join(output_dir, 'predictions_test.csv'), index=False)
    fi.to_csv(os.path.join(output_dir, 'feature_importance_residual.csv'), index=False)
    print(f"[Save] All outputs saved to {output_dir}")


def load_original_model_predictions(test_df: pd.DataFrame) -> pd.DataFrame:
    model_path = os.path.join(BASELINE_DIR, 'model_abs_iv.pkl')
    if not os.path.exists(model_path):
        print("[Warn] 原方案模型未找到，跳过原方案对照")
        return test_df
    with open(model_path, 'rb') as f:
        model_orig = pickle.load(f)

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
    for col in orig_feat_cols:
        if col not in test_df.columns:
            test_df[col] = 0
    X_test = test_df[orig_feat_cols].fillna(0)
    test_df['original_pred_iv'] = model_orig.predict(X_test)
    return test_df


def load_old_two_step_predictions() -> Optional[pd.DataFrame]:
    path = os.path.join(OLD_TWOSTEP_DIR, 'predictions_test.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# =============================================================================
# Main
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # [Checkpoint 1] Data loading + forward table
    # ------------------------------------------------------------------
    print("[Checkpoint 1] 数据加载与远期价格计算")
    df_raw = pd.read_csv('data/raw/50etf_options.csv')
    print(f"  - 原始记录数: {len(df_raw)}")
    print(f"  - 交易日数: {df_raw['trade_date'].nunique()}")
    print(f"  - 唯一合约数: {df_raw['security_id'].nunique()}")

    forward_table = build_forward_table(df_raw)
    n_implied = forward_table['F_implied'].notna().sum()
    n_total = len(forward_table)
    print(f"  - F_implied 计算成功率: {n_implied}/{n_total} ({n_implied/n_total*100:.1f}%)")
    fwd_diff = (forward_table['F_implied'] - forward_table['F_theory']).abs()
    print(f"  - |F_implied - F_theory| median={fwd_diff.median():.4f}, max={fwd_diff.max():.4f}")

    # Merge F to each contract
    df = df_raw.merge(forward_table[['trade_date', 'last_edate', 'F_implied', 'F']],
                      on=['trade_date', 'last_edate'], how='left')

    # ------------------------------------------------------------------
    # [Checkpoint 2] M-W B-Spline fitting
    # ------------------------------------------------------------------
    mw_cache_path = os.path.join(OUTPUT_DIR, 'mw_checkpoint_v2.pkl')
    if os.path.exists(mw_cache_path):
        print("\n[Checkpoint 2] M-W空间B-Spline拟合 (从缓存加载)")
        with open(mw_cache_path, 'rb') as f:
            cache = pickle.load(f)
        df = cache['df']
        daily_mw_data = cache['daily_mw_data']
        print(f"  - 从缓存恢复: {len(df)} 条记录, {len(daily_mw_data)} 条spline数据")
    else:
        print("\n[Checkpoint 2] M-W空间B-Spline拟合")
        df, daily_mw_data = build_residual_series(df, forward_table)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(mw_cache_path, 'wb') as f:
            pickle.dump({'df': df, 'daily_mw_data': daily_mw_data}, f)
        print(f"  - 缓存已保存至 {mw_cache_path}")

    coverage = df['baseline_iv'].notna().mean()
    print(f"  - baseline_iv 覆盖率: {coverage*100:.2f}%")

    # Save M-W surface sample visualization
    sample_date = df['trade_date'].min()
    plot_mw_surface_sample(df, forward_table, sample_date, OUTPUT_DIR)

    # ------------------------------------------------------------------
    # [Checkpoint 3] Feature engineering
    # ------------------------------------------------------------------
    print("\n[Checkpoint 3] 特征工程完成")
    df_feat = build_features_delta_residual(df)
    df = build_target(df_feat)

    # Ensure all feature columns exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    print(f"  - 构造样本数: {len(df)}")
    print(f"  - 特征维度: {len(FEATURE_COLS)}")
    print(f"  - 目标变量统计 (delta_residual): mean={df['target_residual'].mean():.6f}, std={df['target_residual'].std():.6f}")
    print(f"  - 前3行样本:")
    print(df[['security_id', 'trade_date', 'implc_volatlty', 'baseline_iv', 'residual_iv', 'target_residual']].head(3).to_string(index=False))

    # Validation checks
    df = df.drop(columns=['delta', 'gamma', 'theta', 'vega', 'rho'], errors='ignore')
    assert not any(c in df.columns for c in ['delta', 'gamma', 'theta', 'vega', 'rho'])
    assert df['moneyness'].between(0.3, 3.0).all() or df['moneyness'].isna().any()
    assert (df['days_gap'] <= 200).all() or df['days_gap'].isna().any()

    # ------------------------------------------------------------------
    # [Checkpoint 4] Temporal split
    # ------------------------------------------------------------------
    print("\n[Checkpoint 4] 时间划分完成")
    train_df, val_df, test_df = split_temporal(df, TRAIN_END, VAL_END)
    print(f"  - 训练集: {train_df['trade_date'].min()} ~ {train_df['trade_date'].max()}, 样本数={len(train_df)}")
    print(f"  - 验证集: {val_df['trade_date'].min()} ~ {val_df['trade_date'].max()}, 样本数={len(val_df)}")
    print(f"  - 测试集: {test_df['trade_date'].min()} ~ {test_df['trade_date'].max()}, 样本数={len(test_df)}")

    train_max = train_df['trade_date'].max()
    val_min = val_df['trade_date'].min()
    val_max = val_df['trade_date'].max()
    test_min = test_df['trade_date'].min()
    leakage = not (train_max < val_min < test_min)
    print(f"  - 时间泄露检查: {'FAIL' if leakage else 'PASS'}")
    assert not leakage, "Temporal split violated!"

    X_train = train_df[FEATURE_COLS]
    y_train = train_df['target_residual']
    X_val = val_df[FEATURE_COLS]
    y_val = val_df['target_residual']

    # ------------------------------------------------------------------
    # [Checkpoint 5] Model training
    # ------------------------------------------------------------------
    print("\n[Checkpoint 5] 模型训练完成")
    model, best_params = train_delta_residual_model(X_train, y_train, X_val, y_val)

    fi = pd.DataFrame({'feature': FEATURE_COLS, 'importance': model.feature_importances_})
    fi = fi.sort_values('importance', ascending=False).reset_index(drop=True)
    print(f"  - 最优参数: {best_params}")
    print(f"  - Top 5特征: {fi['feature'].head(5).tolist()}")

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"  - 训练RMSE (delta_residual): {train_rmse:.6f}")
    print(f"  - 验证RMSE (delta_residual): {val_rmse:.6f}")

    # ------------------------------------------------------------------
    # [Checkpoint 6] Evaluation
    # ------------------------------------------------------------------
    print("\n[Checkpoint 6] 评估与对照完成")

    # Load original model for comparison
    test_df = load_original_model_predictions(test_df)

    # Two-step v2 prediction
    test_df_pred = predict_two_step_v2(model, test_df, FEATURE_COLS, daily_mw_data, forward_table)

    # Evaluate
    metrics_ts = evaluate_two_step_v2(test_df_pred)
    grouped = grouped_evaluation_v2(test_df_pred)
    metrics_ts.update(grouped)

    print(f"  - 新两步法v2 Test RMSE_IV: {metrics_ts['rmse_iv']:.6f}")
    print(f"  - 新两步法v2 Test MAE_IV: {metrics_ts['mae_iv']:.6f}")
    print(f"  - 新两步法v2 Test R2_IV: {metrics_ts['r2_iv']:.4f}")
    print(f"  - 新两步法v2 Test MAPE_IV: {metrics_ts['mape_iv']:.4f}")
    print(f"  - 新两步法v2 Direction Acc: {metrics_ts['direction_acc']:.4f}")
    print(f"  - Baseline Proxy Corr: {metrics_ts['baseline_proxy_corr']:.4f}")
    print(f"  - Baseline Proxy RMSE: {metrics_ts['baseline_proxy_rmse']:.6f}")
    print(f"  - Residual RMSE: {metrics_ts['rmse_residual']:.6f}")
    print(f"  - Delta Residual RMSE: {metrics_ts['rmse_delta_residual']:.6f}")
    print(f"  - 近月/中月/远月 RMSE: {metrics_ts.get('rmse_near', 0):.5f} / {metrics_ts.get('rmse_mid', 0):.5f} / {metrics_ts.get('rmse_far', 0):.5f}")
    print(f"  - ITM/ATM/OTM RMSE: {metrics_ts.get('rmse_ITM', 0):.5f} / {metrics_ts.get('rmse_ATM', 0):.5f} / {metrics_ts.get('rmse_OTM', 0):.5f}")
    print(f"  - Call/Put RMSE: {metrics_ts.get('rmse_C', 0):.5f} / {metrics_ts.get('rmse_P', 0):.5f}")

    # Original model metrics
    metrics_orig = {}
    if 'original_pred_iv' in test_df_pred.columns:
        mask = test_df_pred['original_pred_iv'].notna() & test_df_pred['true_iv'].notna()
        orig_pred = test_df_pred.loc[mask, 'original_pred_iv'].values
        true_iv = test_df_pred.loc[mask, 'true_iv'].values
        iv_t = test_df_pred.loc[mask, 'implc_volatlty'].values
        metrics_orig['rmse_iv'] = float(np.sqrt(mean_squared_error(true_iv, orig_pred)))
        metrics_orig['mae_iv'] = float(mean_absolute_error(true_iv, orig_pred))
        metrics_orig['r2_iv'] = float(r2_score(true_iv, orig_pred))
        metrics_orig['direction_acc'] = float(np.mean(
            np.sign(orig_pred - iv_t) == np.sign(true_iv - iv_t)
        ))
        print(f"  - 原方案 Test RMSE_IV: {metrics_orig['rmse_iv']:.6f}")
        diff_pct = (metrics_ts['rmse_iv'] - metrics_orig['rmse_iv']) / metrics_orig['rmse_iv'] * 100
        print(f"  - 新两步法v2 vs 原方案差异: {diff_pct:.2f}%")

    # Old two-step metrics
    metrics_old = {}
    old_pred_df = load_old_two_step_predictions()
    if old_pred_df is not None:
        # Merge on common keys
        old_cols = ['security_id', 'trade_date', 'call_put', 'exercise_price']
        if all(c in old_pred_df.columns for c in old_cols) and all(c in test_df_pred.columns for c in old_cols):
            merged = test_df_pred[old_cols + ['true_iv', 'implc_volatlty']].merge(
                old_pred_df[old_cols + ['pred_iv']], on=old_cols, how='inner'
            )
            if len(merged) > 0:
                metrics_old['rmse_iv'] = float(np.sqrt(mean_squared_error(merged['true_iv'], merged['pred_iv'])))
                metrics_old['mae_iv'] = float(mean_absolute_error(merged['true_iv'], merged['pred_iv']))
                metrics_old['r2_iv'] = float(r2_score(merged['true_iv'], merged['pred_iv']))
                metrics_old['direction_acc'] = float(np.mean(
                    np.sign(merged['pred_iv'] - merged['implc_volatlty']) == np.sign(merged['true_iv'] - merged['implc_volatlty'])
                ))
                print(f"  - 旧两步法 Test RMSE_IV: {metrics_old['rmse_iv']:.6f}")
                diff_pct_old = (metrics_ts['rmse_iv'] - metrics_old['rmse_iv']) / metrics_old['rmse_iv'] * 100
                print(f"  - 新两步法v2 vs 旧两步法差异: {diff_pct_old:.2f}%")

    # Assemble full metrics
    full_metrics = {
        'new_two_step_v2': {'test': metrics_ts},
        'original': {'test': metrics_orig},
        'old_two_step': {'test': metrics_old},
        'feature_importance_top10': fi.head(10).to_dict(orient='records'),
        'data_info': {
            'n_samples_total': len(df),
            'n_features': len(FEATURE_COLS),
            'n_contracts': df['security_id'].nunique(),
            'date_range': f"{df['trade_date'].min()}-{df['trade_date'].max()}",
        }
    }

    # Visualization
    print("\n[Visualization] Generating plots...")
    plot_residual_analysis(test_df_pred, OUTPUT_DIR)
    plot_pred_vs_true(test_df_pred, OUTPUT_DIR)
    plot_baseline_proxy(test_df_pred, OUTPUT_DIR)
    plot_timeseries_sample(test_df_pred, OUTPUT_DIR)
    if metrics_orig or metrics_old:
        plot_comparison_table(full_metrics, OUTPUT_DIR)

    # Save
    save_outputs(model, full_metrics, test_df_pred, fi, forward_table, OUTPUT_DIR)

    print("\n[Done] All tasks completed successfully.")


if __name__ == '__main__':
    main()
