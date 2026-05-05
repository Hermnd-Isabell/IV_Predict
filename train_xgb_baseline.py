# -*- coding: utf-8 -*-
"""
50ETF期权合约级IV预测 - XGBoost基线验证
主目标: 预测 delta_iv = IV_{t+1} - IV_t
对照目标: 预测 next_implc_volatlty = IV_{t+1}
"""

import os
import json
import warnings
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RANDOM_STATE = 42

# =============================================================================
# 1. 数据加载
# =============================================================================
def load_raw_data(data_dir: str) -> pd.DataFrame:
    """读取原始数据，返回带列名的DataFrame"""
    csv_path = os.path.join(data_dir, '50etf_options.csv')
    df = pd.read_csv(csv_path)
    return df


# =============================================================================
# 2. 特征工程
# =============================================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    特征工程主函数，输入原始数据，输出带特征的样本集。
    所有特征仅使用 t 或 t-1 时刻信息，严格避免信息泄露。
    """
    df = df.copy()
    df = df.sort_values(['security_id', 'trade_date']).reset_index(drop=True)

    # -------------------------------------------------------------
    # D1. 异常值与边界处理
    # -------------------------------------------------------------
    # 剔除到期日/IV为0的样本（无法构造t+1目标）
    df = df[(df['remaining_time'] > 0) & (df['implc_volatlty'] > 0)].copy()

    # implc_volatlty > 1.0 设为NaN后剔除（极端异常）
    df.loc[df['implc_volatlty'] > 1.0, 'implc_volatlty'] = np.nan
    df = df.dropna(subset=['implc_volatlty']).copy()

    # 标的数据缺失则剔除该交易日所有合约
    spot_missing_dates = df[df['fund_close'].isna()]['trade_date'].unique()
    df = df[~df['trade_date'].isin(spot_missing_dates)].copy()

    # -------------------------------------------------------------
    # 1. 合约固有特征（t时刻直接取）
    # -------------------------------------------------------------
    df['moneyness'] = df['fund_close'] / df['exercise_price']
    df['moneyness_squared'] = df['moneyness'] ** 2
    df['call_put_flag'] = (df['call_put'] == 'C').astype(int)
    df['moneyness_remaining_time'] = df['moneyness'] * df['remaining_time']

    # -------------------------------------------------------------
    # 2. 标的数据特征（t时刻，fund_return需要t-1日信息）
    # -------------------------------------------------------------
    # 先提取每日唯一的标的数据（去重）
    spot_daily = df[['trade_date', 'fund_close', 'fund_volume', 'fund_amount',
                     'fund_high', 'fund_low']].drop_duplicates('trade_date').sort_values('trade_date')
    spot_daily['fund_return'] = spot_daily['fund_close'].pct_change()
    spot_daily['fund_high_low_ratio'] = (spot_daily['fund_high'] - spot_daily['fund_low']) / spot_daily['fund_close']

    # merge回主表
    df = df.merge(
        spot_daily[['trade_date', 'fund_return', 'fund_high_low_ratio']],
        on='trade_date', how='left'
    )

    # -------------------------------------------------------------
    # 3. 历史IV序列特征（按交易日索引滑动窗口，window=5，t/t-1/t-2...）
    # -------------------------------------------------------------
    # 先构造交易日到实际日期的映射，用于计算days_gap
    trade_dates = df['trade_date'].unique()
    trade_dates = np.sort(trade_dates)
    date_map = pd.DataFrame({
        'trade_date': trade_dates,
        'trade_dt': pd.to_datetime(trade_dates, format='%Y%m%d')
    })
    df = df.merge(date_map, on='trade_date', how='left')

    # 按合约分组计算lag特征
    def compute_iv_lags(group):
        group = group.sort_values('trade_date')
        group['iv_t'] = group['implc_volatlty']
        group['iv_t_1'] = group['implc_volatlty'].shift(1)
        group['iv_t_2'] = group['implc_volatlty'].shift(2)
        group['iv_t_3'] = group['implc_volatlty'].shift(3)
        group['iv_t_4'] = group['implc_volatlty'].shift(4)
        group['iv_ma5'] = group['implc_volatlty'].shift(0).rolling(window=5, min_periods=1).mean()
        group['iv_std5'] = group['implc_volatlty'].shift(0).rolling(window=5, min_periods=2).std()
        group['iv_trend5'] = group['implc_volatlty'] - group['implc_volatlty'].shift(4)
        # days_gap: 距上个交易日的实际天数
        group['days_gap'] = (group['trade_dt'] - group['trade_dt'].shift(1)).dt.days
        return group

    df = df.groupby('security_id', group_keys=False).apply(compute_iv_lags)

    # -------------------------------------------------------------
    # 4. 曲面上下文特征（使用 t-1 日截面统计，避免信息泄露）
    # -------------------------------------------------------------
    # 先计算每日截面统计
    daily_stats = []
    for date, day_df in df.groupby('trade_date'):
        stats = {
            'trade_date': date,
            'iv_mean_all': day_df['implc_volatlty'].mean(),
            'iv_std_all': day_df['implc_volatlty'].std(),
            'iv_max_all': day_df['implc_volatlty'].max(),
            'iv_min_all': day_df['implc_volatlty'].min(),
        }
        # atm_iv_call: 对Call在当日fund_close处做线性插值
        call_df = day_df[day_df['call_put'] == 'C'].copy()
        if len(call_df) >= 2:
            call_df = call_df.sort_values('exercise_price')
            fund_close = call_df['fund_close'].iloc[0]
            # 使用 numpy.interp（要求x递增）
            xp = call_df['exercise_price'].values
            fp = call_df['implc_volatlty'].values
            if np.all(np.diff(xp) > 0):
                atm_iv = np.interp(fund_close, xp, fp)
            else:
                # 若行权价有重复，取均值后插值
                call_agg = call_df.groupby('exercise_price')['implc_volatlty'].mean().reset_index().sort_values('exercise_price')
                atm_iv = np.interp(fund_close, call_agg['exercise_price'].values, call_agg['implc_volatlty'].values)
        else:
            atm_iv = np.nan
        stats['atm_iv_call'] = atm_iv
        daily_stats.append(stats)

    daily_stats_df = pd.DataFrame(daily_stats).sort_values('trade_date')

    # 将t-1日的统计merge到t日：需要把 daily_stats 的日期往后推1天作为 merge_date
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

    # -------------------------------------------------------------
    # 5. 宏观特征（t时刻）
    # -------------------------------------------------------------
    df['ten_year'] = df['ten_year'] / 100.0

    # -------------------------------------------------------------
    # 整理特征列
    # -------------------------------------------------------------
    feature_cols = [
        # 1. 合约固有
        'moneyness', 'moneyness_squared', 'remaining_time', 'call_put_flag',
        'exercise_price', 'moneyness_remaining_time',
        # 2. 标的数据
        'fund_return', 'fund_volume', 'fund_high_low_ratio', 'fund_amount',
        # 3. 历史IV
        'iv_t', 'iv_t_1', 'iv_t_2', 'iv_t_3', 'iv_t_4',
        'iv_ma5', 'iv_std5', 'iv_trend5', 'days_gap',
        'iv_change_1d', 'iv_change_2d', 'iv_change_3d', 'iv_change_ma3', 'iv_zscore5',
        'fund_return_iv_interaction',
        # 4. 曲面上下文
        'atm_iv_call_lag1', 'iv_mean_all_lag1', 'iv_std_all_lag1',
        'iv_max_all_lag1', 'iv_min_all_lag1', 'iv_vs_atm_lag1',
        # 5. 宏观
        'ten_year',
    ]

    # 确保所有特征列存在
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    # 用全局均值填充曲面上下文缺失（上市首日无t-1统计）
    for col in ['atm_iv_call_lag1', 'iv_mean_all_lag1', 'iv_std_all_lag1',
                'iv_max_all_lag1', 'iv_min_all_lag1', 'iv_vs_atm_lag1']:
        df[col] = df[col].fillna(df[col].mean())

    # days_gap缺失（上市首日）用1填充
    df['days_gap'] = df['days_gap'].fillna(1)

    # fund_return 缺失（首日）用0填充
    df['fund_return'] = df['fund_return'].fillna(0)

    # 历史IV缺失用当前IV填充（上市前几个交易日）
    for col in ['iv_t_1', 'iv_t_2', 'iv_t_3', 'iv_t_4', 'iv_std5', 'iv_trend5']:
        df[col] = df[col].fillna(df['iv_t'])

    return df


# =============================================================================
# 3. 目标变量构造
# =============================================================================
def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造目标变量，返回含 y 的 DataFrame。
    primary: delta_iv = IV_{t+1} - IV_t
    control: next_implc_volatlty = IV_{t+1}
    """
    df = df.copy()
    df = df.sort_values(['security_id', 'trade_date']).reset_index(drop=True)

    # 计算 next_implc_volatlty
    df['next_implc_volatlty'] = df.groupby('security_id')['implc_volatlty'].shift(-1)
    df['delta_iv'] = df['next_implc_volatlty'] - df['implc_volatlty']

    # 剔除无法构造目标的行
    df = df.dropna(subset=['next_implc_volatlty', 'delta_iv']).copy()

    return df


# =============================================================================
# 4. 时间序列划分
# =============================================================================
def split_temporal(df: pd.DataFrame, train_end: int, val_end: int) -> Tuple:
    """
    时间序列划分，返回(train_df, val_df, test_df)。
    按 trade_date 排序切割。
    """
    train_df = df[df['trade_date'] <= train_end].copy()
    val_df = df[(df['trade_date'] > train_end) & (df['trade_date'] <= val_end)].copy()
    test_df = df[df['trade_date'] > val_end].copy()
    return train_df, val_df, test_df


# =============================================================================
# 5. 模型训练
# =============================================================================
def train_model(X_train, y_train, X_val, y_val, param_grid=None, sample_weight=None):
    """
    训练XGBoost，支持超参数调优（网格搜索）。
    在验证集上选择最优参数。
    """
    if param_grid is None:
        param_grid = {
            'max_depth': [5, 6, 8],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [500, 800],
            'subsample': [0.8, 0.9],
        }

    base_params = {
        'objective': 'reg:squarederror',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'colsample_bytree': 0.8,
    }

    best_model = None
    best_rmse = float('inf')
    best_params = None

    print("[Hyperparameter Tuning] Grid search started...")
    grid = list(ParameterGrid(param_grid))
    # 如果网格太大，可以采样；这里完整搜索
    for i, params in enumerate(grid):
        model = xgb.XGBRegressor(**{**base_params, **params})
        fit_kwargs = {
            'eval_set': [(X_val, y_val)],
            'verbose': False,
        }
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight
        model.fit(X_train, y_train, **fit_kwargs)
        val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        print(f"  [{i+1}/{len(grid)}] params={params}, val_rmse={rmse:.6f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = params

    print(f"[Hyperparameter Tuning] Best params: {best_params}, best_val_rmse: {best_rmse:.6f}")

    # 用最优参数在整个 train+val 上重新训练（可选，但通常更好）
    # 这里保留在train上训练、val上早停的版本
    return best_model, best_params


# =============================================================================
# 6. 评估
# =============================================================================
def evaluate_model(model, X, y, iv_t=None) -> dict:
    """
    评估并返回指标字典。
    若 iv_t 不为None，则同时计算还原IV的指标（用于主实验）。
    """
    preds = model.predict(X)
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y, preds))),
        'mae': float(mean_absolute_error(y, preds)),
        'r2': float(r2_score(y, preds)),
        'direction_acc': float(np.mean(np.sign(preds) == np.sign(y))),
    }

    # 若 iv_t 提供，计算还原IV指标（pred_iv = iv_t + pred_delta_iv）
    if iv_t is not None:
        pred_iv = iv_t + preds
        true_iv = iv_t + y
        metrics['rmse_iv'] = float(np.sqrt(mean_squared_error(true_iv, pred_iv)))
        metrics['mae_iv'] = float(mean_absolute_error(true_iv, pred_iv))
        metrics['mape_iv'] = float(np.mean(np.abs((true_iv - pred_iv) / true_iv)))
        metrics['r2_iv'] = float(r2_score(true_iv, pred_iv))
        metrics['direction_acc_iv'] = float(np.mean(np.sign(pred_iv - iv_t) == np.sign(true_iv - iv_t)))

    return metrics, preds


# =============================================================================
# 7. 分组评估
# =============================================================================
def grouped_evaluation(df_eval: pd.DataFrame) -> dict:
    """按合约、moneyness分箱、remaining_time分箱评估"""
    results = {}

    # 按合约
    by_contract = []
    for sid, g in df_eval.groupby('security_id'):
        rmse = np.sqrt(mean_squared_error(g['true_iv'], g['pred_iv']))
        by_contract.append({'security_id': sid, 'rmse_iv': rmse})
    results['by_contract'] = pd.DataFrame(by_contract)

    # moneyness分箱
    def moneyness_bin(m):
        if m < 0.97:
            return 'ITM'
        elif m <= 1.03:
            return 'ATM'
        else:
            return 'OTM'

    df_eval['moneyness_bin'] = df_eval['moneyness'].apply(moneyness_bin)
    by_moneyness = []
    for b, g in df_eval.groupby('moneyness_bin'):
        rmse = np.sqrt(mean_squared_error(g['true_iv'], g['pred_iv']))
        mae = mean_absolute_error(g['true_iv'], g['pred_iv'])
        by_moneyness.append({'bin': b, 'rmse_iv': rmse, 'mae_iv': mae, 'n': len(g)})
    results['by_moneyness'] = pd.DataFrame(by_moneyness)

    # remaining_time分箱
    def time_bin(t):
        if t <= 30:
            return 'near'
        elif t <= 90:
            return 'mid'
        else:
            return 'far'

    df_eval['time_bin'] = df_eval['remaining_time'].apply(time_bin)
    by_time = []
    for b, g in df_eval.groupby('time_bin'):
        rmse = np.sqrt(mean_squared_error(g['true_iv'], g['pred_iv']))
        mae = mean_absolute_error(g['true_iv'], g['pred_iv'])
        by_time.append({'bin': b, 'rmse_iv': rmse, 'mae_iv': mae, 'n': len(g)})
    results['by_time'] = pd.DataFrame(by_time)

    return results


# =============================================================================
# 8. 可视化
# =============================================================================
def plot_residual_analysis(df_eval: pd.DataFrame, output_dir: str, experiment: str):
    """残差分析图：按moneyness和remaining_time分箱"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # moneyness分箱
    def moneyness_bin(m):
        if m < 0.97:
            return 'ITM'
        elif m <= 1.03:
            return 'ATM'
        else:
            return 'OTM'

    df_eval['moneyness_bin'] = df_eval['moneyness'].apply(moneyness_bin)
    df_eval['time_bin'] = pd.cut(df_eval['remaining_time'], bins=[0, 30, 90, 999], labels=['near', 'mid', 'far'])

    # residual delta by moneyness
    df_eval.boxplot(column='residual_delta', by='moneyness_bin', ax=axes[0, 0])
    axes[0, 0].set_title(f'Residual Delta IV by Moneyness ({experiment})')
    axes[0, 0].set_xlabel('Moneyness Bin')
    axes[0, 0].set_ylabel('Residual (pred - true)')

    # residual delta by time
    df_eval.boxplot(column='residual_delta', by='time_bin', ax=axes[0, 1])
    axes[0, 1].set_title(f'Residual Delta IV by Time to Maturity ({experiment})')
    axes[0, 1].set_xlabel('Time Bin')
    axes[0, 1].set_ylabel('Residual (pred - true)')

    # residual iv by moneyness
    df_eval.boxplot(column='residual_iv', by='moneyness_bin', ax=axes[1, 0])
    axes[1, 0].set_title(f'Residual IV by Moneyness ({experiment})')
    axes[1, 0].set_xlabel('Moneyness Bin')
    axes[1, 0].set_ylabel('Residual (pred - true)')

    # residual iv by time
    df_eval.boxplot(column='residual_iv', by='time_bin', ax=axes[1, 1])
    axes[1, 1].set_title(f'Residual IV by Time to Maturity ({experiment})')
    axes[1, 1].set_xlabel('Time Bin')
    axes[1, 1].set_ylabel('Residual (pred - true)')

    plt.suptitle('')
    plt.tight_layout()
    fname = f'residual_by_moneyness_{experiment.lower()}.png'
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()


def plot_pred_vs_true(df_eval: pd.DataFrame, output_dir: str, experiment: str):
    """预测vs真实散点图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Delta IV
    axes[0].scatter(df_eval['true_delta_iv'], df_eval['pred_delta_iv'], alpha=0.3, s=5)
    lim = [
        min(df_eval['true_delta_iv'].min(), df_eval['pred_delta_iv'].min()),
        max(df_eval['true_delta_iv'].max(), df_eval['pred_delta_iv'].max()),
    ]
    axes[0].plot(lim, lim, 'r--', lw=1)
    axes[0].set_xlabel('True Delta IV')
    axes[0].set_ylabel('Pred Delta IV')
    axes[0].set_title(f'Pred vs True Delta IV ({experiment})')

    # IV
    axes[1].scatter(df_eval['true_iv'], df_eval['pred_iv'], alpha=0.3, s=5)
    lim = [
        min(df_eval['true_iv'].min(), df_eval['pred_iv'].min()),
        max(df_eval['true_iv'].max(), df_eval['pred_iv'].max()),
    ]
    axes[1].plot(lim, lim, 'r--', lw=1)
    axes[1].set_xlabel('True IV')
    axes[1].set_ylabel('Pred IV')
    axes[1].set_title(f'Pred vs True IV ({experiment})')

    plt.tight_layout()
    fname = f'pred_vs_true_{experiment.lower()}.png'
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()


def plot_timeseries_sample(df_eval: pd.DataFrame, output_dir: str):
    """选取1-2个代表性合约，绘制真实IV vs 预测IV的时间序列对比图"""
    # 选取测试集中样本数最多的前2个合约
    top_contracts = df_eval['security_id'].value_counts().head(2).index.tolist()

    fig, axes = plt.subplots(len(top_contracts), 1, figsize=(14, 4 * len(top_contracts)), sharex=True)
    if len(top_contracts) == 1:
        axes = [axes]

    for ax, sid in zip(axes, top_contracts):
        sub = df_eval[df_eval['security_id'] == sid].sort_values('trade_date')
        ax.plot(sub['trade_date'], sub['true_iv'], label='True IV', marker='o', markersize=2)
        ax.plot(sub['trade_date'], sub['pred_iv'], label='Pred IV', marker='x', markersize=2)
        ax.set_title(f'Time Series: {sid}')
        ax.set_ylabel('Implied Volatility')
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Trade Date')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timeseries_sample.png'), dpi=150)
    plt.close()


def plot_comparison_table(metrics_delta: dict, metrics_abs: dict, output_dir: str):
    """主实验vs对照实验对比表"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    rows = [
        ['Metric', 'Primary (Delta IV)', 'Control (Abs IV)'],
        ['Test RMSE (IV restored)', f"{metrics_delta.get('test_rmse_iv', 'N/A'):.5f}", f"{metrics_abs.get('test_rmse_iv', metrics_abs.get('test_rmse', 'N/A')):.5f}"],
        ['Test MAE (IV restored)', f"{metrics_delta.get('test_mae_iv', 'N/A'):.5f}", f"{metrics_abs.get('test_mae_iv', metrics_abs.get('test_mae', 'N/A')):.5f}"],
        ['Test MAPE (IV)', f"{metrics_delta.get('test_mape_iv', 'N/A'):.5f}", f"{metrics_abs.get('test_mape_iv', 'N/A'):.5f}"],
        ['Test R2 (IV restored)', f"{metrics_delta.get('test_r2_iv', 'N/A'):.5f}", f"{metrics_abs.get('test_r2_iv', 'N/A'):.5f}"],
        ['Test Direction Acc (IV)', f"{metrics_delta.get('test_direction_acc_iv', 'N/A'):.5f}", f"{metrics_abs.get('test_direction_acc_iv', metrics_abs.get('test_direction_acc', 'N/A')):.5f}"],
    ]

    table = ax.table(cellText=rows[1:], colLabels=rows[0], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.title('Primary vs Control Experiment Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_table.png'), dpi=150)
    plt.close()


# =============================================================================
# 9. 保存输出
# =============================================================================
def save_outputs(model_delta, model_abs, metrics, df_test_pred,
                 fi_delta, fi_abs, output_dir: str):
    """保存所有输出文件"""
    os.makedirs(output_dir, exist_ok=True)

    # 模型
    with open(os.path.join(output_dir, 'model_delta_iv.pkl'), 'wb') as f:
        pickle.dump(model_delta, f)
    with open(os.path.join(output_dir, 'model_abs_iv.pkl'), 'wb') as f:
        pickle.dump(model_abs, f)

    # 特征重要性
    fi_delta.to_csv(os.path.join(output_dir, 'feature_importance_delta_iv.csv'), index=False)
    fi_abs.to_csv(os.path.join(output_dir, 'feature_importance_abs_iv.csv'), index=False)

    # metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # predictions
    df_test_pred.to_csv(os.path.join(output_dir, 'predictions_test_delta_iv.csv'), index=False)

    print(f"[Save] All outputs saved to {output_dir}")


# =============================================================================
# Main
# =============================================================================
def main():
    data_dir = 'data/raw'
    output_dir = 'data/output/baseline_xgb'
    train_end = 20240630
    val_end = 20241231

    # ------------------------------------------------------------------
    # Checkpoint 1: 数据加载
    # ------------------------------------------------------------------
    print("[Checkpoint 1] 数据加载完成")
    df_raw = load_raw_data(data_dir)
    print(f"  - 原始记录数: {len(df_raw)}")
    print(f"  - 唯一合约数(security_id): {df_raw['security_id'].nunique()}")
    print(f"  - 日期范围: {df_raw['trade_date'].min()} ~ {df_raw['trade_date'].max()}")
    print(f"  - 列名列表: {df_raw.columns.tolist()}")

    # ------------------------------------------------------------------
    # Checkpoint 2: 特征工程 + 目标构造
    # ------------------------------------------------------------------
    print("\n[Checkpoint 2] 特征工程完成")
    df_feat = build_features(df_raw)
    df = build_target(df_feat)

    feature_cols = [
        'moneyness', 'moneyness_squared', 'remaining_time', 'call_put_flag',
        'exercise_price', 'moneyness_remaining_time',
        'fund_return', 'fund_volume', 'fund_high_low_ratio', 'fund_amount',
        'iv_t', 'iv_t_1', 'iv_t_2', 'iv_t_3', 'iv_t_4',
        'iv_ma5', 'iv_std5', 'iv_trend5', 'days_gap',
        'atm_iv_call_lag1', 'iv_mean_all_lag1', 'iv_std_all_lag1',
        'iv_max_all_lag1', 'iv_min_all_lag1', 'iv_vs_atm_lag1',
        'ten_year',
    ]

    print(f"  - 构造样本数: {len(df)}")
    print(f"  - 特征维度: {len(feature_cols)}")
    print(f"  - 特征列名: {feature_cols}")
    print(f"  - 目标变量统计: mean={df['delta_iv'].mean():.6f}, std={df['delta_iv'].std():.6f}, min={df['delta_iv'].min():.6f}, max={df['delta_iv'].max():.6f}")
    print("  - 样本示例（前3行）:")
    print(df[['security_id', 'trade_date', 'implc_volatlty'] + feature_cols[:5]].head(3).to_string(index=False))

    # 验证检查：确保 Greeks 列已从样本中移除
    df = df.drop(columns=['delta', 'gamma', 'theta', 'vega', 'rho'], errors='ignore')
    assert not any(c in df.columns for c in ['delta', 'gamma', 'theta', 'vega', 'rho'])
    assert df['moneyness'].between(0.3, 3.0).all()
    assert (df['remaining_time'] >= 0).all() and (df['remaining_time'] < 365).all()
    assert (df['days_gap'] <= 200).all()
    assert df['atm_iv_call_lag1'].notna().mean() > 0.95

    # ------------------------------------------------------------------
    # Checkpoint 3: 时间划分
    # ------------------------------------------------------------------
    print("\n[Checkpoint 3] 时间划分完成")
    train_df, val_df, test_df = split_temporal(df, train_end, val_end)
    print(f"  - 训练集: {train_df['trade_date'].min()} ~ {train_df['trade_date'].max()}, 样本数={len(train_df)}")
    print(f"  - 验证集: {val_df['trade_date'].min()} ~ {val_df['trade_date'].max()}, 样本数={len(val_df)}")
    print(f"  - 测试集: {test_df['trade_date'].min()} ~ {test_df['trade_date'].max()}, 样本数={len(test_df)}")

    # 泄露检查
    train_max = train_df['trade_date'].max()
    val_min = val_df['trade_date'].min()
    val_max = val_df['trade_date'].max()
    test_min = test_df['trade_date'].min()
    leakage = not (train_max < val_min < test_min)
    print(f"  - 检查：是否存在security_id跨集合泄露？{'Yes' if leakage else 'No'}")
    assert not leakage, "Temporal split violated!"

    # ------------------------------------------------------------------
    # 准备训练数据
    # ------------------------------------------------------------------
    X_train = train_df[feature_cols].values
    y_train_delta = train_df['delta_iv'].values
    y_train_abs = train_df['next_implc_volatlty'].values

    X_val = val_df[feature_cols].values
    y_val_delta = val_df['delta_iv'].values
    y_val_abs = val_df['next_implc_volatlty'].values

    X_test = test_df[feature_cols].values
    y_test_delta = test_df['delta_iv'].values
    y_test_abs = test_df['next_implc_volatlty'].values
    iv_t_test = test_df['implc_volatlty'].values

    # ------------------------------------------------------------------
    # Checkpoint 4: 模型训练（Primary: delta_iv）
    # ------------------------------------------------------------------
    print("\n[Training] Primary experiment: delta_iv")
    model_delta, best_params_delta = train_model(X_train, y_train_delta, X_val, y_val_delta)

    train_pred_delta = model_delta.predict(X_train)
    val_pred_delta = model_delta.predict(X_val)
    train_rmse_delta = np.sqrt(mean_squared_error(y_train_delta, train_pred_delta))
    val_rmse_delta = np.sqrt(mean_squared_error(y_val_delta, val_pred_delta))

    print("\n[Checkpoint 4] 模型训练完成 (Primary)")
    print(f"  - 最优参数: {best_params_delta}")
    print(f"  - 训练RMSE: {train_rmse_delta:.6f}")
    print(f"  - 验证RMSE: {val_rmse_delta:.6f}")

    # ------------------------------------------------------------------
    # 模型训练（Control: abs_iv）
    # ------------------------------------------------------------------
    print("\n[Training] Control experiment: next_implc_volatlty")
    model_abs, best_params_abs = train_model(X_train, y_train_abs, X_val, y_val_abs)

    train_pred_abs = model_abs.predict(X_train)
    val_pred_abs = model_abs.predict(X_val)
    train_rmse_abs = np.sqrt(mean_squared_error(y_train_abs, train_pred_abs))
    val_rmse_abs = np.sqrt(mean_squared_error(y_val_abs, val_pred_abs))

    print("\n[Checkpoint 4] 模型训练完成 (Control)")
    print(f"  - 最优参数: {best_params_abs}")
    print(f"  - 训练RMSE: {train_rmse_abs:.6f}")
    print(f"  - 验证RMSE: {val_rmse_abs:.6f}")

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------
    # Primary
    train_metrics_delta, _ = evaluate_model(model_delta, X_train, y_train_delta, train_df['implc_volatlty'].values)
    val_metrics_delta, _ = evaluate_model(model_delta, X_val, y_val_delta, val_df['implc_volatlty'].values)
    test_metrics_delta, pred_test_delta = evaluate_model(model_delta, X_test, y_test_delta, iv_t_test)

    # Control
    train_metrics_abs, train_pred_abs = evaluate_model(model_abs, X_train, y_train_abs)
    val_metrics_abs, val_pred_abs = evaluate_model(model_abs, X_val, y_val_abs)
    test_metrics_abs, pred_test_abs = evaluate_model(model_abs, X_test, y_test_abs)

    # 对照实验的方向准确率必须用 IV 变化方向计算（pred_iv - iv_t vs true_iv - iv_t）
    train_metrics_abs['direction_acc'] = float(np.mean(np.sign(train_pred_abs - train_df['implc_volatlty'].values) == np.sign(y_train_abs - train_df['implc_volatlty'].values)))
    val_metrics_abs['direction_acc'] = float(np.mean(np.sign(val_pred_abs - val_df['implc_volatlty'].values) == np.sign(y_val_abs - val_df['implc_volatlty'].values)))

    # 对照实验还原IV指标（用于比较）
    pred_iv_from_abs = pred_test_abs
    true_iv_test = y_test_abs
    test_metrics_abs['rmse_iv'] = float(np.sqrt(mean_squared_error(true_iv_test, pred_iv_from_abs)))
    test_metrics_abs['mae_iv'] = float(mean_absolute_error(true_iv_test, pred_iv_from_abs))
    test_metrics_abs['mape_iv'] = float(np.mean(np.abs((true_iv_test - pred_iv_from_abs) / true_iv_test)))
    test_metrics_abs['r2_iv'] = float(r2_score(true_iv_test, pred_iv_from_abs))
    test_metrics_abs['direction_acc'] = float(np.mean(np.sign(pred_iv_from_abs - iv_t_test) == np.sign(true_iv_test - iv_t_test)))

    # ------------------------------------------------------------------
    # Checkpoint 5: 评估完成
    # ------------------------------------------------------------------
    print("\n[Checkpoint 5] 评估完成")
    print(f"  - 主实验(ΔIV)指标: { {k: f'{v:.5f}' for k, v in test_metrics_delta.items() if not k.endswith('_iv')} }")
    print(f"  - 对照实验(绝对IV)指标: { {k: f'{v:.5f}' for k, v in test_metrics_abs.items() if not k.endswith('_iv')} }")

    # 特征重要性
    fi_delta = pd.DataFrame({
        'feature': feature_cols,
        'importance': model_delta.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    fi_abs = pd.DataFrame({
        'feature': feature_cols,
        'importance': model_abs.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    print(f"  - Top 5特征 (Primary): {fi_delta['feature'].head(5).tolist()}")
    print(f"  - Top 5特征 (Control): {fi_abs['feature'].head(5).tolist()}")

    # ------------------------------------------------------------------
    # 构造预测结果表
    # ------------------------------------------------------------------
    df_test_pred = test_df[['security_id', 'trade_date', 'call_put', 'exercise_price',
                            'last_edate', 'remaining_time', 'moneyness', 'implc_volatlty']].copy()
    df_test_pred['iv_t'] = test_df['implc_volatlty'].values
    df_test_pred['true_delta_iv'] = y_test_delta
    df_test_pred['pred_delta_iv'] = pred_test_delta
    df_test_pred['true_iv'] = y_test_abs
    df_test_pred['pred_iv'] = iv_t_test + pred_test_delta
    df_test_pred['residual_iv'] = df_test_pred['pred_iv'] - df_test_pred['true_iv']
    df_test_pred['residual_delta'] = df_test_pred['pred_delta_iv'] - df_test_pred['true_delta_iv']

    # ------------------------------------------------------------------
    # 分组评估
    # ------------------------------------------------------------------
    grouped = grouped_evaluation(df_test_pred)
    print("\n[Grouped Evaluation] By Moneyness:")
    print(grouped['by_moneyness'].to_string(index=False))
    print("\n[Grouped Evaluation] By Time to Maturity:")
    print(grouped['by_time'].to_string(index=False))

    # ------------------------------------------------------------------
    # 组装 metrics.json
    # ------------------------------------------------------------------
    metrics_json = {
        "primary_experiment": {
            "target": "delta_iv",
            "train": {k: float(v) for k, v in train_metrics_delta.items() if not k.endswith('_iv')},
            "val": {k: float(v) for k, v in val_metrics_delta.items() if not k.endswith('_iv')},
            "test": {k: float(v) for k, v in test_metrics_delta.items() if not k.endswith('_iv')},
        },
        "control_experiment": {
            "target": "abs_iv",
            "train": {
                "rmse": train_metrics_abs['rmse'],
                "mae": train_metrics_abs['mae'],
                "mape": train_metrics_abs.get('mape', 0),
                "r2": train_metrics_abs['r2'],
                "direction_acc": train_metrics_abs.get('direction_acc_iv', train_metrics_abs.get('direction_acc', 0)),
            },
            "val": {
                "rmse": val_metrics_abs['rmse'],
                "mae": val_metrics_abs['mae'],
                "mape": val_metrics_abs.get('mape', 0),
                "r2": val_metrics_abs['r2'],
                "direction_acc": val_metrics_abs.get('direction_acc_iv', val_metrics_abs.get('direction_acc', 0)),
            },
            "test": {
                "rmse": test_metrics_abs['rmse'],
                "mae": test_metrics_abs['mae'],
                "mape": test_metrics_abs['mape_iv'],
                "r2": test_metrics_abs['r2'],
                "direction_acc": test_metrics_abs.get('direction_acc_iv', test_metrics_abs.get('direction_acc', 0)),
            },
        },
        "comparison": {
            "test_rmse_iv_primary": test_metrics_delta['rmse_iv'],
            "test_rmse_iv_control": test_metrics_abs['rmse_iv'],
            "primary_vs_control": "similar" if abs(test_metrics_delta['rmse_iv'] - test_metrics_abs['rmse_iv']) < 0.01 else ("primary better" if test_metrics_delta['rmse_iv'] < test_metrics_abs['rmse_iv'] else "control better"),
        },
        "feature_importance_top10": fi_delta.head(10).to_dict(orient='records'),
        "data_info": {
            "n_samples_total": len(df),
            "n_features": len(feature_cols),
            "n_contracts": df['security_id'].nunique(),
            "date_range": f"{df['trade_date'].min()}-{df['trade_date'].max()}"
        }
    }

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------
    print("\n[Visualization] Generating plots...")
    plot_residual_analysis(df_test_pred, output_dir, 'Primary')
    plot_pred_vs_true(df_test_pred, output_dir, 'Primary')
    plot_timeseries_sample(df_test_pred, output_dir)
    plot_comparison_table(
        {f"test_{k}": v for k, v in test_metrics_delta.items()},
        {f"test_{k}": v for k, v in test_metrics_abs.items()},
        output_dir
    )

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------
    save_outputs(model_delta, model_abs, metrics_json, df_test_pred,
                 fi_delta, fi_abs, output_dir)

    print("\n[Done] All tasks completed successfully.")


if __name__ == '__main__':
    main()
