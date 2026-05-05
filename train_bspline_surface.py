"""
50ETF期权IV曲面预测 - B-Spline + XGBoost MultiOutputRegressor
方案B: 每天拟合IV曲面得到20维系数，预测系数向量，再还原曲面插值到任意合约
"""

import os
import json
import pickle
import warnings
from datetime import datetime
from typing import Tuple, Optional, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 全局常量
# =============================================================================
STRIKE_NODES = np.array([2.20, 2.30, 2.40, 2.50, 2.60])
N_MATURITIES = 4
N_STRIKES = 5
N_COEFFS = N_MATURITIES * N_STRIKES  # 20

TRAIN_END = '2024-06-30'
VAL_END = '2024-12-31'
TEST_START = '2025-01-01'

OUTPUT_DIR = 'data/output/bspline_surface/'
BASELINE_DIR = 'data/output/baseline_xgb/'

# =============================================================================
# Step 1: B-Spline 曲面拟合
# =============================================================================

def load_raw_data(path: str = 'data/raw/50etf_options.csv') -> pd.DataFrame:
    """读取原始期权数据"""
    df = pd.read_csv(path)
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    # 剔除 Greeks
    greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
    df = df.drop(columns=[c for c in greeks if c in df.columns], errors='ignore')
    return df


def fit_bspline_surface(df_day: pd.DataFrame, call_put: str,
                        strike_nodes: np.ndarray = STRIKE_NODES) -> Optional[np.ndarray]:
    """
    对单日单类型的合约拟合B-Spline曲面
    沿行权价方向用 CubicSpline，在标准节点插值
    返回20维系数向量 [mat1_k1..mat1_k5, mat2_k1..mat4_k5]
    """
    df_sub = df_day[df_day['call_put'] == call_put].copy()
    # 剔除到期日和IV=0
    df_sub = df_sub[(df_sub['remaining_time'] > 0) & (df_sub['implc_volatlty'] > 0)]

    if len(df_sub) < 10:
        return None

    # 按 last_edate 排序（近月到远月）
    maturities = sorted(df_sub['last_edate'].unique())
    if len(maturities) != N_MATURITIES:
        mat_info = df_sub.groupby('last_edate')['remaining_time'].first().sort_values()
        maturities = mat_info.index.tolist()[:N_MATURITIES]

    coeffs = []
    for mat in maturities:
        df_mat = df_sub[df_sub['last_edate'] == mat].sort_values('exercise_price')
        if len(df_mat) < 3:
            coeffs.extend([np.nan] * len(strike_nodes))
            continue

        x = df_mat['exercise_price'].values.astype(float)
        y = df_mat['implc_volatlty'].values.astype(float)

        # 去重（同一行权价可能有多个合约，取平均）
        uniq = pd.DataFrame({'x': x, 'y': y}).groupby('x')['y'].mean().reset_index().sort_values('x')
        xu, yu = uniq['x'].values, uniq['y'].values

        try:
            # CubicSpline 仅在数据范围内插值，范围外用边界值填充，避免外推爆炸
            cs = CubicSpline(xu, yu, extrapolate=False)
            iv_interp = cs(strike_nodes)
            # 范围外填充边界最近值
            for i in range(len(iv_interp)):
                if not np.isfinite(iv_interp[i]):
                    if strike_nodes[i] <= xu[0]:
                        iv_interp[i] = yu[0]
                    else:
                        iv_interp[i] = yu[-1]
        except Exception:
            iv_interp = np.interp(strike_nodes, xu, yu,
                                  left=yu[0], right=yu[-1])
        coeffs.extend(iv_interp.tolist())

    return np.array(coeffs, dtype=float)


def build_coefficient_series(df: pd.DataFrame,
                             strike_nodes: np.ndarray = STRIKE_NODES) -> pd.DataFrame:
    """
    构造系数时间序列
    每天2条样本（Call/Put各一），每条20维y
    """
    records = []
    grouped = df.groupby(['trade_date', 'call_put'])

    for (date, cp), group in grouped:
        coeffs = fit_bspline_surface(group, cp, strike_nodes)
        if coeffs is not None:
            rec = {'trade_date': date, 'call_put': cp}
            for i in range(len(coeffs)):
                rec[f'coeff_{i}'] = coeffs[i]
            # 记录当天该类型的近月剩余时间（用于后续特征）
            # 用原始 group（不过滤 remaining_time=0）确保4个到期月都存在
            mat_info_raw = group.groupby('last_edate')['remaining_time'].first().sort_values()
            if len(mat_info_raw) > 0:
                rec['near_month_rt'] = mat_info_raw.iloc[0]
                rec['far_month_rt'] = mat_info_raw.iloc[-1]
                # 记录4个到期月的 remaining_time（用于后续还原曲面）
                for idx in range(N_MATURITIES):
                    if idx < len(mat_info_raw):
                        edate = mat_info_raw.index[idx]
                        rt = mat_info_raw.iloc[idx]
                    else:
                        # 兜底：若不足4个，复制最远月
                        edate = mat_info_raw.index[-1]
                        rt = mat_info_raw.iloc[-1]
                    rec[f'mat_{idx}_edate'] = edate
                    rec[f'mat_{idx}_rt'] = rt
            records.append(rec)

    coeff_df = pd.DataFrame(records)
    coeff_df = coeff_df.sort_values(['trade_date', 'call_put']).reset_index(drop=True)
    return coeff_df


def fill_coeff_na_with_train_mean(coeff_df: pd.DataFrame,
                                  train_mask: pd.Series) -> pd.DataFrame:
    """用训练集均值填充所有NaN"""
    coeff_cols = [c for c in coeff_df.columns if c.startswith('coeff_')]
    train_means = coeff_df.loc[train_mask, coeff_cols].mean()
    for c in coeff_cols:
        coeff_df[c] = coeff_df[c].fillna(train_means[c])
    return coeff_df


# =============================================================================
# Step 2: 特征工程
# =============================================================================

def build_features(coeff_df: pd.DataFrame, fund_df: pd.DataFrame) -> pd.DataFrame:
    """
    特征工程：系数历史 + 标的数据 + 曲面上下文
    返回包含 X 和 y（20维系数目标）的 DataFrame
    """
    df = coeff_df.copy()
    df = df.sort_values(['call_put', 'trade_date']).reset_index(drop=True)

    coeff_cols = [c for c in df.columns if c.startswith('coeff_')]
    n_coeff = len(coeff_cols)

    # ---------- 从当日20维系数提取统计量 ----------
    coeff_mat = df[coeff_cols].values  # (N, 20)

    # 整体统计
    df['coeff_mean'] = np.mean(coeff_mat, axis=1)
    df['coeff_std'] = np.std(coeff_mat, axis=1)
    df['coeff_max'] = np.max(coeff_mat, axis=1)
    df['coeff_min'] = np.min(coeff_mat, axis=1)

    # 近月/远月均值（每个到期月5个节点）
    df['near_month_mean'] = np.mean(coeff_mat[:, 0:5], axis=1)
    df['m2_month_mean'] = np.mean(coeff_mat[:, 5:10], axis=1)
    df['m3_month_mean'] = np.mean(coeff_mat[:, 10:15], axis=1)
    df['far_month_mean'] = np.mean(coeff_mat[:, 15:20], axis=1)

    # 微笑曲率（每个到期月：(最低K + 最高K)/2 - 中间K），再取4个到期月平均
    curvatures = []
    for i in range(N_MATURITIES):
        c0 = coeff_mat[:, i * 5 + 0]      # K=2.2
        c2 = coeff_mat[:, i * 5 + 2]      # K=2.4 (中间)
        c4 = coeff_mat[:, i * 5 + 4]      # K=2.6
        curvatures.append((c0 + c4) / 2.0 - c2)
    df['smile_curvature'] = np.mean(curvatures, axis=0)

    # 期限斜率
    df['term_slope'] = df['far_month_mean'] - df['near_month_mean']

    # 25-delta skew 代理（近月 OTM-ITM 价差）
    # Call: K=2.6 OTM, K=2.2 ITM; Put: 相反。这里用绝对价差作为 skew 代理
    df['skew_25_proxy'] = coeff_mat[:, 4] - coeff_mat[:, 0]  # K2.6 - K2.2 近月

    # ---------- 历史滞后与滚动统计 ----------
    stat_cols = ['coeff_mean', 'near_month_mean', 'far_month_mean',
                 'term_slope', 'smile_curvature', 'skew_25_proxy']

    for cp in ['C', 'P']:
        mask = df['call_put'] == cp
        idx = df.index[mask]
        for col in stat_cols:
            vals = df.loc[idx, col].values
            for lag in [1, 2, 3, 4]:
                lagged = np.empty_like(vals)
                lagged[:lag] = np.nan
                lagged[lag:] = vals[:-lag]
                df.loc[idx, f'{col}_lag{lag}'] = lagged

            # 滚动窗口3
            ma3 = pd.Series(vals).rolling(window=3, min_periods=1).mean().values
            std3 = pd.Series(vals).rolling(window=3, min_periods=1).std().values
            trend3 = vals - np.concatenate([[np.nan] * 2, vals[:-2]])
            df.loc[idx, f'{col}_ma3'] = ma3
            df.loc[idx, f'{col}_std3'] = std3.fillna(0) if hasattr(std3, 'fillna') else np.where(np.isnan(std3), 0, std3)
            df.loc[idx, f'{col}_trend3'] = trend3

    # ---------- 合并标的数据 ----------
    df = df.merge(fund_df, on='trade_date', how='left')

    # ---------- 曲面上下文（t-1日）----------
    # 需要为每天每类型构造 t-1 的曲面统计
    for cp in ['C', 'P']:
        mask = df['call_put'] == cp
        idx = df.index[mask].values
        sorted_idx = idx[np.argsort(df.loc[idx, 'trade_date'].values)]

        for col in ['atm_iv', 'skew_25', 'term_slope']:
            src_col = {'atm_iv': 'near_month_mean', 'skew_25': 'skew_25_proxy', 'term_slope': 'term_slope'}[col]
            vals = df.loc[sorted_idx, src_col].values
            lagged = np.empty_like(vals)
            lagged[0] = np.nan
            lagged[1:] = vals[:-1]
            df.loc[sorted_idx, f'{col}_lag1'] = lagged

    # ---------- 时间特征 ----------
    for cp in ['C', 'P']:
        mask = df['call_put'] == cp
        idx = df.index[mask].values
        sorted_idx = idx[np.argsort(df.loc[idx, 'trade_date'].values)]
        dates = df.loc[sorted_idx, 'trade_date']
        days_gap = (dates.diff().dt.days).values
        df.loc[sorted_idx, 'days_gap'] = days_gap
    df['days_gap'] = df['days_gap'].fillna(1.0)

    # remaining_time_proxy: 近月剩余时间
    df['remaining_time_proxy'] = df['near_month_rt'].fillna(30)

    # ---------- 构造目标 y = 下一天的20维系数变化量 (delta_coeff) ----------
    for cp in ['C', 'P']:
        mask = df['call_put'] == cp
        idx = df.index[mask].values
        sorted_idx = idx[np.argsort(df.loc[idx, 'trade_date'].values)]
        for c in coeff_cols:
            vals = df.loc[sorted_idx, c].values
            target = np.empty_like(vals)
            target[:-1] = vals[1:]  # 绝对系数 coeff_{t+1}
            target[-1] = np.nan
            df.loc[sorted_idx, f'target_{c}'] = target

    # Call/Put 编码
    df['call_put_flag'] = (df['call_put'] == 'P').astype(int)

    return df


def extract_daily_fund_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """提取每日标的数据（去重）"""
    fund_cols = ['trade_date', 'fund_close', 'fund_high', 'fund_low',
                 'fund_volume', 'fund_amount', 'ten_year']
    fund_df = df_raw[fund_cols].drop_duplicates(subset=['trade_date']).copy()
    fund_df = fund_df.sort_values('trade_date').reset_index(drop=True)

    fund_df['fund_return'] = fund_df['fund_close'].pct_change()
    fund_df['fund_high_low_ratio'] = (fund_df['fund_high'] - fund_df['fund_low']) / fund_df['fund_close']
    return fund_df


# =============================================================================
# Step 3: 时间划分
# =============================================================================

def split_temporal(df: pd.DataFrame, train_end: str = TRAIN_END,
                   val_end: str = VAL_END) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按交易日时间切分"""
    train_end_dt = pd.Timestamp(train_end)
    val_end_dt = pd.Timestamp(val_end)

    train_df = df[df['trade_date'] <= train_end_dt].copy()
    val_df = df[(df['trade_date'] > train_end_dt) & (df['trade_date'] <= val_end_dt)].copy()
    test_df = df[df['trade_date'] > val_end_dt].copy()
    return train_df, val_df, test_df


# =============================================================================
# Step 4: 训练 MultiOutput XGBoost
# =============================================================================

def get_feature_columns(df: pd.DataFrame) -> list:
    """获取输入特征列（排除目标、ID、日期、辅助列）"""
    exclude = {'trade_date', 'call_put'}
    exclude.update([c for c in df.columns if c.startswith('target_')])
    # 保留 coeff_* 作为核心特征（当日20维曲面形状是最强信号）
    exclude.update([c for c in df.columns if c.startswith('mat_')])
    exclude.update({'near_month_rt', 'far_month_rt'})

    # 选择数值列
    feat_cols = [c for c in df.columns
                 if c not in exclude
                 and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    # 确保 call_put_flag 在前
    if 'call_put_flag' in feat_cols:
        feat_cols.remove('call_put_flag')
        feat_cols = ['call_put_flag'] + feat_cols
    return feat_cols


class PCAMultiOutputWrapper:
    """包装器：predict 自动做 PCA 逆变换"""
    def __init__(self, model, pca):
        self.model = model
        self.pca = pca
        self.estimators_ = model.estimators_

    def predict(self, X):
        pca_pred = self.model.predict(X)
        return self.pca.inverse_transform(pca_pred)


def train_multioutput_xgb(X_train: pd.DataFrame, y_train: pd.DataFrame,
                          X_val: pd.DataFrame, y_val: pd.DataFrame,
                          method: str = 'B'):
    """
    训练MultiOutputRegressor
    method='B'：直接预测20维
    method='C'：PCA降维到5维主成分，预测后再逆变换
    """
    if method == 'C':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=5)
        y_train_pca = pca.fit_transform(y_train.values)
        y_val_pca = pca.transform(y_val.values)

        param_grid = {
            'estimator__max_depth': [3, 5, 6],
            'estimator__learning_rate': [0.03, 0.05, 0.1],
            'estimator__n_estimators': [300, 500, 800],
        }
        best_rmse = float('inf')
        best_inner = None
        best_params = None

        base_params = {
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1,
        }

        print("[Grid Search PCA-C] 开始超参数搜索...")
        for params in ParameterGrid(param_grid):
            est_params = {**base_params}
            for k, v in params.items():
                est_params[k.replace('estimator__', '')] = v
            base = xgb.XGBRegressor(**est_params)
            model = MultiOutputRegressor(base, n_jobs=-1)
            model.fit(X_train.values, y_train_pca)
            pred_val_pca = model.predict(X_val.values)
            pred_val = pca.inverse_transform(pred_val_pca)
            rmse = np.sqrt(mean_squared_error(y_val.values, pred_val))
            print(f"  Params={params}, Val RMSE={rmse:.6f}")
            if rmse < best_rmse:
                best_rmse = rmse
                best_inner = model
                best_params = params

        wrapped = PCAMultiOutputWrapper(best_inner, pca)
        return wrapped, best_params, best_rmse

    # 方案B：直接预测20维，网格搜索
    param_grid = {
        'estimator__max_depth': [3, 5, 6],
        'estimator__learning_rate': [0.03, 0.05, 0.1],
        'estimator__n_estimators': [300, 500, 800],
    }
    best_rmse = float('inf')
    best_model = None
    best_params = None

    base_params = {
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1,
    }

    print("[Grid Search B] 开始超参数搜索...")
    for params in ParameterGrid(param_grid):
        est_params = {**base_params}
        for k, v in params.items():
            est_params[k.replace('estimator__', '')] = v
        base = xgb.XGBRegressor(**est_params)
        model = MultiOutputRegressor(base, n_jobs=-1)
        model.fit(X_train.values, y_train.values)
        pred_val = model.predict(X_val.values)
        rmse = np.sqrt(mean_squared_error(y_val.values, pred_val))
        print(f"  Params={params}, Val RMSE={rmse:.6f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = params

    return best_model, best_params, best_rmse


# =============================================================================
# Step 5: 还原曲面与插值
# =============================================================================

class BilinearSurfaceInterpolator:
    """
    自定义双线性插值器
    K方向：对每个到期月线性插值，范围外取边界值（避免外推爆炸）
    T方向：在到期月间线性插值
    """
    def __init__(self, coeff_vector: np.ndarray,
                 strike_nodes: np.ndarray = STRIKE_NODES,
                 maturities: np.ndarray = None):
        self.strike_nodes = np.asarray(strike_nodes, dtype=float)
        if maturities is None:
            maturities = np.array([30, 60, 90, 120], dtype=float)
        self.maturities = np.asarray(maturities, dtype=float)
        self.maturities = np.unique(self.maturities[np.isfinite(self.maturities)])
        if len(self.maturities) < 2:
            self.maturities = np.array([30, 60, 90, 120], dtype=float)

        n_mat = len(self.maturities)
        n_needed = len(self.strike_nodes) * n_mat
        if len(coeff_vector) >= n_needed:
            self.iv_grid = coeff_vector[:n_needed].reshape(n_mat, len(self.strike_nodes))
        else:
            pad = np.full(n_needed - len(coeff_vector), np.mean(coeff_vector))
            full_coeff = np.concatenate([coeff_vector, pad])
            self.iv_grid = full_coeff.reshape(n_mat, len(self.strike_nodes))

    def __call__(self, points):
        """
        points: array-like of shape (N, 2) with columns [K, T]
        returns: array of shape (N,)
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")

        K_vals = points[:, 0]
        T_vals = points[:, 1]
        n = len(K_vals)

        # K方向线性插值（每个到期月单独做），范围外取边界值
        iv_per_mat = np.empty((len(self.maturities), n))
        for i in range(len(self.maturities)):
            iv_per_mat[i, :] = np.interp(
                K_vals, self.strike_nodes, self.iv_grid[i],
                left=self.iv_grid[i, 0], right=self.iv_grid[i, -1]
            )

        # T方向线性插值
        result = np.empty(n)
        for j in range(n):
            T = T_vals[j]
            if T <= self.maturities[0]:
                result[j] = iv_per_mat[0, j]
            elif T >= self.maturities[-1]:
                result[j] = iv_per_mat[-1, j]
            else:
                result[j] = np.interp(T, self.maturities, iv_per_mat[:, j])
        return result


def restore_surface(coeff_vector: np.ndarray,
                    strike_nodes: np.ndarray = STRIKE_NODES,
                    maturities: np.ndarray = None) -> Callable:
    """
    从20维系数还原IV曲面函数
    返回插值函数 f(K, T) -> IV
    """
    return BilinearSurfaceInterpolator(coeff_vector, strike_nodes, maturities)


def interpolate_iv(coeff_vector: np.ndarray, K: float, T: float,
                   maturities: np.ndarray = None) -> float:
    """对单个 (K, T) 查询 IV"""
    interp = restore_surface(coeff_vector, STRIKE_NODES, maturities)
    return float(interp([[K, T]])[0])


# =============================================================================
# 评估
# =============================================================================

def enforce_monotonicity(coeff_vector: np.ndarray, call_put: str) -> np.ndarray:
    """
    对20维系数做单调性后处理，降低无套利违规率
    Call: 每个到期月5个节点随K增加非递增
    Put: 每个到期月5个节点随K增加非递减
    """
    vec = coeff_vector.copy()
    for i in range(N_MATURITIES):
        seg = vec[i * 5:(i + 1) * 5]
        if call_put == 'C':
            # 非递增：从高到低
            for j in range(1, 5):
                if seg[j] > seg[j - 1]:
                    seg[j] = seg[j - 1]
        else:
            # 非递减：从低到高
            for j in range(1, 5):
                if seg[j] < seg[j - 1]:
                    seg[j] = seg[j - 1]
        vec[i * 5:(i + 1) * 5] = seg
    return vec


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.DataFrame,
                   df_test_meta: pd.DataFrame,
                   df_raw_full: pd.DataFrame,
                   baseline_model=None,
                   apply_monotonic: bool = True) -> dict:
    """
    评估：还原曲面 -> 插值到合约 -> 计算RMSE
    关键修正：对每条测试样本（日期t），使用t+1日的合约和到期月结构进行评估
    """
    pred_coeff = model.predict(X_test.values)
    true_coeff = y_test.values

    metrics = {}

    # 1. 系数空间 RMSE（在标准节点上）
    metrics['coeff_rmse'] = float(np.sqrt(mean_squared_error(true_coeff, pred_coeff)))
    metrics['coeff_mae'] = float(mean_absolute_error(true_coeff, pred_coeff))

    # 2. 还原到合约层面的 IV（t+1日）
    pred_iv_list = []
    true_iv_list = []
    residual_list = []
    meta_records = []

    for i in range(len(df_test_meta)):
        row_meta = df_test_meta.iloc[i]
        date_t = row_meta['trade_date']
        cp = row_meta['call_put']

        # t+1 日的日期
        date_t1 = date_t + pd.Timedelta(days=1)
        # 找到实际下一个交易日（跳过周末节假日）
        # 从 coeff_df 或 raw 数据中找下一个存在的交易日
        future_dates = df_raw_full[df_raw_full['trade_date'] > date_t]['trade_date'].unique()
        if len(future_dates) == 0:
            continue
        future_dates = np.sort(future_dates)
        # 找最近的未来日期（如果在1-7天内）
        next_trade_date = None
        for fd in future_dates:
            gap = (fd - date_t).days
            if 1 <= gap <= 10:
                next_trade_date = fd
                break
        if next_trade_date is None:
            next_trade_date = future_dates[0]

        # t+1 日的原始合约
        day_contracts = df_raw_full[
            (df_raw_full['trade_date'] == next_trade_date) &
            (df_raw_full['call_put'] == cp) &
            (df_raw_full['remaining_time'] > 0) &
            (df_raw_full['implc_volatlty'] > 0)
        ].copy()

        if len(day_contracts) == 0:
            continue

        # t+1 日的4个到期月 remaining_time
        mat_info = day_contracts.groupby('last_edate')['remaining_time'].first().sort_values()
        mat_rts = []
        for j in range(N_MATURITIES):
            if j < len(mat_info):
                mat_rts.append(float(mat_info.iloc[j]))
            else:
                mat_rts.append(float(mat_info.iloc[-1]) if len(mat_info) > 0 else 30 + j * 30)
        mat_rts = np.array(mat_rts)

        # 预测系数（对应t+1日的绝对系数）
        pred_vec = pred_coeff[i].copy()
        if apply_monotonic:
            pred_vec = enforce_monotonicity(pred_vec, cp)

        interp = restore_surface(pred_vec, STRIKE_NODES, mat_rts)

        for _, contract in day_contracts.iterrows():
            K = contract['exercise_price']
            T = contract['remaining_time']
            true_iv = contract['implc_volatlty']

            try:
                pred_iv = float(interp([[K, T]])[0])
            except Exception:
                pred_iv = np.nan

            if not np.isfinite(pred_iv):
                pred_iv = float(np.mean(pred_vec))

            pred_iv_list.append(pred_iv)
            true_iv_list.append(true_iv)
            residual_list.append(pred_iv - true_iv)

            meta_records.append({
                'trade_date_t': date_t,
                'trade_date_t1': next_trade_date,
                'call_put': cp,
                'security_id': contract.get('security_id', ''),
                'exercise_price': K,
                'remaining_time': T,
                'last_edate': contract.get('last_edate', ''),
                'true_iv': true_iv,
                'pred_iv': pred_iv,
                'residual': pred_iv - true_iv,
            })

    pred_iv_arr = np.array(pred_iv_list)
    true_iv_arr = np.array(true_iv_list)

    mask = np.isfinite(pred_iv_arr) & np.isfinite(true_iv_arr)
    pred_iv_arr = pred_iv_arr[mask]
    true_iv_arr = true_iv_arr[mask]

    metrics['rmse_iv'] = float(np.sqrt(mean_squared_error(true_iv_arr, pred_iv_arr)))
    metrics['mae_iv'] = float(mean_absolute_error(true_iv_arr, pred_iv_arr))
    metrics['r2_iv'] = float(r2_score(true_iv_arr, pred_iv_arr))
    metrics['mape_iv'] = float(np.mean(np.abs((true_iv_arr - pred_iv_arr) / (true_iv_arr + 1e-8))) * 100)

    pred_df = pd.DataFrame(meta_records)
    if len(pred_df) > 0:
        pred_df = pred_df.loc[mask].reset_index(drop=True)

        # 分箱 RMSE
        fund_close_map = df_raw_full.groupby('trade_date')['fund_close'].first().to_dict()
        pred_df['fund_close'] = pred_df['trade_date_t1'].map(fund_close_map)
        pred_df['moneyness'] = pred_df['fund_close'] / pred_df['exercise_price']

        def maturity_bin(rt):
            if rt <= 30:
                return 'near'
            elif rt <= 90:
                return 'mid'
            else:
                return 'far'

        pred_df['maturity_bin'] = pred_df['remaining_time'].apply(maturity_bin)

        def moneyness_bin(m):
            if m > 1.05:
                return 'ITM'
            elif m < 0.95:
                return 'OTM'
            else:
                return 'ATM'

        pred_df['moneyness_bin'] = pred_df['moneyness'].apply(moneyness_bin)

        for b in ['near', 'mid', 'far']:
            sub = pred_df[pred_df['maturity_bin'] == b]
            if len(sub) > 0:
                metrics[f'rmse_{b}'] = float(np.sqrt(mean_squared_error(sub['true_iv'], sub['pred_iv'])))
            else:
                metrics[f'rmse_{b}'] = np.nan

        for b in ['ITM', 'ATM', 'OTM']:
            sub = pred_df[pred_df['moneyness_bin'] == b]
            if len(sub) > 0:
                metrics[f'rmse_{b}'] = float(np.sqrt(mean_squared_error(sub['true_iv'], sub['pred_iv'])))
            else:
                metrics[f'rmse_{b}'] = np.nan

        for cp in ['C', 'P']:
            sub = pred_df[pred_df['call_put'] == cp]
            if len(sub) > 0:
                metrics[f'rmse_{cp}'] = float(np.sqrt(mean_squared_error(sub['true_iv'], sub['pred_iv'])))
            else:
                metrics[f'rmse_{cp}'] = np.nan

        # 无套利检查（在t+1日预测曲面上）
        violations = check_arbitrage(pred_df)
        metrics['arbitrage_violation_rate'] = float(violations['violation_rate'])

    return metrics, pred_df


def check_arbitrage(pred_df: pd.DataFrame) -> dict:
    """
    无套利检查：
    - Call: IV 随 K 增加应基本递减
    - Put: IV 随 K 增加应基本递增
    """
    if len(pred_df) == 0:
        return {'violation_rate': 0.0}

    # 对每天每个到期月每个类型，检查单调性
    # 但 pred_df 是合约级别的，我们需要看预测 IV 在行权价方向的单调性
    # 由于插值是连续的，这里简化：按 (date, call_put, last_edate) 分组，检查 pred_iv 随 K 的单调性
    vcount = 0
    vtotal = 0

    date_col = 'trade_date_t1' if 'trade_date_t1' in pred_df.columns else 'trade_date'
    for (date, cp, edate), group in pred_df.groupby([date_col, 'call_put', 'last_edate']):
        if len(group) < 3:
            continue
        g = group.sort_values('exercise_price')
        Ks = g['exercise_price'].values
        IVs = g['pred_iv'].values
        # 计算一阶差分
        diffs = np.diff(IVs)
        if cp == 'C':
            # Call: IV 应随 K 增加而递减，所以 diffs 应大多为负
            # 如果递增次数超过 30%，视为违规
            n_up = np.sum(diffs > 0)
        else:
            # Put: IV 应随 K 增加而递增，所以 diffs 应大多为正
            n_up = np.sum(diffs < 0)

        if len(diffs) > 0 and n_up / len(diffs) > 0.3:
            vcount += 1
        vtotal += 1

    rate = vcount / max(vtotal, 1)
    return {'violation_rate': rate}


# =============================================================================
# 可视化
# =============================================================================

def plot_surface_sample(coeff_call: np.ndarray, coeff_put: np.ndarray,
                        mat_rts: np.ndarray, date_str: str, output_path: str):
    """某天Call/Put的IV曲面热力图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    K_grid = np.linspace(2.0, 3.0, 50)
    T_grid = np.linspace(mat_rts.min(), mat_rts.max(), 50)
    KK, TT = np.meshgrid(K_grid, T_grid)

    for ax, coeff, title in [(axes[0], coeff_call, 'Call'),
                              (axes[1], coeff_put, 'Put')]:
        interp = restore_surface(coeff, STRIKE_NODES, mat_rts)
        Z = interp(np.vstack([KK.ravel(), TT.ravel()]).T).reshape(KK.shape)
        im = ax.contourf(KK, TT, Z, levels=20, cmap='viridis')
        ax.set_xlabel('Strike K')
        ax.set_ylabel('Time to Maturity T')
        ax.set_title(f'{title} IV Surface ({date_str})')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_pred_vs_true(pred_df: pd.DataFrame, output_path: str):
    """预测vs真实散点图"""
    plt.figure(figsize=(6, 6))
    plt.scatter(pred_df['true_iv'], pred_df['pred_iv'], alpha=0.3, s=5)
    lim = [pred_df['true_iv'].min(), pred_df['true_iv'].max()]
    plt.plot(lim, lim, 'r--', lw=1)
    plt.xlabel('True IV')
    plt.ylabel('Predicted IV')
    plt.title('Predicted vs True IV (B-Spline Surface)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_residual_by_maturity(pred_df: pd.DataFrame, output_path: str):
    """按期限分箱的残差分布"""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = ['near', 'mid', 'far']
    data = [pred_df[pred_df['maturity_bin'] == b]['residual'].values for b in bins]
    ax.boxplot(data, labels=['Near (<=30)', 'Mid (30-90)', 'Far (>90)'])
    ax.axhline(0, color='r', linestyle='--')
    ax.set_ylabel('Residual (Pred - True)')
    ax.set_title('Residual Distribution by Maturity')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_comparison_table(metrics_new: dict, metrics_old: dict, output_path: str):
    """新旧方案指标对比表"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    rows = [['Metric', 'New (B-Spline)', 'Old (Baseline)', 'Improvement %']]
    for k in ['rmse_iv', 'mae_iv', 'r2_iv']:
        v_new = metrics_new.get(k, np.nan)
        v_old = metrics_old.get(k, np.nan)
        if k == 'r2_iv':
            imp = (v_new - v_old) / max(abs(v_old), 1e-6) * 100
        else:
            imp = (v_old - v_new) / max(v_old, 1e-6) * 100
        rows.append([k, f'{v_new:.5f}', f'{v_old:.5f}', f'{imp:.2f}%'])
    table = ax.table(cellText=rows, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.title('New vs Old Method Comparison')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# 保存输出
# =============================================================================

def save_outputs(model, metrics: dict, pred_df: pd.DataFrame,
                 feature_importance: pd.DataFrame, coeff_df: pd.DataFrame,
                 output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # 模型
    with open(os.path.join(output_dir, 'model_bspline_multioutput.pkl'), 'wb') as f:
        pickle.dump(model, f)

    # 系数时间序列
    coeff_df.to_csv(os.path.join(output_dir, 'bspline_coefficients.csv'), index=False)

    # 特征重要性
    feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

    # 指标
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    # 预测结果
    pred_df.to_csv(os.path.join(output_dir, 'predictions_test.csv'), index=False)


def load_baseline_metrics() -> dict:
    path = os.path.join(BASELINE_DIR, 'metrics.json')
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        # 兼容原方案的 metrics 结构
        if 'control_experiment' in data and 'test' in data['control_experiment']:
            test = data['control_experiment']['test']
            return {
                'test': {
                    'rmse_iv': test.get('rmse', test.get('rmse_abs_iv', np.nan)),
                    'mae_iv': test.get('mae', test.get('mae_abs_iv', np.nan)),
                    'r2_iv': test.get('r2', test.get('r2_abs_iv', np.nan)),
                }
            }
        return data
    return {}


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # [Checkpoint 1] 数据加载与B-Spline拟合
    # ------------------------------------------------------------------
    print("[Checkpoint 1] 数据加载与B-Spline拟合")
    df_raw = load_raw_data()
    print(f"  - 原始记录数: {len(df_raw)}")
    print(f"  - 交易日数: {df_raw['trade_date'].nunique()}")

    daily_counts = df_raw.groupby(['trade_date', 'call_put']).size().reset_index(name='count')
    call_avg = daily_counts[daily_counts['call_put'] == 'C']['count'].mean()
    put_avg = daily_counts[daily_counts['call_put'] == 'P']['count'].mean()
    print(f"  - Call/Put日均合约数: {call_avg:.1f} / {put_avg:.1f}")

    # 提取标的数据
    fund_df = extract_daily_fund_data(df_raw)

    # B-Spline 拟合
    coeff_df = build_coefficient_series(df_raw, STRIKE_NODES)
    success_days = coeff_df['trade_date'].nunique()
    print(f"  - B-Spline拟合成功天数: {success_days}")
    print(f"  - 系数矩阵形状: {len(coeff_df)} 行 x {N_COEFFS} 维 (Call+Put)")

    # 保存某天拟合效果图
    sample_date = coeff_df['trade_date'].iloc[0]
    sample_row_c = coeff_df[(coeff_df['trade_date'] == sample_date) & (coeff_df['call_put'] == 'C')]
    sample_row_p = coeff_df[(coeff_df['trade_date'] == sample_date) & (coeff_df['call_put'] == 'P')]
    if len(sample_row_c) > 0 and len(sample_row_p) > 0:
        mat_rts = np.array([sample_row_c[f'mat_{i}_rt'].values[0] for i in range(N_MATURITIES)])
        coeff_c = sample_row_c[[c for c in sample_row_c.columns if c.startswith('coeff_')]].values[0]
        coeff_p = sample_row_p[[c for c in sample_row_p.columns if c.startswith('coeff_')]].values[0]
        plot_surface_sample(coeff_c, coeff_p, mat_rts,
                            sample_date.strftime('%Y%m%d'),
                            os.path.join(OUTPUT_DIR, 'surface_sample.png'))

    # ------------------------------------------------------------------
    # [Checkpoint 2] 特征工程完成
    # ------------------------------------------------------------------
    print("\n[Checkpoint 2] 特征工程完成")
    df_feat = build_features(coeff_df, fund_df)

    # 填充NaN（用训练集均值，但先划分再填充）
    train_df, val_df, test_df = split_temporal(df_feat)
    train_mask = df_feat['trade_date'] <= pd.Timestamp(TRAIN_END)
    df_feat = fill_coeff_na_with_train_mean(df_feat, train_mask)

    # 重新划分（因为fill不影响日期）
    train_df, val_df, test_df = split_temporal(df_feat)

    # 去掉有NaN目标的行（最后一天没有次日目标）
    target_cols = [c for c in df_feat.columns if c.startswith('target_')]
    train_df = train_df.dropna(subset=target_cols).copy()
    val_df = val_df.dropna(subset=target_cols).copy()
    test_df = test_df.dropna(subset=target_cols).copy()

    feat_cols = get_feature_columns(train_df)
    print(f"  - 构造样本数: {len(train_df)} (train) + {len(val_df)} (val) + {len(test_df)} (test)")
    print(f"  - 输入特征维度: {len(feat_cols)}")
    print(f"  - 输出维度: {len(target_cols)}")
    print(f"  - 特征列名示例: {feat_cols[:10]}")

    # ------------------------------------------------------------------
    # [Checkpoint 3] 时间划分完成
    # ------------------------------------------------------------------
    print("\n[Checkpoint 3] 时间划分完成")
    print(f"  - 训练集: {train_df['trade_date'].min().date()} ~ {train_df['trade_date'].max().date()}, 样本数={len(train_df)}")
    print(f"  - 验证集: {val_df['trade_date'].min().date()} ~ {val_df['trade_date'].max().date()}, 样本数={len(val_df)}")
    print(f"  - 测试集: {test_df['trade_date'].min().date()} ~ {test_df['trade_date'].max().date()}, 样本数={len(test_df)}")
    leakage = not (train_df['trade_date'].max() < val_df['trade_date'].min() < test_df['trade_date'].min())
    print(f"  - 检查：是否存在交易日跨集合泄露？ {'Yes' if leakage else 'No'}")

    X_train, y_train = train_df[feat_cols], train_df[target_cols]
    X_val, y_val = val_df[feat_cols], val_df[target_cols]
    X_test, y_test = test_df[feat_cols], test_df[target_cols]

    # 再次确保无NaN
    X_train = X_train.fillna(X_train.median())
    X_val = X_val.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())

    # ------------------------------------------------------------------
    # [Checkpoint 4] 模型训练完成
    # ------------------------------------------------------------------
    print("\n[Checkpoint 4] 模型训练完成")
    method = 'C'
    model, best_params, best_val_rmse = train_multioutput_xgb(X_train, y_train, X_val, y_val, method=method)
    print(f"  - 方案: MultiOutputRegressor (method={method})")
    print(f"  - 最优参数: {best_params}")

    pred_train = model.predict(X_train.values)
    pred_val = model.predict(X_val.values)
    train_rmse = np.sqrt(mean_squared_error(y_train.values, pred_train))
    print(f"  - 训练20维平均RMSE: {train_rmse:.6f}")
    print(f"  - 验证20维平均RMSE: {best_val_rmse:.6f}")

    # 特征重要性（平均20个输出维度）
    importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
    fi_df = pd.DataFrame({'feature': feat_cols, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=False).reset_index(drop=True)
    print(f"  - Top 5特征（按平均重要性）: {fi_df['feature'].head(5).tolist()}")

    # ------------------------------------------------------------------
    # [Checkpoint 5] 评估与对照完成
    # ------------------------------------------------------------------
    print("\n[Checkpoint 5] 评估与对照完成")

    # 评估新方案（传入完整原始数据，用于查询t+1日合约）
    metrics_new, pred_df = evaluate_model(
        model, X_test, y_test, test_df,
        df_raw.copy()
    )

    print(f"  - 新方案 Test RMSE_IV: {metrics_new['rmse_iv']:.6f}")
    print(f"  - 新方案 Test R2_IV: {metrics_new['r2_iv']:.4f}")
    print(f"  - 无套利违规率: {metrics_new.get('arbitrage_violation_rate', 0)*100:.2f}%")
    print(f"  - 近月/中月/远月 RMSE: {metrics_new.get('rmse_near', 0):.5f} / {metrics_new.get('rmse_mid', 0):.5f} / {metrics_new.get('rmse_far', 0):.5f}")
    print(f"  - Call/Put RMSE: {metrics_new.get('rmse_C', 0):.5f} / {metrics_new.get('rmse_P', 0):.5f}")

    # 对照原方案
    baseline_metrics = load_baseline_metrics()
    comparison = {}
    if baseline_metrics and 'test' in baseline_metrics:
        old_test = baseline_metrics['test']
        old_rmse = old_test.get('rmse_iv', old_test.get('rmse_abs_iv', np.nan))
        old_r2 = old_test.get('r2_iv', old_test.get('r2_abs_iv', np.nan))
        comparison['baseline_rmse_iv'] = old_rmse
        comparison['new_rmse_iv'] = metrics_new['rmse_iv']
        if old_rmse and old_rmse > 0:
            comparison['relative_improvement_pct'] = (old_rmse - metrics_new['rmse_iv']) / old_rmse * 100
        else:
            comparison['relative_improvement_pct'] = np.nan
        print(f"  - 原方案 Test RMSE_IV: {old_rmse}")
        print(f"  - 相对改善: {comparison.get('relative_improvement_pct', np.nan):.2f}%")
    else:
        print("  - 原方案 metrics.json 未找到，跳过对照")
        comparison['note'] = 'baseline metrics not found'

    # 保存对照
    with open(os.path.join(OUTPUT_DIR, 'metrics_comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2, default=str)

    # 可视化
    if len(pred_df) > 0:
        plot_pred_vs_true(pred_df, os.path.join(OUTPUT_DIR, 'pred_vs_true.png'))
        plot_residual_by_maturity(pred_df, os.path.join(OUTPUT_DIR, 'residual_by_maturity.png'))

    if baseline_metrics:
        old_test = baseline_metrics.get('test', {})
        plot_comparison_table(metrics_new, old_test, os.path.join(OUTPUT_DIR, 'comparison_table.png'))

    # 保存所有输出
    save_outputs(model, metrics_new, pred_df, fi_df, coeff_df, OUTPUT_DIR)
    print(f"\n所有输出已保存到: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
