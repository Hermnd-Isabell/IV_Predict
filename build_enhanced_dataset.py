"""
build_enhanced_dataset.py
将 PI-LCDM 合成曲面特征与 XGBoost 训练数据拼接。

位置：E:\\Codes\\IV_Predict\\build_enhanced_dataset.py
输出：E:\\Codes\\IV_Predict\\data\\enhanced_dataset_v2.pkl

注：本脚本不依赖 torch，直接通过 zipfile+pickle+numpy 读取 .pt 文件。
"""

import os
import re
import zipfile
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict


# ==================== 路径配置 ====================

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

REAL_DATA_PATH = r'E:\Codes\IV_Predict\data\output\two_step_v2\mw_checkpoint_v2.pkl'
SYNTH_FEATURES_PATH = r'E:\Codes\Fin-diffusion\results\synthetic_variants\regime2_surface_features.pt'
REGIME_LABELS_PATH = r'E:\Codes\Fin-diffusion\data\regime_labels.pt'
OUTPUT_PATH = r'E:\Codes\IV_Predict\data\enhanced_dataset_v2.pkl'

# 你的训练/验证/测试时间划分（根据实际修改）
TRAIN_END = 20240630
VAL_END = 20241231


# ==================== 加载函数 ====================

def load_real_data():
    """加载你的 XGBoost 训练数据（合约-日期粒度）"""
    with open(REAL_DATA_PATH, 'rb') as f:
        checkpoint = pickle.load(f)

    df = checkpoint['df'] if isinstance(checkpoint, dict) else checkpoint
    df['trade_date'] = df['trade_date'].astype(int)

    print(f"[load] Real data: {len(df)} rows, {df['trade_date'].nunique()} unique dates")
    return df


def load_synthetic_features():
    """加载 PI-LCDM 提取的合成曲面特征（不依赖 torch）"""
    with zipfile.ZipFile(SYNTH_FEATURES_PATH, 'r') as z:
        pkl_bytes = z.read('regime2_surface_features/data.pkl')
        data = pickle.loads(pkl_bytes, encoding='latin1')

    # data['features'] 是 dict[str, np.ndarray]，shape=(n_samples,)
    # data['metadata'] 是 list[dict]，length=n_samples
    feature_names = data['feature_names']
    features_dict = data['features']
    metadata = data['metadata']
    n_samples = len(metadata)

    # 转换为 list[dict]，每个 dict 对应一个合成样本
    features_list = []
    for i in range(n_samples):
        feat = {name: float(features_dict[name][i]) for name in feature_names}
        features_list.append(feat)

    # 特征名映射：扩散模型输出名 -> XGBoost 特征名
    # 扩散模型特征: atm_iv_short, atm_iv_mid, atm_iv_long, atm_iv_avg, term_slope, ...
    # XGBoost 需要的: atm_iv_call_lag1, iv_mean_all_lag1, iv_std_all_lag1, iv_max_all_lag1, iv_min_all_lag1
    def map_variant(feat):
        return {
            'atm_iv_call_lag1': feat.get('atm_iv_avg', feat.get('atm_iv_mid', 0.0)),
            'iv_mean_all_lag1': feat.get('surf_mean', 0.0),
            'iv_std_all_lag1': feat.get('surf_std', 0.0),
            'iv_max_all_lag1': feat.get('surf_max', 0.0),
            'iv_min_all_lag1': feat.get('surf_min', 0.0),
        }

    mapped_features = [map_variant(f) for f in features_list]

    synth_by_date = defaultdict(list)
    for feat, meta in zip(mapped_features, metadata):
        date_val = int(meta['date'])  # 确保是 int
        synth_by_date[date_val].append(feat)

    print(f"[load] Synthetic variants: {n_samples} total, {len(synth_by_date)} unique dates")
    return synth_by_date


def get_regime2_dates():
    """从 Phase 1 的 regime_labels.pt 读取 Regime 2 日期列表（不依赖 torch）"""
    with zipfile.ZipFile(REGIME_LABELS_PATH, 'r') as z:
        pkl_bytes = z.read('regime_labels/data.pkl')
        raw_data = z.read('regime_labels/data/0')

    # 从 data.pkl 中提取日期字符串（8位数字）
    dates_str = re.findall(rb'\d{8}', pkl_bytes)
    dates = [int(d.decode('ascii')) for d in dates_str]

    # 从 data/0 读取 regime_ids（int64 数组）
    regime_ids = np.frombuffer(raw_data, dtype=np.int64)

    if len(dates) != len(regime_ids):
        # 如果数量不匹配，可能是 dates_str 包含了其他数字
        # 取前 min(len(dates), len(regime_ids)) 个
        n = min(len(dates), len(regime_ids))
        dates = dates[:n]
        regime_ids = regime_ids[:n]

    regime2_dates = {int(d) for d, r in zip(dates, regime_ids) if int(r) == 2}
    print(f"[load] Regime 2 dates from regime_labels.pt: {len(regime2_dates)}")
    return regime2_dates


# ==================== 校验函数 ====================

def validate_iv_vs_atm_shift(df_enhanced):
    """校验 iv_vs_atm_lag1 分布偏移"""
    real_mask = df_enhanced['is_synthetic'] == False
    synth_mask = df_enhanced['is_synthetic'] == True

    if synth_mask.sum() == 0:
        print("[validate] No synthetic samples found.")
        return 0.0

    real_mean = df_enhanced.loc[real_mask, 'iv_vs_atm_lag1'].mean()
    synth_mean = df_enhanced.loc[synth_mask, 'iv_vs_atm_lag1'].mean()
    shift = synth_mean - real_mean

    print(f"[validate] Real iv_vs_atm mean: {real_mean:.4f}")
    print(f"[validate] Synth iv_vs_atm mean: {synth_mean:.4f}")
    print(f"[validate] Shift: {shift:.4f} ({shift/(abs(real_mean)+1e-8)*100:.1f}%)")

    if abs(shift) > 0.02:
        print("[WARNING] iv_vs_atm_lag1 shift > 2%! Consider recalibrating PI-LCDM ATM extraction.")
    else:
        print("[PASS] iv_vs_atm_lag1 shift within tolerance.")

    return shift


def validate_regime2_split(common_dates, train_end=TRAIN_END, val_end=VAL_END):
    """校验 Regime 2 日期在训练/验证/测试中的分布"""
    train_dates = {d for d in common_dates if d <= train_end}
    val_dates = {d for d in common_dates if train_end < d <= val_end}
    test_dates = {d for d in common_dates if d > val_end}

    print(f"[validate] Regime 2 in TRAIN: {len(train_dates)} dates")
    print(f"[validate] Regime 2 in VAL:   {len(val_dates)} dates")
    print(f"[validate] Regime 2 in TEST:  {len(test_dates)} dates")

    if len(train_dates) < 10:
        print("[FATAL] Too few Regime 2 dates in training set! Synthetic data won't help training.")
        return False

    print("[PASS] Sufficient Regime 2 dates in training set.")
    return True


def validate_synthetic_ratio(n_real, n_synthetic):
    """校验合成比例"""
    ratio = n_synthetic / (n_real + n_synthetic) * 100
    print(f"[validate] Synthetic ratio: {ratio:.1f}%")

    if ratio > 95:
        print("[WARNING] Synthetic ratio > 95%! Consider reducing k_variants or weight.")
        return False
    elif ratio > 90:
        print("[WARNING] Synthetic ratio > 90%. Monitor feature importance.")
    else:
        print("[PASS] Synthetic ratio within acceptable range.")

    return True


# ==================== 核心构造逻辑 ====================

def build_enhanced_dataset(synthetic_weight=0.3):
    """
    构造增强数据集。

    逻辑：
    1. 加载真实合约数据
    2. 构造 target_residual（如果 mw_checkpoint_v2.pkl 中没有）
    3. 对 Regime 2 的每个日期，加载 K 个合成曲面特征
    4. 对该日期的每个合约，生成 K 个合成样本（替换 _lag1，重算 iv_vs_atm_lag1）
    5. 合并，标记 is_synthetic 和 sample_weight
    """

    df_real = load_real_data()
    synth_by_date = load_synthetic_features()
    regime2_dates = get_regime2_dates()

    # === 特征工程与目标变量构造（如果 mw_checkpoint_v2.pkl 中尚未完成） ===
    if 'target_residual' not in df_real.columns or 'iv_t' not in df_real.columns:
        print("[build] Running feature engineering from train_two_step_v2...")
        import sys
        sys.path.insert(0, r'E:\Codes\IV_Predict')
        from train_two_step_v2 import build_features_delta_residual, build_target
        df_real = build_features_delta_residual(df_real)
        df_real = build_target(df_real)
        print(f"[build] After feature engineering: {len(df_real)} rows")

    # 取交集：Regime 2 日期 & 有合成变体 & 有真实合约
    real_dates = set(df_real['trade_date'].unique())
    common_dates = regime2_dates & real_dates & set(synth_by_date.keys())

    print(f"[build] Common Regime 2 dates with synthetic: {len(common_dates)}")

    # 校验日期分布
    split_ok = validate_regime2_split(common_dates)

    # A. 真实样本（全部保留）
    df_real['is_synthetic'] = False
    df_real['sample_weight'] = 1.0
    enhanced_rows = [df_real.copy()]

    # B. 合成样本（仅对 Regime 2 日期）
    n_synthetic = 0

    for date_int in sorted(common_dates):
        day_contracts = df_real[df_real['trade_date'] == date_int].copy()
        if len(day_contracts) == 0:
            continue

        variants = synth_by_date[date_int]

        for variant in variants:
            synth = day_contracts.copy()

            # 替换 5 个 _lag1 特征（来自合成 IVS）
            synth['atm_iv_call_lag1'] = variant['atm_iv_call_lag1']
            synth['iv_mean_all_lag1'] = variant['iv_mean_all_lag1']
            synth['iv_std_all_lag1'] = variant['iv_std_all_lag1']
            synth['iv_max_all_lag1'] = variant['iv_max_all_lag1']
            synth['iv_min_all_lag1'] = variant['iv_min_all_lag1']

            # 重算 iv_vs_atm_lag1 = 当日合约 IV (iv_t) - 合成 ATM_{t-1}
            synth['iv_vs_atm_lag1'] = synth['iv_t'] - variant['atm_iv_call_lag1']

            # 标记与权重
            synth['is_synthetic'] = True
            synth['sample_weight'] = synthetic_weight

            enhanced_rows.append(synth)
            n_synthetic += len(synth)

    # 合并
    df_enhanced = pd.concat(enhanced_rows, ignore_index=True)

    # 校验 iv_vs_atm 偏移
    validate_iv_vs_atm_shift(df_enhanced)

    # 校验合成比例
    n_real = len(df_enhanced[df_enhanced['is_synthetic'] == False])
    n_total = len(df_enhanced)
    ratio_ok = validate_synthetic_ratio(n_real, n_synthetic)

    print(f"\n[build] Real: {n_real}, Synthetic: {n_synthetic}, Total: {n_total}")

    # 保存
    output = {
        'df': df_enhanced,
        'feature_cols': FEATURE_COLS,
        'n_real': n_real,
        'n_synthetic': n_synthetic,
        'split_ok': split_ok,
        'ratio_ok': ratio_ok,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(output, f)

    print(f"[build] Saved to {OUTPUT_PATH}")
    return df_enhanced


def get_training_data_for_xgboost(enhanced_path=OUTPUT_PATH):
    """
    在你的 train_two_step_v2.py 中，替换原来的数据加载逻辑：

    原代码：
        with open('mw_checkpoint_v2.pkl', 'rb') as f:
            data = pickle.load(f)
        df = data['df']

    替换为：
        from build_enhanced_dataset import get_training_data_for_xgboost
        X, y, w, df = get_training_data_for_xgboost('data/enhanced_dataset_v2.pkl')
    """
    with open(enhanced_path, 'rb') as f:
        data = pickle.load(f)

    df = data['df']

    # 确保所有特征列存在
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in enhanced dataset: {missing_cols}")

    X = df[FEATURE_COLS]
    y = df['target_residual']
    w = df['sample_weight']

    # 删除 NaN
    mask = X.notna().all(axis=1) & y.notna() & w.notna()
    X, y, w, df = X[mask], y[mask], w[mask], df[mask]

    return X, y, w, df


if __name__ == '__main__':
    build_enhanced_dataset(synthetic_weight=0.3)
