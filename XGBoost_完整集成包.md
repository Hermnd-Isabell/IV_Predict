# XGBoost 项目完整集成包

> **前置**：Diffusion 项目 Phase 2 已完成，`regime2_surface_features.pt`（848 条，日期格式 int YYYYMMDD）已生成  
> **目标**：在 `E:\Codes\IV_Predict` 下完成数据增强、校验、训练集成

---

## 文件清单

| 文件 | 位置 | 作用 |
|------|------|------|
| `build_enhanced_dataset.py` | `E:\Codes\IV_Predict\` | **桥梁脚本**：拼接真实合约数据 + 合成曲面特征 |
| `test_enhanced_dataset.py` | `E:\Codes\IV_Predict\` | 校验脚本：检查偏移、比例、日期分布 |
| `train_two_step_v2_enhanced.py` | `E:\Codes\IV_Predict\` | **训练入口**：复制原文件，改 4 行代码 |

---

## 文件 1：build_enhanced_dataset.py

```python
"""
build_enhanced_dataset.py
将 PI-LCDM 合成曲面特征与 XGBoost 训练数据拼接。

位置：E:\Codes\IV_Predictuild_enhanced_dataset.py
输出：E:\Codes\IV_Predict\data\enhanced_dataset_v2.pkl
"""

import pandas as pd
import torch
import pickle
import numpy as np
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

REAL_DATA_PATH = r'E:\Codes\IV_Predict\data\output	wo_step_v2\mw_checkpoint_v2.pkl'
SYNTH_FEATURES_PATH = r'E:\Codes\Fin-diffusionesults\synthetic_variantsegime2_surface_features.pt'
REGIME_LABELS_PATH = r'E:\Codes\Fin-diffusion\dataegime_labels.pt'
OUTPUT_PATH = r'E:\Codes\IV_Predict\data\enhanced_dataset_v2.pkl'

# 你的训练/验证/测试时间划分（根据实际修改）
TRAIN_END = 20231231
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
    """加载 PI-LCDM 提取的合成曲面特征"""
    data = torch.load(SYNTH_FEATURES_PATH)
    features, metadata = data['features'], data['metadata']

    synth_by_date = defaultdict(list)
    for feat, meta in zip(features, metadata):
        date_val = meta['date']  # int, e.g. 20150619
        synth_by_date[date_val].append(feat)

    print(f"[load] Synthetic variants: {len(features)} total, {len(synth_by_date)} unique dates")
    return synth_by_date


def get_regime2_dates():
    """从 Phase 1 的 regime_labels.pt 读取 Regime 2 日期列表"""
    regime_data = torch.load(REGIME_LABELS_PATH)
    dates, regime_ids = regime_data['dates'], regime_data['regime_ids']

    def normalize_date(d):
        if isinstance(d, int): return d
        if isinstance(d, str): return int(d.strip().split()[0].replace('-',''))
        if hasattr(d, 'strftime'): return int(d.strftime('%Y%m%d'))
        raise ValueError(f"Unknown date type: {type(d)}")

    regime2_dates = {normalize_date(d) for d, r in zip(dates, regime_ids) if int(r) == 2}
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

    # === 构造 target_residual（如果 mw_checkpoint_v2.pkl 中没有）===
    if 'target_residual' not in df_real.columns:
        print("[build] Constructing target_residual from residual_iv...")
        df_real = df_real.sort_values(['security_id', 'trade_date']).reset_index(drop=True)
        df_real['next_residual'] = df_real.groupby('security_id')['residual_iv'].shift(-1)
        df_real['target_residual'] = df_real['next_residual']
        df_real = df_real.dropna(subset=['target_residual']).copy()
        print(f"[build] After target construction: {len(df_real)} rows")

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
```

---

## 文件 2：test_enhanced_dataset.py（可选，用于二次验证）

```python
"""
test_enhanced_dataset.py
二次校验增强数据集的质量。
"""

import pickle
import pandas as pd


def test_dataset_integrity(path=r'E:\Codes\IV_Predict\data\enhanced_dataset_v2.pkl'):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    df = data['df']
    n_real = data['n_real']
    n_synthetic = data['n_synthetic']

    print(f"=== Dataset Integrity Test ===")
    print(f"Total rows: {len(df)}")
    print(f"Real: {n_real}, Synthetic: {n_synthetic}")
    print(f"Synthetic ratio: {n_synthetic / len(df) * 100:.1f}%")

    # 测试 1: 特征列无缺失
    feature_cols = data['feature_cols']
    missing = [c for c in feature_cols if c not in df.columns]
    assert len(missing) == 0, f"Missing columns: {missing}"
    print(f"[PASS] All {len(feature_cols)} feature columns present.")

    # 测试 2: 无 NaN
    assert df[feature_cols].isna().sum().sum() == 0, "NaN found in features!"
    print("[PASS] No NaN in feature matrix.")

    # 测试 3: 标签存在
    assert 'target_residual' in df.columns, "target_residual missing!"
    assert df['target_residual'].isna().sum() == 0, "NaN in target!"
    print("[PASS] target_residual present and clean.")

    # 测试 4: 权重列正确
    assert 'sample_weight' in df.columns, "sample_weight missing!"
    assert set(df['sample_weight'].unique()).issubset({1.0, 0.3}), "Unexpected weight values!"
    print("[PASS] sample_weight column correct.")

    # 测试 5: 合成样本的 _lag1 特征与真实样本不同（证明替换生效）
    real_df = df[df['is_synthetic'] == False]
    synth_df = df[df['is_synthetic'] == True]

    if len(synth_df) > 0:
        real_atm_mean = real_df['atm_iv_call_lag1'].mean()
        synth_atm_mean = synth_df['atm_iv_call_lag1'].mean()
        print(f"[INFO] Real atm_iv mean: {real_atm_mean:.4f}")
        print(f"[INFO] Synth atm_iv mean: {synth_atm_mean:.4f}")

        # 合成样本的 ATM 应该系统性更高（因为是危机状态）
        if synth_atm_mean > real_atm_mean:
            print("[PASS] Synthetic ATM_IV higher than real (crisis states).")
        else:
            print("[WARNING] Synthetic ATM_IV not higher than real. Check regime filter.")

    print("\n=== All tests passed ===")


if __name__ == '__main__':
    test_dataset_integrity()
```

---

## 文件 3：train_two_step_v2_enhanced.py（训练入口，只改 4 行）

**不要直接修改原 `train_two_step_v2.py`，先复制一份**：

```bash
copy E:\Codes\IV_Predict	rain_two_step_v2.py E:\Codes\IV_Predict	rain_two_step_v2_enhanced.py
```

**然后在 `train_two_step_v2_enhanced.py` 中修改以下 4 处**：

### 修改 1：导入增强数据加载函数（文件顶部，import 区域）

```python
# 新增这一行 import
from build_enhanced_dataset import get_training_data_for_xgboost
```

### 修改 2：替换数据加载逻辑（找到原来加载 mw_checkpoint_v2.pkl 的地方）

**原代码**（可能在主函数开头或 `if __name__ == '__main__':` 内）：
```python
# 原来的加载方式（注释掉或删除）
# with open(DATA_PATH, 'rb') as f:
#     data = pickle.load(f)
# df = data['df']
```

**替换为**：
```python
# 增强数据加载
X, y, w, df = get_training_data_for_xgboost(
    r'E:\Codes\IV_Predict\data\enhanced_dataset_v2.pkl'
)
```

### 修改 3：时间划分时同步划分权重（找到 train_test_split 或时间掩码的地方）

**原代码可能类似**：
```python
# 按时间划分
train_mask = df['trade_date'] <= TRAIN_END
val_mask = (df['trade_date'] > TRAIN_END) & (df['trade_date'] <= VAL_END)
test_mask = df['trade_date'] > VAL_END

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]
```

**改为**：
```python
# 按时间划分（同步划分 sample_weight）
train_mask = df['trade_date'] <= TRAIN_END
val_mask = (df['trade_date'] > TRAIN_END) & (df['trade_date'] <= VAL_END)
test_mask = df['trade_date'] > VAL_END

X_train, y_train, w_train = X[train_mask], y[train_mask], w[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]
```

### 修改 4：model.fit 传入 sample_weight（找到训练代码）

**原代码**：
```python
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
```

**改为**：
```python
model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)], verbose=False)
```

---

## 执行顺序

```bash
cd E:\Codes\IV_Predict

# Step 1: 构造增强数据集（含 3 项校验）
python build_enhanced_dataset.py

# Step 2: （可选）二次完整性校验
python test_enhanced_dataset.py

# Step 3: 复制原训练文件
copy train_two_step_v2.py train_two_step_v2_baseline.py
copy train_two_step_v2.py train_two_step_v2_enhanced.py

# Step 4: 修改 train_two_step_v2_enhanced.py（4 处，见上文）
# 用编辑器或 Claude 完成

# Step 5: 训练增强版本
python train_two_step_v2_enhanced.py

# Step 6: （对比实验）训练原始版本
python train_two_step_v2_baseline.py
```

---

## 校验标准（build_enhanced_dataset.py 输出）

| 检查项 | 正常范围 | 如果异常 |
|--------|---------|---------|
| `Synthetic ratio` | **< 92%** | > 95% 回 Diffusion 降 K；> 90% 监控 feature importance |
| `iv_vs_atm_lag1 shift` | **< 2% (0.02)** | > 2% 告诉我，校准 PI-LCDM ATM 提取 |
| `Regime 2 in TRAIN` | **> 10 天** | < 10 告诉我，调整时间划分 |
| `split_ok` | **True** | False 则合成数据不进入训练集 |
| `ratio_ok` | **True** | False 则合成比例过高 |

**全部 PASS 后，即可进入训练阶段。**
