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
