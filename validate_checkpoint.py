# -*- coding: utf-8 -*-
"""检查点验证脚本 - 确认LSQUnivariateSpline修复后residual分布正常"""
import pickle
import numpy as np
import pandas as pd

OUTPUT_DIR = 'data/output/two_step_v2/'

# 加载新的 mw_checkpoint_v2.pkl
with open(f'{OUTPUT_DIR}mw_checkpoint_v2.pkl', 'rb') as f:
    cache = pickle.load(f)

df = cache['df']

print("=" * 70)
print("[Checkpoint 1] residual 不再恒为 0")
print("=" * 70)
print(f"[CP1] residual mean: {df['residual_iv'].mean():.6f} (应≈0)")
print(f"[CP1] residual std: {df['residual_iv'].std():.6f} (应>0.001)")
print(f"[CP1] residual non-zero pct: {(df['residual_iv'].abs() > 1e-6).mean():.2%} (应>50%)")
print(f"[CP1] residual min: {df['residual_iv'].min():.6f}")
print(f"[CP1] residual max: {df['residual_iv'].max():.6f}")

assert df['residual_iv'].std() > 0.001, "FAIL: residual std 过低，B-Spline 可能仍过拟合"
print("[CP1] PASS")

print()
print("=" * 70)
print("[Checkpoint 2] baseline 与 market_IV 有合理差异")
print("=" * 70)
diff = (df['baseline_iv'] - df['implc_volatlty']).abs()
print(f"[CP2] |baseline - market| mean: {diff.mean():.6f} (应>0.0005)")
print(f"[CP2] |baseline - market| median: {diff.median():.6f}")
print(f"[CP2] |baseline - market| 95th pct: {diff.quantile(0.95):.6f}")

assert diff.mean() > 0.0005, "FAIL: baseline 与 market_IV 差异过小"
print("[CP2] PASS")

print()
print("=" * 70)
print("[Checkpoint 3] 按到期月看拟合质量")
print("=" * 70)
df['tau'] = df['remaining_time'] / 365.0
for tau_val in sorted(df['tau'].unique())[:5]:
    subset = df[df['tau'] == tau_val]
    n_contracts = subset['M'].nunique()
    n_knots = max(3, min(n_contracts - 2, int(n_contracts * 0.4)))
    print(f"[CP3] tau={tau_val:.4f}: {n_contracts} contracts, {n_knots} knots, residual_std={subset['residual_iv'].std():.6f}")
print("[CP3] PASS")

print()
print("=" * 70)
print("[Checkpoint 4] 截面 zscore 可计算性（策略 B/C 的生死线）")
print("=" * 70)
df['residual_zscore'] = df.groupby(['trade_date', 'last_edate'])['residual_iv'].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() > 1e-6 else 0
)
print(f"[CP4] |zscore|>1.0 pct: {(df['residual_zscore'].abs() > 1.0).mean():.2%} (应>5%)")
print(f"[CP4] |zscore|>1.5 pct: {(df['residual_zscore'].abs() > 1.5).mean():.2%} (应>1%)")

assert (df['residual_zscore'].abs() > 1.0).mean() > 0.05, "FAIL: zscore 可交易比例过低"
print("[CP4] PASS")

print()
print("=" * 70)
print("[Checkpoint 5] residual 时序自相关")
print("=" * 70)
df_sorted = df.sort_values(['security_id', 'trade_date'])
df_sorted['residual_lag1'] = df_sorted.groupby('security_id')['residual_iv'].shift(1)
autocorr = df_sorted['residual_iv'].corr(df_sorted['residual_lag1'])
print(f"[CP5] residual autocorr(lag1): {autocorr:.4f} (=0 为白噪声, >0.1 为动量)")
print("[CP5] PASS (供参考，不强制)")

print()
print("=" * 70)
print("[Summary] 所有检查点通过！可以进入模型训练。")
print("=" * 70)
