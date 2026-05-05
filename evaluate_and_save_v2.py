# -*- coding: utf-8 -*-
"""
评估新模型并生成 v2 输出文件
- model_residual_v2.pkl
- predictions_test_v2.csv
- metrics_v2.json
"""
import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

warnings.filterwarnings('ignore')
OUTPUT_DIR = 'data/output/two_step_v2/'

# 加载 checkpoint
with open(f'{OUTPUT_DIR}mw_checkpoint_v2.pkl', 'rb') as f:
    cache = pickle.load(f)
df = cache['df']
daily_mw_data = cache['daily_mw_data']

print(f"[Load] 数据记录数: {len(df)}, spline数据: {len(daily_mw_data)}")

# 特征列（复用原方案）
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

# 确保特征存在
for col in FEATURE_COLS:
    if col not in df.columns:
        df[col] = 0

# 加载 forward_table
forward_table = pd.read_csv(f'{OUTPUT_DIR}forward_table.csv')

# 加载原始 baseline 模型用于对比
with open('data/output/baseline_xgb/model_abs_iv.pkl', 'rb') as f:
    baseline_model = pickle.load(f)

# 构造目标变量: residual_{t+1}（绝对残差）
df = df.sort_values(['security_id', 'trade_date']).reset_index(drop=True)
df['next_residual'] = df.groupby('security_id')['residual_iv'].shift(-1)
df['next_implc_volatlty'] = df.groupby('security_id')['implc_volatlty'].shift(-1)
df['next_baseline_iv'] = df.groupby('security_id')['baseline_iv'].shift(-1)
df['target_residual'] = df['next_residual']
df = df.dropna(subset=['target_residual', 'next_implc_volatlty', 'next_baseline_iv']).copy()

# 时间划分
TRAIN_END = 20240630
VAL_END = 20241231
train_df = df[df['trade_date'] <= TRAIN_END].copy()
val_df = df[(df['trade_date'] > TRAIN_END) & (df['trade_date'] <= VAL_END)].copy()
test_df = df[df['trade_date'] > VAL_END].copy()

print(f"[Split] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# 训练残差预测模型
print("\n[Train] 训练残差预测模型 (y = residual_{t+1})...")
X_train = train_df[FEATURE_COLS].fillna(0)
y_train = train_df['target_residual']
X_val = val_df[FEATURE_COLS].fillna(0)
y_val = val_df['target_residual']

model = xgb.XGBRegressor(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective='reg:squarederror', random_state=42, n_jobs=-1
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# 评估残差模型
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
val_r2 = r2_score(y_val, y_pred_val)

print(f"[Eval] Train RMSE (residual): {train_rmse:.6f}")
print(f"[Eval] Val RMSE (residual): {val_rmse:.6f}")
print(f"[Eval] Val R2 (residual): {val_r2:.4f}")
print(f"[Eval] Target residual std: {y_train.std():.6f}")
print(f"[Eval] Signal-to-Noise: {y_train.std()/val_rmse:.2f}")

# Feature importance
fi = pd.DataFrame({'feature': FEATURE_COLS, 'importance': model.feature_importances_})
fi = fi.sort_values('importance', ascending=False).reset_index(drop=True)
print(f"[Eval] Top 10 features:")
for i, row in fi.head(10).iterrows():
    print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")

# 保存模型
model_path = f'{OUTPUT_DIR}model_residual_v2.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"\n[Save] Model saved to {model_path}")

# 测试集预测
X_test = test_df[FEATURE_COLS].fillna(0)
y_test = test_df['target_residual']
y_pred_test = model.predict(X_test)

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\n[Eval] Test RMSE (residual): {test_rmse:.6f}")
print(f"[Eval] Test MAE (residual): {test_mae:.6f}")
print(f"[Eval] Test R2 (residual): {test_r2:.4f}")

# 总体 IV 预测评估
# pred_IV = baseline_IV_t (用T日baseline近似) + pred_residual
test_df = test_df.copy()
test_df['pred_residual'] = y_pred_test
test_df['pred_iv'] = test_df['baseline_iv'] + test_df['pred_residual']
test_df['true_iv'] = test_df['next_implc_volatlty']

# 过滤有效数据
valid = test_df.dropna(subset=['pred_iv', 'true_iv', 'implc_volatlty']).copy()

rmse_iv = np.sqrt(mean_squared_error(valid['true_iv'], valid['pred_iv']))
mae_iv = mean_absolute_error(valid['true_iv'], valid['pred_iv'])
r2_iv = r2_score(valid['true_iv'], valid['pred_iv'])
mape_iv = np.mean(np.abs((valid['true_iv'] - valid['pred_iv']) / valid['true_iv']))
direction_acc = np.mean(np.sign(valid['pred_iv'] - valid['implc_volatlty']) == np.sign(valid['true_iv'] - valid['implc_volatlty']))

print(f"\n[Eval] Test RMSE (IV): {rmse_iv:.6f}")
print(f"[Eval] Test MAE (IV): {mae_iv:.6f}")
print(f"[Eval] Test R2 (IV): {r2_iv:.4f}")
print(f"[Eval] Test MAPE (IV): {mape_iv:.4f}")
print(f"[Eval] Direction Acc: {direction_acc:.4f}")

# Baseline proxy corr
baseline_proxy_corr = np.corrcoef(valid['baseline_iv'], valid['next_baseline_iv'])[0, 1]
print(f"[Eval] Baseline Proxy Corr: {baseline_proxy_corr:.4f}")

# 保存 predictions_test_v2.csv
pred_cols = ['security_id', 'trade_date', 'call_put', 'exercise_price', 'last_edate',
             'remaining_time', 'moneyness', 'implc_volatlty', 'baseline_iv', 'residual_iv',
             'true_iv', 'pred_iv', 'pred_residual', 'target_residual']
pred_df = valid[pred_cols].copy()
pred_df['residual_iv'] = valid['residual_iv']
pred_df['iv_residual'] = pred_df['pred_iv'] - pred_df['true_iv']

pred_path = f'{OUTPUT_DIR}predictions_test_v2.csv'
pred_df.to_csv(pred_path, index=False)
print(f"\n[Save] Predictions saved to {pred_path}")

# 保存 metrics_v2.json
metrics = {
    'model_info': {
        'target': 'residual_{t+1}',
        'n_estimators': 500,
        'max_depth': 5,
        'learning_rate': 0.05,
    },
    'residual_model': {
        'train_rmse': float(train_rmse),
        'val_rmse': float(val_rmse),
        'val_r2': float(val_r2),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'target_std': float(y_train.std()),
        'signal_to_noise': float(y_train.std() / val_rmse),
        'predictable': bool(val_rmse < y_train.std() * 0.8),
    },
    'iv_prediction': {
        'test_rmse_iv': float(rmse_iv),
        'test_mae_iv': float(mae_iv),
        'test_r2_iv': float(r2_iv),
        'test_mape_iv': float(mape_iv),
        'direction_acc': float(direction_acc),
        'baseline_proxy_corr': float(baseline_proxy_corr),
    },
    'feature_importance_top10': fi.head(10).to_dict(orient='records'),
    'notes': {
        'residual_predictable': bool(val_rmse < y_train.std() * 0.8),
        'strategy_b_c_fallback': 'pure_cross_sectional' if val_rmse >= y_train.std() * 0.8 else 'residual_enhanced',
        'strategy_a_signal': 'pred_residual' if val_rmse >= y_train.std() * 0.8 else 'pred_residual_t1 - residual_t',
    }
}

metrics_path = f'{OUTPUT_DIR}metrics_v2.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"[Save] Metrics saved to {metrics_path}")

print("\n" + "=" * 70)
print("[Summary] 训练完成！")
print("=" * 70)
print(f"残差可预测性: {'是' if val_rmse < y_train.std() * 0.8 else '否'} (RMSE={val_rmse:.4f}, Std={y_train.std():.4f})")
print(f"策略 B/C: {'残差增强' if val_rmse < y_train.std() * 0.8 else '纯截面排序'}")
print(f"策略 A 信号: {'pred_residual_t1 - residual_t' if val_rmse < y_train.std() * 0.8 else 'pred_IV - baseline_IV'}")
print(f"测试集 RMSE_IV: {rmse_iv:.6f}")
print(f"测试集 R2_IV: {r2_iv:.4f}")
