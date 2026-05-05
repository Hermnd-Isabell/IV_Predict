# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **50ETF option implied volatility (IV) prediction** project. The core task is to build an XGBoost regression model that predicts the **next-day implied volatility change (ΔIV)** for each option contract individually. This is a contract-level panel data problem: approximately 40 contracts per day (4 expiries × 5 strikes × 2 types), each treated as an independent sample.

The authoritative project specification is in `prompt_xgb_baseline_iv_v2.md`, which defines the feature engineering requirements, model parameters, evaluation metrics, output formats, and code structure in exhaustive detail. Read this file before writing any code.

## Data

- **Raw data**: `data/raw/50etf_options.csv` (343,158 rows, 30 columns, 4,856 unique contracts)
- **Date range**: 2015-02-09 to 2026-01-30
- **Format**: CSV with headers, no tab-delimited files despite what the prompt says
- **Key columns**: `security_id`, `trade_date`, `call_put`, `exercise_price`, `remaining_time`, `implc_volatlty`, `fund_close`, `fund_volume`, `ten_year`, plus Greeks (`delta`, `gamma`, `theta`, `vega`, `rho`)

## Critical Constraints (Non-Negotiable)

1. **No lookahead leakage**: All features must come from time `t` or earlier. Never use `t+1` information.
2. **No Greeks as features**: `delta`, `gamma`, `theta`, `vega`, `rho` are derived from IV via Black-Scholes and are strictly prohibited as model inputs to avoid circular reasoning.
3. **No random shuffling**: Data splits must be temporal. Train → Validation → Test in chronological order.
4. **No `security_id` as feature**: Do not let the model memorize specific contracts.
5. **Exclude expiration days**: Drop rows where `remaining_time == 0` or `implc_volatlty == 0`, because next-day target `y` cannot be constructed.
6. **Panel data structure**: Do not aggregate or compress daily data. Each contract-day is a separate sample.

## Expected Code Architecture

The prompt mandates a specific modular function structure:

- `load_raw_data(data_dir)` → `pd.DataFrame`
- `build_features(df)` → `pd.DataFrame` (feature engineering)
- `build_target(df)` → `pd.Series` (construct `delta_iv` and `next_implc_volatlty`)
- `split_temporal(df, train_end, val_end)` → `(train_df, val_df, test_df)`
- `train_model(X_train, y_train, X_val, y_val)` → `xgb.XGBRegressor` (with hyperparameter tuning)
- `evaluate_model(model, X_test, y_test, iv_t_test)` → `dict`
- `save_outputs(model, metrics, predictions, feature_importance, output_dir)`

Feature engineering categories to implement:
1. **Contract intrinsic**: `moneyness` (`fund_close / exercise_price`), `moneyness_squared`, `remaining_time`, `call_put`, `exercise_price`, interaction `moneyness × remaining_time`
2. **Underlying (spot) features**: `fund_return`, `fund_volume`, `fund_high_low_ratio`, `fund_amount`
3. **Historical IV lags** (per-contract, window=5): `iv_t` through `iv_t_4`, `iv_ma5`, `iv_std5`, `iv_trend5`, `days_gap`
4. **Surface context** (using `t-1` cross-sectional stats): `atm_iv_call_lag1` (linear interpolation), `iv_mean_all_lag1`, `iv_std_all_lag1`, `iv_max_all_lag1`, `iv_min_all_lag1`, `iv_vs_atm_lag1`
5. **Macro**: `ten_year / 100`

## Model & Training

- **Baseline**: `xgboost.XGBRegressor`
- **Suggested base params**: `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `objective='reg:squarederror'`, `random_state=42`
- **Hyperparameter search space**: `max_depth: [3,5,6,8]`, `learning_rate: [0.01,0.05,0.1]`, `n_estimators: [300,500,800]`, `subsample: [0.7,0.8,0.9]`
- **Temporal split**: Train (2015-02 ~ 2024-06), Validation (2024-07 ~ 2024-12), Test (2025-01 ~ 2026-01)
- **Two experiments**:
  - **Primary**: Predict `delta_iv = IV_{t+1} - IV_t`
  - **Control**: Predict absolute `next_implc_volatlty = IV_{t+1}` (using identical features)

## Evaluation Metrics

- **Primary (ΔIV)**: RMSE_ΔIV, MAE_ΔIV, R²_ΔIV, direction accuracy (sign of ΔIV)
- **Control (absolute IV)**: RMSE_IV, MAE_IV, MAPE_IV, R²_IV, direction accuracy
  - For control experiment, also report restored IV metrics where `pred_IV = IV_t + pred_ΔIV`
- **Grouped analysis**: By `security_id`, by moneyness bin (ITM/ATM/OTM), by remaining time (near/mid/far)

### Success Thresholds
- Test RMSE_ΔIV < 0.03
- Direction accuracy_ΔIV > 55%
- Test restored RMSE_IV < 0.05
- Top-5 feature importance must include at least one historical IV feature
- Control experiment RMSE_IV should **not** be dramatically better than primary (signals leakage)

## Outputs

All outputs go to `data/output/baseline_xgb/`:

- `model_delta_iv.pkl` / `model_abs_iv.pkl` — trained models
- `feature_importance_delta_iv.csv` / `feature_importance_abs_iv.csv`
- `metrics.json` — full metrics for train/val/test
- `predictions_test_delta_iv.csv` — test predictions with required columns: `security_id`, `trade_date`, `call_put`, `exercise_price`, `last_edate`, `remaining_time`, `moneyness`, `iv_t`, `true_delta_iv`, `pred_delta_iv`, `true_iv`, `pred_iv`, `residual_iv`, `residual_delta`
- Visualization PNGs (dpi ≥ 150): residual analysis, pred vs true scatter, time-series sample plots, comparison table

## Runtime Checkpoints

The code must print checkpoint logs at these stages:
1. `[Checkpoint 1] 数据加载完成` — raw record count, unique contracts, date range, columns
2. `[Checkpoint 2] 特征工程完成` — sample count, feature dimensions, target stats, first 3 rows
3. `[Checkpoint 3] 时间划分完成` — split ranges, sample counts, leakage check
4. `[Checkpoint 4] 模型训练完成` — best params, train/val RMSE
5. `[Checkpoint 5] 评估完成` — primary/control metrics, top 5 features

## Development Environment

This is a plain Python project with no build system, package manager, or test suite defined yet. Use standard data science libraries (`pandas`, `numpy`, `xgboost`, `scikit-learn`, `matplotlib`). There is no `requirements.txt` or `pyproject.toml` currently; install dependencies ad-hoc or create one if needed.
