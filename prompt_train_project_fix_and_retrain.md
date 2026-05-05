# 训练项目：修复 B-Spline 过拟合 + 重新训练路线 B 残差预测模型

## 一、问题背景（必须阅读）

当前 `train_two_step_v2.py` 中的 `fit_mw_surface` 使用 `CubicSpline(Mu, Wu)` 做精确插值。由于查询点 `M_query` 与拟合点 `Mu` 来自同一组合约，导致：

```
cs(M_i) = W_i = IV^2 * tau
baseline_IV = sqrt(W_i/tau) = IV
residual = market_IV - baseline_IV = 0
```

**后果**：
- 路线 B 的 residual 恒为 0，残差模型训练数据 y 全是噪声
- 策略 B/C 的 `residual_zscore` 全为 NaN/0，无法交易
- 旧权重 `model_residual.pkl` 是在退化数据上训练的，必须废弃

**修复目标**：用 `LSQUnivariateSpline` 替代 `CubicSpline`，通过减少内部节点数强制平滑，让 baseline 成为"无套利基准"而非"市场复制机"。

---

## 二、修改范围

### 2.1 修改文件
- `train_two_step_v2.py`：修改 `fit_mw_surface` 函数
- `utils/bspline_mw.py`（或存放 B-Spline 工具的文件）：同步修改

### 2.2 保留不变
- 数据加载逻辑（`load_data`, `preprocess`）
- 特征工程（`build_features`）
- XGBoost 超参数
- 训练/验证/测试集划分
- 评估指标计算

---

## 三、核心代码修改

### 3.1 替换 `fit_mw_surface`

**原代码（删除或注释掉）**：
```python
from scipy.interpolate import CubicSpline

def fit_mw_surface(df_mat, F, tau):
    uniq = df_mat.drop_duplicates(subset=['M']).sort_values('M')
    Mu = uniq['M'].values
    Wu = uniq['W'].values
    cs = CubicSpline(Mu, Wu)  # 精确插值，问题根源
    return cs, F, tau
```

**新代码**：
```python
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline
import numpy as np

def fit_mw_surface_v2(df_mat, F, tau):
    """
    用 LSQUnivariateSpline 拟合 M-W 空间 B-Spline 曲面。
    节点数按合约密度动态调整，避免精确插值导致 residual 恒为 0。

    Parameters:
        df_mat: 当日某到期月的合约 DataFrame，必须包含列 ['M', 'W', 'IV']
        F: 隐含远期价格（Put-Call Parity 反推）
        tau: 年化剩余期限

    Returns:
        cs: LSQUnivariateSpline 对象（支持 cs(x) 调用）
        F: 远期价格
        tau: 期限
    """
    # 去重并按 M 排序
    uniq = df_mat.drop_duplicates(subset=['M']).sort_values('M')
    Mu = uniq['M'].values
    Wu = uniq['W'].values

    n = len(Mu)

    # 合约数不足时无法拟合 B-Spline，返回 None（调用方应跳过该到期月）
    if n < 4:
        return None, F, tau

    # 动态节点数：合约越多节点越多，但始终少于数据点
    # 经验公式：内部节点数 = max(3, min(n-2, int(n * 0.4)))
    n_knots = max(3, min(n - 2, int(n * 0.4)))

    # 在 Mu 的分布范围内均匀放置内部节点（避开边界 10%，防止外推压力）
    # 节点必须严格位于数据范围内，且互不相同
    t = np.quantile(Mu, np.linspace(0.10, 0.90, n_knots))
    t = np.unique(t)

    # 确保至少 3 个内部节点（LSQUnivariateSpline 要求）
    if len(t) < 3:
        # 数据点太少且集中，补充等距节点
        m_min, m_max = Mu.min(), Mu.max()
        t = np.linspace(m_min + 0.01, m_max - 0.01, 3)

    # 拟合 LSQUnivariateSpline（k=3 为三次样条）
    try:
        cs = LSQUnivariateSpline(Mu, Wu, t=t, k=3)
    except Exception as e:
        # 数值问题回退：UnivariateSpline 带小平滑参数
        print(f"[WARN] LSQUnivariateSpline failed for tau={tau:.4f}, n={n}, fallback to UnivariateSpline: {e}")
        cs = UnivariateSpline(Mu, Wu, s=1e-5)

    return cs, F, tau
```

### 3.2 修改主训练循环中的调用点

找到 `train_two_step_v2.py` 中调用 `fit_mw_surface` 的位置，替换为 `fit_mw_surface_v2`，并处理 `None` 回退：

```python
# 原调用
cs, F, tau = fit_mw_surface(df_mat, F, tau)

# 新调用
cs, F, tau = fit_mw_surface_v2(df_mat, F, tau)
if cs is None:
    # 合约数不足，跳过该到期月
    df_mat['baseline_IV'] = np.nan
    df_mat['residual'] = np.nan
    continue
```

### 3.3 修改 `interpolate_baseline`（如有需要）

`interpolate_baseline` 函数本身不需要改——`LSQUnivariateSpline` 和 `CubicSpline` 都支持 `cs(x)` 调用。但请确认你的 `interpolate_baseline` 没有硬编码 `CubicSpline` 类型检查。

---

## 四、重新计算全局 Baseline

修改代码后，**必须重新跑完整数据流**，生成新的 `mw_checkpoint.pkl`：

```bash
python train_two_step_v2.py --stage compute_baseline --output /data/output/two_step_v2/
```

或手动执行主脚本的前半段（只计算 baseline，不训练模型）。

**关键**：不要加载旧的 `mw_checkpoint.pkl`，因为里面的 `baseline_IV` 是 CubicSpline 生成的（恒等于 market_IV）。

---

## 五、重新训练残差预测模型

### 5.1 训练数据变化

修复后，`residual` 不再恒为 0，XGBoost 现在有真实的 y 可学：

```python
# 特征 X：保持不变（moneyness, tau, iv_lag1, iv_std5, 等）
# 目标 y（二选一，建议先尝试 y = residual_{t+1}）：

# 方案 A：预测残差绝对值
y = residual_{t+1}

# 方案 B：预测残差变化（如果方案 A 效果差再试）
# y = residual_{t+1} - residual_t
```

**注意**：不要复用旧的 `model_residual.pkl` 权重。旧模型是在 `residual=0` 的数据上训练的，输入任何样本都会输出接近 0 的预测，完全失效。

### 5.2 训练脚本修改点

在 `train_two_step_v2.py` 的训练阶段：

```python
# 加载新计算的 baseline 和 residual
df = pd.read_pickle('/data/output/two_step_v2/mw_checkpoint_v2.pkl')

# 确认 residual 分布正常
assert df['residual'].std() > 0.001, "residual 标准差过低，B-Spline 可能仍过拟合"

# 构建训练集
X = df[feature_cols]
y = df['residual'].shift(-1)  # 预测次日残差（或 delta_residual）

# 删除 NaN
mask = X.notna().all(axis=1) & y.notna()
X, y = X[mask], y[mask]

# 训练/验证/测试划分（保持与原方案一致的时间顺序划分）
# ...

# 训练 XGBoost
import xgboost as xgb
model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train, 
          eval_set=[(X_val, y_val)], 
          early_stopping_rounds=50,
          verbose=True)

# 保存新权重
model.save_model('/data/output/two_step_v2/model_residual_v2.json')
# 或 pickle
import pickle
with open('/data/output/two_step_v2/model_residual_v2.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### 5.3 评估新模型

```python
# 测试集预测
y_pred = model.predict(X_test)

# 关键指标
from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"[Model Eval] Test RMSE (residual): {rmse:.6f}")
print(f"[Model Eval] Test R2 (residual): {r2:.4f}")
print(f"[Model Eval] residual std: {y_test.std():.6f}")
print(f"[Model Eval] Signal-to-Noise (std/rmse): {y_test.std()/rmse:.2f}")

# 通过标准：
# - RMSE < residual_std * 0.8（预测优于"预测不变"基准）
# - R2 > 0.01（至少学到一点信号）
# - 如果 RMSE > residual_std，说明残差不可预测，策略 B/C 的残差预测层无价值
```

---

## 六、验证检查点（必须打印）

在重新计算 baseline 后、训练模型前，执行以下检查：

```python
# [Checkpoint 1] residual 不再恒为 0
print(f"[CP1] residual mean: {df['residual'].mean():.6f} (应=0)")
print(f"[CP1] residual std: {df['residual'].std():.6f} (应>0.001)")
print(f"[CP1] residual non-zero pct: {(df['residual'].abs() > 1e-6).mean():.2%} (应>50%)")

# [Checkpoint 2] baseline 与 market_IV 有合理差异
diff = (df['baseline_IV'] - df['implc_volatlty']).abs()
print(f"[CP2] |baseline - market| mean: {diff.mean():.6f} (应>0.0005)")
print(f"[CP2] |baseline - market| median: {diff.median():.6f}")
print(f"[CP2] |baseline - market| 95th pct: {diff.quantile(0.95):.6f}")

# [Checkpoint 3] 按到期月看节点数和拟合质量
for tau_val in sorted(df['tau'].unique())[:5]:
    subset = df[df['tau'] == tau_val]
    n_contracts = subset['M'].nunique()
    n_knots = max(3, min(n_contracts - 2, int(n_contracts * 0.4)))
    print(f"[CP3] tau={tau_val:.4f}: {n_contracts} contracts, {n_knots} knots, residual_std={subset['residual'].std():.6f}")

# [Checkpoint 4] 截面 zscore 可计算性（策略 B/C 的生死线）
df['residual_zscore'] = df.groupby(['trade_date', 'tau'])['residual'].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() > 1e-6 else 0
)
print(f"[CP4] |zscore|>1.0 pct: {(df['residual_zscore'].abs() > 1.0).mean():.2%} (应>5%)")
print(f"[CP4] |zscore|>1.5 pct: {(df['residual_zscore'].abs() > 1.5).mean():.2%} (应>1%)")

# [Checkpoint 5] residual 时序自相关（策略 A 的信号质量参考）
df_sorted = df.sort_values(['security_id', 'trade_date'])
df_sorted['residual_lag1'] = df_sorted.groupby('security_id')['residual'].shift(1)
autocorr = df_sorted['residual'].corr(df_sorted['residual_lag1'])
print(f"[CP5] residual autocorr(lag1): {autocorr:.4f} (=0 为白噪声, >0.1 为动量)")

# [Checkpoint 6] 路线 B 总体 RMSE（baseline + 残差预测）
# 在模型训练后计算
# baseline_proxy_corr = corr(baseline_IV_t, baseline_IV_{t+1}_true)
# total_rmse = sqrt(baseline_rmse^2 + residual_rmse^2)  # 近似
```

**通过标准**：
- CP1: residual std > 0.001
- CP2: mean diff > 0.0005
- CP4: |zscore|>1.0 的比例 > 5%
- CP5: autocorr 供参考，不强制通过/不通过

如果任何检查点不通过，先调试 `fit_mw_surface_v2` 的节点数逻辑（调整 `0.4` 系数或边界 quantile `0.10/0.90`），不要进入模型训练。

---

## 七、输出文件清单

训练完成后，`/data/output/two_step_v2/` 目录下应包含：

| 文件 | 说明 | 是否新文件 |
|------|------|-----------|
| `mw_checkpoint_v2.pkl` | 新 baseline 和 residual（LSQUnivariateSpline 生成） | 是，覆盖旧文件 |
| `model_residual_v2.pkl` / `.json` | 新残差预测模型权重 | 是，旧权重废弃 |
| `predictions_test_v2.csv` | 测试集预测结果（含 baseline_IV, pred_residual, pred_IV） | 是 |
| `metrics_v2.json` | 模型评估指标（RMSE, R2, baseline_proxy_corr 等） | 是 |

---

## 八、迁移到回测平台

训练完成后，将以下文件复制到回测平台：

```bash
# 从训练项目
/data/output/two_step_v2/model_residual_v2.pkl
/data/output/two_step_v2/predictions_test_v2.csv
/data/output/two_step_v2/mw_checkpoint_v2.pkl  # 可选，如果回测平台需要重新计算 baseline

# 到回测平台
/data/output/two_step_v2/
```

**回测平台需要同步修改**：
- `fit_mw_surface` -> `fit_mw_surface_v2`（如果回测平台独立计算 baseline）
- 或直接使用 `predictions_test_v2.csv` 中的 `baseline_IV` 和 `pred_residual`

---

## 九、关键提醒

1. **不要保留旧权重**：旧 `model_residual.pkl` 是在 residual=0 的退化数据上训练的，输入任何样本都会输出=0，完全失效。必须重新训练。

2. **节点数是核心超参数**：`int(n * 0.4)` 是经验值。如果检查点 1 的 residual std 仍然 < 0.001，说明节点数仍然太多，降到 `int(n * 0.3)` 或 `int(n * 0.2)`。如果 residual std > 0.01，说明过度平滑，升到 `int(n * 0.5)`。

3. **近月合约的特殊性**：近月合约（remaining_time <= 30）通常只有 3-8 个，`n_knots` 会被压到 3。这可能导致近月 baseline 过度平滑。如果检查点 3 显示近月 residual_std 异常低，可考虑对近月单独用 `UnivariateSpline(s=1e-4)` 而非 LSQUnivariateSpline。

4. **路线 B 的 RMSE 可能恶化**：修复前 0.0299 是虚假的（baseline 完美复制 market）。修复后 baseline 不再完美，总体 RMSE 可能上升到 0.032-0.035。但如果残差模型能学到有效信号（RMSE_residual < residual_std * 0.8），策略层面仍有价值。

5. **如果残差模型训练后 RMSE > residual_std**：说明残差是纯噪声，不可预测。此时策略 B/C 应退化为"纯 B-Spline 截面排序"（不预测残差变化，只做当前 residual 的截面多空），策略 A 的信号也简化为 `pred_IV - baseline_IV`（不减去 residual_t）。

---

## 十、执行顺序

1. **备份旧文件**：`mv mw_checkpoint.pkl mw_checkpoint_old.pkl`
2. **修改 `fit_mw_surface` 为 `fit_mw_surface_v2`**（LSQUnivariateSpline）
3. **重新计算全局 baseline**（生成 `mw_checkpoint_v2.pkl`）
4. **运行检查点 1-6**，确认 residual 分布正常
5. **重新训练残差预测模型**（生成 `model_residual_v2.pkl`）
6. **评估新模型**（RMSE, R2, signal-to-noise）
7. **生成测试集预测**（`predictions_test_v2.csv`）
8. **迁移到回测平台**，跑策略 A/B/C 回测

**请先执行步骤 1-4（修复 + 验证），确认检查点全部通过后，再进入模型训练。**