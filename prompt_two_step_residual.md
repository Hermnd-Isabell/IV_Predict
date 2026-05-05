# 50ETF期权IV预测 - 路线A：B-Spline基准 + 逐合约残差预测

## 任务目标
在已完成的逐合约XGBoost基线（直接预测IV，RMSE≈0.028）和B-Spline曲面探索（无套利保证但精度差）基础上，升级为 **两步法**：

1. **B-Spline 做无套利基准**：每天用CubicSpline拟合Call/Put的IV曲面，插值到每个合约位置，得到"基准IV"（保证无套利）
2. **XGBoost 预测个体残差**：`residual = market_IV - baseline_IV`，预测每个合约明天的残差，保留个体时序信息
3. **合成最终预测**：`pred_IV = pred_baseline_IV + pred_residual`

**核心认知**：B-Spline擅长"画轮廓"（保证曲面形状合理），XGBoost擅长"补细节"（捕捉个体合约的时序规律）。两者各司其职。

## 数据路径
- 原始数据：`/data/raw/50etf_options.csv`
- 原方案输出（对照用）：`/data/output/baseline_xgb/`
  - 必须存在：`model_abs_iv.pkl`（原逐合约XGBoost模型）
  - 必须存在：`predictions_test_delta_iv.csv`（原方案测试集预测结果）
- 新方案输出：`/data/output/two_step_residual/`

## 核心约束（不可更改）
1. **剔除Greeks**：delta/gamma/theta/vega/rho 禁止作为模型输入（内生性问题）
2. **截面统计用t-1日**：防止信息泄露
3. **不把security_id作为模型输入**
4. **时间序列划分**：按时间切分，禁止随机打乱
5. **保留原方案对照**：必须对比"两步法" vs "原方案"的测试集RMSE

---

## 5步流水线架构

### Step 1: B-Spline 基准IV计算（每天收盘后，Call/Put各一次）

**输入**：某一天同一类型（Call或Put）的所有合约
  - 行权价 K：exercise_price
  - 剩余期限 T：remaining_time
  - IV：implc_volatlty

**拟合过程**：
1. 对每个到期月（如1503/1504/1506/1509），收集该月所有行权价的IV
2. 在该到期月内，用 **CubicSpline** 沿行权价方向拟合 IV = f(K)
3. **关键**：不是只插值到固定节点，而是**插值到当天该类型所有合约的实际行权价位置**
   - 例如：当天Call有 K=2.20, 2.25, 2.30, 2.35, 2.40, 2.45, 2.50... 
   - 对每个K，用CubicSpline插值得到 `baseline_iv(K)`
4. 若某合约的K在拟合范围外（小于最小K或大于最大K），用**边界值外推**（flat extrapolation：取最近边界值，不做线性外推）
5. 若某天某到期月合约数 < 3，该月所有合约的baseline_iv用**同天同类型其他到期月的ATM IV**填充

**输出**：
  - 每个合约当天的 `baseline_iv`（标量）
  - 同时保存该天B-Spline曲面的系数（用于T+1日快速插值）

**无套利保证**：
  - Call：同一到期月内，baseline_iv 随 K 非递增（若Spline输出违反，强制单调调整）
  - Put：同一到期月内，baseline_iv 随 K 非递减

### Step 2: 残差计算与构造训练样本

**残差定义**：
```
residual_iv = implc_volatlty - baseline_iv
```

**解释**：
- residual > 0：该合约被市场"高估"（IV高于无套利基准）
- residual < 0：该合约被市场"低估"
- residual ≈ 0：该合约定价接近无套利曲面

**训练样本构造**（与原方案一致，只是y改为residual）：

**输入特征 X（与原方案相同）**：

| 特征类别 | 具体特征 | 时序来源 | 说明 |
|---------|---------|---------|------|
| 合约固有 | moneyness, moneyness², remaining_time, call_put, exercise_price, moneyness×remaining_time | t | 直接取 |
| 标的数据 | fund_return, fund_volume, fund_high_low_ratio, fund_amount | t | 外生变量 |
| 历史IV | iv_t, iv_t-1, iv_t-2, iv_t-3, iv_t-4, iv_ma5, iv_std5, iv_trend5 | t, t-1... | 按交易日索引滑动窗口 |
| 历史残差 | residual_t, residual_t-1, residual_t-2, residual_ma3, residual_std3 | t, t-1... | **新增关键特征** |
| 曲面上下文 | atm_iv_call_lag1, iv_mean_all_lag1, iv_std_all_lag1, iv_vs_atm_lag1 | t-1 | 防止信息泄露 |
| 宏观 | ten_year | t | 年化利率 |
| 时间 | days_gap | t | 距上个交易日天数 |

**输出 y**：
  - `residual_{t+1} = IV_{t+1} - baseline_{t+1}`
  - 注意：baseline_{t+1} 是T+1日用T+1日所有合约重新拟合的B-Spline，不是用T日的曲面外推

**关键处理**：
- 到期日（remaining_time=0）样本剔除（无法构造T+1的y）
- implc_volatlty=0 的样本剔除
- 若T+1日某合约不存在（如到期退市），该样本剔除

### Step 3: XGBoost 残差预测模型训练

**模型**：单输出 XGBRegressor（回到原方案架构，只是y改为residual）

```python
import xgboost as xgb

model_residual = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)
```

**超参数调优**：
- max_depth: [3, 5, 6, 8]
- learning_rate: [0.01, 0.05, 0.1]
- n_estimators: [300, 500, 800]
- subsample: [0.7, 0.8, 0.9]

**训练逻辑**：
- 与原方案完全一致
- 只是目标变量从 `IV_{t+1}` 改为 `residual_{t+1}`
- 特征工程代码可大量复用原方案

### Step 4: 两步法预测（T日 → T+1日）

**预测流程**：

```
T日收盘后：
  1. 用T日所有合约拟合 B-Spline → 得到T日曲面
  2. 用T日曲面插值，预测T+1日每个合约的 baseline_IV_pred
     （这里用T日曲面作为T+1日基准的代理，因为T+1日市场数据未知）

  3. 用XGBoost预测T+1日每个合约的 residual_pred

  4. 合成预测：pred_IV = baseline_IV_pred + residual_pred
```

**重要修正**：
- **训练阶段**：y = `IV_{t+1} - baseline_{t+1}`（用T+1日真实B-Spline计算）
- **预测阶段**：baseline 用 **T日曲面外推** 或 **T日同位置插值** 作为代理
- 更精确的做法：预测 residual 后，在T+1日开盘前用T日曲面插值到T+1日合约列表（T+1日合约列表已知，只是价格未知）

**简化处理（推荐）**：
- 训练时：y = `IV_{t+1} - baseline_{t+1}`
- 预测时：baseline 用 **T日曲面在T+1日合约K上的插值值**（假设T+1日合约列表与T日相同或已知）
- 若T+1日有新合约上市，用T日曲面外推或全局均值

### Step 5: 评估与对照

**评估指标**（与原方案完全一致）：
1. **Test RMSE_IV**：合成pred_IV vs 真实IV
2. **Test MAE_IV**
3. **Test R²_IV**
4. **方向准确率**：sign(pred_IV - IV_t) == sign(true_IV - IV_t)
5. **分箱RMSE**：按moneyness / remaining_time / call_put 分箱
6. **残差预测RMSE**：pred_residual vs true_residual（评估残差模型本身）
7. **无套利违规率**：对合成pred_IV重新拟合B-Spline，检查是否违反单调性

**对照实验**：
- **对照1**：与原方案（直接预测IV）对比RMSE
- **对照2**：与"纯B-Spline"（只用baseline，不加残差）对比RMSE
- **对照3**：残差模型的特征重要性 vs 原方案特征重要性

---

## 数据划分（与原方案完全一致）

- 训练集：2015-02 ~ 2024-06（约前80%时间）
- 验证集：2024-07 ~ 2024-12（中间10%）
- 测试集：2025-01 ~ 2026-01（最后20%）

**注意**：B-Spline拟合在划分前全局计算（不泄露，因为只用当日截面数据），残差模型训练在划分后进行。

---

## 输出文件要求

保存到 `/data/output/two_step_residual/`：

1. `model_residual.pkl` - 训练好的残差预测XGBoost
2. `feature_importance_residual.csv` - 残差模型特征重要性
3. `metrics.json` - 完整评估指标（含两步法、纯B-Spline、原方案对照）
4. `predictions_test.csv` - 测试集预测结果（必须包含以下列）：
   - security_id, trade_date, call_put, exercise_price, remaining_time, moneyness
   - iv_t（T日IV）, true_iv（T+1日真实IV）
   - baseline_iv（T+1日B-Spline基准）, true_residual（T+1日真实残差）
   - pred_residual（预测残差）, pred_iv（合成预测IV = baseline + pred_residual）
   - residual_residual（残差预测误差）, iv_residual（IV预测误差）
5. `residual_analysis.png` - 残差分布分析（真实残差vs预测残差散点图）
6. `pred_vs_true.png` - 合成IV预测vs真实散点图
7. `comparison_table.png` - 三步对比表（原方案 vs 纯B-Spline vs 两步法）
8. `baseline_surface_sample.png` - 某天B-Spline基准曲面可视化

---

## 代码结构规范（必须模块化）

```python
def load_raw_data(path) -> pd.DataFrame:
    """读取原始数据，返回带列名的DataFrame"""

def compute_baseline_iv(df_day, call_put) -> pd.Series:
    """
    对单日单类型计算B-Spline基准IV
    输入：当天该类型所有合约（含K, T, IV）
    输出：每个合约的baseline_iv（Series，index与输入对齐）
    内部逻辑：
      1. 按到期月分组
      2. 每组用CubicSpline沿K拟合
      3. 插值到每个合约的实际K
      4. 范围外用边界值保护
      5. 单调性约束（Call非递增，Put非递减）
    """

def build_residual_series(df) -> pd.DataFrame:
    """
    全局计算所有合约的baseline_iv和residual_iv
    输出：在原df基础上增加baseline_iv, residual_iv列
    """

def build_features_residual(df) -> pd.DataFrame:
    """
    特征工程（与原方案一致，但增加历史残差特征）
    输出：X（特征矩阵）, y_residual（residual_{t+1}）, y_iv（IV_{t+1}，用于对照）
    """

def split_temporal(df, train_end, val_end) -> tuple:
    """按trade_date时间切分，与原方案一致"""

def train_residual_model(X_train, y_train, X_val, y_val) -> xgb.XGBRegressor:
    """训练XGBoost残差预测模型，支持超参数调优"""

def predict_two_step(model_residual, df_test_t, df_test_t1) -> pd.DataFrame:
    """
    两步法预测：
    输入：T日数据（用于构造baseline），T+1日数据（用于获取真实baseline和评估）
    输出：pred_iv = baseline_t1 + pred_residual
    """

def evaluate_two_step(predictions_df, baseline_only_df, original_df) -> dict:
    """
    评估：
    1. 两步法指标（合成IV）
    2. 纯B-Spline指标（只用baseline_iv）
    3. 原方案指标（加载/model_abs_iv.pkl在相同测试集预测）
    返回对比字典
    """

def save_outputs(model, metrics, predictions, output_dir):
    """保存所有输出文件"""
```

---

## 检查点（必须打印）

```
[Checkpoint 1] 数据加载与B-Spline基准计算
  - 原始记录数: {N}
  - 交易日数: {T}
  - B-Spline拟合成功天数: {success_days}
  - baseline_iv 计算覆盖率: {coverage}%（NaN比例应<5%）
  - residual_iv 统计: mean={x}, std={y}, min={z}, max={w}
  - residual分布: 应近似以0为中心对称
  - 某天(如20150209)Call曲面可视化: baseline_surface_sample.png

[Checkpoint 2] 特征工程完成
  - 构造样本数: {N_samples}
  - 特征维度: {N_features}
  - 新增特征确认: residual_t, residual_t-1, residual_ma3 等已包含
  - 目标变量统计（residual_{t+1}）: mean={x}, std={y}
  - 样本示例（前3行）:
    ...

[Checkpoint 3] 时间划分完成
  - 训练集: {train_start} ~ {train_end}, 样本数={n_train}
  - 验证集: {val_start} ~ {val_end}, 样本数={n_val}
  - 测试集: {test_start} ~ {test_end}, 样本数={n_test}

[Checkpoint 4] 模型训练完成
  - 最优参数: {params}
  - 训练RMSE（残差）: {x}
  - 验证RMSE（残差）: {y}
  - 残差模型Top 5特征: {feature_list}

[Checkpoint 5] 评估与对照完成
  - 两步法 Test RMSE_IV: {rmse_two_step}
  - 两步法 Test R²_IV: {r2_two_step}
  - 两步法 Test MAE_IV: {mae_two_step}
  - 两步法方向准确率: {dir_acc_two_step}
  - 纯B-Spline Test RMSE_IV: {rmse_baseline_only}
  - 原方案 Test RMSE_IV: {rmse_original}
  - 两步法 vs 原方案差异: {diff_pct}%
  - 残差模型 Test RMSE（预测残差）: {rmse_residual}
  - 无套利违规率（对合成IV检查）: {violation_rate}%
  - 近月/中月/远月 RMSE: {near} / {mid} / {far}
  - Call/Put RMSE: {call} / {put}
```

---

## 中间结果验证清单

- [ ] `baseline_iv` 覆盖率 > 95%（极少NaN）
- [ ] `residual_iv` 均值 ≈ 0（B-Spline基准应无偏）
- [ ] `residual_iv` 标准差 < 0.05（残差应比原始IV波动小）
- [ ] Call的baseline_iv随K非递增；Put的baseline_iv随K非递减
- [ ] 特征中 **不包含** delta/gamma/theta/vega/rho
- [ ] 特征中 **不包含** t+1日信息（如IV_{t+1}）
- [ ] residual_{t+1} 与 residual_t 的相关系数 > 0.3（验证残差有持续性）

---

## 成功标准

- [ ] **两步法 Test RMSE_IV ≤ 原方案 Test RMSE_IV × 1.1**（允许10%以内损耗，换取无套利保证）
- [ ] **两步法 Test RMSE_IV < 纯B-Spline Test RMSE_IV**（残差模型必须有用）
- [ ] **残差模型 Test RMSE（预测残差）< residual标准差 × 0.8**（优于用0猜测）
- [ ] **无套利违规率 < 5%**（B-Spline基准保证）
- [ ] **方向准确率 > 50%**（残差方向有可预测性）
- [ ] **特征重要性Top 5中至少包含1个残差历史特征**（验证残差时序有用）

---

## 异常处理规范

- `implc_volatlty = 0`：不参与B-Spline拟合，baseline设为同到期月均值
- 某天某到期月合约数 < 3：该月baseline用同类型其他到期月ATM IV填充
- 合约K在拟合范围外：用最近边界值（不做线性外推）
- 历史残差窗口不足：用可用天数计算
- T+1日合约退市：该样本剔除

---

## Reviewer快速验证脚本

```python
import json, pandas as pd, os

# 1. 检查输出完整性
assert os.path.exists('/data/output/two_step_residual/model_residual.pkl')
assert os.path.exists('/data/output/two_step_residual/metrics.json')
assert os.path.exists('/data/output/two_step_residual/predictions_test.csv')

# 2. 检查关键指标
with open('/data/output/two_step_residual/metrics.json') as f:
    m = json.load(f)
assert m['two_step']['test']['rmse_iv'] < 0.035, "两步法RMSE过高"
assert m['two_step']['test']['rmse_iv'] < m['baseline_only']['test']['rmse_iv'], "残差模型无效"
assert m['two_step']['test']['direction_acc'] > 0.50, "方向准确率过低"
assert m['two_step']['test']['arbitrage_violation_rate'] < 0.05, "无套利保证失效"

# 3. 检查对照结果
print(f"两步法RMSE: {m['two_step']['test']['rmse_iv']:.4f}")
print(f"原方案RMSE: {m['original']['test']['rmse_iv']:.4f}")
print(f"相对差异: {(m['two_step']['test']['rmse_iv'] - m['original']['test']['rmse_iv']) / m['original']['test']['rmse_iv'] * 100:.1f}%")

# 4. 检查残差模型有效性
df = pd.read_csv('/data/output/two_step_residual/predictions_test.csv')
residual_rmse = ((df['pred_residual'] - df['true_residual']) ** 2).mean() ** 0.5
print(f"残差预测RMSE: {residual_rmse:.4f}")
assert residual_rmse < df['true_residual'].std() * 0.8, "残差模型不如猜0"

# 5. 检查特征重要性
fi = pd.read_csv('/data/output/two_step_residual/feature_importance_residual.csv')
print("Top 5 features:", fi['feature'].head(5).tolist())
assert any('residual' in f for f in fi['feature'].head(10)), "残差特征未进入Top10"
```

---

## 备注

1. **代码复用**：本方案的特征工程与原方案高度一致，可直接复用 `build_features()` 逻辑，只需增加 `residual_t`, `residual_t-1` 等历史残差特征。
2. **B-Spline实现**：可直接复用之前 `fit_bspline_surface()` 函数，但修改插值目标为"每个合约的实际K"而非"固定节点"。
3. **baseline代理问题**：预测阶段无法获得T+1日真实B-Spline，用T日曲面在T+1日合约K上插值作为代理。这是工程妥协，若T+1日有新合约上市，用全局训练集均值填充。
4. **内存优化**：若34万条数据压力大，B-Spline拟合可按天循环处理，无需一次性加载全部到内存。
