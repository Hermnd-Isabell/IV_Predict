# 50ETF期权IV曲面预测 - B-Spline + XGBoost MultiOutput（方案B）

## 任务目标
在已完成的XGBoost基线（逐合约预测）基础上，升级为 **B-Spline IV曲面拟合 + XGBoost MultiOutputRegressor 系数预测**。

核心变化：
- 原方案：逐合约独立预测（34万样本，y为标量IV）
- 新方案：每天拟合IV曲面得到20维系数，预测系数向量，再还原曲面插值到任意合约

## 数据路径
- 原始数据：`/data/raw/50etf_options.csv`
- 原方案输出（对照用）：`/data/output/baseline_xgb/`
- 新方案输出：`/data/output/bspline_surface/`

## 数据结构（已确认）
- 面板数据，约34万条，2667个交易日
- 字段：security_id, symbol, trade_date, call_put, exercise_price, remaining_time, last_edate, implc_volatlty, fund_close, 以及Greeks/价格等
- 交易日不连续（周末/节假日无数据），按交易日索引处理

## 核心约束（不可更改）
1. **Call/Put 分开拟合**：两张独立曲面
2. **剔除Greeks**：delta/gamma/theta/vega/rho 禁止作为输入
3. **截面统计用t-1日**：防止信息泄露
4. **不把security_id作为模型输入**
5. **时间序列划分**：按时间切分，禁止随机打乱
6. **保留原方案对照**：必须对比新方案 vs 原方案的测试集RMSE

---

## 5步流水线架构

### Step 1: B-Spline 曲面拟合（每天收盘后，Call/Put各一次）

**输入**：某一天同一类型（Call或Put）的所有合约
  - 行权价 K：exercise_price
  - 剩余期限 T：remaining_time
  - IV：implc_volatlty

**拟合过程**：
1. 对每个到期月（如1503/1504/1506/1509），收集该月所有行权价的IV
2. 在该到期月内，用 **CubicSpline** 沿行权价方向拟合 IV = f(K)
3. 在 **5个标准行权价节点** 上插值：K in {2.20, 2.30, 2.40, 2.50, 2.60}
4. 得到该到期月的5个IV值
5. 4个到期月 x 5个节点 = **20维系数向量 a_t**

**输出**：
  - Call曲面系数：`a_t_call` = [a1, a2, ..., a20]
  - Put曲面系数：`a_t_put` = [a1, a2, ..., a20]

**注意**：
- 若某天某到期月合约数 < 3，该月5个节点用全局训练集均值填充
- 若某天总合约数 < 10，该天整体剔除（不进入训练集）
- 到期日（remaining_time=0）的合约 **不参与拟合**（IV=0无意义）

### Step 2: 构造训练样本（系数时间序列）

**每条样本 = 一个交易日 x 一个类型（Call或Put）**

**输入特征 X（约50维）**：

| 特征类别 | 具体特征 | 时序来源 | 说明 |
|---------|---------|---------|------|
| 系数历史 | a_t, a_t-1, a_t-2, a_t-3, a_t-4 | t, t-1, t-2, t-3, t-4 | 20维 x 5天 = 100维（可降维） |
| 系数统计 | ma3, std3, trend3 | t窗口 | 20维 x 3 = 60维 |
| 标的数据 | fund_return, fund_volume, fund_high_low_ratio | t | 3维 |
| 宏观 | ten_year | t | 1维 |
| 曲面上下文 | atm_iv_lag1, skew_25_lag1, term_slope_lag1 | t-1 | 3维 |
| 时间 | days_gap, remaining_time_proxy | t | 2维 |

**简化建议**：为避免维度爆炸，对20维系数历史做 **逐维度统计**（而非展开200维）：
- 系数向量均值、标准差、最大值、最小值
- 近月(到期月1)系数均值、远月(到期月4)系数均值
- 微笑曲率代理：(a_虚值 + a_实值)/2 - a_平值

**输出 y**：
  - `a_{t+1}` = 下一天的20维系数向量

### Step 3: XGBoost 训练（MultiOutputRegressor，方案B）

**模型**：
```python
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

model = MultiOutputRegressor(
    xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
)
```

**预留方案C接口**：
```python
# 代码中预留切换参数
method = 'B'  # 当前用B；若改为'C'，启用PCA降维到5维再预测
if method == 'C':
    from sklearn.decomposition import PCA
    pca = PCA(n_components=5)
    # 预测5个主成分，再还原20维
```

**超参数调优**：
- max_depth: [3, 5, 6]
- learning_rate: [0.03, 0.05, 0.1]
- n_estimators: [300, 500, 800]

### Step 4: 还原 IV 曲面

**输入**：预测的系数 `a_{t+1}^pred`（20维）

**还原**：
- 4个到期月 x 5个标准节点 = 一张完整IV网格
- 对任意查询 (K, T)，在网格上做 **双线性插值** 得到 IV_pred

### Step 5: 应用到具体合约

**输入**：还原的曲面 + 测试集合约参数（K, T, type）

**输出**：
- 每个合约的 `pred_iv = interpolate(surface, K, T)`
- 与真实IV对比计算RMSE

---

## 数据划分（时间序列）

- 训练集：2015-02 ~ 2024-06（约前80%交易日）
- 验证集：2024-07 ~ 2024-12（中间10%）
- 测试集：2025-01 ~ 2026-01（最后20%）

**关键**：划分在 **交易日级别** 进行（不是合约级别）。所有Call/Put样本按交易日一起进入同一集合。

---

## 对照实验（必须执行）

### 对照1：与原方案对比
1. 检查 `/data/output/baseline_xgb/` 是否存在 `model_abs_iv.pkl`
2. 若存在：加载原模型，在 **相同测试集** 上预测IV
3. 若不存在：快速训练一个简化版逐合约XGB（仅用于对比）
4. 输出 `metrics_comparison.json`

### 对照2：新方案内部验证
- 用预测的系数还原曲面后，在 **标准节点** 上计算RMSE
- 在 **非标准节点（实际合约位置）** 上计算插值RMSE
- 对比两者差异（验证插值误差）

---

## 评估指标

### 主指标
1. **Test RMSE_IV**：还原IV vs 真实IV的均方根误差
2. **Test MAE_IV**：平均绝对误差
3. **Test R2_IV**：决定系数
4. **分箱RMSE**：
   - 按 remaining_time 分：近月(<=30) / 中月(30-90) / 远月(>90)
   - 按 moneyness 分：ITM / ATM / OTM
   - 按类型分：Call / Put

### 无套利检查
5. **无套利违规率**：
   - 同一到期月内，IV随K是否单调？（Call: 应基本递减；Put: 应基本递增）
   - 不同到期月间，近月IV是否不应系统性地低于远月IV？（允许正常期限结构）
   - 统计预测曲面中违反上述逻辑的比例

### 对照指标
6. **Primary vs Control RMSE差异**：(新方案RMSE - 原方案RMSE) / 原方案RMSE

---

## 输出文件要求

保存到 `/data/output/bspline_surface/`：

1. `model_bspline_multioutput.pkl` - 训练好的MultiOutput XGBoost
2. `bspline_coefficients.csv` - 每天Call/Put的20维系数时间序列
3. `feature_importance.csv` - 20个输出维度的平均特征重要性
4. `metrics.json` - 完整评估指标
5. `metrics_comparison.json` - 新方案 vs 原方案对比
6. `predictions_test.csv` - 测试集预测结果（含security_id, trade_date, true_iv, pred_iv, residual）
7. `surface_sample.png` - 某天Call/Put的IV曲面3D/热力图可视化
8. `pred_vs_true.png` - 预测vs真实散点图
9. `residual_by_maturity.png` - 按期限分箱的残差分布
10. `comparison_table.png` - 新旧方案指标对比表

---

## 代码结构规范（必须模块化）

```python
def load_raw_data(path) -> pd.DataFrame:
    """读取原始期权数据"""

def fit_bspline_surface(df_day, call_put, strike_nodes) -> np.ndarray:
    """
    对单日单类型的合约拟合B-Spline曲面
    返回20维系数向量
    """

def build_coefficient_series(df, strike_nodes) -> pd.DataFrame:
    """
    构造系数时间序列
    每天2条样本（Call/Put各一），每条20维y
    """

def build_features(coeff_df, fund_df) -> pd.DataFrame:
    """
    特征工程：系数历史 + 标的数据 + 曲面上下文
    返回X（输入）和y（20维系数）
    """

def split_temporal(df, train_end, val_end) -> tuple:
    """按交易日时间切分"""

def train_multioutput_xgb(X_train, y_train, X_val, y_val, method='B'):
    """
    训练MultiOutputRegressor
    method='B'：直接预测20维
    method='C'：PCA降维（预留接口）
    """

def restore_surface(coeff_vector, strike_nodes, maturities) -> callable:
    """
    从20维系数还原IV曲面函数
    返回插值函数 f(K, T) -> IV
    """

def evaluate_model(model, X_test, y_test, df_test_raw, baseline_model=None) -> dict:
    """
    评估：还原曲面 -> 插值到合约 -> 计算RMSE
    若提供baseline_model，同时评估对照
    """

def save_outputs(model, metrics, predictions, output_dir):
    """保存所有输出"""
```

---

## 检查点（必须打印）

```
[Checkpoint 1] 数据加载与B-Spline拟合
  - 原始记录数: {N}
  - 交易日数: {T}
  - Call/Put日均合约数: {call_avg} / {put_avg}
  - B-Spline拟合成功天数: {success_days}
  - 系数矩阵形状: {T_success} x 20 x 2 (Call+Put)
  - 某天(如20150209)拟合效果图: surface_sample.png

[Checkpoint 2] 特征工程完成
  - 构造样本数: {N_samples}（每天2条 x 天数）
  - 输入特征维度: {N_features}
  - 输出维度: 20
  - 特征列名示例: [coeff_mean, coeff_std, near_month_iv, far_month_iv, smile_curvature, fund_return, ...]
  - 样本示例（前2行）:
    ...

[Checkpoint 3] 时间划分完成
  - 训练集: {train_start} ~ {train_end}, 样本数={n_train}
  - 验证集: {val_start} ~ {val_end}, 样本数={n_val}
  - 测试集: {test_start} ~ {test_end}, 样本数={n_test}
  - 检查：是否存在交易日跨集合泄露？{Yes/No}

[Checkpoint 4] 模型训练完成
  - 方案: MultiOutputRegressor (method={B/C})
  - 最优参数: {params}
  - 训练20维平均RMSE: {x}
  - 验证20维平均RMSE: {y}
  - Top 5特征（按平均重要性）: {feature_list}

[Checkpoint 5] 评估与对照完成
  - 新方案 Test RMSE_IV: {rmse_new}
  - 新方案 Test R2_IV: {r2_new}
  - 原方案 Test RMSE_IV: {rmse_old}（若基线存在）
  - 相对改善: {(rmse_old - rmse_new)/rmse_old * 100}%
  - 无套利违规率: {violation_rate}%
  - 近月/中月/远月 RMSE: {rmse_near} / {rmse_mid} / {rmse_far}
  - Call/Put RMSE: {rmse_call} / {rmse_put}
```

---

## 中间结果验证清单

- [ ] `coefficients` 矩阵无NaN比例 > 95%
- [ ] 每天Call和Put的系数向量均值差异合理（Put IV系统高于Call）
- [ ] `days_gap` 分布正常（1-3为主，偶尔7-10）
- [ ] 特征中 **不包含** delta/gamma/theta/vega/rho
- [ ] 特征中 **不包含** t+1日信息
- [ ] 测试集交易日严格晚于训练/验证集

---

## 成功标准

- [ ] **新方案 Test RMSE_IV <= 原方案 Test RMSE_IV**（空间结构应过滤噪声）
- [ ] **远月合约（>90天）RMSE 显著优于原方案**（原方案对稀疏数据预测差）
- [ ] **无套利违规率 < 10%**（CubicSpline天然平滑）
- [ ] **系数时间序列可视化合理**：曲面随时间平滑演变，无突变
- [ ] **特征重要性Top 5中至少2个与系数/IV历史相关**

---

## 异常处理规范

- `implc_volatlty = 0`：不参与B-Spline拟合
- 某天某到期月合约数 < 3：该月5节点用训练集全局均值填充
- 某天总合约数 < 10：剔除该天
- 历史窗口不足5天：用可用天数计算，不填充
- 标的数据缺失：剔除该交易日所有样本

---

## Reviewer快速验证脚本

```python
import json, pandas as pd

# 1. 检查输出完整性
import os
assert os.path.exists('/data/output/bspline_surface/model_bspline_multioutput.pkl')
assert os.path.exists('/data/output/bspline_surface/metrics.json')

# 2. 检查关键指标
with open('/data/output/bspline_surface/metrics.json') as f:
    m = json.load(f)
assert m['test']['rmse_iv'] < 0.05, "RMSE too high"
assert m['test']['r2_iv'] > 0.3, "R2 too low"
assert m['arbitrage_violation_rate'] < 0.10, "Too many arbitrage violations"

# 3. 检查对照结果
with open('/data/output/bspline_surface/metrics_comparison.json') as f:
    c = json.load(f)
print(f"Improvement over baseline: {c['relative_improvement_pct']:.2f}%")

# 4. 检查特征重要性
fi = pd.read_csv('/data/output/bspline_surface/feature_importance.csv')
print("Top 5 features:", fi['feature'].head(5).tolist())
```

---

## 备注

1. **代码修改预留**：`train_multioutput_xgb` 函数中 `method` 参数当前设为 `'B'`。若后续需切换为方案C（PCA降维），只需传入 `method='C'`，函数内部自动启用PCA路径，外部pipeline无需改动。
2. **内存管理**：若34万条数据加载内存压力大，可分块读取或只加载必要列（trade_date, call_put, exercise_price, remaining_time, implc_volatlty, fund_close, fund_volume, ten_year）。
3. **可视化**：`surface_sample.png` 建议画热力图（X轴=行权价，Y轴=到期时间，颜色=IV值），直观展示曲面形态。
