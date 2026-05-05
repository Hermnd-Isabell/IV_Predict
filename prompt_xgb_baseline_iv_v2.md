
# 50ETF期权合约级IV预测 - XGBoost基线验证（修订版）

## 任务目标
构建一个XGBoost回归模型，预测50ETF期权每个合约在下一个交易日的**隐含波动率变化量（ΔIV）**。
这是一个**合约级**预测任务：同一天40个合约 = 40条独立样本，各自预测自己的ΔIV，不做任何数据压缩/聚合。

**核心原则**：
- 严格避免信息泄露：所有特征必须来自 **t 时刻或更早**，绝不使用 t+1 的信息
- 避免内生性问题：**剔除所有由IV派生的Greeks特征**（delta/gamma/theta/vega/rho），仅使用外生变量
- 主目标为 **ΔIV = IV_{t+1} - IV_t**，同时保留绝对IV预测作为对照实验

## 数据路径
- 原始数据目录: `/data/raw/`
- 文件格式: CSV/TXT，无表头，字段按顺序如下（制表符分隔）:
  `security_id, symbol, trade_date, call_put, open, high, low, close, volume, amount, open_interest, pre_settle_price, settle_price, list_date, exercise_price, remaining_time, last_edate, delta, gamma, rho, theta, vega, implc_volatlty, fund_open, fund_high, fund_low, fund_close, fund_volume, fund_amount, ten_year`
- **注意**：虽然原始数据包含Greeks字段，但基线模型**禁止使用**这些字段作为输入特征

## 数据结构说明
- 面板数据（Panel Data）：每天约40个合约（4到期月 × 5行权价 × 2类型）
- 交易日不连续（存在周末/节假日），按**交易日索引**处理，绝不填充非交易日
- trade_date格式: YYYYMMDD
- call_put: 'C'=Call, 'P'=Put

## 特征工程要求

### 1. 合约固有特征（直接取，不压缩）
- `moneyness` = fund_close / exercise_price
- `moneyness_squared` = moneyness²（捕捉波动率微笑曲率）
- `remaining_time` = remaining_time（距到期天数）
- `call_put` = 1 if 'C' else 0
- `exercise_price` = exercise_price
- **交互项**: `moneyness × remaining_time`（捕捉skew随期限变化）

### 2. 标的数据特征（外生变量）
- `fund_return` = (当日fund_close - 昨日fund_close) / 昨日fund_close
- `fund_volume` = fund_volume
- `fund_high_low_ratio` = (fund_high - fund_low) / fund_close
- `fund_amount` = fund_amount

### 3. 历史IV序列特征（按交易日索引滑动窗口，window=5）
对每个合约单独构造：
- `iv_t` = 当日implc_volatlty
- `iv_t_1` = 前1个交易日implc_volatlty
- `iv_t_2` = 前2个交易日implc_volatlty
- `iv_t_3` = 前3个交易日implc_volatlty
- `iv_t_4` = 前4个交易日implc_volatlty
- `iv_ma5` = 过去5个交易日implc_volatlty均值
- `iv_std5` = 过去5个交易日implc_volatlty标准差
- `iv_trend5` = iv_t - iv_t_4（趋势）
- `days_gap` = (当日trade_date - 前1个交易日trade_date).days（距上个交易日的实际天数，假期长度）

### 4. 曲面上下文特征（使用 t-1 日数据，避免信息泄露）
对每个样本，计算**上一个交易日（t-1）**所有合约的统计量：
- `atm_iv_call_lag1` = 在t-1日的fund_close处对所有Call的implc_volatlty做线性插值
- `iv_mean_all_lag1` = t-1日所有合约implc_volatlty均值
- `iv_std_all_lag1` = t-1日所有合约implc_volatlty标准差
- `iv_max_all_lag1` = t-1日所有合约implc_volatlty最大值
- `iv_min_all_lag1` = t-1日所有合约implc_volatlty最小值
- `iv_vs_atm_lag1` = t-1日该合约implc_volatlty - atm_iv_call_lag1

**注意**：若使用当日（t）截面统计，必须在计算时**排除目标合约自身**，避免自我引用。

### 5. 宏观特征
- `ten_year` = ten_year / 100（年化无风险利率）

### 6. 明确剔除的特征（内生性问题）
以下字段**禁止**作为模型输入：
- `delta`, `gamma`, `theta`, `vega`, `rho`（均由IV通过BS公式推导，存在循环论证）

## 输出目标（y）

### 主目标（Primary）
`delta_iv` = implc_volatlty_{t+1} - implc_volatlty_t

### 对照目标（Secondary，用于对比实验）
`next_implc_volatlty` = implc_volatlty_{t+1}

**要求**：
1. 先以 `delta_iv` 为主目标训练模型
2. 再用相同特征以 `next_implc_volatlty` 为目标训练对照模型
3. 对比两者在测试集上的表现

## 数据划分（时间序列划分，绝不随机打乱）
- 按trade_date排序，按时间顺序切分
- 训练集: 2015-02 ~ 2024-06（约前80%时间）
- 验证集: 2024-07 ~ 2024-12（中间10%）
- 测试集: 2025-01 ~ 2026-01（最后20%，或按实际数据最新日期调整）
- 注意：划分时保留每个合约的时序连续性
- **不把 security_id 作为模型输入特征**，避免模型记忆特定合约

## 模型要求
- 基线模型: `xgboost.XGBRegressor`
- 参数建议（可调整）:
  ```python
  model = xgb.XGBRegressor(
      n_estimators=500,
      max_depth=6,
      learning_rate=0.05,
      subsample=0.8,
      colsample_bytree=0.8,
      objective='reg:squarederror',
      random_state=42,
      n_jobs=-1
  )
  ```
- 必须做超参数调优（GridSearchCV或Optuna），调优范围:
  - max_depth: [3, 5, 6, 8]
  - learning_rate: [0.01, 0.05, 0.1]
  - n_estimators: [300, 500, 800]
  - subsample: [0.7, 0.8, 0.9]

## 评估指标

### 主实验（预测 ΔIV）
1. **RMSE_ΔIV**: sqrt(mean((pred_ΔIV - true_ΔIV)²))
2. **MAE_ΔIV**: mean(abs(pred_ΔIV - true_ΔIV))
3. **R²_ΔIV**: 决定系数
4. **方向准确率_ΔIV**: sign(pred_ΔIV) == sign(true_ΔIV) 的比例（预测IV涨跌方向是否正确）

### 对照实验（预测绝对IV）
5. **RMSE_IV**: sqrt(mean((pred_IV - true_IV)²))，其中 pred_IV = IV_t + pred_ΔIV
6. **MAE_IV**: mean(abs(pred_IV - true_IV))
7. **MAPE_IV**: mean(abs((pred_IV - true_IV) / true_IV))
8. **R²_IV**: 决定系数
9. **方向准确率_IV**: sign(pred_IV - IV_t) == sign(true_IV - IV_t) 的比例

### 分组评估
10. **按合约分组**: 每个security_id单独计算RMSE_IV，观察不同到期月/行权价的预测难度差异
11. **按moneyness分箱**: ITM(m<0.97) / ATM(0.97≤m≤1.03) / OTM(m>1.03) 分别评估
12. **按remaining_time分箱**: 近月(≤30天) / 中月(30-90天) / 远月(>90天) 分别评估

## 额外分析要求
1. **特征重要性**: 输出XGBoost的feature_importances_，排序展示Top 15特征
2. **残差分析**: 
   - 残差 = pred_IV - true_IV（或 pred_ΔIV - true_ΔIV）
   - 按moneyness分箱观察残差分布
   - 按remaining_time分箱观察残差分布
3. **预测vs真实散点图**: 测试集上 pred_IV vs true_IV 的散点图 + 对角线
4. **时间序列可视化**: 选取1-2个代表性合约，绘制真实IV vs 预测IV的时间序列对比图
5. **对照实验对比表**: 主实验(ΔIV) vs 对照实验(绝对IV) 的各指标对比

## 输出文件要求
所有输出保存到 `/data/output/baseline_xgb/`:
1. `model_delta_iv.pkl` - 主实验（ΔIV）训练好的XGBoost模型
2. `model_abs_iv.pkl` - 对照实验（绝对IV）训练好的XGBoost模型
3. `feature_importance_delta_iv.csv` - 主实验特征重要性排序
4. `feature_importance_abs_iv.csv` - 对照实验特征重要性排序
5. `metrics.json` - 训练/验证/测试集的完整评估指标（含主实验和对照实验）
6. `predictions_test_delta_iv.csv` - 测试集预测结果（含security_id, trade_date, IV_t, true_ΔIV, pred_ΔIV, true_IV, pred_IV, 残差）
7. `residual_analysis.png` - 残差分析图
8. `pred_vs_true.png` - 预测vs真实散点图
9. `timeseries_sample.png` - 时间序列对比图
10. `comparison_table.png` - 主实验vs对照实验对比表

## 代码规范
- 使用Python，pandas/numpy/xgboost/sklearn/matplotlib
- 所有函数必须有docstring
- 数据读取和特征工程必须模块化（函数化），方便后续替换模型
- 必须处理异常值（如implc_volatlty=0的到期日数据）
- 日志输出训练进度和关键指标
- 特征工程代码需注释说明每个特征的时序来源（t/t-1/t-2等）

## 成功标准
- **主实验（ΔIV）**: 测试集 RMSE_ΔIV < 0.03（IV日均波动约0.02-0.05）
- **方向准确率_ΔIV > 55%**（优于随机猜测）
- **对照实验（绝对IV）**: 测试集 RMSE_IV < 0.05
- **特征重要性Top 5中至少包含1个历史IV特征**（验证IV的时序可预测性）
- **对照实验RMSE_IV不显著优于主实验**（若绝对IV模型远好于ΔIV模型，说明存在信息泄露）



## 附录：输出规范与检查清单（供Reviewer验证用）

Claude Code 在执行本任务时，必须遵循以下输出规范，以便后续人工/自动化检查。

### A. 代码结构规范

#### A1. 模块化要求
代码必须按以下函数结构组织，每个函数有明确输入输出：

```python
def load_raw_data(data_dir: str) -> pd.DataFrame:
    """读取原始数据，返回带列名的DataFrame"""
    pass

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """特征工程主函数，输入原始数据，输出带特征的样本集"""
    pass

def build_target(df: pd.DataFrame) -> pd.Series:
    """构造目标变量，返回y（delta_iv）"""
    pass

def split_temporal(df: pd.DataFrame, train_end: int, val_end: int) -> tuple:
    """时间序列划分，返回(train_df, val_df, test_df)"""
    pass

def train_model(X_train, y_train, X_val, y_val) -> xgb.XGBRegressor:
    """训练XGBoost，支持超参数调优"""
    pass

def evaluate_model(model, X_test, y_test, iv_t_test) -> dict:
    """评估并返回指标字典"""
    pass

def save_outputs(model, metrics, predictions, feature_importance, output_dir: str):
    """保存所有输出文件"""
    pass
```

#### A2. 日志输出要求
代码运行时必须打印以下关键检查点（print或logging）：

```
[Checkpoint 1] 数据加载完成
  - 原始记录数: {N}
  - 唯一合约数(security_id): {M}
  - 日期范围: {start_date} ~ {end_date}
  - 列名列表: [...]

[Checkpoint 2] 特征工程完成
  - 构造样本数: {N_samples}
  - 特征维度: {N_features}
  - 特征列名: [col1, col2, ...]
  - 目标变量统计: mean={x}, std={y}, min={z}, max={w}
  - 样本示例（前3行）:
    ...

[Checkpoint 3] 时间划分完成
  - 训练集: {train_start} ~ {train_end}, 样本数={n_train}
  - 验证集: {val_start} ~ {val_end}, 样本数={n_val}
  - 测试集: {test_start} ~ {test_end}, 样本数={n_test}
  - 检查：是否存在security_id跨集合泄露？{Yes/No}

[Checkpoint 4] 模型训练完成
  - 最优参数: {params}
  - 训练RMSE: {x}
  - 验证RMSE: {y}

[Checkpoint 5] 评估完成
  - 主实验(ΔIV)指标: {metrics_delta}
  - 对照实验(绝对IV)指标: {metrics_abs}
  - Top 5特征: {feature_list}
```

### B. 中间结果检查点

#### B1. 特征工程后必须验证
- [ ] `moneyness` 范围合理：0.8 ~ 1.3（50ETF期权 typical range）
- [ ] `remaining_time` 无负值，最大值 < 365
- [ ] `days_gap` 分布：大部分为1-3，偶尔有7-10（长假），无异常大值（如>30）
- [ ] `iv_mean_all_lag1` 非空比例 > 95%（t-1日有数据的合约足够多）
- [ ] 样本中 **不包含** delta/gamma/theta/vega/rho 列（Greeks已剔除）
- [ ] 样本中 **不包含** implc_volatlty_{t+1} 作为输入特征（目标泄露检查）

#### B2. 目标变量必须验证
- [ ] `delta_iv` 统计：均值 ≈ 0，标准差 ≈ 0.02-0.05
- [ ] `delta_iv` 分布近似对称（IV涨跌概率接近50%）
- [ ] 到期日（remaining_time=0）的样本 **必须剔除**（无T+1数据，无法构造y）

#### B3. 时间划分必须验证
- [ ] 训练集最晚日期 < 验证集最早日期 < 测试集最早日期
- [ ] 同一security_id在训练/验证/测试中的时间无重叠
- [ ] 测试集中不存在"训练阶段完全没见过"的新security_id（如果存在，需单独报告）

### C. 输出文件格式规范

#### C1. predictions_test_delta_iv.csv
必须包含以下列（按顺序）：

| 列名 | 类型 | 说明 |
|------|------|------|
| security_id | str | 合约代码 |
| trade_date | int | 日期YYYYMMDD |
| call_put | str | C/P |
| exercise_price | float | 行权价 |
| last_edate | int | 到期日 |
| remaining_time | int | 剩余天数 |
| moneyness | float | S/K |
| iv_t | float | T日IV |
| true_delta_iv | float | 真实ΔIV = IV_{t+1} - IV_t |
| pred_delta_iv | float | 预测ΔIV |
| true_iv | float | 真实IV_{t+1} |
| pred_iv | float | 还原IV = IV_t + pred_delta_iv |
| residual_iv | float | pred_iv - true_iv |
| residual_delta | float | pred_delta_iv - true_delta_iv |

#### C2. metrics.json
必须包含以下结构：

```json
{
  "primary_experiment": {
    "target": "delta_iv",
    "train": {"rmse": x, "mae": x, "r2": x, "direction_acc": x},
    "val": {"rmse": x, "mae": x, "r2": x, "direction_acc": x},
    "test": {"rmse": x, "mae": x, "r2": x, "direction_acc": x}
  },
  "control_experiment": {
    "target": "abs_iv",
    "train": {"rmse": x, "mae": x, "mape": x, "r2": x, "direction_acc": x},
    "val": {"rmse": x, "mae": x, "mape": x, "r2": x, "direction_acc": x},
    "test": {"rmse": x, "mae": x, "mape": x, "r2": x, "direction_acc": x}
  },
  "comparison": {
    "test_rmse_iv_primary": x,  // 主实验还原的IV RMSE
    "test_rmse_iv_control": x,  // 对照实验直接预测的IV RMSE
    "primary_vs_control": "primary better/worse/similar"
  },
  "feature_importance_top10": [
    {"feature": "iv_t", "importance": 0.35},
    ...
  ],
  "data_info": {
    "n_samples_total": x,
    "n_features": x,
    "n_contracts": x,
    "date_range": "20150209-20260130"
  }
}
```

#### C3. 可视化规范
所有图表必须：
- 有清晰的标题（含实验名称：Primary/Control）
- 有x轴/y轴标签
- 有图例（如适用）
- 保存为PNG，dpi≥150
- 文件名反映内容：
  - `residual_by_moneyness_primary.png`
  - `residual_by_moneyness_control.png`
  - `pred_vs_true_primary.png`
  - `pred_vs_true_control.png`
  - `timeseries_sample_{security_id}.png`

### D. 异常值与边界情况处理规范

#### D1. 必须处理的情况
- [ ] `implc_volatlty = 0`（到期日或深度虚值）：**剔除该样本**（无法构造有意义的ΔIV）
- [ ] `implc_volatlty > 1.0`（极端异常）：**设为NaN并剔除**，或记录为异常
- [ ] `volume = 0`（无成交）：保留样本，但标记为"零成交"，观察模型表现
- [ ] 某个交易日截面合约数 < 3（无法计算可靠的截面统计）：用全局均值填充或标记

#### D2. 缺失值处理
- [ ] 历史IV窗口不足5天（合约上市初期）：用**可用天数**计算，不填充
- [ ] t-1日截面统计缺失（如t为合约上市首日）：用**全局训练集统计量**填充
- [ ] 标的数据缺失（如fund_close为NaN）：**剔除该交易日所有合约样本**

### E. Reviewer快速验证清单

当Claude Code完成任务后，Reviewer执行以下检查：

```bash
# 1. 检查输出目录完整性
ls /data/output/baseline_xgb/
# 期望看到: model_delta_iv.pkl, model_abs_iv.pkl, metrics.json, 
#           predictions_test_delta_iv.csv, *.png

# 2. 检查metrics.json关键数值
python -c "
import json
with open('/data/output/baseline_xgb/metrics.json') as f:
    m = json.load(f)
assert m['primary_experiment']['test']['rmse'] < 0.03, 'Primary RMSE too high'
assert m['primary_experiment']['test']['direction_acc'] > 0.55, 'Direction acc too low'
assert m['comparison']['test_rmse_iv_primary'] < 0.05, 'Restored IV RMSE too high'
print('All checks passed!')
"

# 3. 检查predictions.csv列完整性
python -c "
import pandas as pd
df = pd.read_csv('/data/output/baseline_xgb/predictions_test_delta_iv.csv')
required_cols = ['security_id', 'trade_date', 'iv_t', 'true_delta_iv', 
               'pred_delta_iv', 'true_iv', 'pred_iv']
assert all(c in df.columns for c in required_cols), 'Missing columns'
assert 'delta' not in df.columns, 'Greeks leaked into output!'
print('Column check passed!')
"

# 4. 检查特征重要性Top 5
python -c "
import pandas as pd
fi = pd.read_csv('/data/output/baseline_xgb/feature_importance_delta_iv.csv')
top5 = fi['feature'].head(5).tolist()
print('Top 5 features:', top5)
assert any('iv' in f for f in top5), 'No IV feature in top 5!'
"
```

### F. 代码可读性规范

- 所有特征构造逻辑必须在代码中**注释时序来源**（t/t-1/t-2）
- 关键步骤必须打印**数据形状**（如 `print(f"After feature engineering: {df.shape}")`）
- 超参数调优过程必须保存**搜索日志**（如Optuna的study对象或GridSearchCV的cv_results_）
- 随机种子固定：`random_state=42` 在所有随机操作中一致
