# 50ETF期权IV预测 - 路线B: Moneyness-W空间B-Spline + 残差预测(最终版)

## 任务目标
在路线A(两步法残差预测, RMSE=0.0337)基础上, 通过以下核心改进修复baseline代理错配, 目标逼近理论最优RMSE=0.0255:

1. **空间坐标升级**: 绝对行权价K -> Log-moneyness `M = ln(K/F_implied)`
2. **远期价格升级**: 理论远期 `S*e^(r*tau)` -> 隐含远期 `F_implied`(Put-Call Parity反推)
3. **时间坐标升级**: 直接IV插值 -> 总方差 `W = sigma^2 * tau` 空间插值
4. **边界外推升级**: CubicSpline外推 -> 平坦外推(零斜率边界保护)
5. **预测目标保持**: `delta_residual = residual_{t+1} - residual_t`

**合成公式**:
```
pred_IV = interp_baseline_t(M_{t+1}, tau_{t+1}) + residual_t + pred_delta_residual
```

## 数据路径
- 原始数据: `/data/raw/50etf_options.csv`
- 原方案输出(对照用): `/data/output/baseline_xgb/`(需存在 `model_abs_iv.pkl`)
- 旧两步法输出(对照用): `/data/output/two_step_residual/`(可选, 用于对比)
- 新方案输出: `/data/output/two_step_v2/`

## 核心约束(不可更改)
1. **剔除Greeks**: delta/gamma/theta/vega/rho 禁止作为模型输入
2. **截面统计用t-1日**: 防止信息泄露
3. **不把security_id作为模型输入**
4. **时间序列划分**: 按时间切分, 禁止随机打乱
5. **保留原方案对照**: 必须对比新方案 vs 原方案 vs 旧两步法的测试集RMSE

---

## 数学符号与定义

| 符号 | 定义 | 计算方式 |
|------|------|----------|
| `S` | 标的价格 | `fund_close` |
| `r` | 无风险利率 | `ten_year / 100`(年化) |
| `tau` | 年化剩余期限 | `remaining_time / 365` |
| `F_implied` | 隐含远期价格 | Put-Call Parity反推(见下文) |
| `M` | Log-moneyness | `ln(K / F_implied)` |
| `W` | 总隐含方差 | `IV^2 * tau` |
| `baseline_IV` | B-Spline基准IV | 从(M,W)曲面还原: `sqrt(W_interp / tau)` |
| `residual` | 个体定价偏差 | `market_IV - baseline_IV` |
| `delta_residual` | 残差日变化 | `residual_{t+1} - residual_t` |

---

## 5步流水线架构

### Step 1: 隐含远期价格 F_implied 计算(每天每到期月)

**目标**: 为每个交易日、每个到期月计算一个统一的 `F_implied`, 用于该月所有合约的Moneyness计算.

**算法**:
```python
def compute_implied_forward(df_day, maturity_group, r, tau):
    # 1. 分离Call和Put
    calls = df_day[(maturity_group) & (call_put == 'C')]
    puts  = df_day[(maturity_group) & (call_put == 'P')]
    
    # 2. 找到共有的行权价(ATM附近优先)
    common_strikes = set(calls.exercise_price) & set(puts.exercise_price)
    
    # 3. 对每对Call/Put, 用Put-Call Parity反推F
    F_list = []
    for K in common_strikes:
        C = calls[calls.exercise_price == K].close.values[0]
        P = puts[puts.exercise_price == K].close.values[0]
        F = K + np.exp(r * tau) * (C - P)
        F_list.append(F)
    
    # 4. 取中位数(防异常值)
    if len(F_list) >= 2:
        F_implied = np.median(F_list)
    elif len(F_list) == 1:
        F_implied = F_list[0]
    else:
        # 退化为理论远期
        S = df_day.fund_close.iloc[0]
        F_implied = S * np.exp(r * tau)
    
    return F_implied
```

**边界情况处理**:
- 某天某到期月只有Call或只有Put: 退化为理论远期 `F = S*e^(r*tau)`
- 新到期月首日上市: 用理论远期初始化, 次日有交易后改用隐含远期
- `C-P` 异常大的配对(流动性极差): 在取中位数前剔除 `|C-P| > 3 * median(|C-P|)` 的异常值

**输出**:
- 全局表 `forward_table`: columns = `[trade_date, last_edate, F_implied, F_theory]`
- 每个合约关联其到期月对应的 `F_implied`

---

### Step 2: Moneyness-W空间 B-Spline 曲面拟合(每天Call/Put各一次)

**输入**: 某一天同一类型(Call或Put)的所有合约
  - `M = ln(K / F_implied)`: Log-moneyness
  - `tau = remaining_time / 365`: 年化剩余期限
  - `W = IV^2 * tau`: 总隐含方差

**拟合过程**:
1. **按到期月分组**: 每天可能有多个到期月(如近月、次近月、远月1、远月2)
2. **每组内M方向CubicSpline**: 在该到期月内, 用 `CubicSpline(M, W)` 拟合 `W = f(M)`
   - 要求: M必须严格递增(若不满足, 先按M排序)
   - 节点数: 使用合约实际K对应的M值作为节点
3. **保存每日曲面**: 对每个到期月保存 `(M_nodes, W_nodes, Spline_object)`

**关键约束**:
- `W` 必须为正(IV>0, tau>0). 若Spline输出W<0, 强制设为 `W = 1e-6`
- 同一到期月内, Call的 `W` 随 `M` 应基本非递增(深度实值W高, 虚值W低); Put相反. 但**不强制单调调整**(W空间天然比IV空间平滑)

**输出**:
- 每天Call/Put各一个曲面对象(含多个到期月的Spline)
- 每个合约当天的 `baseline_W`(在自身M位置上插值得到)
- 还原: `baseline_IV = sqrt(baseline_W / tau)`

---

### Step 3: 残差计算与构造训练样本

**残差定义**:
```
residual = market_IV - baseline_IV
```

**训练样本构造**(与原方案一致, y改为delta_residual):

**输入特征 X**:

| 特征类别 | 具体特征 | 时序来源 | 说明 |
|---------|---------|---------|------|
| 合约固有 | moneyness, moneyness^2, remaining_time, call_put, moneyness*remaining_time | t | 直接取 |
| 标的数据 | fund_return, fund_volume, fund_high_low_ratio, fund_amount | t | 外生变量 |
| 历史IV | iv_t, iv_t-1...iv_t-4, iv_ma5, iv_std5 | t, t-1... | 按交易日索引滑动窗口 |
| **历史残差** | **residual_t, residual_t-1, residual_t-2, residual_ma3, residual_std3** | **t, t-1...** | **核心特征** |
| 曲面上下文 | atm_iv_call_lag1, iv_mean_all_lag1, iv_std_all_lag1 | t-1 | 防止信息泄露 |
| 宏观 | ten_year | t | 年化利率 |
| 时间 | days_gap | t | 距上个交易日天数 |

**输出 y**:
  - `delta_residual = residual_{t+1} - residual_t`

**关键处理**:
- 到期日(remaining_time=0)样本剔除
- `implc_volatlty=0` 的样本剔除
- 若T+1日某合约不存在(到期退市), 该样本剔除
- 历史残差窗口不足: 用可用天数计算

---

### Step 4: XGBoost 残差变化预测模型训练

**模型**: 单输出 XGBRegressor(预测标量 delta_residual)

```python
import xgboost as xgb

model_delta_residual = xgb.XGBRegressor(
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

**超参数调优**(与原方案一致):
- max_depth: [3, 5, 6, 8]
- learning_rate: [0.01, 0.05, 0.1]
- n_estimators: [300, 500, 800]
- subsample: [0.7, 0.8, 0.9]

**训练逻辑**:
- 与原方案完全一致, 只是目标变量从 `IV_{t+1}` 改为 `delta_residual`
- 特征工程代码大量复用原方案, 增加 `residual_t`, `residual_t-1` 等历史残差特征

---

### Step 5: 两步法预测(T日 -> T+1日)

**预测流程**:

```
T日收盘后:
  1. 用T日所有合约拟合 M-W B-Spline -> 保存T日曲面(各到期月的 Spline对象)
  2. 计算T日每个合约的 residual_t = IV_t - baseline_IV_t

T+1日开盘前(预测阶段):
  3. 加载T日保存的曲面 S_t
  4. 对T+1日每个合约(K, tau_{t+1}已知, 价格未知):
     a. 计算 M_{t+1} = ln(K / F_t_implied)  
        (F_{t+1}未知, 用T日隐含远期作为代理; 短期变化极小)
     b. M方向插值: 在S_t的对应到期月CubicSpline上查询 W_at_M
        - 若 M_{t+1} < M_min: W = S_t(M_min)  (平坦外推)
        - 若 M_{t+1} > M_max: W = S_t(M_max)  (平坦外推)
     c. tau方向插值(跨到期月):
        - 若 tau_{t+1} 恰好匹配某到期月: 直接用该月Spline
        - 若 tau_{t+1} 落在两个到期月之间: 在W空间线性插值
          W_interp = W_near + (W_far - W_near) * (tau_{t+1} - tau_near) / (tau_far - tau_near)
        - 若 tau_{t+1} > 所有到期月: 用最远到期月的W(平坦外推)
        - 若 tau_{t+1} < 最小到期月: 用最近到期月的W
     d. 还原 baseline_IV = sqrt(max(W_interp, 1e-6) / tau_{t+1})
  
  5. 用XGBoost预测 delta_residual_pred
  
  6. 合成预测:
     pred_IV = baseline_IV + residual_t + delta_residual_pred
```

**重要说明**:
- **F代理**: T+1日 `F_{t+1}` 未知, 但1天内变化极小(主要由标的价格变动主导). 若标的价格大幅跳空, 误差被残差模型部分吸收.
- **新合约处理**: T+1日全新行权价上市 -> M在范围外 -> 平坦外推至边界W
- **新到期月处理**: T+1日新远月上市 -> 无T日Spline -> 用同类型最远到期月外推, 或全局训练集均值
- **到期退市处理**: T+1日到期的合约 -> 无预测需求 -> 剔除

---

## 数据划分(与原方案完全一致)

- 训练集: 2015-02 ~ 2024-06(约前80%时间)
- 验证集: 2024-07 ~ 2024-12(中间10%)
- 测试集: 2025-01 ~ 2026-01(最后20%)

**注意**:
- `F_implied` 和 `baseline_IV` 在划分前全局计算(不泄露, 只用当日截面数据)
- 残差模型训练在划分后进行

---

## 对照实验(必须执行)

### 对照1: 与原方案对比
1. 加载 `/data/output/baseline_xgb/model_abs_iv.pkl`
2. 在**相同测试集**上预测IV
3. 输出RMSE对比

### 对照2: 与旧两步法对比
1. 若存在 `/data/output/two_step_residual/predictions_test.csv`
2. 对比新方案 vs 旧两步法(0.0337)的RMSE

### 对照3: baseline代理质量评估
- 在测试集上计算 `baseline_interp`(T日曲面插值到T+1日) vs `true_baseline_{t+1}`(T+1日真实B-Spline)的相关系数
- 目标: 相关系数 > 0.95

### 对照4: 三方案汇总表
| 方法 | Test RMSE_IV | Test R^2 | 近月RMSE | 远月RMSE | 无套利违规率 |
|------|-------------|---------|---------|---------|------------|
| 原方案(直接预测IV) | 0.0278 | 0.843 | - | - | - |
| 旧两步法(IV_t代理) | 0.0337 | 0.770 | 0.061 | - | 90.8% |
| **新两步法(M+W空间)** | **目标<=0.0292** | **目标>0.80** | **目标<0.05** | - | **记录** |

---

## 评估指标

### 主指标
1. **Test RMSE_IV**: 合成pred_IV vs 真实IV
2. **Test MAE_IV**
3. **Test R^2_IV**
4. **方向准确率**: sign(pred_IV - IV_t) == sign(true_IV - IV_t)

### 分箱指标
5. **分箱RMSE**:
   - 按 remaining_time 分: 近月(<=30) / 中月(30-90) / 远月(>90)
   - 按 moneyness 分: ITM(M<-0.1) / ATM(|M|<=0.1) / OTM(M>0.1)
   - 按类型分: Call / Put

### baseline代理质量
6. **baseline代理相关系数**: `corr(baseline_interp, true_baseline)`
7. **baseline代理RMSE**: `RMSE(baseline_interp, true_baseline)`

### 无套利记录(非硬性指标)
8. **无套利违规率**: 对合成pred_IV重新计算M-W B-Spline, 检查Call随M非递增/Put随M非递减的比例
   - 已知市场本身约85%违规, **不强制要求<5%**, 只记录供分析

---

## 输出文件要求

保存到 `/data/output/two_step_v2/`:

1. `model_delta_residual.pkl` - 训练好的delta_residual XGBoost
2. `forward_table.csv` - 每天每到期月的F_implied记录
3. `feature_importance_residual.csv` - 特征重要性
4. `metrics.json` - 完整评估指标(含三方案对照)
5. `predictions_test.csv` - 测试集预测结果(必须包含以下列):
   - security_id, trade_date, call_put, exercise_price, remaining_time, moneyness
   - iv_t, true_iv, F_implied, baseline_iv_t, baseline_iv_t1_true, baseline_iv_t1_interp
   - residual_t, true_residual_t1, pred_delta_residual, pred_residual_t1, pred_iv
   - iv_residual(IV预测误差), baseline_proxy_residual(baseline代理误差)
6. `residual_analysis.png` - 残差分析(真实vs预测delta_residual散点图)
7. `pred_vs_true.png` - 合成IV预测vs真实散点图
8. `comparison_table.png` - 三方案对比表
9. `baseline_proxy_quality.png` - baseline代理质量图(interp vs true散点图)
10. `m_w_surface_sample.png` - 某天Call/Put的M-W曲面可视化(M为X轴, W为Y轴, 各到期月不同颜色)

---

## 代码结构规范(必须模块化)

```python
def load_raw_data(path) -> pd.DataFrame:
    """读取原始数据"""

def compute_implied_forward(df_day, maturity_group, r, tau) -> float:
    """
    对单日单到期月计算隐含远期价格
    使用Put-Call Parity: F = K + e^(r*tau)(C - P)
    取ATM附近多对合约的中位数
    """

def build_forward_table(df) -> pd.DataFrame:
    """
    全局计算每天每到期月的F_implied
    输出: forward_table [trade_date, last_edate, F_implied, F_theory]
    """

def fit_mw_surface(df_day, call_put, forward_table_day) -> dict:
    """
    对单日单类型拟合M-W空间B-Spline曲面
    输入: 当天该类型所有合约 + 当天各到期月的F_implied
    输出: {last_edate: CubicSpline_object, ...}
    内部逻辑:
      1. 计算 M = ln(K/F), W = IV^2*tau
      2. 按到期月分组
      3. 每组用CubicSpline(M, W)
      4. 保存Spline对象和(M_min, M_max, W_min, W_max)
    """

def interpolate_baseline(spline_dict, K, tau, F_proxy, tau_nodes) -> float:
    """
    从M-W曲面插值得到baseline_IV
    输入: T日保存的曲面字典, T+1日合约参数(K, tau), F代理
    步骤:
      1. M = ln(K / F_proxy)
      2. 找到tau对应的到期月(或相邻两个月)
      3. M方向: CubicSpline插值(范围外平坦保护)
      4. tau方向: W空间线性插值(跨到期月)
      5. 还原 IV = sqrt(max(W, 1e-6) / tau)
    """

def build_residual_series(df, forward_table) -> pd.DataFrame:
    """
    全局计算所有合约的baseline_IV和residual
    输出: 增加 baseline_iv, residual 列
    """

def build_features_delta_residual(df) -> pd.DataFrame:
    """
    特征工程(复用原方案, 增加历史残差特征)
    输出: X, y_delta_residual, y_iv(用于对照)
    """

def split_temporal(df, train_end, val_end) -> tuple:
    """按交易日时间切分"""

def train_delta_residual_model(X_train, y_train, X_val, y_val) -> xgb.XGBRegressor:
    """训练XGBoost delta_residual预测模型, 支持超参数调优"""

def predict_two_step_v2(model, df_test_t, df_test_t1, spline_dict_t, forward_t) -> pd.DataFrame:
    """
    路线B两步法预测:
    pred_IV = interp_baseline(M_{t+1}, tau_{t+1}) + residual_t + pred_delta_residual
    """

def evaluate_two_step_v2(predictions_df, original_model=None) -> dict:
    """
    评估:
    1. 新方案指标
    2. 原方案对照(若提供模型)
    3. baseline代理质量
    4. 三方案对比表
    """

def save_outputs(model, metrics, predictions, output_dir):
    """保存所有输出"""
```

---

## 检查点(必须打印)

```
[Checkpoint 1] 数据加载与隐含远期计算
  - 原始记录数: {N}
  - 交易日数: {T}
  - 到期月数: {M}(每天平均到期月数)
  - F_implied 计算成功率: {success_rate}%(理论远期 fallback比例)
  - F_implied vs F_theory 差异统计: mean_diff={x}, std_diff={y}
    (验证隐含远期与理论远期差异是否合理, 应<0.05)
  - 某天(如20150209)各到期月F_implied示例: [2.331, 2.335, 2.340, 2.345]

[Checkpoint 2] M-W空间B-Spline拟合
  - Call/Put日均合约数: {call_avg} / {put_avg}
  - M-W曲面拟合成功天数: {success_days}
  - M范围覆盖: [M_min, M_max] = [{min_M}, {max_M}](应覆盖-0.3~0.3)
  - W范围覆盖: [W_min, W_max] = [{min_W}, {max_W}]
  - W单调性检查(Call随M非递增违规率): {call_violation}%
  - W单调性检查(Put随M非递减违规率): {put_violation}%
  - residual统计: mean={x}, std={y}, min={z}, max={w}
    (residual应近似以0为中心, std < 0.05)
  - 某天M-W曲面可视化: m_w_surface_sample.png

[Checkpoint 3] 特征工程完成
  - 构造样本数: {N_samples}
  - 特征维度: {N_features}
  - 新增特征确认: residual_t, residual_t-1, residual_ma3 已包含
  - 目标变量 delta_residual 统计: mean={x}, std={y}, min={z}, max={w}
  - delta_residual vs 0 的RMSE(基准线): {naive_rmse}
  - 样本示例(前3行):
    ...

[Checkpoint 4] 模型训练完成
  - 最优参数: {params}
  - 训练RMSE(delta_residual): {x}
  - 验证RMSE(delta_residual): {y}
  - 验证RMSE vs 基准线改善: {improvement}%
  - Top 5特征: {feature_list}
  - residual特征是否进入Top 5: {Yes/No}

[Checkpoint 5] 评估与对照完成
  - 新方案 Test RMSE_IV: {rmse_new}
  - 新方案 Test R^2_IV: {r2_new}
  - 新方案 Test MAE_IV: {mae_new}
  - 新方案方向准确率: {dir_acc_new}
  - 原方案 Test RMSE_IV: {rmse_original}
  - 旧两步法 Test RMSE_IV: {rmse_old_two_step}(若存在)
  - 新方案 vs 原方案差异: {diff_vs_original_pct}%
  - 新方案 vs 旧两步法差异: {diff_vs_old_pct}%
  - baseline代理相关系数: {baseline_corr}
  - baseline代理RMSE: {baseline_rmse}
  - 近月/中月/远月 RMSE: {near} / {mid} / {far}
  - Call/Put RMSE: {call} / {put}
  - 无套利违规率(记录): {violation_rate}%
```

---

## 中间结果验证清单

- [ ] `F_implied` 与 `F_theory` 差异均值 < 0.05(隐含远期合理)
- [ ] `F_implied` 无NaN(所有到期月都有F值)
- [ ] M范围覆盖 > 95%合约([-0.3, 0.3]内)
- [ ] `residual` 均值 ~= 0(B-Spline基准应无偏)
- [ ] `residual` 标准差 < 0.05(残差比原始IV波动小)
- [ ] `delta_residual` 标准差 < `residual` 标准差(日变化应更小)
- [ ] 特征中 **不包含** delta/gamma/theta/vega/rho
- [ ] 特征中 **不包含** t+1日信息(如IV_{t+1})
- [ ] 测试集交易日严格晚于训练/验证集

---

## 成功标准(优先级排序)

1. **新方案 RMSE_IV < 旧两步法 0.0337**(必须优于路线A)
2. **新方案 RMSE_IV <= 原方案 0.0278 * 1.05 = 0.0292**(接近原方案精度)
3. **baseline代理相关系数 > 0.95**(期限对齐有效)
4. **近月RMSE < 0.05**(修复路线A的近月惨败)
5. **残差模型 RMSE(delta_residual) < residual标准差 * 0.5**(优于随机游走假设)
6. **特征重要性Top 5中至少包含1个残差历史特征**(验证残差时序有用)

---

## 异常处理规范

- `implc_volatlty = 0`: 不参与B-Spline拟合, baseline设为同到期月均值
- 某天某到期月合约数 < 3: 该月baseline用同类型其他到期月ATM附近W填充
- 合约M在拟合范围外: 用最近边界W值(平坦外推, 零斜率)
- tau在拟合范围外: 用最近到期月W值
- 历史残差窗口不足: 用可用天数计算
- T+1日新合约上市(全新行权价): 用T日M范围边界W值
- T+1日新到期月上市: 用同类型最远到期月W外推
- T+1日合约退市: 该样本剔除

---

## Reviewer快速验证脚本

```python
import json, pandas as pd, os

# 1. 检查输出完整性
assert os.path.exists('/data/output/two_step_v2/model_delta_residual.pkl')
assert os.path.exists('/data/output/two_step_v2/metrics.json')
assert os.path.exists('/data/output/two_step_v2/predictions_test.csv')

# 2. 检查关键指标
with open('/data/output/two_step_v2/metrics.json') as f:
    m = json.load(f)

assert m['two_step_v2']['test']['rmse_iv'] < 0.0337, "未优于旧两步法"
assert m['two_step_v2']['test']['rmse_iv'] <= 0.0292, "未达原方案105%目标"
assert m['two_step_v2']['test']['baseline_proxy_corr'] > 0.95, "baseline代理质量不足"
assert m['two_step_v2']['test']['rmse_near'] < 0.05, "近月RMSE未改善"

# 3. 检查对照结果
print(f"新方案RMSE: {m['two_step_v2']['test']['rmse_iv']:.4f}")
print(f"原方案RMSE: {m['original']['test']['rmse_iv']:.4f}")
print(f"相对差异: {(m['two_step_v2']['test']['rmse_iv'] - m['original']['test']['rmse_iv']) / m['original']['test']['rmse_iv'] * 100:.1f}%")

# 4. 检查残差模型有效性
df = pd.read_csv('/data/output/two_step_v2/predictions_test.csv')
delta_rmse = ((df['pred_delta_residual'] - (df['true_residual_t1'] - df['residual_t'])) ** 2).mean() ** 0.5
residual_std = df['residual_t'].std()
print(f"delta_residual RMSE: {delta_rmse:.4f}")
print(f"residual std: {residual_std:.4f}")
assert delta_rmse < residual_std * 0.5, "残差模型不如随机游走"

# 5. 检查特征重要性
fi = pd.read_csv('/data/output/two_step_v2/feature_importance_residual.csv')
print("Top 5 features:", fi['feature'].head(5).tolist())
assert any('residual' in f for f in fi['feature'].head(10)), "残差特征未进入Top10"

# 6. 检查baseline代理质量
proxy_rmse = ((df['baseline_iv_t1_interp'] - df['baseline_iv_t1_true']) ** 2).mean() ** 0.5
proxy_corr = df['baseline_iv_t1_interp'].corr(df['baseline_iv_t1_true'])
print(f"baseline代理RMSE: {proxy_rmse:.4f}")
print(f"baseline代理相关系数: {proxy_corr:.4f}")
assert proxy_corr > 0.95, "baseline代理相关系数不足"
```

---

## 备注

1. **代码复用**: 本方案特征工程与原方案高度一致, 直接复用 `build_features()` 逻辑, 增加 `residual_t`, `residual_t-1`, `residual_ma3` 等历史残差特征.
2. **曲面保存**: 每天Call/Put的B-Spline对象建议用 `pickle` 保存为轻量字典 `{last_edate: (M_nodes, W_nodes, CubicSpline)}`, 预测时重建即可, 无需保存整个对象.
3. **内存优化**: `F_implied` 计算和M-W曲面拟合按天循环处理, 无需一次性加载全部数据.
4. **关键认知**: M-W空间的B-Spline不是"预测"对象, 而是"基准计算"工具. 预测的是**delta_residual**, 曲面只负责提供准确的baseline代理.
5. **无套利说明**: 市场数据本身约85%存在微观结构噪声导致的"违规", 本方案记录违规率但不作为硬性指标. B-Spline的平滑性会自然降低预测输出的违规率, 但不应以牺牲精度为代价强行满足.