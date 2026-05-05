# 策略C：双信号组合策略（Double-Sort 条件筛选）

## 任务目标
基于 **XGBoost 预测偏差** 与 **B-Spline 相对定价偏差** 两个正交信号，构建 **Double-Sort 条件筛选** 的 Delta-Neutral Long-Short 回测策略。

**核心原则**：
- **废弃简单加权 Ensemble**（截面均值回归与时序动量逻辑冲突，直接相加会互相抵消）。
- **唯一实现方式**：Double-Sort 条件筛选——先用 B-Spline 找出显著定价偏差的合约，再用 XGBoost 判断其修复方向，**仅当两信号方向一致时才开仓**。

---

## 一、数据加载与前置计算

### 1.1 输入数据路径
```
/data/output/baseline_xgb/model_abs_iv.pkl          # XGBoost 模型（原方案）
/data/output/two_step_v2/predictions_test.csv        # B-Spline 输出（含 baseline_IV, residual）
/data/processed/options_panel_test.csv              # 原始测试集面板数据
```

若 `predictions_test.csv` 不存在，需先用路线 B 的 M-W B-Spline 逻辑重新计算：
- 每天每到期月用 Put-Call Parity 反推 `F_implied`（取 ATM 附近多对 C/P 的 `K + e^(rτ)(C-P)` 中位数）。
- 计算 `M = ln(K/F_implied)`，`W = IV²·τ`。
- 每日每到期月在 M 方向做 CubicSpline + 平坦外推，得到 `baseline_IV`。
- `residual = market_IV - baseline_IV`。

### 1.2 XGBoost 预测值计算
使用 `model_abs_iv.pkl` 对测试集逐合约预测 `pred_IV`。
**关键**：计算 **预测残差** 而非绝对 IV：
```python
pred_residual = pred_IV - baseline_IV
```
其中 `baseline_IV` 为 T 日 B-Spline 曲面在对应 `(M, τ)` 处的插值结果。

---

## 二、信号定义与方向一致性约束

### 2.1 B-Spline 截面信号（均值回归逻辑）
对每天每个到期月内的所有合约，计算残差的截面 Z-Score：
```python
residual_zscore = (residual - residual_mean_tau) / residual_std_tau
```
- `residual_mean_tau`：该到期月当日所有合约 residual 的均值（理论上应接近 0，用于纠偏）。
- `residual_std_tau`：该到期月当日所有合约 residual 的标准差。

**筛选阈值**：仅保留 `|residual_zscore| > 1.5` 的合约（显著定价偏差）。

### 2.2 XGBoost 时序信号（修复方向逻辑）
XGBoost 预测的是 T+1 日残差 `pred_residual_{t+1}`。
**方向一致性约束**（核心过滤器）：

| 当前状态 | B-Spline 信号 | 要求 XGBoost 预测方向 | 逻辑 |
|---------|--------------|---------------------|------|
| `residual > 0`（高估） | 做空 | `pred_residual < residual`（预测 residual 下降，向基准回归） | 预测修复 |
| `residual < 0`（低估） | 做多 | `pred_residual > residual`（预测 residual 上升，向基准回归） | 预测修复 |

**仅当 XGBoost 预测方向与 residual 回归方向一致时，才保留该合约进入组合。**
若 XGBoost 预测 residual 继续偏离（如当前高估但预测更高），**即使偏离显著也剔除**——避免逆势操作。

### 2.3 信号隔夜衰减分析（必须记录）
在回测前执行诊断测试：
```python
# 计算 T 日收盘 residual 与 T+1 日开盘收益的相关性（如有分钟数据）
# 若仅有日度数据，计算：
overnight_decay = (residual_t - residual_{t+1}) / residual_t
```
在回测报告中输出 **"隔夜衰减率均值"** 和 **"信号半衰期估计"**。若衰减率 > 50%，需在文档中明确标注："策略基于日度快照，存在信号隔夜衰减风险。"

---

## 三、流动性过滤与合约筛选

在进入 Double-Sort 前，先剔除流动性不足的合约：
```python
liquidity_mask = (
    (volume > 该到期月当日成交量中位数 * 0.3) &
    (open_interest > 100) &
    (abs(residual) > 0.003) &           # 避免在权利金极低的合约上交易
    (remaining_time > 5)                # 剔除最后5天的极度近月合约（防止 Gamma 风险）
)
```

---

## 四、Double-Sort 组合构建

### 4.1 第一步：B-Spline 显著偏离池
每天，对每个到期月：
1. 计算所有合约的 `residual_zscore`。
2. 保留 `|residual_zscore| > 1.5` 且通过流动性过滤的合约。
3. 分为两个子池：
   - **高估池**：`residual_zscore > 1.5`（候选空头）
   - **低估池**：`residual_zscore < -1.5`（候选多头）

### 4.2 第二步：XGBoost 方向一致性过滤
在高估池和低估池内，分别应用方向一致性约束：
- **高估池**：仅保留 `pred_residual < residual` 的合约。
- **低估池**：仅保留 `pred_residual > residual` 的合约。

### 4.3 第三步：分档与等权配置
- **多头组合**：在通过过滤的低估池中，按 `residual_zscore` 升序取 **Top 10%**（最被低估）。
- **空头组合**：在通过过滤的高估池中，按 `residual_zscore` 降序取 **Top 10%**（最被高估）。
- 每个组合内合约 **等权配置名义本金**（或按 Vega 等权，见下文）。

### 4.4 Delta 对冲
对每个合约，使用原始数据中的 `delta` 字段（或从 `baseline_IV` 通过 BS 公式反推）。

**组合层面**：
```python
net_delta_long = sum(多头合约 weight_i * delta_i)
net_delta_short = sum(空头合约 weight_i * delta_i)
net_delta_total = net_delta_long - net_delta_short
```
- 若 `|net_delta_total| > 0.1`（组合层面 10% Delta 暴露），用 50ETF 现货或期货对冲至 `|net_delta| < 0.05`。
- 记录每日 `net_delta_total` 作为风控指标。

### 4.5 Vega 暴露监控（不强制中性，但必须记录）
```python
net_vega_total = sum(多头 weight_i * vega_i) - sum(空头 weight_i * vega_i)
```
在回测输出中增加每日 `net_vega_total`。若回测发现 Vega 暴露与收益高度相关（|corr| > 0.3），需在分析文档中标注："收益可能主要由 Vega 敞口驱动，而非 Alpha。"

---

## 五、收益与成本计算

### 5.1 Delta-Hedged 日收益（单合约）
```python
option_pnl = (option_close_{t+1} - option_close_t) - delta_t * (fund_close_{t+1} - fund_close_t)
```
整体组合收益 = 多头组合 option_pnl 之和 - 空头组合 option_pnl 之和。

### 5.2 阶梯滑点（按 Moneyness 区分）
根据合约的 moneyness `M = ln(K/F)` 设定不同滑点率：

| Moneyness 区间 | 滑点率（单边） |
|---------------|--------------|
| ATM (`|M| < 0.05`) | 0.05% |
| 轻度虚值 (`0.05 ≤ |M| < 0.15`) | 0.10% |
| 深度虚值 (`|M| ≥ 0.15`) | 0.25% |

**说明**：深度虚值合约买卖价差极大，IV 反推不稳定，必须施加更高摩擦成本。

交易成本从每日调仓的合约中扣除（开仓 + 平仓双边）。

### 5.3 保证金成本
国内 ETF 期权保证金规则（空头）：
```python
margin_short = max(
    权利金_market + 标的收盘价 * 保证金比例 - 虚值额,
    权利金_market + 标的收盘价 * 保证金比例 * 0.5
)
# 保证金比例取 12%（交易所标准）
```
多头仅支付权利金，无额外保证金。

在回测指标中计算 **ROE（保证金收益率）**：
```python
roe = net_pnl / total_margin_occupied
```
其中 `total_margin_occupied` 为每日空头组合保证金占用之和（多头权利金支出也计入资金占用）。

---

## 六、回测输出要求

### 6.1 主回测表：`backtest_results.csv`
每日一行，字段包括：
- `trade_date`
- `n_long` / `n_short`：多空持仓数量
- `pnl_before_cost`：滑点前毛收益
- `trading_cost`：总交易成本（滑点）
- `pnl_after_cost`：净收益
- `margin_occupied`：保证金占用
- `roe`：当日 ROE
- `net_delta`：组合净 Delta
- `net_vega`：组合净 Vega
- `avg_residual_zscore_long` / `avg_residual_zscore_short`：持仓平均 Z-Score
- `direction_agreement_rate`：当日通过方向一致性过滤的合约比例

### 6.2 诊断分析表：`signal_diagnostics.csv`
- `residual_autocorr_lag1`：residual 日变化自相关（检验噪声属性）
- `pred_residual_corr_with_actual`：预测残差与实际残差的相关性
- `overnight_decay_rate`：信号隔夜衰减率
- `xgb_signal_alpha`：XGBoost 信号单独回测的日收益（用于对比）
- `bspline_signal_alpha`：B-Spline 信号单独回测的日收益（用于对比）

### 6.3 策略对比表：`strategy_comparison.csv`
同时记录策略 A（纯 XGBoost）、策略 B（纯 B-Spline）、策略 C（Double-Sort）的关键指标：

| 指标 | 策略A | 策略B | 策略C |
|-----|------|------|------|
| 日均收益 | | | |
| 年化夏普比率 | | | |
| 年化 ROE | | | |
| 最大回撤 | | | |
| 胜率 | | | |
| 日均换手率 | | | |
| 收益相关系数(A,C) | | | |
| 收益相关系数(B,C) | | | |
| 净 Delta 暴露均值 | | | |
| 净 Vega 暴露均值 | | | |

### 6.4 分维度拆解表
按以下维度拆解策略 C 的表现：
- **按 Moneyness**：ATM / 轻度虚值 / 深度虚值
- **按期限**：近月（≤30天）/ 中月（30-90天）/ 远月（>90天）
- **按方向一致性强度**：`|pred_residual - residual|` 分档（修复幅度越大，置信度越高）

---

## 七、关键约束与提醒

1. **绝对禁止简单 Ensemble**：不要在代码中实现 `0.5*zscore(xgb) + 0.5*zscore(bspline)` 的线性加权。两种信号物理意义不同（时序动量 vs 截面均值回归），直接相加会导致逻辑冲突和信号抵消。

2. **方向一致性是核心 Alpha 来源**：策略 C 的价值不在于"两个信号叠加"，而在于 **XGBoost 为 B-Spline 的均值回归交易提供了"何时修复"的时间维度过滤**。如果 XGBoost 预测 residual 继续偏离，说明当前定价偏差可能不是噪声而是真实信息（如重大事件导致微笑变形），此时应回避。

3. **近月特殊处理**：对于 `remaining_time ≤ 30` 的合约，在计算 `baseline_IV` 时，T+1 日代理改用 `baseline_IV_t * sqrt(τ_t / τ_{t+1})`（W 守恒假设），而非直接 M-W 曲面插值。这已在路线 B 中被证明可降低近月误差。

4. **残差模型已死，不要试图预测 delta_residual**：XGBoost 的 `pred_IV` 是完整 IV 预测，不是残差预测。在策略 C 中，我们利用的是 `pred_IV` 与 `baseline_IV` 的相对关系（`pred_residual`），而非单独训练残差模型。

5. **Theta Bleed 已规避**：由于信号基于 **residual 的变化**（`pred_residual - residual`），而非绝对 IV 的变化（`pred_IV - IV_t`），近月合约的期限衰减已被 B-Spline baseline 吸收，不会系统性影响信号。

6. **如果 Double-Sort 后合约数量过少**（如每天多空各 < 3 只），放宽 `|residual_zscore| > 1.5` 至 `> 1.0`，或放宽流动性阈值，但必须在报告中记录调整。

---

## 八、执行顺序

1. 加载数据并计算 `baseline_IV`、`residual`、`pred_IV`、`pred_residual`。
2. 执行信号诊断（隔夜衰减、自相关、预测相关性）。
3. 运行策略 C 回测主循环（Double-Sort + 方向一致性 + Delta 对冲）。
4. 同时运行策略 A 和策略 B 的简化回测（用于对比表）。
5. 输出三张 CSV 和一份 Markdown 分析报告。

**预期目标**：
- 策略 C 的夏普比率应 **高于策略 A 和策略 B 的单独夏普**（证明 Double-Sort 的过滤价值）。
- 策略 C 与策略 A/B 的收益相关性应 **< 0.5**（证明 Alpha 来源正交）。
- 净 Delta 暴露均值应 **< 0.05**（证明对冲有效）。
- 若策略 C 夏普低于策略 B，说明 XGBoost 时序信号对截面均值回归无增益，需重新审视方向一致性逻辑。
