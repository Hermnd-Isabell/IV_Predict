# 50ETF期权策略A(修订版): XGBoost残差预测偏差(Delta-Neutral Long-Short)

## 任务目标
基于已训练完成的逐合约XGBoost IV预测模型(RMSE=0.0278)和M-W B-Spline无套利曲面, 构建**残差预测偏差驱动的Delta-Neutral多空策略**.

**核心修正(相比初版)**:
1. **绕过Theta Bleed**: 不再使用 `pred_IV - IV_t`(包含期限衰减), 改用 **残差变化** `pred_residual_{t+1} - residual_t` 作为信号
2. **Vega暴露监控**: 记录组合净Vega, 评估Vega裸露对收益的影响
3. **阶梯滑点**: 按Moneyness设定不同滑点率(ATM 0.5%, OTM 3%)
4. **保证金成本**: 计算空头保证金占用, 输出ROE而非简单资金回报

**信号定义**:
```
residual_t        = IV_t - baseline_IV_t          (T日定价偏差)
pred_residual_t1  = pred_IV_{t+1} - baseline_IV_{t+1}  (预测T+1日定价偏差, 用T+1日真实baseline或代理)
signal            = pred_residual_t1 - residual_t      (预测残差变化方向 = 纯Alpha)
```

- `signal > 0`: 预测残差走阔(定价偏差扩大) -> **做多(Long)**
- `signal < 0`: 预测残差收敛(定价偏差缩小) -> **做空(Short)**

## 数据路径
- 原始数据: `/data/raw/50etf_options.csv`
- XGBoost模型: `/data/output/baseline_xgb/model_abs_iv.pkl`
- B-Spline输出: `/data/output/two_step_v2/predictions_test.csv`(含baseline_IV, residual)
- 策略输出: `/data/output/strategy_a_v2/`

## 核心约束
1. **交易成本**: 50ETF期权手续费按实际市场标准(开平仓各3元/张)
2. **阶梯滑点**: ATM(|M|<0.05) 0.5%, 轻度虚值(0.05<=|M|<0.15) 1.0%, 深度虚值(|M|>=0.15) 3.0%
3. **Delta对冲**: 每日收盘重新平衡, 确保组合Delta~=0
4. **Vega监控**: 记录但不强制Vega中性(数据vega存在内生性问题)
5. **保证金**: 计算空头保证金占用, 输出ROE
6. **持仓周期**: 日度调仓

---

## 策略架构

### Step 1: 残差信号构造(每日收盘后)

**前提**: B-Spline baseline_IV 和 residual 已全局计算(复用two_step_v2代码)

```python
# T日已知
residual_t = IV_t - baseline_IV_t

# T+1日预测
pred_IV_t1 = xgb_model.predict(X_t)  # XGBoost预测绝对IV
baseline_IV_t1 = interpolate_mw_surface(...)  # B-Spline T+1日baseline(用T日曲面代理或真实拟合)
pred_residual_t1 = pred_IV_t1 - baseline_IV_t1

# 核心信号: 残差变化
signal = pred_residual_t1 - residual_t
```

**过滤条件**:
- 只保留 `|signal| > threshold` 的合约(threshold = 0.003 或按分位数)
- 剔除 `implc_volatlty = 0` 的合约
- 剔除到期日(remaining_time <= 3)的合约(规避末日轮)
- 剔除当日 volume < 该到期月中位数×0.3 的合约(流动性过滤)
- 剔除 open_interest < 100 的合约

---

### Step 2: 组合构建(Delta-Neutral Long-Short)

**排序与分档**:
```python
# 每天对所有合约按 signal 排序
df['rank'] = df.groupby('trade_date')['signal'].rank(pct=True)

# 多头组合(预测残差走阔): Top 10% signal
df_long = df[df['rank'] >= 0.90]

# 空头组合(预测残差收敛): Bottom 10% signal
df_short = df[df['rank'] <= 0.10]
```

**权重分配**:
- **等权重**: 默认方案
- **信号加权**: 按 `|signal|` 加权
- **Vega加权**: 按 `1/vega` 加权(尝试实现Vega中性, 可选)

**Delta对冲计算**:
```python
# 组合净Delta
portfolio_delta = sum(long_weights * long_deltas) - sum(short_weights * short_deltas)

# 50ETF现货对冲
hedge_shares = -portfolio_delta
hedge_cost = abs(hedge_shares) * fund_close * stock_commission
```

**Vega监控**:
```python
portfolio_vega = sum(long_weights * long_vegas) - sum(short_weights * short_vegas)
# 记录但不强制对冲
```

---

### Step 3: 收益与成本计算

**期权收益**:
```python
option_pnl = (close_{t+1} - close_t) - delta_t * (fund_close_{t+1} - fund_close_t)
gross_pnl = sum(long_weights * long_option_pnl) - sum(short_weights * short_option_pnl)
```

**阶梯滑点**:
```python
def get_slippage(moneyness_abs):
    if moneyness_abs < 0.05: return 0.005    # ATM: 0.5%
    elif moneyness_abs < 0.15: return 0.010  # 轻度虚值: 1.0%
    else: return 0.030                         # 深度虚值: 3.0%

slippage = sum(contract_value * get_slippage(abs(moneyness)))
```

**保证金计算(空头)**:
```python
# 国内交易所标准(简化)
def calc_margin(short_contracts, S, K, call_put):
    if call_put == 'C':
        margin = max(premium + S * 0.10 - max(K - S, 0), premium + S * 0.05)
    else:
        margin = max(premium + S * 0.10 - max(S - K, 0), premium + S * 0.05)
    return margin

total_margin = sum(calc_margin(short_contracts))
roe = net_pnl / total_margin  # 净资产收益率
```

**净收益**:
```python
option_trade_cost = (num_long + num_short) * contract_fee
net_pnl = gross_pnl - option_trade_cost - hedge_cost - slippage
```

---

### Step 4: 回测框架

**时间范围**: 2025-01 ~ 2026-01(测试集)

**每日流程**:
```
T日收盘:
  1. 计算T日所有合约的 residual_t = IV_t - baseline_IV_t
  2. XGBoost预测T+1日 pred_IV, B-Spline计算 baseline_IV_{t+1}
  3. 计算 signal = pred_residual_t1 - residual_t
  4. 排序, 构建Top 10% Long / Bottom 10% Short
  5. 计算组合Delta, Vega, 50ETF对冲仓位, 空头保证金

T+1日收盘:
  6. 计算Delta-Hedged收益, 扣除成本, 记录ROE
  7. 平仓或移仓
```

---

## 评估指标

### 收益指标
1. **日均净收益**
2. **年化收益**
3. **年化波动率**
4. **年化夏普比率**
5. **信息比率**
6. **ROE(净资产收益率)**: net_pnl / margin_required

### 风险指标
7. **最大回撤**
8. **Calmar比率**

### 策略质量指标
9. **多头胜率 / 空头胜率**
10. **日均换手率**
11. **Delta暴露均值**
12. **Vega暴露均值**: 检验Vega裸露程度
13. **Gamma暴露**

### 分箱分析
14. **按moneyness分箱**: ITM / ATM / OTM
15. **按remaining_time分箱**: 近月 / 中月 / 远月
16. **按call_put分箱**: Call vs Put
17. **按|signal|大小分箱**: 强信号 vs 弱信号

---

## 输出文件要求

保存到 `/data/output/strategy_a_v2/`:

1. `backtest_results.csv`:
   - trade_date, gross_pnl, net_pnl, long_pnl, short_pnl
   - num_long, num_short, avg_signal_long, avg_signal_short
   - portfolio_delta, portfolio_vega, hedge_shares, hedge_cost
   - option_trade_cost, slippage_cost, total_margin, roe
   - cumulative_pnl, drawdown

2. `daily_positions.csv`:
   - trade_date, security_id, call_put, exercise_price, position_type
   - weight, delta, vega, signal, residual_t, pred_residual_t1
   - entry_iv, exit_iv, moneyness, slippage_rate

3. `metrics.json`
4. `equity_curve.png`
5. `drawdown_curve.png`
6. `monthly_returns.png`
7. `pnl_decomposition.png`
8. `moneyness_pnl.png`
9. `maturity_pnl.png`
10. `signal_strength_pnl.png`: 按|signal|分箱收益
11. `vega_exposure.png`: 每日Vega暴露时间序列

---

## 代码结构规范

```python
def load_model_and_data(model_path, bspline_path, data_path):
    # 加载XGBoost模型, B-Spline输出, 原始数据

def compute_residual_signal(df, xgb_model, bspline_dict):
    # 计算 residual_t, pred_residual_t1, signal
    # signal = pred_residual_t1 - residual_t

def apply_liquidity_filter(df, volume_pct=0.3, min_oi=100):
    # 流动性过滤: volume > 到期月中位数*volume_pct, OI > min_oi

def build_portfolio(df, signal_col='signal', long_pct=0.10, short_pct=0.10):
    # 构建Delta-Neutral Long-Short组合

def compute_delta_hedge(long_df, short_df):
    # 计算组合净Delta和50ETF对冲仓位

def compute_vega_exposure(long_df, short_df):
    # 计算组合净Vega(仅监控, 不强制对冲)

def calc_margin(short_contracts, S):
    # 计算空头保证金占用(国内交易所标准)

def get_slippage(moneyness_abs):
    # 阶梯滑点: ATM 0.5%, 轻度虚值1%, 深度虚值3%

def calculate_pnl(long_df, short_df, hedge_shares, df_next_day):
    # 计算Delta-Hedged收益, 扣除成本, 计算ROE

def run_backtest(df, xgb_model, bspline_dict, start_date, end_date):
    # 主回测循环

def compute_metrics(backtest_df):
    # 计算夏普, 信息比率, 最大回撤, ROE等

def plot_results(backtest_df, output_dir):
    # 绘制权益曲线, 回撤, Vega暴露等

def save_outputs(backtest_df, positions_df, metrics, output_dir):
    # 保存所有输出
```

---

## 检查点(必须打印)

```
[Checkpoint 1] 模型与数据加载
  - XGBoost模型加载成功
  - B-Spline输出记录数: {N}
  - residual_t 统计: mean={mean}, std={std}

[Checkpoint 2] 信号构造
  - signal = pred_residual_t1 - residual_t
  - signal均值: {mean}, 标准差: {std}
  - |signal| > 0.003 的合约比例: {pct}%
  - 流动性过滤后保留合约比例: {liq_pct}%

[Checkpoint 3] 组合构建示例(某天)
  - Long合约数: {n_long}, 平均signal: {avg_long}
  - Short合约数: {n_short}, 平均signal: {avg_short}
  - 组合净Delta(对冲前): {raw_delta}
  - 组合净Vega(监控): {raw_vega}
  - 50ETF对冲仓位: {hedge_shares}
  - 空头保证金: {margin}

[Checkpoint 4] 回测完成
  - 总交易日: {T}
  - 日均毛收益: {gross_pnl}
  - 日均净收益: {net_pnl}
  - 日均ROE: {roe}
  - 年化夏普比率: {sharpe}
  - 最大回撤: {max_dd}
  - 信息比率: {ir}

[Checkpoint 5] 分箱与信号分析
  - ITM/ATM/OTM日均收益: {itm} / {atm} / {otm}
  - 近月/中月/远月日均收益: {near} / {mid} / {far}
  - 强|signal|(Top 20%)日均收益: {strong_signal}
  - 弱|signal|(Bottom 20%)日均收益: {weak_signal}
  - Vega暴露与收益相关系数: {vega_corr}
```

---

## 成功标准

- [ ] **日均净收益 > 0**
- [ ] **年化夏普比率 > 0.5**
- [ ] **信息比率 > 0.3**
- [ ] **ROE > 0**(考虑保证金成本后仍有正收益)
- [ ] **多头胜率 > 50% 且 空头胜率 > 50%**
- [ ] **Delta暴露均值 < 0.1**
- [ ] **Vega暴露与收益相关系数 < 0.3**(Vega不主导收益)
- [ ] **最大回撤 < 20%**

---

## 备注

1. **残差变化信号**: 这是本修订版的核心改进. 通过 `pred_residual_t1 - residual_t`, 我们消除了期限衰减(Theta Bleed)的干扰, 信号只反映定价偏差的纯Alpha变化.
2. **Vega监控而非对冲**: 由于vega字段由BS公式反推存在内生性问题, 我们不强制Vega中性, 但记录Vega暴露. 如果回测发现Vega暴露与收益高度相关(|corr|>0.3), 可在后续版本中升级为Vega约束优化.
3. **保证金简化**: 国内期权保证金公式复杂(涉及虚值额、标的保证金比例等), 本Prompt使用简化版. 实盘需按交易所最新标准调整.
4. **B-Spline代理**: T+1日baseline_IV可用T日曲面代理(如two_step_v2中的interpolate_baseline), 或每日重新拟合. 后者更准确但计算量大.
Strategy A V2 saved.