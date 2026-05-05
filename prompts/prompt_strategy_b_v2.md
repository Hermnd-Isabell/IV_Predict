# 50ETF期权策略B(修订版): B-Spline截面相对定价偏差(Delta-Neutral Long-Short)

## 任务目标
基于已构建的M-W空间B-Spline无套利IV曲面, 构建**截面相对定价偏差驱动的Delta-Neutral多空策略**. 核心逻辑:

如果某合约的市场IV显著偏离B-Spline无套利基准IV, 则该合约存在相对定价偏差

**核心修正(相比初版)**:
1. **信号衰减测试**: 增加隔夜信号衰减分析, 评估T日收盘residual到T+1日开盘是否已失效
2. **流动性过滤**: 剔除低流动性合约(volume < 到期月中位数×0.3, OI < 100)
3. **阶梯滑点**: ATM 0.5%, 轻度虚值1%, 深度虚值3%
4. **保证金成本**: 计算空头保证金占用, 输出ROE

**信号定义**:
```
residual = market_IV - baseline_IV  (截面定价偏差)
residual_zscore = (residual - 到期月均值) / 到期月标准差  (标准化)
```

- `residual > 0` (residual_zscore > 0): 相对高估 -> **做空(Short)**
- `residual < 0` (residual_zscore < 0): 相对低估 -> **做多(Long)**

**关键认知**: residual本质是市场微观结构噪声(买卖价差/流动性分层/做市商差异). 其时序变化不可预测, 但截面分布可能包含可套利信息. **隔夜衰减风险**: 这种噪声的半衰期可能只有几分钟, T+1日开盘时可能已均值回归.

## 数据路径
- 原始数据: `/data/raw/50etf_options.csv`
- B-Spline输出: `/data/output/two_step_v2/predictions_test.csv`(含baseline_IV, residual)
- 策略输出: `/data/output/strategy_b_v2/`

## 核心约束
1. **交易成本**: 50ETF期权手续费(开平仓各3元/张)
2. **阶梯滑点**: ATM 0.5%, 轻度虚值1%, 深度虚值3%
3. **Delta对冲**: 每日收盘重新平衡
4. **流动性过滤**: volume > 到期月中位数×0.3, OI > 100
5. **保证金**: 计算空头保证金, 输出ROE
6. **持仓周期**: 日度调仓

---

## 策略架构

### Step 1: 信号构造与衰减测试(每日收盘后)

**信号计算**:
```python
# 全局计算(复用two_step_v2)
residual = market_IV - baseline_IV

# 按到期月标准化(消除不同到期月的尺度差异)
df['residual_zscore'] = df.groupby(['trade_date', 'last_edate'])['residual'].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
)

# 或按交易日全局标准化
df['residual_zscore_global'] = df.groupby('trade_date')['residual'].transform(
    lambda x: (x - x.mean()) / x.std()
)
```

**信号衰减测试(关键新增)**:
```python
# 检验T日收盘residual与T+1日开盘收益的关系
# 若数据中有开盘价, 计算:
overnight_return = (open_{t+1} - close_t) / close_t  # 期权隔夜收益率

# 衰减率统计
decay_corr = residual_t.corr(overnight_return)  # 应接近0或为负(反向)
decay_rate = (residual_t - residual_{t+1_open}) / residual_t  # 信号隔夜衰减比例
```

**过滤条件**:
- 只保留 `|residual_zscore| > threshold` 的合约(threshold = 1.0 或 1.5)
- 剔除 `implc_volatlty = 0` 的合约
- 剔除到期日(remaining_time <= 3)的合约
- **流动性过滤**: volume > 该到期月当日中位数 × 0.3
- **持仓过滤**: open_interest > 100
- 剔除B-Spline拟合失败的合约(baseline_IV为NaN)

---

### Step 2: 组合构建(Delta-Neutral Long-Short)

**排序与分档**:
```python
# 每天对所有合约按 residual 排序(residual越大=越高估=做空)
df['rank'] = df.groupby('trade_date')['residual'].rank(pct=True)

# 多头组合(相对低估): Bottom 10% residual(最负)
df_long = df[df['rank'] <= 0.10]

# 空头组合(相对高估): Top 10% residual(最正)
df_short = df[df['rank'] >= 0.90]
```

**权重分配**:
- **等权重**: 默认
- **信号加权**: 按 `|residual_zscore|` 加权(偏差越大, 权重越高)
- **流动性加权**: 按 `volume` 或 `open_interest` 加权

**Delta对冲**:
```python
portfolio_delta = sum(long_weights * long_deltas) - sum(short_weights * short_deltas)
hedge_shares = -portfolio_delta
hedge_cost = abs(hedge_shares) * fund_close * stock_commission
```

---

### Step 3: 收益与成本计算

**期权收益**:
```python
option_pnl = (close_{t+1} - close_t) - delta_t * (fund_close_{t+1} - fund_close_t)
long_pnl = sum(long_weights * long_option_pnl)
short_pnl = -sum(short_weights * short_option_pnl)
gross_pnl = long_pnl + short_pnl
```

**阶梯滑点**:
```python
def get_slippage(moneyness_abs):
    if moneyness_abs < 0.05: return 0.005
    elif moneyness_abs < 0.15: return 0.010
    else: return 0.030
```

**保证金与ROE**:
```python
def calc_margin(short_contracts, S, K, call_put, premium):
    if call_put == 'C':
        margin = max(premium + S * 0.10 - max(K - S, 0), premium + S * 0.05)
    else:
        margin = max(premium + S * 0.10 - max(S - K, 0), premium + S * 0.05)
    return margin

total_margin = sum(calc_margin(short_contracts))
roe = net_pnl / total_margin
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
  1. 用T日所有合约拟合M-W B-Spline曲面
  2. 计算每个合约的 baseline_IV 和 residual
  3. 按到期月标准化得到 residual_zscore
  4. 流动性过滤, 剔除低流动性合约
  5. 按residual排序, 构建Bottom 10% Long / Top 10% Short
  6. 计算组合Delta, 50ETF对冲仓位, 空头保证金

T+1日收盘:
  7. 计算Delta-Hedged收益, 扣除成本
  8. 记录净收益, ROE, 持仓明细
  9. 信号衰减测试: residual_t vs T+1日收益
```

---

## 评估指标

### 收益指标
1. **日均净收益**
2. **年化收益**
3. **年化波动率**
4. **年化夏普比率**
5. **信息比率**
6. **ROE**

### 风险指标
7. **最大回撤**
8. **Calmar比率**

### 策略质量指标
9. **多头胜率 / 空头胜率**
10. **多空胜率差异**
11. **日均换手率**
12. **Delta暴露均值**

### 分箱分析
13. **按moneyness分箱**: ITM / ATM / OTM
14. **按remaining_time分箱**: 近月 / 中月 / 远月
15. **按call_put分箱**: Call vs Put
16. **按|residual|大小分箱**: 大偏差 vs 小偏差

### 特有指标(新增)
17. **信号衰减率**: (residual_t - residual_{t+1}) / residual_t
18. **隔夜衰减相关系数**: residual_t 与 overnight_return 的相关系数
19. **大|residual|合约收益**: 检验信号强度与收益关系
20. **流动性分箱收益**: 高流动性 vs 低流动性合约收益差异

---

## 输出文件要求

保存到 `/data/output/strategy_b_v2/`:

1. `backtest_results.csv`:
   - trade_date, gross_pnl, net_pnl, long_pnl, short_pnl
   - num_long, num_short, avg_residual_long, avg_residual_short
   - portfolio_delta, hedge_shares, hedge_cost
   - option_trade_cost, slippage_cost, total_margin, roe
   - cumulative_pnl, drawdown
   - signal_decay_rate, overnight_decay_corr

2. `daily_positions.csv`:
   - trade_date, security_id, call_put, exercise_price, position_type
   - weight, delta, residual, residual_zscore, baseline_iv, market_iv
   - volume, open_interest, moneyness, slippage_rate

3. `metrics.json`
4. `equity_curve.png`
5. `drawdown_curve.png`
6. `monthly_returns.png`
7. `pnl_decomposition.png`
8. `residual_signal_pnl.png`: 按|residual|分箱收益
9. `signal_decay_analysis.png`: 信号衰减分析(residual_t vs T+1收益散点图)
10. `liquidity_pnl.png`: 按流动性分箱收益
11. `moneyness_pnl.png`
12. `maturity_pnl.png`

---

## 代码结构规范

```python
def load_bspline_data(data_path, bspline_path):
    # 加载原始数据和B-Spline输出

def compute_baseline_and_residual(df):
    # 用M-W B-Spline计算baseline_IV和residual
    # 复用two_step_v2的fit_mw_surface和interpolate_baseline

def construct_residual_signal(df):
    # 构造residual信号和zscore标准化
    # 按到期月标准化

def apply_liquidity_filter(df, volume_pct=0.3, min_oi=100):
    # 流动性过滤

def analyze_signal_decay(df):
    # 信号衰减测试
    # residual_t vs T+1日收益的相关性
    # 返回衰减统计

def build_portfolio(df, signal_col='residual', long_pct=0.10, short_pct=0.10):
    # 构建Delta-Neutral Long-Short组合
    # residual越大=越高估=做空
    # 返回: long_df, short_df

def compute_delta_hedge(long_df, short_df):
    # 计算组合净Delta和50ETF对冲仓位

def get_slippage(moneyness_abs):
    # 阶梯滑点

def calc_margin(short_contracts, S, K, call_put, premium):
    # 计算空头保证金

def calculate_pnl(long_df, short_df, hedge_shares, df_next_day):
    # 计算Delta-Hedged收益和净收益

def run_backtest(df, start_date, end_date, fee_rate=0.0):
    # 主回测循环

def compute_metrics(backtest_df):
    # 计算夏普, 信息比率, 最大回撤, 衰减统计等

def plot_results(backtest_df, output_dir):
    # 绘制权益曲线, 回撤, 信号衰减分析等

def save_outputs(backtest_df, positions_df, metrics, output_dir):
    # 保存所有输出
```

---

## 检查点(必须打印)

```
[Checkpoint 1] B-Spline与信号加载
  - 原始数据记录数: {N}
  - B-Spline拟合成功天数: {success_days}
  - residual统计: mean={mean}, std={std}, min={min}, max={max}
  - residual分布: 应近似以0为中心对称

[Checkpoint 2] 信号构造与衰减测试
  - |residual_zscore| > 1.0 的合约比例: {pct}%
  - 各到期月residual标准差均值: {avg_std}
  - 流动性过滤后保留比例: {liq_pct}%
  - **信号衰减相关系数**: {decay_corr}(residual_t vs T+1收益)
  - **隔夜衰减率均值**: {decay_rate_mean}

[Checkpoint 3] 组合构建示例(某天)
  - Long合约数: {n_long}, 平均residual: {avg_long}(应为负)
  - Short合约数: {n_short}, 平均residual: {avg_short}(应为正)
  - 组合净Delta(对冲前): {raw_delta}
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

[Checkpoint 5] 信号有效性检验
  - residual与次日收益相关系数: {corr}
  - 大|residual|合约(Top 20%)日均收益: {big_residual_pnl}
  - 小|residual|合约(Bottom 20%)日均收益: {small_residual_pnl}
  - 高流动性合约日均收益: {high_liq_pnl}
  - 低流动性合约日均收益: {low_liq_pnl}

[Checkpoint 6] 分箱分析
  - ITM/ATM/OTM日均收益: {itm} / {atm} / {otm}
  - 近月/中月/远月日均收益: {near} / {mid} / {far}
  - Call/Put日均收益: {call} / {put}
```

---

## 成功标准

- [ ] **日均净收益 > 0**
- [ ] **年化夏普比率 > 0.5**
- [ ] **信息比率 > 0.3**
- [ ] **ROE > 0**
- [ ] **residual与次日收益相关系数 < -0.05**(高估合约确实下跌)
- [ ] **大|residual|合约收益显著优于小|residual|合约**
- [ ] **高流动性合约收益优于低流动性合约**(排除流动性幻觉)
- [ ] **Delta暴露均值 < 0.1**
- [ ] **最大回撤 < 20%**

---

## 备注

1. **信号衰减风险**: 本策略最大的不确定性是residual的隔夜衰减. 如果衰减相关系数接近0或为负, 说明T日收盘信号到T+1日开盘已失效, 策略可能无法盈利. 这是微观结构噪声的本质特征.
2. **B-Spline复用**: 直接复用two_step_v2的fit_mw_surface和interpolate_baseline函数.
3. **流动性约束**: 近月合约在最后3-5天Gamma极大, Bid-Ask Spread极宽. 流动性过滤可有效规避末日轮的高成本陷阱.
4. **与策略A对比**: 策略B的信号是截面的(同一天内不同合约比较), 策略A的信号是时序的(预测残差变化). 两者alpha来源不同.
5. **日度数据局限**: 我们只有日度数据, 无法执行盘中建仓. 如果信号衰减严重, 策略盈利可能仅限于理论层面. 在论文/专利中需明确标注这一局限性.
Strategy B V2 saved.