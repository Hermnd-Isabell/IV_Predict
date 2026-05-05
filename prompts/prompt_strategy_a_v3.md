# 策略A（v3）: XGBoost 预测动量（残差不可预测修订版）

## 核心前提
残差不可预测（SNR=0.97，RMSE 0.0098 > std 0.0071）。策略放弃"预测残差变化"，改为"XGBoost 预测绝对 IV 与 B-Spline baseline 的偏离"。

## 信号定义
```python
pred_residual = pred_IV_{t+1} - baseline_IV_{t+1}
```
- `pred_IV_{t+1}`：XGBoost 逐合约预测 T+1 日 IV（复用原方案 `model_abs_iv.pkl`）
- `baseline_IV_{t+1}`：用 T 日 B-Spline 曲面插值到 T+1 日 `(M_{t+1}, tau_{t+1})` 的 baseline
- `pred_residual`：XGBoost 预测的 T+1 日定价偏差

**交易逻辑**：
- `pred_residual > 0`：XGBoost 预测相对高估 → **做空(Short)**
- `pred_residual < 0`：XGBoost 预测相对低估 → **做多(Long)**

## 组合构建
1. 每天收盘后，计算所有合约的 `pred_residual`
2. 流动性过滤：`volume > 到期月中位数×0.3`，`OI > 100`，`remaining_time > 3`
3. 按 `pred_residual` 排序：
   - 多头：Bottom 10%（最负 = 预测最低估）
   - 空头：Top 10%（最正 = 预测最高估）
4. Delta 对冲：组合净 Delta 用 50ETF 现货对冲至 |net_delta| < 0.05
5. Vega 监控：记录但不强制中性

## 收益计算
```python
option_pnl = (close_{t+1} - close_t) - delta_t * (fund_close_{t+1} - fund_close_t)
gross_pnl = sum(long_pnl) - sum(short_pnl)
```

## 阶梯滑点
| Moneyness | 滑点率 |
|-----------|--------|
| ATM (\|M\|<0.05) | 0.05% |
| 轻度虚值 (0.05≤\|M\|<0.15) | 0.10% |
| 深度虚值 (\|M\|≥0.15) | 0.25% |

## 输出
- `backtest_results.csv`：每日收益、Delta/Vega 暴露、保证金、ROE
- `daily_positions.csv`：持仓明细
- 图表：权益曲线、回撤、Vega 暴露、分箱收益

## 成功标准
- 日均净收益 > 0
- 年化夏普 > 0.5
- Delta 暴露均值 < 0.1
- Vega-收益相关系数 < 0.3
