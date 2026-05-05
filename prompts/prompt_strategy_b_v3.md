# 策略B（v3）: B-Spline 纯截面均值回归（残差不可预测修订版）

## 核心前提
残差不可预测（SNR=0.97）。放弃"预测残差变化"，只做**当前截面的定价偏差排序**，赌 residual 向 baseline 回归。

## 信号定义
```python
residual = market_IV - baseline_IV  # 当前定价偏差
residual_zscore = (residual - 到期月均值) / 到期月标准差
```

**交易逻辑**：
- `residual > 0`（或 `zscore > 0`）：相对高估 → **做空(Short)**，赌向 baseline 回归
- `residual < 0`（或 `zscore < 0`）：相对低估 → **做多(Long)**，赌向 baseline 回归

## 组合构建
1. 每天收盘后，用 M-W B-Spline 计算所有合约的 `baseline_IV` 和 `residual`
2. 按到期月计算 `residual_zscore`
3. 流动性过滤：`volume > 到期月中位数×0.3`，`OI > 100`，`remaining_time > 3`
4. 按 `residual` 排序：
   - 多头：Bottom 10% residual（最负 = 最低估）
   - 空头：Top 10% residual（最正 = 最高估）
5. Delta 对冲至 |net_delta| < 0.05

## 关键：不预测，只排序
- 不加载 `model_residual_v2.pkl`
- 不使用 `pred_residual`
- 信号就是今天的 `residual`，赌明天向 0 回归

## 阶梯滑点
同策略A。

## 信号衰减分析（必须输出）
```python
# residual_t 与 T+1 日 delta-hedged 收益的相关性
ic = residual_t.corr(pnl_{t+1})
overnight_decay = (residual_t - residual_{t+1}) / residual_t
```
在回测报告中标注："策略基于日度快照，若隔夜衰减率 > 50%，信号可能已失效。"

## 输出
同策略A，增加：
- `signal_decay_analysis.png`：residual_t 与次日收益散点图
- `residual_signal_pnl.png`：按 \|residual\| 分箱收益

## 成功标准
- 日均净收益 > 0
- 年化夏普 > 0.5
- residual 与次日收益相关系数 < -0.05（高估确实下跌）
- 大 \|residual\| 合约收益显著优于小 \|residual\|
- Delta 暴露均值 < 0.1
