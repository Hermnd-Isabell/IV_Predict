# 策略C（v3）: B-Spline 显著偏离筛选（严格版均值回归）

## 核心前提
残差不可预测，XGBoost 方向过滤无意义。策略退化为"纯 B-Spline 显著偏离池内的截面排序"。

## 信号定义（两步筛选）

**Step 1：显著偏离池**
```python
pool = {contracts: |residual_zscore| > 1.5}
```
- 只保留定价偏差超过 1.5 个标准差的合约
- 这些合约的 market IV 与 baseline IV 偏离"明显离谱"

**Step 2：池内排序**
```python
# 在 pool 内按 residual 排序
long = pool[residual < 0].nsmallest(int(len(pool)*0.1), 'residual')  # 最低估
short = pool[residual > 0].nlargest(int(len(pool)*0.1), 'residual')   # 最高估
```

## 组合构建
1. 每天计算所有合约的 `residual` 和 `residual_zscore`
2. 第一层过滤：`\|zscore\| > 1.5`（显著偏离池）
3. 第二层过滤：流动性过滤（volume/OI/remaining_time）
4. 在池内按 `residual` 排序取 Top/Bottom 10% 多空
5. Delta 对冲至 |net_delta| < 0.05

## 废弃旧逻辑
- ❌ 废弃 XGBoost 方向一致性过滤（`pred_residual` 与 `residual` 方向比较）
- ❌ 废弃 `model_residual_v2.pkl`
- ✅ 只依赖 `baseline_IV` 和 `residual`

## 与策略B的区别
| | 策略B | 策略C |
|---|---|---|
| 筛选 | 全截面排序 | 先筛显著偏离池，再排序 |
| 交易频率 | 高（每天~20%合约） | 低（可能每天~5-10%合约） |
| 信号强度 | 弱 | 强（只交易"明显离谱"的） |
| 本质 | 宽松均值回归 | 严格均值回归 |

## 输出
同策略B，增加：
- 每日通过 `\|zscore\|>1.5` 的合约数时间序列
- 池内平均 `\|residual\|` 变化

## 成功标准
- 日均净收益 > 0
- 年化夏普 > 策略B（证明严格筛选有价值）
- 若夏普 < 策略B，说明筛选过度，丢失了有效信号
- Delta 暴露均值 < 0.1
