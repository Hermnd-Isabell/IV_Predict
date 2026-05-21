# 任务：复现论文 Figure 7 —— 每日 RMSE/MAPE 时间序列分析
# 基于 Zhang et al. (2023) Section 4.3 的评估方法

## 目标

加载 Step 2 DNN 模型的逐日评估结果，完成以下工作：
1. 绘制论文 Figure 7 风格的每日 RMSE 和 MAPE 时间序列图（三种模型对比）
2. 识别极端误差日期（Top 20 RMSE/MAPE）
3. 将极端误差日期与 2002-2004 年重大历史事件对照，分析误差飙升是否对应特定市场事件
4. 输出统计摘要（均值/中位数/标准差/95%分位数/超过2×均值的天数比例）

---

## 输入数据

Step 2 输出的评估结果文件，路径：
```
/data/output/step2/results_sam.npz
/data/output/step2/results_pca.npz
/data/output/step2/results_vae.npz
```

每个 npz 文件必须包含以下数组：
- `rmse_daily`: 每日 RMSE 数组，shape=(n_test_days,)
- `mape_daily`: 每日 MAPE 数组，shape=(n_test_days,)
- `dates_test`: 测试日期数组，shape=(n_test_days,)，可以是字符串或 datetime

**如果 npz 中没有 `dates_test`**：
- 请从 Step 1 的 `sam_features.npz` 中提取 `dates_test`
- 或者根据训练/验证/测试划分比例推算测试期日期范围

---

## 2002-2004 年重大历史事件对照表

以下事件已内置在代码中，用于误差峰值对照：

| 日期 | 事件 | 市场影响 |
|------|------|---------|
| 2002-07-10 | WorldCom 破产（美国最大企业破产案）| 会计丑闻引发市场恐慌，VIX 飙升 |
| 2002-08-05 | VIX 飙升至 45+ | 极端恐慌，期权 IV 剧烈波动 |
| 2002-10-09 | SPX 跌至 776（熊市低点）| 9/11 后熊市底部，IV 极高 |
| 2003-03-20 | 伊拉克战争爆发 | 地缘政治冲击，IV 脉冲式上升 |
| 2003-05-01 | 布什"任务完成"演讲 | 战争预期缓和，IV 回落 |
| 2003-07-15 | 伊拉克战争后 VIX 持续回落 | 市场进入复苏期，微笑结构简化 |
| 2003-10-27 | SPX 突破 1050 | 复苏确认，低波动期开始 |
| 2004-03-20 | 马德里爆炸案 | 欧洲恐怖主义，短暂冲击 |
| 2004-05-01 | Abu Ghraib 丑闻曝光 | 伊拉克局势恶化预期 |
| 2004-06-30 | 伊拉克主权移交 | 政治不确定性 |
| 2004-07-30 | 9/11 委员会报告发布 | 安全担忧 |
| 2004-10-08 | SPX 突破 1120 | 大选前市场乐观 |
| 2004-11-02 | 美国总统大选（布什 vs 克里）| 政治不确定性峰值 |

**论文的对应发现**：论文 Figure 7 显示 2020 年 3 月（COVID）所有模型误差飙升。你的数据对应 2002-2004 年的类似冲击事件。

---

## 实现步骤

### Step 1: 加载数据

```python
import numpy as np
import pandas as pd

def load_results(feature_type):
    data = np.load(f"/data/output/step2/results_{feature_type.lower()}.npz", allow_pickle=True)
    rmse = data["rmse_daily"]
    mape = data["mape_daily"]
    dates = data.get("dates_test", None)
    if dates is None:
        # 从 Step 1 补日期
        d1 = np.load("/data/output/step1/sam_features.npz", allow_pickle=True)
        dates = d1["dates_test"]
    dates = pd.to_datetime([str(d) for d in dates])
    return dates, rmse, mape

# 加载三种模型
for ft in ["SAM", "PCA", "VAE"]:
    dates, rmse, mape = load_results(ft)
    # 构建 DataFrame
```

### Step 2: 构建统一 DataFrame

将三种模型的每日误差合并到一个 DataFrame：
```
trade_date | model | rmse | mape
2004-01-02 | SAM   | 0.045 | 0.312
2004-01-02 | PCA   | 0.042 | 0.298
2004-01-02 | VAE   | 0.043 | 0.305
...
```

### Step 3: 绘制 Figure 7

复现论文 Figure 7 风格：

**子图 (a): 每日 RMSE**
- X 轴：日期（2002-2004）
- Y 轴：RMSE
- 三条线：SAM-DNN（红色）、PCA-DNN（绿色）、VAE-DNN（蓝色）
- 标注：网格线、图例、月份刻度
- 标题："(a) Daily RMSE in the Test Period"

**子图 (b): 每日 MAPE**
- 同上，Y 轴为 MAPE
- 标题："(b) Daily MAPE in the Test Period"

**样式要求**：
- 图大小：14×10 英寸
- DPI：300
- 颜色：SAM=红色(#d62728)，PCA=绿色(#2ca02c)，VAE=蓝色(#1f77b4)
- X 轴日期格式："%Y-%m"，每 2 个月一个主刻度

### Step 4: 极端误差分析

对每种模型，找出 RMSE 和 MAPE 最高的 **Top 20** 日期：

```python
for model in ["SAM", "PCA", "VAE"]:
    sub = df[df["model"] == model]
    top_rmse = sub.nlargest(20, "rmse")[["trade_date", "rmse", "mape"]]
    top_mape = sub.nlargest(20, "mape")[["trade_date", "rmse", "mape"]]
```

输出 CSV：`/data/output/step2/extreme_error_dates.csv`

### Step 5: 历史事件对照

对每个历史事件，检查事件日期前后 **±3 天** 窗口内是否有误差峰值：

```python
events = {
    "2002-07-10": "WorldCom 破产",
    "2002-10-09": "SPX 熊市低点",
    "2003-03-20": "伊拉克战争爆发",
    # ... 完整列表见上文
}

for event_date, event_name in events.items():
    ed = pd.to_datetime(event_date)
    mask = (df["trade_date"] >= ed - pd.Timedelta(days=3)) &            (df["trade_date"] <= ed + pd.Timedelta(days=3))
    window = df[mask]
    # 取窗口内 RMSE 最高的记录
```

输出 CSV：`/data/output/step2/error_event_correlation.csv`

### Step 6: 统计摘要

对每种模型打印：
```
SAM-DNN RMSE:
  均值: {mean:.4f}
  中位数: {median:.4f}
  标准差: {std:.4f}
  最小值: {min:.4f} (日期: {date})
  最大值: {max:.4f} (日期: {date})
  95% 分位数: {p95:.4f}
  超过 2×均值的天数: {n} / {total} ({pct}%)

  [Top 5 极端日期]
  1. 2002-07-15: RMSE=0.152, MAPE=0.89
  2. ...
```

---

## 输出文件清单

| 文件 | 路径 | 内容 |
|------|------|------|
| Figure 7 | `/data/output/step2/figure7_daily_rmse_mape.png` | 每日 RMSE/MAPE 时间序列 |
| 极端误差 | `/data/output/step2/extreme_error_dates.csv` | Top 20 RMSE/MAPE 日期 |
| 事件对照 | `/data/output/step2/error_event_correlation.csv` | 历史事件窗口内误差峰值 |

---

## 检查点（必须打印）

```
[Checkpoint 1] 数据加载
  SAM: {n_sam} 天, PCA: {n_pca} 天, VAE: {n_vae} 天
  测试期范围: {start_date} ~ {end_date}

[Checkpoint 2] Figure 7 绘制
  保存路径: /data/output/step2/figure7_daily_rmse_mape.png

[Checkpoint 3] 极端误差分析
  SAM Top 1 RMSE: {date} = {rmse:.4f}
  PCA Top 1 RMSE: {date} = {rmse:.4f}
  VAE Top 1 RMSE: {date} = {rmse:.4f}

[Checkpoint 4] 历史事件对照
  事件窗口覆盖天数: {n_covered}
  事件窗口内 RMSE > 2×均值的天数: {n_extreme}
  典型事件-误差关联:
    - {event_name} ({event_date}): 窗口内最高 RMSE = {rmse:.4f} ({model})

[Checkpoint 5] 统计摘要
  SAM: mean={mean:.4f}, median={median:.4f}, std={std:.4f}, max={max:.4f}
  PCA: mean={mean:.4f}, median={median:.4f}, std={std:.4f}, max={max:.4f}
  VAE: mean={mean:.4f}, median={median:.4f}, std={std:.4f}, max={max:.4f}
```

---

## 预期发现（参考）

基于你目前 RMSE ~0.07 的结果，预期：

1. **2002 年 7-10 月**（WorldCom + 熊市低点）：RMSE 可能飙升至 0.12-0.18
2. **2003 年 3 月**（伊拉克战争）：短期脉冲式上升至 0.10+
3. **2003 年下半年**：市场复苏，RMSE 趋于稳定（0.04-0.06）
4. **2004 年大选前后**：政治不确定性，RMSE 小幅上升至 0.08-0.10

如果误差峰值与上述事件不吻合，说明模型误差主要由**随机噪声**驱动而非**事件冲击**，这将进一步支持"小样本下 LSTM 过拟合"的判断。

---

## 执行顺序

1. 确认 `results_sam.npz`、`results_pca.npz`、`results_vae.npz` 存在且包含 `rmse_daily`、`mape_daily`
2. 如果缺少 `dates_test`，从 Step 1 补日期
3. 加载三种模型结果，构建统一 DataFrame
4. 绘制 Figure 7 并保存 PNG
5. 提取每种模型 Top 20 极端误差日期，保存 CSV
6. 对照历史事件表，分析事件窗口内误差峰值，保存 CSV
7. 打印统计摘要和检查点

**请先确认 Step 2 的 npz 文件中是否包含 `rmse_daily`、`mape_daily` 和 `dates_test`，如果缺少日期数组，先从 Step 1 补全后再执行。**
