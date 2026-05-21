# Step 0: 单日 IV 曲面插值 —— DFW vs NW
# 复现 Zhang et al. (2023) Section 2.3

## 目标
对每一天的 50ETF 期权合约，用 DFW 多项式和 NW 核回归插值到固定 154 维网格，输出论文 Table 1 的 RMSE 对比。

---

## 输入数据格式

单文件 CSV，每行一个合约：
```csv
trade_date,call_put,exercise_price,remaining_time,implc_volatlty,fund_close
2024-01-02,C,2.200,44,0.185,2.412
2024-01-02,P,2.200,44,0.192,2.412
...
```

**必需列**：trade_date, call_put, exercise_price, remaining_time, implc_volatlty, fund_close

---

## 固定网格 I0（论文定义）

```python
import numpy as np

# Moneyness 网格（非均匀，ATM 附近密集）
m_grid = np.log(np.array([0.6, 0.8, 0.9, 0.95, 0.975, 1.0, 1.025, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0]))

# 期限网格（天）
tau_days_all = np.array([10, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730])

# 50ETF 适配：根据实际最大期限截断
def get_tau_grid(df):
    max_days = df["remaining_time"].max()
    tau_days = tau_days_all[tau_days_all <= max_days]
    return tau_days / 365.0

# 总网格点 = len(m_grid) * len(tau_grid) = 154（若 tau 全）或更少（若截断）
```

---

## 预处理：计算 Moneyness

```python
import pandas as pd
import numpy as np

def preprocess(df):
    # 简化：F ~ fund_close（忽略股息利率，或粗略估计）
    # 如需精确，用 Put-Call Parity 反推 F
    df["F"] = df["fund_close"]
    df["moneyness"] = np.log(df["exercise_price"] / df["F"])
    df["tau"] = df["remaining_time"] / 365.0
    return df
```

---

## 方法 A: DFW 多项式插值

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def dfw_interpolate(df_day, m_query, tau_query, floor=0.01):
    """
    sigma(m,tau) = max(floor, a0 + a1*m + a2*tau + a3*m^2 + a4*tau^2 + a5*m*tau)
    """
    m = df_day["moneyness"].values
    tau = df_day["tau"].values
    iv = df_day["implc_volatlty"].values

    X = np.column_stack([np.ones(len(m)), m, tau, m**2, tau**2, m*tau])
    model = LinearRegression(fit_intercept=False)
    model.fit(X, iv)

    Xq = np.column_stack([np.ones(len(m_query)), m_query, tau_query, 
                          m_query**2, tau_query**2, m_query*tau_query])
    iv_pred = Xq @ model.coef_
    return np.maximum(iv_pred, floor)
```

---

## 方法 B: NW 核回归（五折 CV 选带宽）

```python
from sklearn.model_selection import KFold

def nw_interpolate(df_day, m_query, tau_query, h1, h2):
    """
    sigma_hat(m,tau) = sum_i g((m-m_i)/h1, (tau-tau_i)/h2) * sigma_i / sum_i g(...)
    g(x,y) = (1/2pi) * exp(-x^2/2 - y^2/2)
    """
    m_obs = df_day["moneyness"].values
    tau_obs = df_day["tau"].values
    iv_obs = df_day["implc_volatlty"].values

    iv_pred = np.zeros(len(m_query))
    for j in range(len(m_query)):
        u = (m_query[j] - m_obs) / h1
        v = (tau_query[j] - tau_obs) / h2
        w = np.exp(-0.5 * (u**2 + v**2))
        s = w.sum()
        iv_pred[j] = (w * iv_obs).sum() / s if s > 1e-10 else iv_obs.mean()
    return iv_pred


def nw_cv_bandwidth(df_day, h1_grid, h2_grid, n_splits=5):
    """
    五折交叉验证选最优 (h1, h2)。
    论文原文："on each day we apply five-fold cross-validation"
    """
    m = df_day["moneyness"].values
    tau = df_day["tau"].values
    iv = df_day["implc_volatlty"].values

    if len(m) < n_splits:
        return 0.1, 0.1

    best_mse = float("inf")
    best_h1, best_h2 = h1_grid[0], h2_grid[0]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for h1 in h1_grid:
        for h2 in h2_grid:
            mse_list = []
            for train_idx, val_idx in kf.split(m):
                df_train = pd.DataFrame({"moneyness": m[train_idx], "tau": tau[train_idx], "implc_volatlty": iv[train_idx]})
                pred = nw_interpolate(df_train, m[val_idx], tau[val_idx], h1, h2)
                mse_list.append(mean_squared_error(iv[val_idx], pred))

            avg_mse = np.mean(mse_list)
            if avg_mse < best_mse:
                best_mse, best_h1, best_h2 = avg_mse, h1, h2

    return best_h1, best_h2
```

---

## 主流程：逐日插值 + RMSE 对比

```python
def run_step0(df_raw):
    df = preprocess(df_raw)
    tau_grid = get_tau_grid(df)

    # 构建完整网格坐标
    M_grid, Tau_grid = np.meshgrid(m_grid, tau_grid, indexing="ij")
    m_flat = M_grid.ravel()
    tau_flat = Tau_grid.ravel()

    results = []
    grid_data = []

    for date in sorted(df["trade_date"].unique()):
        df_day = df[df["trade_date"] == date]
        if len(df_day) < 6:
            continue

        # DFW
        iv_dfw = dfw_interpolate(df_day, m_flat, tau_flat)
        rmse_dfw = np.sqrt(mean_squared_error(
            df_day["implc_volatlty"],
            dfw_interpolate(df_day, df_day["moneyness"], df_day["tau"])
        ))

        # NW（CV 选带宽）
        h1_grid = np.linspace(0.02, 0.30, 10)
        h2_grid = np.linspace(0.02, 0.30, 10)
        h1_best, h2_best = nw_cv_bandwidth(df_day, h1_grid, h2_grid)
        iv_nw = nw_interpolate(df_day, m_flat, tau_flat, h1_best, h2_best)
        rmse_nw = np.sqrt(mean_squared_error(
            df_day["implc_volatlty"],
            nw_interpolate(df_day, df_day["moneyness"], df_day["tau"], h1_best, h2_best)
        ))

        results.append({
            "date": date, "n": len(df_day),
            "dfw_rmse": rmse_dfw, "nw_rmse": rmse_nw,
            "h1": h1_best, "h2": h2_best
        })

        # 保存 154 维网格（DFW 版本，用于后续步骤）
        grid_data.append({
            "date": date, "m_grid": m_grid, "tau_grid": tau_grid,
            "iv_dfw": iv_dfw, "iv_nw": iv_nw
        })

    # 输出 Table 1
    res_df = pd.DataFrame(results)
    print("=" * 50)
    print("Table 1: Average RMSE for NW and DFW")
    print("=" * 50)
    print(f"DFW: {res_df['dfw_rmse'].mean():.4f}")
    print(f"NW:  {res_df['nw_rmse'].mean():.4f}")
    print("=" * 50)

    return res_df, grid_data
```

---

## 输出文件

```python
# 1. 每日 RMSE 对比（Table 1）
res_df.to_csv("/data/output/step0/table1_nw_dfw.csv", index=False)

# 2. 每天 154 维 DFW 网格（Step 1 的输入）
grid_records = []
for g in grid_data:
    for i in range(len(g["iv_dfw"])):
        grid_records.append({
            "trade_date": g["date"],
            "grid_idx": i,
            "m": g["m_grid"][i % len(m_grid)],
            "tau": g["tau_grid"][i // len(m_grid)],
            "iv_dfw": g["iv_dfw"][i],
            "iv_nw": g["iv_nw"][i]
        })
grid_df = pd.DataFrame(grid_records)
grid_df.to_parquet("/data/output/step0/daily_grid_154.parquet")
```

---

## 检查点（必须打印）

```
[Checkpoint 1] 数据加载
  - 总交易日: {n_days}
  - 每日平均合约数: {avg_contracts}
  - moneyness 范围: [{m_min:.3f}, {m_max:.3f}]
  - tau 范围: [{tau_min:.3f}, {tau_max:.3f}] 年

[Checkpoint 2] 网格构建
  - m_grid 点数: {len(m_grid)}
  - tau_grid 点数: {len(tau_grid)}
  - 总网格点: {len(m_grid)*len(tau_grid)} (论文 154)

[Checkpoint 3] DFW 插值
  - 平均 RMSE: {dfw_rmse_mean:.4f}
  - 中位数 RMSE: {dfw_rmse_median:.4f}

[Checkpoint 4] NW 插值（五折 CV）
  - 平均 RMSE: {nw_rmse_mean:.4f}
  - 中位数 RMSE: {nw_rmse_median:.4f}
  - 平均最优带宽 h1: {h1_mean:.3f}
  - 平均最优带宽 h2: {h2_mean:.3f}

[Checkpoint 5] 输出文件
  - table1_nw_dfw.csv: {n_rows} 行
  - daily_grid_154.parquet: {n_grid_points} 个网格点
```

---

## 执行顺序

1. 加载 50etf_options.csv
2. preprocess() 计算 moneyness 和 tau
3. run_step0() 逐日插值
4. 检查 Table 1 RMSE 是否合理（预期 DFW < NW，或接近论文 0.018 vs 0.026）
5. 保存 daily_grid_154.parquet（供 Step 1 使用）

**下一步**：拿到此输出后，再进入 Step 1（特征提取 SAM/PCA/VAE）。
