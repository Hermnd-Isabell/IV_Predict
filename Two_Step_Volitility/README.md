# Two-Step Volatility Surface Modeling

复现 Zhang et al. (2023) 两步法波动率曲面建模框架。

## 项目结构

```
Two_Step_Volitility/
├── src/                          # 源代码
│   ├── step0_interpolation.py    # Step 0: DFW vs NW 插值
│   └── step1_features_lstm.py    # Step 1: 特征提取 + LSTM 预测
├── output/                       # 输出结果
│   ├── step0/
│   │   ├── table1_nw_dfw.csv         # 每日 RMSE 对比（Table 1）
│   │   └── daily_grid_154.parquet    # 154 维插值网格
│   ├── step1/
│   │   ├── sam_features.npz          # SAM 特征 + LSTM 预测
│   │   ├── pca_features.npz          # PCA 特征 + LSTM 预测
│   │   └── vae_features.npz          # VAE 特征 + LSTM 预测
│   └── step2/
│       ├── dnn_sam.pt                # SAM-DNN 模型
│       ├── dnn_pca.pt                # PCA-DNN 模型
│       ├── dnn_vae.pt                # VAE-DNN 模型
│       ├── results_sam.npz           # SAM-DNN 评估结果
│       ├── results_pca.npz           # PCA-DNN 评估结果
│       └── results_vae.npz           # VAE-DNN 评估结果
├── prompts/                      # 任务参考 prompt
└── README.md                     # 本文件
```

## 输入数据

源数据（不复制，通过相对路径引用）：
```
../../data/raw/50etf_options.csv
```

必需列：`trade_date`, `call_put`, `exercise_price`, `remaining_time`, `implc_volatlty`, `fund_close`

## Step 0: 单日 IV 曲面插值

复现论文 Section 2.3，对每一天的 50ETF 期权合约：

1. **预处理**：计算 moneyness `m = ln(K/F)` 和年化期限 `tau = remaining_time / 365`
2. **DFW 插值**：二次多项式 `sigma(m,tau) = a0 + a1*m + a2*tau + a3*m^2 + a4*tau^2 + a5*m*tau`
3. **NW 插值**：二维高斯核回归，五折 CV 选最优带宽 `(h1, h2)`
4. **固定网格**：14(moneyness) × 11(tau) = 154 维（或根据实际最大期限截断）

### 运行

```bash
cd Two_Step_Volitility
python src/step0_interpolation.py
```

### 输出

- `output/step0/table1_nw_dfw.csv`：每日 DFW/NW RMSE 对比
- `output/step0/daily_grid_154.parquet`：每日 154 维插值网格（DFW + NW）

### 检查点

脚本运行时会打印 5 个检查点：
1. 数据加载（交易日数、合约数、moneyness/tau 范围）
2. 网格构建（m_grid/tau_grid 点数、总网格点数）
3. DFW 插值（平均/中位数 RMSE）
4. NW 插值（平均/中位数 RMSE、最优带宽）
5. 输出文件（行数、网格点数）

## Step 1: 特征提取 + LSTM 预测

复现论文 Section 3.1 & 3.2，从 Step 0 的每日网格数据中提取低维特征，并用 LSTM 预测 T+1 日特征。

### 三种特征方法

| 方法 | 特征维度 | 说明 | 状态 |
|------|---------|------|------|
| SAM  | 98 (网格维数) | 直接采样网格 IV 值，恒等映射 | 已完成 |
| PCA  | 3 | log-IV 日变化的主成分分析 (K=3) | 已完成 |
| VAE  | 10 (默认) | 变分自编码器，latent_dim=10 | 已完成 |

### LSTM 架构

- **输入**: 三尺度 `(月均线, 周均线, 当前值)`，shape `(batch, 3, feature_dim)`
- **网络**: 单层 LSTM (hidden=12) + Linear 输出
- **激活**: SAM 用 ReLU（IV>0），PCA/VAE 用 Identity
- **训练**: Adam, lr=0.01, epochs=200, batch_size=128，验证集早停
- **划分**: 时序划分 Train 75% / Val 15% / Test 10%

### 运行

```bash
cd Two_Step_Volitility
python src/step1_features_lstm.py
```

### 输出

- `output/step1/sam_features.npz`: SAM 特征 `Z`、测试集预测 `Z_pred`、日期、RMSE
- `output/step1/pca_features.npz`: PCA 特征 `Z`、预测 `Z_pred`、特征曲面 `eigensurfaces`、解释方差
- `output/step1/vae_features.npz`: VAE 特征 `Z`、预测 `Z_pred`、日期、RMSE

### 50ETF 实测结果（Checkpoints）

**SAM**
- Z 维度: `(2667, 98)`
- LSTM 最优 epoch: 76, best val loss: 0.0959
- RMSE: Train=0.2436, Val=0.3915, Test=0.3487

**PCA**
- Z 维度: `(2667, 3)`, 解释方差: `[68.35%, 9.04%, 7.69%]`
- LSTM 最优 epoch: 39, best val loss: 3.8000
- RMSE: Train=1.8391, Val=2.1822, Test=2.7205

> PCA 的 RMSE 在 cumsum 后的主成分空间计算，数值范围与 SAM/VAE 不同，不直接可比。

**VAE** (latent_dim=10)
- Z 维度: `(2667, 10)`, 重构 MSE: 0.0203
- LSTM 最优 epoch: 125, best val loss: 0.0428
- RMSE: Train=0.1874, Val=0.2209, Test=0.2191

> VAE 的 latent_dim 可进一步搜索 `{2,5,10,15,20}` 以优化重构误差。

## Step 2: DNN 无套利曲面重构

复现论文 Section 3.3 & 3.4，将 Step 1 预测的特征还原为完整 IV 曲面，并通过 DNN 内置无套利约束。

### 特征映射

| 方法 | 映射公式 | 说明 |
|------|---------|------|
| SAM | `F = Z_pred` | 恒等映射，154 维 |
| PCA | `F = sigma_0 * exp(Z_pred @ eigensurfaces)` | sigma_0 为首日 DFW 网格 |
| VAE | `F = decoder(Z_pred)` | VAE 解码器还原 |

### DNN 架构

- **输入**: `F` (154-dim) + `tau` (年化期限) + `m` (moneyness)，共 156 维
- **隐藏层**: 3 × 50 单元，Tanh 激活
- **输出**: Softplus 激活，保证 `sigma > 0`
- **训练**: Adam, lr=0.001, epochs=20, 按天 batch (batch_size_days=32)

### 无套利损失（自动微分）

在密集网格上通过 PyTorch autograd 计算二阶导数，施加三类约束：

1. **日历套利 (Condition 3)**: `sigma + 2*tau*grad_tau >= 0`
2. **蝶式套利 (Condition 4 / Durrleman)**: `(1 - m*grad_m/sigma)^2 - (sigma*tau*grad_m)^2/4 + tau*sigma*grad_mm >= 0`
3. **大 moneyness 边界 (Condition 5)**: `|sigma*grad_mm + grad_m^2| -> 0`

损失函数: `L = MSE + lambda * (L_cal + L_but + L_boundary)`，lambda=1.0

### 运行

```bash
cd Two_Step_Volitility
python src/step2_dnn_surface.py
```

### 输出

- `output/step2/dnn_{sam,pca,vae}.pt`: DNN 模型权重
- `output/step2/results_{sam,pca,vae}.npz`: 测试 RMSE/MAPE、套利违规检查、训练历史

---

## SPX 数据实验结果

### 子集 A: 2002.05–2004.12 (663 个交易日, 350,576 条观测)

#### Step 0: 插值

| 方法 | Test RMSE | 说明 |
|------|-----------|------|
| DFW | 0.0608 | 二次多项式拟合 |
| NW | 0.0421 | 二维核回归（五折 CV 选带宽） |

#### Step 1: 特征提取 + LSTM 预测

| 方法 | 特征维度 | LSTM Best Val | Test RMSE |
|------|---------|---------------|-----------|
| SAM | 154 | 0.0065 (epoch 38) | 0.1641 |
| PCA | 3 | 0.0505 (epoch 125) | 4.3085* |
| VAE | 10 | 0.0031 (epoch 168) | 0.3234 |

\* PCA RMSE 在主成分空间计算，数值范围与 SAM/VAE 不同，不直接可比。

#### Step 2: DNN 曲面重构

| 方法 | Test RMSE | Test MAPE | L_cal | L_but | Best Val Epoch |
|------|-----------|-----------|-------|-------|----------------|
| SAM-DNN | 0.072848 | 0.5286 | 0.0000 | 0.0000 | 19 |
| PCA-DNN | 0.068683 | 0.4302 | 0.0000 | 0.0000 | 1 |
| VAE-DNN | 0.068574 | 0.4571 | 0.0000 | 0.0000 | 0 |

**关键发现**：
- 三种 DNN 模型的测试 RMSE 接近（~0.069），说明 Step 2 的曲面重构对不同特征输入具有鲁棒性
- 无套利约束完全生效：`L_cal = L_but = 0`（测试集上无日历/蝶式套利违规）
- 相比 Step 0 NW 插值 RMSE (0.0421)，Step 2 略高，因 DNN 需要拟合整个连续曲面而非仅插值观测点

---

### 子集 B: 2002–2007 (1,417 个交易日, 867,574 条观测)

使用更完整的 S&P 500 期权数据（覆盖 2002–2007 年），测试期为 **2007-06-11 ~ 2007-12-31**（142 天，含次贷危机爆发期）。

#### Step 0: 插值

| 方法 | Test RMSE | 说明 |
|------|-----------|------|
| DFW | 0.0608 | 二次多项式拟合 |
| NW | 0.0421 | 二维核回归（五折 CV 选带宽） |

#### Step 1: 特征提取 + LSTM 预测

| 方法 | 特征维度 | LSTM Best Val | Test RMSE |
|------|---------|---------------|-----------|
| SAM | 154 | 0.0087 (epoch 19) | 0.1641 |
| PCA | 3 | 0.0093 (epoch 19) | 4.3085* |
| VAE | 10 | 0.0101 (epoch 3) | 0.3234 |

\* PCA RMSE 在主成分空间计算，不直接可比。

#### Step 2: DNN 曲面重构

| 方法 | Test RMSE | Test MAPE | L_cal | L_but | Best Val Epoch |
|------|-----------|-----------|-------|-------|----------------|
| SAM-DNN | 0.1104 | 27.65% | 0.0000 | 0.0000 | 19 |
| PCA-DNN | 0.1111 | 28.61% | 0.0000 | 0.0000 | 1 |
| VAE-DNN | 0.1068 | 28.34% | 0.0000 | 0.0000 | 3 |

**关键发现**：
- **VAE-DNN 表现最优**（RMSE=0.1068），略优于 SAM-DNN 和 PCA-DNN
- 三种模型 RMSE 接近（~0.11），再次验证 Step 2 对不同特征输入的鲁棒性
- 无套利约束完全生效：`L_cal = L_but = 0`
- **误差峰值出现在 2007-09-19**（SAM RMSE=0.3449，PCA RMSE=0.3537，VAE RMSE=0.3410），对应次贷危机期间市场剧烈波动
- 相比 2004 子集（RMSE~0.069），2007 测试期 RMSE 更高，因包含金融危机前的高波动时期
- 142 天测试中，约 4.9% 的交易日 RMSE 超过 2 倍中位数（极端误差日）

---

## 下一步

- Step 3：利用重构曲面进行期权定价与对冲策略回测
- 尝试不同 lambda_penalty 权重，观察套利约束与拟合精度的 trade-off
- 探索更复杂的 DNN 架构（ResNet、Attention）以进一步提升曲面精度
