# 任务：ResNet 验证 —— 替换 Step 2 MLP 曲面重构模型

## 目标

在已完成的模块化架构基础上，实现 **ResNetSurface** 替换 **MLPSurface (DNN_Surface)**，运行完整三步流水线，验证 Step 2 架构切换能否打破 0.11 RMSE 瓶颈。

## 背景与动机

GRU 验证揭示的关键洞察：
- Step 1 (GRU) Test RMSE: 0.0918（比 LSTM 提升 44%）
- Step 2 (MLP) Test RMSE: 0.1102（几乎不变）

**结论**：MLP 重构层是信息瓶颈——它通过 Softplus + 无套利惩罚强制平滑，抹平了 Step 1 的高频信号。

**ResNet 的理论优势**：
- **跳跃连接（Skip Connection）**：允许梯度直接传播，保留深层网络中的特征信息
- **残差学习**：学习 "F(x) = H(x) - x" 而非直接学习 H(x)，更容易优化
- **预期效果**：更好地保留 Step 1 预测的 154 维离散 IV 点信息，降低插值损失

---

## Step 1: 实现 ResNetSurface

### 文件：`src/models/surfaces/resnet.py`

```python
import torch
import torch.nn as nn
from ..base import BaseSurface

class ResidualBlock(nn.Module):
    """
    ResNet 残差块：
    x -> Linear -> Activation -> Linear -> (+ x) -> Activation
    """
    def __init__(self, dim: int, activation: str = "tanh"):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(0.1)
        else:
            self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        out = out + residual  # 跳跃连接
        out = self.act(out)
        return out


class ResNetSurface(BaseSurface):
    """
    ResNet 曲面重构模型。
    输入: F (n_grid维) + tau (1维) + m (1维)
    输出: sigma (1维), 通过 Softplus 保证非负
    """
    def __init__(self, input_dim: int, 
                 hidden_dims: list = [50, 50, 50], 
                 num_blocks: int = 3,
                 activation: str = "tanh",
                 output_activation: str = "softplus"):
        super().__init__(input_dim, output_activation)

        self.hidden_dims = hidden_dims
        self.num_blocks = num_blocks

        # 输入层: input_dim -> hidden_dims[0]
        self.input_fc = nn.Linear(input_dim, hidden_dims[0])

        # 残差块堆叠
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[0], activation)
            for _ in range(num_blocks)
        ])

        # 输出层: hidden_dims[-1] -> 1
        self.output_fc = nn.Linear(hidden_dims[-1], 1)

        # 激活函数
        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(0.1)
        else:
            self.act = nn.Tanh()

    def forward(self, F: torch.Tensor, tau: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        参数:
            F: (batch, n_grid) 预测的离散 IV 点
            tau: (batch, 1) 查询期限
            m: (batch, 1) 查询 moneyness
        返回:
            (batch, 1) IV 值
        """
        # 拼接输入
        x = torch.cat([F, tau, m], dim=1)  # (batch, input_dim)

        # 输入层
        x = self.act(self.input_fc(x))

        # 残差块
        for block in self.residual_blocks:
            x = block(x)

        # 输出层 + Softplus
        x = self.output_fc(x)
        return self.activation(x)
```

**关键设计点**：
- `ResidualBlock` 保持维度不变（dim -> dim），确保跳跃连接 `x + residual` 可行
- `num_blocks` 控制残差层深度（建议 2-4 层）
- `hidden_dims` 控制宽度（建议 [50, 50, 50] 或 [64, 64, 64]）
- 与 MLP 的接口完全一致：`forward(F, tau, m) -> sigma`

---

## Step 2: 注册 ResNet 到模型注册表

### 修改 `src/models/__init__.py`

```python
from .surfaces.mlp import MLPSurface
from .surfaces.resnet import ResNetSurface

STEP2_MODELS = {
    "mlp": MLPSurface,
    "resnet": ResNetSurface,
    "kan": KANSurface,  # 预留
}
```

---

## Step 3: 创建 ResNet 配置文件

### 文件：`config_resnet.json`

```json
{
    "step0": {
        "interpolation_method": "NW",
        "use_synthetic": false,
        "data_path": "/data/raw/spx_options_2002_2007.csv",
        "output_dir": "/data/output/resnet_test/step0/"
    },
    "step1": {
        "model_type": "gru",
        "model_kwargs": {
            "hidden_dim": 12,
            "num_layers": 1,
            "dropout": 0.0
        },
        "output_activation": "relu",
        "feature_type": "SAM",
        "train_kwargs": {
            "epochs": 200,
            "batch_size": 128,
            "learning_rate": 0.01,
            "optimizer": "adam"
        },
        "temporal_split": {
            "train_ratio": 0.75,
            "val_ratio": 0.15,
            "test_ratio": 0.10
        },
        "output_dir": "/data/output/resnet_test/step1/"
    },
    "step2": {
        "model_type": "resnet",
        "model_kwargs": {
            "hidden_dims": [64, 64, 64],
            "num_blocks": 3,
            "activation": "tanh"
        },
        "output_activation": "softplus",
        "train_kwargs": {
            "epochs": 20,
            "batch_size": 1024,
            "learning_rate": 0.001,
            "lambda_penalty": 1.0,
            "optimizer": "adam"
        },
        "output_dir": "/data/output/resnet_test/step2/"
    }
}
```

**与 MLP 的关键差异**：
- `model_type`: `"resnet"`
- `hidden_dims`: `[64, 64, 64]`（比 MLP 的 [50,50,50] 略宽，补偿残差连接的维度保持）
- `num_blocks`: `3`（残差块数量）
- `activation`: `"tanh"`

**超参数调优建议**（如果第一次结果不理想）：
- `num_blocks`: 尝试 2, 3, 4
- `hidden_dims`: 尝试 [50,50,50], [64,64,64], [128,128,128]
- `activation`: 尝试 "tanh", "relu", "leaky_relu"

---

## Step 4: 运行完整流水线

### 命令

```bash
python main.py --config config_resnet.json
```

### 预期执行流程

```
[Step 0] NW 插值 → daily_grid_154.parquet (与 GRU 版本相同)
[Step 1] GRU 训练 → 200 epochs → gru_features.npz
[Step 2] ResNet 曲面重构 → 20 epochs
         - 输入: F(98维) + tau + m
         - 模型: ResNetSurface(hidden_dims=[64,64,64], num_blocks=3)
         - 输出: results_resnet_sam.npz
```

---

## Step 5: 结果对比

### 对比脚本

```python
import numpy as np

# 加载三种结果
lstm_mlp = np.load("/data/output/step2/results_sam.npz", allow_pickle=True)
gru_mlp = np.load("/data/output/gru_test/step2/results_sam.npz", allow_pickle=True)
gru_resnet = np.load("/data/output/resnet_test/step2/results_sam.npz", allow_pickle=True)

print("=" * 60)
print("Step 2 架构对比: MLP vs ResNet")
print("=" * 60)
print(f"{'配置':<25} {'Test RMSE':<12} {'Test MAPE':<12} {'L_cal':<8} {'L_but':<8}")
print("-" * 60)
print(f"{'LSTM + MLP (基准)':<25} {lstm_mlp['rmse']:.4f}       {lstm_mlp['mape']:.4f}       0        0")
print(f"{'GRU + MLP':<25} {gru_mlp['rmse']:.4f}       {gru_mlp['mape']:.4f}       0        0")
print(f"{'GRU + ResNet':<25} {gru_resnet['rmse']:.4f}       {gru_resnet['mape']:.4f}       0        0")
print("=" * 60)

# 计算改进
print(f"
改进幅度:")
print(f"  GRU + MLP vs LSTM + MLP:  RMSE {(gru_mlp['rmse']/lstm_mlp['rmse']-1)*100:+.1f}%")
print(f"  GRU + ResNet vs GRU + MLP: RMSE {(gru_resnet['rmse']/gru_mlp['rmse']-1)*100:+.1f}%")
print(f"  GRU + ResNet vs LSTM + MLP: RMSE {(gru_resnet['rmse']/lstm_mlp['rmse']-1)*100:+.1f}%")
```

### 预期结果

| 配置 | Step 1 RMSE | Step 2 RMSE | Step 2 MAPE | 预期 |
|------|------------|------------|------------|------|
| LSTM + MLP | 0.1641 | **0.1104** | 27.65% | 基准 |
| GRU + MLP | 0.0918 | **0.1102** | 27.72% | Step 1 提升被 MLP 抹平 |
| GRU + ResNet | 0.0918 | **?** | **?** | **关键测试** |

**三种情景**：

| 情景 | ResNet RMSE | 解读 | 下一步 |
|------|------------|------|--------|
| **A: 显著改善** | < 0.10 (↓>10%) | 跳跃连接有效保留特征信息 | 继续调优 ResNet 超参数，尝试更深的网络 |
| **B: 轻微改善** | 0.105-0.110 (↓0-5%) | 残差连接有微弱效果，但瓶颈仍在 | 尝试 KAN 或 Neural Operator |
| **C: 无改善/恶化** | ≥ 0.110 | ResNet 的归纳偏置不适合 IV 曲面 | 回到论文主线，接受 0.11 为当前数据极限 |

**理论预期**：情景 A 或 B 更可能。ResNet 的跳跃连接应该比 MLP 的逐层压缩更能保留 Step 1 的 98 维特征信息。

---

## 检查点（必须打印）

```
[Checkpoint 1] ResNet 实现
  - resnet.py 是否存在: ✅/❌
  - ResidualBlock 是否正确定义 (dim->dim + skip): ✅/❌
  - ResNetSurface 是否继承 BaseSurface: ✅/❌
  - forward 签名是否匹配 (F, tau, m) -> sigma: ✅/❌
  - STEP2_MODELS 是否注册 "resnet": ✅/❌

[Checkpoint 2] 配置加载
  - config_resnet.json 是否有效 JSON: ✅/❌
  - get_step2_model("resnet") 是否返回 ResNetSurface 类: ✅/❌
  - model_kwargs 是否正确传递 (hidden_dims, num_blocks): ✅/❌

[Checkpoint 3] Step 2 训练
  - ResNet 训练是否启动（打印 epoch 0/20）: ✅/❌
  - 训练损失是否下降（非 NaN/Inf）: ✅/❌
  - 无套利惩罚项是否收敛到 0: ✅/❌
  - 最优 epoch 是否在合理范围（5-20）: ✅/❌

[Checkpoint 4] 结果对比
  - LSTM + MLP Test RMSE: {lstm_rmse:.4f}
  - GRU + MLP Test RMSE: {gru_rmse:.4f}
  - GRU + ResNet Test RMSE: {resnet_rmse:.4f}
  - 改进幅度: {improvement:+.1f}%
  - 无套利违规 L_cal / L_but: {lcal:.6f} / {lbut:.6f}

[Checkpoint 5] 结论
  - ResNet 是否打破 0.11 瓶颈: ✅/❌
  - 如果否，下一步建议: [调参 / 尝试 KAN / 接受现状]
```

---

## 故障排查

| 问题 | 原因 | 解决 |
|------|------|------|
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | hidden_dims 长度与 num_blocks 不匹配 | 确保 hidden_dims 是列表且所有元素相同（如 [64,64,64]） |
| 训练损失 NaN | 残差连接导致梯度爆炸 | 加 BatchNorm 或降低学习率到 0.0005 |
| RMSE 比 MLP 还高 | ResNet 过拟合或欠拟合 | 调 num_blocks (2→4) 或 hidden_dims ([50]→[128]) |
| 无套利惩罚不收敛 | ResNet 太深，Softplus 梯度消失 | 减少 num_blocks 到 2，或改用 ReLU + 输出层 Softplus |

---

## 超参数调优网格（如果第一次结果不理想）

如果第一次 `config_resnet.json` 的 RMSE ≥ 0.11，尝试以下组合：

```python
# 调参脚本
configs = [
    {"hidden_dims": [50, 50, 50], "num_blocks": 2, "activation": "tanh"},
    {"hidden_dims": [64, 64, 64], "num_blocks": 2, "activation": "tanh"},
    {"hidden_dims": [64, 64, 64], "num_blocks": 4, "activation": "tanh"},
    {"hidden_dims": [128, 128, 128], "num_blocks": 2, "activation": "relu"},
    {"hidden_dims": [64, 64, 64], "num_blocks": 3, "activation": "leaky_relu"},
]

for cfg in configs:
    # 修改 config_resnet.json 中的 model_kwargs
    # 运行 main.py
    # 记录 RMSE
    pass
```

---

## 成功后下一步

| ResNet 结果 | 下一步 |
|------------|--------|
| RMSE < 0.10 (显著改善) | 继续调优 ResNet 超参数，尝试更深的网络（num_blocks=4, hidden_dims=[128,128,128]） |
| RMSE 0.105-0.110 (轻微改善) | 尝试 KAN（Kolmogorov-Arnold Network）—— 可解释性更强的替代方案 |
| RMSE ≥ 0.110 (无改善) | 回到论文主线完成 DM 检验和汇报，接受 0.11 为当前数据极限 |

---

## 执行顺序

1. **填充 `src/models/surfaces/resnet.py`**（复制上方代码）
2. **修改 `src/models/__init__.py`** 注册 ResNet
3. **创建 `config_resnet.json`**（复制上方配置）
4. **运行 `python main.py --config config_resnet.json`**
5. **对比 LSTM+MLP / GRU+MLP / GRU+ResNet 结果**
6. **打印检查点 4-5**

**预计总耗时：20-40 分钟（ResNet 训练 20 epochs 约 10-15 分钟，与 MLP 相当）**
