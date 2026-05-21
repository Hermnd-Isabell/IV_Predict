# 任务：KAN (Kolmogorov-Arnold Network) 验证 —— Step 2 前沿架构测试

## 目标

在模块化架构基础上，实现 **KANSurface** 替换 **MLPSurface/ResNetSurface**，测试 Kolmogorov-Arnold Network 在 IV 曲面重构中的效果。

## 背景：为什么选择 KAN？

**KAN 的核心优势**（Liu et al., 2024, Nature）：
- **可解释性**：用可学习的激活函数（B-spline 基函数）替代固定激活函数，每个连接都是一个 1D 函数
- **准确性**：在小样本上优于 MLP，因为参数效率更高
- **无套利直觉**：KAN 的 1D 函数天然适合建模 `σ(m, τ)` 的偏度和期限结构

**与 ResNet/MLP 的区别**：
- MLP：`σ = W₃·act(W₂·act(W₁·x))` —— 固定激活函数
- ResNet：`σ = W₃·act(ResBlock(ResBlock(W₁·x)))` —— 跳跃连接
- **KAN**：`σ = Σᵢ φᵢ(Σⱼ ψᵢⱼ(xⱼ))` —— 每个连接是可学习的 1D 函数

## Step 1: KAN 核心实现

### 文件：`src/models/surfaces/kan.py`

```python
import torch
import torch.nn as nn
import numpy as np
from ..base import BaseSurface

class KANLayer(nn.Module):
    """
    KAN 层：每个输入-输出连接是一个可学习的 1D B-spline 函数。
    简化版：用 MLP 近似 KAN 的 1D 函数（避免复杂的 B-spline 网格实现）。
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 为每个连接创建一个小的 MLP（近似 1D 函数）
        # 实际 KAN 用 B-spline，这里用 3 层 MLP 近似
        self.functions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(in_dim * out_dim)
        ])

        # 基函数（线性部分）
        self.base_weight = nn.Parameter(torch.randn(out_dim, in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_dim)
        out: (batch, out_dim)
        """
        batch_size = x.shape[0]

        # 基函数输出
        base = torch.matmul(x, self.base_weight.t())  # (batch, out_dim)

        # KAN 函数输出
        kan_out = torch.zeros(batch_size, self.out_dim, device=x.device)
        idx = 0
        for i in range(self.out_dim):
            for j in range(self.in_dim):
                # 对每个输入维度应用独立的 1D 函数
                x_j = x[:, j:j+1]  # (batch, 1)
                kan_out[:, i] += self.functions[idx](x_j).squeeze(-1)
                idx += 1

        return base + kan_out


class KANSurface(BaseSurface):
    """
    KAN 曲面重构模型。
    架构：输入 -> KANLayer -> KANLayer -> Softplus
    """
    def __init__(self, input_dim: int,
                 hidden_dim: int = 32,
                 kan_hidden: int = 16,
                 num_layers: int = 2,
                 output_activation: str = "softplus"):
        super().__init__(input_dim, output_activation)

        layers = []
        dims = [input_dim] + [hidden_dim] * num_layers + [1]

        for i in range(len(dims) - 1):
            layers.append(KANLayer(dims[i], dims[i+1], kan_hidden))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i+1]))  # 稳定训练

        self.layers = nn.Sequential(*layers)

    def forward(self, F: torch.Tensor, tau: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        x = torch.cat([F, tau, m], dim=1)
        x = self.layers(x)
        return self.activation(x)
```

**注意**：这是**简化版 KAN**，用小型 MLP 近似 1D 函数。真正的 KAN 使用 B-spline 基函数（实现复杂，需要网格参数）。简化版已能验证 KAN 的"可学习连接"直觉。

## Step 2: 注册与配置

### 修改 `src/models/__init__.py`

```python
from .surfaces.kan import KANSurface

STEP2_MODELS = {
    "mlp": MLPSurface,
    "resnet": ResNetSurface,
    "kan": KANSurface,
}
```

### 配置文件：`config_kan.json`

```json
{
    "step1": {
        "model_type": "gru",
        "model_kwargs": {"hidden_dim": 12, "num_layers": 1, "dropout": 0.0},
        "output_activation": "relu",
        "feature_type": "SAM",
        "train_kwargs": {"epochs": 200, "batch_size": 128, "learning_rate": 0.01}
    },
    "step2": {
        "model_type": "kan",
        "model_kwargs": {
            "hidden_dim": 32,
            "kan_hidden": 16,
            "num_layers": 2
        },
        "output_activation": "softplus",
        "train_kwargs": {
            "epochs": 20,
            "batch_size": 1024,
            "learning_rate": 0.001,
            "lambda_penalty": 1.0
        }
    }
}
```

## Step 3: 运行与对比

```bash
python main.py --config config_kan.json
```

**预期**：KAN 的参数效率可能在小样本（1,417 天）上优于 MLP，但训练更慢（每个连接独立前向传播）。

## 检查点

```
[Checkpoint] KAN 训练
  - 训练是否启动: ✅/❌
  - 损失是否下降: ✅/❌
  - Test RMSE vs MLP: {kan_rmse:.4f} vs {mlp_rmse:.4f}
  - 改进: {improvement:+.1f}%
```
