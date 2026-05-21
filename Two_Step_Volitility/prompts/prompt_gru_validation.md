# 任务：GRU 验证 —— 最小可行架构切换测试

## 目标

在已完成的模块化架构基础上，实现 **GRUPredictor** 替换 **LSTMPredictor**，运行完整三步流水线，验证架构切换机制端到端可用。

**预期**：GRU 与 LSTM 接口完全一致，只需将 `nn.LSTM` 替换为 `nn.GRU`，超参数相同。如果 GRU-DNN 能正常训练并输出 Test RMSE，说明整个 `model_class` + `model_kwargs` 切换机制工作正常。

---

## 背景：现有模块化架构

```
src/
├── models/
│   ├── predictors/
│   │   ├── lstm.py          # LSTMPredictor (完整实现)
│   │   ├── gru.py           # GRUPredictor (预留框架，目前只有 pass)
│   │   └── transformer.py   # 预留框架
│   └── surfaces/
│       ├── mlp.py           # MLPSurface (完整实现)
│       ├── resnet.py        # 预留框架
│       └── kan.py           # 预留框架
├── config.json              # 默认 LSTM + MLP
└── main.py                  # 配置驱动入口
```

---

## Step 1: 实现 GRUPredictor（5 分钟）

### 文件：`src/models/predictors/gru.py`

```python
import torch
import torch.nn as nn
from ..base import BasePredictor

class GRUPredictor(BasePredictor):
    """
    GRU 时序预测模型。
    与 LSTM 接口完全一致：输入 (batch, seq_len, input_dim)，输出 (batch, output_dim)。
    """
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_dim: int = 12, num_layers: int = 1, dropout: float = 0.0,
                 output_activation: str = "relu"):
        super().__init__(input_dim, output_dim, output_activation)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU 与 LSTM 参数完全一致
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (batch, seq_len, input_dim)
        返回:
            (batch, output_dim)
        """
        # GRU 前向传播
        # out: (batch, seq_len, hidden_dim)
        # hn: (num_layers, batch, hidden_dim)
        out, hn = self.gru(x)

        # 取最后一个时间步的隐藏状态
        last_hidden = out[:, -1, :]  # (batch, hidden_dim)

        # 全连接层 + 激活函数
        return self.activation(self.fc(last_hidden))
```

**关键验证点**：
- GRU 的 `forward` 签名与 LSTM 完全一致：`(batch, seq_len, input_dim) → (batch, output_dim)`
- `BasePredictor` 的 `__init__` 已处理 `output_activation`（relu/identity/tanh 等）
- 无需修改 `train_step1()` 或 `main.py` 的任何代码

---

## Step 2: 创建 GRU 配置文件

### 文件：`config_gru.json`

```json
{
    "step0": {
        "interpolation_method": "NW",
        "use_synthetic": false,
        "data_path": "/data/raw/spx_options_2002_2007.csv",
        "output_dir": "/data/output/gru_test/step0/"
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
        "output_dir": "/data/output/gru_test/step1/"
    },
    "step2": {
        "model_type": "mlp",
        "model_kwargs": {
            "hidden_dims": [50, 50, 50],
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
        "output_dir": "/data/output/gru_test/step2/"
    }
}
```

**注意**：
- `step1.model_type` 改为 `"gru"`
- `step1.model_kwargs` 与 LSTM 完全一致（hidden_dim=12, num_layers=1）
- `step2` 保持不变（仍用 MLP/DNN）
- 输出目录改为 `gru_test/` 避免覆盖原有 LSTM 结果

---

## Step 3: 运行完整流水线

### 命令

```bash
# 方式 1: 通过 main.py 配置驱动
python main.py --config config_gru.json

# 方式 2: 如果 main.py 不支持 --config 参数，手动修改 config.json 中的 model_type 为 "gru"
cp config_gru.json config.json
python main.py
```

### 预期执行流程

```
[Step 0] NW 插值 → daily_grid_154.parquet (与 LSTM 版本相同)
[Step 1] GRU 训练 → 200 epochs
         - 输入: (batch, 3, 98)
         - 模型: GRUPredictor(hidden_dim=12, num_layers=1)
         - 输出: gru_features.npz
[Step 2] MLP 曲面重构 → 20 epochs
         - 输入: F(98维) + tau + m
         - 模型: MLPSurface(hidden_dims=[50,50,50])
         - 输出: results_gru_sam.npz
```

---

## Step 4: 结果对比

### 对比脚本

```python
import numpy as np

# 加载 LSTM 结果
lstm = np.load("/data/output/step2/results_sam.npz", allow_pickle=True)
# 加载 GRU 结果
gru = np.load("/data/output/gru_test/step2/results_sam.npz", allow_pickle=True)

print("=== LSTM vs GRU 对比 ===")
print(f"LSTM Test RMSE: {lstm['rmse']:.4f}")
print(f"GRU  Test RMSE: {gru['rmse']:.4f}")
print(f"LSTM Test MAPE: {lstm['mape']:.4f}")
print(f"GRU  Test MAPE: {gru['mape']:.4f}")
print(f"RMSE 差异: {gru['rmse'] - lstm['rmse']:.4f} ({(gru['rmse']/lstm['rmse']-1)*100:+.1f}%)")
```

### 预期结果

| 指标 | LSTM-DNN (基准) | GRU-DNN (验证) | 预期差异 |
|------|----------------|---------------|---------|
| Test RMSE | ~0.1104 | ~0.105-0.115 | ±5%（GRU 可能略优或持平） |
| Test MAPE | ~27.65% | ~26-29% | ±5% |
| 训练速度 | 基准 | **快 10-20%** | GRU 门控更少，计算量略低 |
| 无套利违规 | 0 / 0 | **0 / 0** | Step 2 相同，应一致 |

**GRU 的理论优势**：
- 门控机制比 LSTM 简单（没有 forget gate + input gate + output gate 的复杂交互）
- 在短序列（seq_len=3）上，GRU 可能更稳定
- 训练速度略快（参数少 25%）

**如果 GRU RMSE 与 LSTM 差距 > 15%**：检查 GRU 的超参数是否需要调优（如 hidden_dim 从 12 调到 24）

---

## 检查点（必须打印）

```
[Checkpoint 1] GRU 实现
  - gru.py 是否存在: ✅/❌
  - GRUPredictor 是否继承 BasePredictor: ✅/❌
  - forward 签名是否匹配: (batch, seq_len, input_dim) -> (batch, output_dim): ✅/❌
  - 是否能被 STEP1_MODELS["gru"] 正确加载: ✅/❌

[Checkpoint 2] 配置加载
  - config_gru.json 是否有效 JSON: ✅/❌
  - get_step1_model("gru") 是否返回 GRUPredictor 类: ✅/❌
  - model_kwargs 是否正确传递给 GRU __init__: ✅/❌

[Checkpoint 3] Step 1 训练
  - GRU 训练是否启动（打印 epoch 0/200）: ✅/❌
  - 训练损失是否下降（非 NaN/Inf）: ✅/❌
  - 最优 epoch 是否在合理范围（50-200）: ✅/❌
  - 输出文件 gru_features.npz 是否生成: ✅/❌

[Checkpoint 4] Step 2 训练
  - MLP 训练是否启动（打印 epoch 0/20）: ✅/❌
  - 无套利惩罚项是否收敛到 0: ✅/❌
  - 输出文件 results_gru_sam.npz 是否生成: ✅/❌

[Checkpoint 5] 结果对比
  - LSTM Test RMSE: {lstm_rmse:.4f}
  - GRU  Test RMSE: {gru_rmse:.4f}
  - 差异: {diff:.4f} ({pct:+.1f}%)
  - 无套利违规 L_cal / L_but: {lcal:.6f} / {lbut:.6f}
  - 结论: GRU 切换机制 ✅/❌
```

---

## 故障排查

| 问题 | 原因 | 解决 |
|------|------|------|
| `KeyError: 'gru'` | STEP1_MODELS 未注册 GRU | 检查 `models/__init__.py` 是否导入 GRUPredictor |
| `TypeError: forward() takes 2 positional arguments` | BasePredictor 接口不匹配 | 检查 GRUPredictor.forward 是否接收 `self, x` |
| 训练损失 NaN | 学习率过高或梯度爆炸 | 降低 lr 到 0.001，或加 gradient clipping |
| RMSE 远高于 LSTM (> 0.15) | GRU 欠拟合 | 增大 hidden_dim 到 24，或增加 num_layers 到 2 |
| 训练速度比 LSTM 慢 | batch_size 太小或数据加载瓶颈 | 确认 batch_size=128，检查 DataLoader num_workers |

---

## 成功后下一步

GRU 验证通过后，证明模块化架构**完全可用**。下一步优先级：

1. **P0（立即）**: 实现 **Transformer**（Informer 风格）—— 真正的架构升级
2. **P1（并行）**: 实现 **ResNetSurface** 替换 MLP—— 测试 Step 2 切换
3. **P2（汇报前）**: 完成 DM 检验 + 无套利统计—— 论文实证完整性

---

## 执行顺序

1. **填充 `src/models/predictors/gru.py`**（复制上方代码）
2. **创建 `config_gru.json`**（复制上方配置）
3. **确认 `models/__init__.py` 已注册 GRU**（添加 `"gru": GRUPredictor`）
4. **运行 `python main.py --config config_gru.json`**
5. **对比 LSTM vs GRU 结果**
6. **打印检查点 5**

**预计总耗时：15-30 分钟（GRU 训练 200 epochs 约 10-15 分钟）**
