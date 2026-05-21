# 任务：将现有代码改造为支持多架构切换的模块化系统

## 背景

我已复现 Zhang et al. (2023) 的两步法 IV 曲面预测框架（Step 0: DFW/NW 插值 → Step 1: LSTM 特征预测 → Step 2: DNN 无套利曲面重构）。现在希望在**不删除原有 LSTM/DNN 代码**的前提下，获得**通过配置文件切换网络架构**的能力，以便后续实验对比：
- Step 1 候选：LSTM / GRU / Transformer (Informer) / TCN
- Step 2 候选：MLP (原 DNN) / ResNet / KAN / Neural Operator

## 对现有分析的确认与补充

你之前的分析准确识别了以下硬编码点：
- Step 1: `train_lstm()` 内部硬编码 `model = LSTMPredictor(...)`
- Step 2: `train_dnn()` 内部硬编码 `model = DNN_Surface(...)`，且 4 个辅助函数类型注解写死 `DNN_Surface`

**但以下关键点需要补充**，否则新增架构时会遇到运行时错误：

### 补充 1: Step 1 输入格式差异（必须处理）

不同架构对输入张量形状要求不同：

| 架构 | 期望输入形状 | 特殊要求 |
|------|------------|---------|
| LSTM/GRU | `(batch, seq_len=3, input_dim)` | 无 |
| Transformer | `(batch, seq_len, input_dim)` | 需要位置编码 (Positional Encoding)，seq_len 可扩展至 22 |
| TCN | `(batch, input_dim, seq_len)` | 通道优先 (channels first) |

**要求**：每个模型类内部自行处理输入格式转换，外部 `train_step1()` 只负责传递 `(batch, seq_len, input_dim)`。

### 补充 2: 超参数传递机制（必须统一）

不同架构的超参数完全不同，不能硬编码在 train 函数中：

```python
# LSTM
{"hidden_dim": 12, "num_layers": 1, "dropout": 0.0}

# Transformer  
{"d_model": 64, "nhead": 4, "num_encoder_layers": 2, "dim_feedforward": 128, "dropout": 0.1}

# GRU
{"hidden_dim": 12, "num_layers": 1, "dropout": 0.0}

# TCN
{"num_channels": [64, 64, 64], "kernel_size": 3, "dropout": 0.1}
```

**要求**：通过统一的 `model_kwargs: dict` 传入，train 函数不感知具体参数名。

### 补充 3: 模型保存/加载兼容性（必须处理）

不同架构的 `state_dict` 键名不同，需要保存时记录模型类型，加载时自动重建：

```python
# 保存
checkpoint = {
    "state_dict": model.state_dict(),
    "model_type": "transformer",  # 关键！
    "model_kwargs": {"d_model": 64, "nhead": 4, ...},
    "feature_type": "SAM",
}
torch.save(checkpoint, "model.pt")

# 加载
checkpoint = torch.load("model.pt")
model_class = STEP1_MODELS[checkpoint["model_type"]]
model = model_class(input_dim, output_dim, **checkpoint["model_kwargs"])
model.load_state_dict(checkpoint["state_dict"])
```

### 补充 4: 激活函数解耦（必须处理）

现有代码中激活函数与 `feature_type` 耦合：
```python
if feature_type in ("PCA", "VAE"):
    model.activation = nn.Identity()
```

**要求**：激活函数作为模型构造参数传入，或在模型类内部根据 `output_activation` 参数设置，train 函数完全不处理激活函数。

### 补充 5: 配置文件模板（必须提供）

提供 `config.json` 模板，支持一键切换：

```json
{
    "step1": {
        "model_type": "lstm",
        "model_kwargs": {
            "hidden_dim": 12,
            "num_layers": 1,
            "dropout": 0.0
        },
        "output_activation": "relu",
        "train_kwargs": {
            "epochs": 200,
            "batch_size": 128,
            "learning_rate": 0.01,
            "optimizer": "adam"
        }
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
        }
    }
}
```

## 执行方案（分优先级）

### P0: 最小侵入式改造（30 分钟）

**目标**：让 `train_step1()` 和 `train_step2()` 接收 `model_class` 和 `model_kwargs` 参数，原有代码逻辑完全保留。

**具体改动**：

1. **Step 1 — `train_step1()` 签名改造**：
```python
def train_step1(
    X_train, y_train, X_val, y_val,
    model_class=LSTMPredictor,        # 新增
    model_kwargs=None,                # 新增
    output_activation="relu",         # 新增，替代 feature_type 耦合
    train_kwargs=None,                # 新增
    device="cpu"
):
    model_kwargs = model_kwargs or {}
    train_kwargs = train_kwargs or {"epochs": 200, "batch_size": 128, "lr": 0.01}

    model = model_class(
        input_dim=X_train.shape[-1],
        output_dim=y_train.shape[-1],
        output_activation=output_activation,
        **model_kwargs
    ).to(device)

    # ... 原有训练逻辑不变
```

2. **Step 2 — `train_step2()` 签名改造**：
```python
def train_step2(
    train_data, val_data, n_grid,
    model_class=DNN_Surface,          # 新增
    model_kwargs=None,                # 新增
    output_activation="softplus",     # 新增
    train_kwargs=None,                # 新增
    device="cpu"
):
    model_kwargs = model_kwargs or {}
    train_kwargs = train_kwargs or {"epochs": 20, "batch_size": 1024, "lr": 0.001, "lambda_penalty": 1.0}

    model = model_class(
        input_dim=n_grid + 2,  # F + tau + m
        output_activation=output_activation,
        **model_kwargs
    ).to(device)

    # ... 原有训练逻辑不变
```

3. **Step 2 辅助函数类型注解改为 `nn.Module`**：
```python
def compute_arbitrage_penalties(model: nn.Module, ...): ...
def day_loss(model: nn.Module, ...): ...
def evaluate_step2(model: nn.Module, ...): ...
def check_arbitrage_violation(model: nn.Module, ...): ...
```

### P1: 创建模型注册表 + 目录结构（1 小时）

**目标**：新建 `src/models/` 目录，所有架构统一管理，通过字符串配置切换。

**目录结构**：
```
src/
├── models/
│   ├── __init__.py              # 注册表 + 工厂函数
│   ├── base.py                  # 抽象基类（可选）
│   ├── predictors/              # Step 1 时序预测模型
│   │   ├── __init__.py
│   │   ├── lstm.py              # 迁移原有 LSTMPredictor
│   │   ├── gru.py               # 新增（预留）
│   │   └── transformer.py       # 新增（预留）
│   └── surfaces/                # Step 2 曲面重构模型
│       ├── __init__.py
│       ├── mlp.py               # 迁移原有 DNN_Surface
│       ├── resnet.py            # 新增（预留）
│       └── kan.py               # 新增（预留）
```

**`models/__init__.py` 注册表**：
```python
from .predictors.lstm import LSTMPredictor
from .predictors.gru import GRUPredictor
from .predictors.transformer import TransformerPredictor
from .surfaces.mlp import MLPSurface
from .surfaces.resnet import ResNetSurface

STEP1_MODELS = {
    "lstm": LSTMPredictor,
    "gru": GRUPredictor,
    "transformer": TransformerPredictor,
}

STEP2_MODELS = {
    "mlp": MLPSurface,
    "resnet": ResNetSurface,
}

def get_step1_model(model_type: str):
    if model_type not in STEP1_MODELS:
        raise ValueError(f"Unknown Step 1 model: {model_type}. Available: {list(STEP1_MODELS.keys())}")
    return STEP1_MODELS[model_type]

def get_step2_model(model_type: str):
    if model_type not in STEP2_MODELS:
        raise ValueError(f"Unknown Step 2 model: {model_type}. Available: {list(STEP2_MODELS.keys())}")
    return STEP2_MODELS[model_type]
```

### P2: 统一模型类接口契约（必须保证）

**Step 1 模型接口**（所有预测模型必须满足）：
```python
class BasePredictor(nn.Module):
    """Step 1 预测模型基类"""

    def __init__(self, input_dim: int, output_dim: int, output_activation: str = "relu", **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # 子类实现具体网络

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x (batch, seq_len, input_dim)
        输出: (batch, output_dim)
        """
        raise NotImplementedError
```

**Step 2 模型接口**（所有曲面模型必须满足）：
```python
class BaseSurface(nn.Module):
    """Step 2 曲面重构模型基类"""

    def __init__(self, input_dim: int, output_activation: str = "softplus", **kwargs):
        super().__init__()
        self.input_dim = input_dim  # = n_grid + 2
        # 子类实现具体网络

    def forward(self, F: torch.Tensor, tau: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        输入: 
            F: (batch, n_grid) 预测的离散 IV 点
            tau: (batch, 1) 查询期限
            m: (batch, 1) 查询 moneyness
        输出: (batch, 1) IV 值
        """
        raise NotImplementedError
```

### P3: 配置文件驱动 main()（30 分钟）

**目标**：`main.py` 读取 `config.json`，自动选择模型架构，无需改代码。

```python
import json
from models import get_step1_model, get_step2_model

# 加载配置
with open("config.json") as f:
    config = json.load(f)

# Step 1
step1_cfg = config["step1"]
model_class = get_step1_model(step1_cfg["model_type"])
model = train_step1(
    X_train, y_train, X_val, y_val,
    model_class=model_class,
    model_kwargs=step1_cfg["model_kwargs"],
    output_activation=step1_cfg["output_activation"],
    train_kwargs=step1_cfg["train_kwargs"],
)

# Step 2
step2_cfg = config["step2"]
model_class = get_step2_model(step2_cfg["model_type"])
model = train_step2(
    train_data, val_data, n_grid,
    model_class=model_class,
    model_kwargs=step2_cfg["model_kwargs"],
    output_activation=step2_cfg["output_activation"],
    train_kwargs=step2_cfg["train_kwargs"],
)
```

## 检查点（必须打印）

```
[Checkpoint 1] P0 完成
  - train_step1() 是否接受 model_class + model_kwargs: ✅/❌
  - train_step2() 是否接受 model_class + model_kwargs: ✅/❌
  - 4 个辅助函数类型注解是否改为 nn.Module: ✅/❌
  - 原有 LSTM/DNN 训练是否仍正常: ✅/❌

[Checkpoint 2] P1 完成
  - src/models/ 目录是否存在: ✅/❌
  - STEP1_MODELS 注册表是否包含 lstm: ✅/❌
  - STEP2_MODELS 注册表是否包含 mlp: ✅/❌
  - 原有 LSTMPredictor 是否已迁移到 models/predictors/lstm.py: ✅/❌
  - 原有 DNN_Surface 是否已迁移到 models/surfaces/mlp.py: ✅/❌

[Checkpoint 3] P2 完成
  - BasePredictor 接口是否定义: ✅/❌
  - BaseSurface 接口是否定义: ✅/❌
  - LSTMPredictor 是否继承 BasePredictor: ✅/❌
  - MLPSurface 是否继承 BaseSurface: ✅/❌

[Checkpoint 4] P3 完成
  - config.json 模板是否生成: ✅/❌
  - main.py 是否通过 config 加载模型: ✅/❌
  - 运行 `python main.py --config config.json` 是否成功: ✅/❌

[Checkpoint 5] 新增架构兼容性验证
  - 预留 gru.py 框架是否符合 BasePredictor: ✅/❌
  - 预留 transformer.py 框架是否符合 BasePredictor: ✅/❌
  - 预留 resnet.py 框架是否符合 BaseSurface: ✅/❌
```

## 输出文件

```
src/
├── models/
│   ├── __init__.py
│   ├── base.py
│   ├── predictors/
│   │   ├── __init__.py
│   │   ├── lstm.py          # 原 LSTMPredictor（迁移）
│   │   ├── gru.py           # 空框架（预留）
│   │   └── transformer.py   # 空框架（预留）
│   └── surfaces/
│       ├── __init__.py
│       ├── mlp.py           # 原 DNN_Surface（迁移）
│       ├── resnet.py        # 空框架（预留）
│       └── kan.py           # 空框架（预留）
├── config.json              # 默认配置（LSTM + MLP）
├── config_transformer.json  # 示例配置（Transformer + MLP）
└── main.py                  # 配置文件驱动入口
```

## 关键约束

1. **不删除原有 LSTM/DNN 代码**：只迁移到 models/ 目录，原有功能完全保留
2. **向后兼容**：原有 `python main.py` 无参数运行时应使用默认配置（LSTM + MLP）
3. **预留空框架**：GRU/Transformer/ResNet/KAN 文件可以只有类定义和 `raise NotImplementedError`，后续再实现
4. **错误处理**：如果 config.json 中的 model_type 不在注册表中，必须抛出自描述错误（列出可用选项）

## 执行顺序

1. **P0**（立即执行）：改造 train 函数签名，验证原有 LSTM/DNN 仍正常
2. **P1**（接着执行）：创建 models/ 目录，迁移原有代码，建立注册表
3. **P2**（同步执行）：定义 BasePredictor/BaseSurface 接口，确保预留框架符合契约
4. **P3**（最后执行）：生成 config.json 模板，改造 main.py 为配置驱动

**请先执行 P0，确认原有训练流程不受影响后，再进入 P1-P3。**
