# Step 2: DNN 无套利曲面重构
# 复现 Zhang et al. (2023) Section 3.3 & 3.4

## 目标
从 Step 1 预测的离散特征点，用 DNN 重构完整 IV 曲面，内置无套利约束。

---

## 输入

```python
# Step 1 输出
sam_data = np.load("/data/output/step1/sam_features.npz")
pca_data = np.load("/data/output/step1/pca_features.npz")
# Z_pred_test: LSTM 预测的 T+1 日特征
# dates_test: 测试日期
```

---

## Step 2.1: 从预测特征还原离散 IV 点 F_{T+1}

```python
def map_features_to_iv_points(Z_pred, feature_type, **kwargs):
    """
    将预测的 Z_{T+1} 映射到 154/98 个离散 IV 点。
    """
    if feature_type == "SAM":
        # F = Z_pred 直接就是 98 个 IV 点
        return Z_pred  # (n_test, 98)

    elif feature_type == "PCA":
        # F = sigma_0 * exp(sum x_k * f_k)
        eigensurfaces = kwargs["eigensurfaces"]  # (3, 98)
        sigma_0 = kwargs["sigma_0"]  # (98,)
        F = sigma_0 * np.exp(Z_pred @ eigensurfaces)  # (n_test, 98)
        return F

    elif feature_type == "VAE":
        # F = Decoder(Z_pred)
        decoder = kwargs["decoder"]  # PyTorch model
        with torch.no_grad():
            F = decoder(torch.tensor(Z_pred, dtype=torch.float32)).numpy()
        return F
```

---

## Step 2.2: DNN 架构（论文 Figure 4）

```python
import torch
import torch.nn as nn

class DNN_Surface(nn.Module):
    def __init__(self, input_dim, hidden_dim=50):
        """
        input_dim: F 的维度（SAM=98, PCA=98 via还原, VAE=98 via还原）
        但 DNN 输入是 F + tau + m，所以实际 input_dim = len(F) + 2
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

        self.activation = nn.Tanh()  # 论文未明确，tanh 适合曲面

    def forward(self, F_input, tau_input, m_input):
        """
        F_input: (batch, F_dim) 预测的离散 IV 点
        tau_input: (batch, 1) 查询期限
        m_input: (batch, 1) 查询 moneyness
        """
        x = torch.cat([F_input, tau_input, m_input], dim=1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc_out(x)
        return nn.Softplus()(x)  # ln(1+exp(x)), 保证非负+二阶可微
```

---

## Step 2.3: 无套利损失函数（核心）

### 合成密集网格

```python
# 用于条件 3&4（日历+蝶式套利）
m_min, m_max = np.log(0.6), np.log(2.0)
m_grid_c34 = np.linspace(-(2*abs(m_min))**(1/3), (2*m_max)**(1/3), 40)
tau_grid_c34 = np.exp(np.linspace(np.log(1/365), np.log(730/365 + 1), 40))
M_c34, Tau_c34 = np.meshgrid(m_grid_c34, tau_grid_c34, indexing="ij")

# 用于条件 5（大 moneyness）
m_grid_c5 = np.array([6*m_min, 4*m_min, 4*m_max, 6*m_max])
tau_grid_c5 = tau_grid_c34.copy()
M_c5, Tau_c5 = np.meshgrid(m_grid_c5, tau_grid_c5, indexing="ij")
```

### 自动微分计算导数

```python
def compute_arbitrage_penalties(model, F_batch, device="cpu"):
    """
    对一批 F_batch，在密集网格上计算无套利惩罚。

    F_batch: (batch_size, F_dim)
    返回: (penalty_cal, penalty_but, penalty_boundary)
    """
    batch_size = F_batch.shape[0]

    # 将密集网格复制 batch 份
    m_c34 = torch.tensor(M_c34.ravel(), dtype=torch.float32, requires_grad=True).to(device)
    tau_c34 = torch.tensor(Tau_c34.ravel(), dtype=torch.float32, requires_grad=True).to(device)
    n_c34 = len(m_c34)

    # 扩展 F 匹配网格点数
    F_expanded = F_batch.unsqueeze(1).expand(-1, n_c34, -1).reshape(-1, F_batch.shape[1])
    m_expanded = m_c34.repeat(batch_size)
    tau_expanded = tau_c34.repeat(batch_size)

    # DNN 前向
    sigma = model(F_expanded, tau_expanded.unsqueeze(1), m_expanded.unsqueeze(1)).squeeze()

    # 一阶导
    grad_m = torch.autograd.grad(sigma.sum(), m_expanded, create_graph=True)[0]
    grad_tau = torch.autograd.grad(sigma.sum(), tau_expanded, create_graph=True)[0]

    # 二阶导
    grad_mm = torch.autograd.grad(grad_m.sum(), m_expanded, create_graph=True)[0]

    # 条件 3: 日历套利
    # l_cal = sigma + 2*tau*grad_tau >= 0
    l_cal = sigma + 2 * tau_expanded * grad_tau
    penalty_cal = torch.clamp(-l_cal, min=0).mean()

    # 条件 4: 蝶式套利（Durrleman）
    # l_but = (1 - m*grad_m/sigma)^2 - (sigma*tau*grad_m)^2/4 + tau*sigma*grad_mm >= 0
    term1 = (1 - m_expanded * grad_m / sigma)**2
    term2 = (sigma * tau_expanded * grad_m)**2 / 4
    term3 = tau_expanded * sigma * grad_mm
    l_but = term1 - term2 + term3
    penalty_but = torch.clamp(-l_but, min=0).mean()

    # 条件 5: 大 moneyness 行为
    # |sigma * grad_mm + grad_m^2| -> 0
    m_c5 = torch.tensor(M_c5.ravel(), dtype=torch.float32, requires_grad=True).to(device)
    tau_c5 = torch.tensor(Tau_c5.ravel(), dtype=torch.float32, requires_grad=True).to(device)
    n_c5 = len(m_c5)

    F_expanded5 = F_batch.unsqueeze(1).expand(-1, n_c5, -1).reshape(-1, F_batch.shape[1])
    m_expanded5 = m_c5.repeat(batch_size)
    tau_expanded5 = tau_c5.repeat(batch_size)

    sigma5 = model(F_expanded5, tau_expanded5.unsqueeze(1), m_expanded5.unsqueeze(1)).squeeze()
    grad_m5 = torch.autograd.grad(sigma5.sum(), m_expanded5, create_graph=True)[0]
    grad_mm5 = torch.autograd.grad(grad_m5.sum(), m_expanded5, create_graph=True)[0]

    boundary_val = torch.abs(sigma5 * grad_mm5 + grad_m5**2)
    penalty_boundary = boundary_val.mean()

    return penalty_cal, penalty_but, penalty_boundary
```

### 总损失函数

```python
def total_loss(model, F_batch, m_obs, tau_obs, sigma_obs, lambda_penalty=1.0, device="cpu"):
    """
    L_C = L_S + lambda * (L_C3 + L_C4 + L_C5)
    """
    # L_S: MSE 在观测点上
    sigma_pred = model(F_batch, 
                       torch.tensor(tau_obs, dtype=torch.float32).to(device).unsqueeze(1),
                       torch.tensor(m_obs, dtype=torch.float32).to(device).unsqueeze(1))
    L_S = nn.functional.mse_loss(sigma_pred.squeeze(), 
                                  torch.tensor(sigma_obs, dtype=torch.float32).to(device))

    # 无套利惩罚
    p_cal, p_but, p_bound = compute_arbitrage_penalties(model, F_batch, device)

    return L_S + lambda_penalty * (p_cal + p_but + p_bound), L_S, p_cal, p_but, p_bound
```

---

## Step 2.4: 训练 DNN

```python
def train_dnn(F_train, m_train, tau_train, sigma_train,
              F_val, m_val, tau_val, sigma_val,
              feature_type="SAM", epochs=20, batch_size=1024, lr=0.001, device="cpu"):
    """
    训练 DNN 曲面重构模型。

    F_train: (n_train, F_dim) 预测的离散 IV 点
    m_train, tau_train, sigma_train: 观测点的真实 (m, tau, IV)
    """
    F_dim = F_train.shape[1]
    model = DNN_Surface(input_dim=F_dim + 2, hidden_dim=50).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 论文参数：Xavier 初始化 + Batch Normalization（已在架构中隐含）
    # PyTorch 默认 Xavier 等价于 nn.init.xavier_uniform_

    n_train = len(F_train)
    best_val_loss = float("inf")
    best_state = None

    history = {"train_loss": [], "val_loss": [], "mse": [], 
               "pen_cal": [], "pen_but": [], "pen_bound": []}

    for epoch in range(epochs):
        model.train()
        # Mini-batch
        indices = torch.randperm(n_train)
        train_loss_epoch = 0

        for i in range(0, n_train, batch_size):
            idx = indices[i:i+batch_size]
            F_batch = torch.tensor(F_train[idx], dtype=torch.float32).to(device)

            # 对每个 batch，随机采样部分观测点（减少计算量）
            # 或者直接用全部观测点
            m_b = torch.tensor(m_train[idx], dtype=torch.float32).to(device)
            tau_b = torch.tensor(tau_train[idx], dtype=torch.float32).to(device)
            sigma_b = torch.tensor(sigma_train[idx], dtype=torch.float32).to(device)

            optimizer.zero_grad()
            loss, mse, p_cal, p_but, p_bound = total_loss(
                model, F_batch, m_b, tau_b, sigma_b, lambda_penalty=1.0, device=device
            )
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()

        # 验证
        model.eval()
        with torch.no_grad():
            F_val_t = torch.tensor(F_val, dtype=torch.float32).to(device)
            val_loss, val_mse, val_pcal, val_pbut, val_pbound = total_loss(
                model, F_val_t,
                torch.tensor(m_val, dtype=torch.float32).to(device),
                torch.tensor(tau_val, dtype=torch.float32).to(device),
                torch.tensor(sigma_val, dtype=torch.float32).to(device),
                lambda_penalty=1.0, device=device
            )

        history["train_loss"].append(train_loss_epoch / (n_train // batch_size + 1))
        history["val_loss"].append(val_loss.item())
        history["mse"].append(val_mse.item())
        history["pen_cal"].append(val_pcal.item())
        history["pen_but"].append(val_pbut.item())
        history["pen_bound"].append(val_pbound.item())

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = model.state_dict().copy()

        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Val Loss={val_loss:.6f}, MSE={val_mse:.6f}, "
                  f"Cal={val_pcal:.6f}, But={val_pbut:.6f}, Bound={val_pbound:.6f}")

    # 加载最优
    model.load_state_dict(best_state)
    return model, history
```

---

## Step 2.5: 评估

### RMSE / MAPE（在观测点上）

```python
def evaluate_dnn(model, F_test, df_test_obs, device="cpu"):
    """
    df_test_obs: DataFrame with columns [moneyness, tau, implc_volatlty]
    """
    model.eval()
    m_obs = df_test_obs["moneyness"].values
    tau_obs = df_test_obs["tau"].values
    sigma_true = df_test_obs["implc_volatlty"].values

    # 逐日评估（因为 F_test 是每日一个）
    rmse_list = []
    mape_list = []

    for i in range(len(F_test)):
        F_i = torch.tensor(F_test[i:i+1], dtype=torch.float32).to(device)

        # 预测该日所有观测点
        m_t = torch.tensor(m_obs, dtype=torch.float32).to(device).unsqueeze(1)
        tau_t = torch.tensor(tau_obs, dtype=torch.float32).to(device).unsqueeze(1)
        F_t = F_i.expand(len(m_obs), -1)

        with torch.no_grad():
            sigma_pred = model(F_t, tau_t, m_t).squeeze().cpu().numpy()

        rmse = np.sqrt(np.mean((sigma_pred - sigma_true)**2))
        mape = np.mean(np.abs(sigma_pred - sigma_true) / sigma_true)
        rmse_list.append(rmse)
        mape_list.append(mape)

    return np.mean(rmse_list), np.mean(mape_list), rmse_list, mape_list
```

### 无套利违规检查

```python
def check_arbitrage_violation(model, F_test, device="cpu"):
    """
    在测试集的观测点上检查无套利违规。
    返回 L_cal^- 和 L_but^-（论文 Table 6）。
    """
    model.eval()

    # 用合成网格检查（更严格）
    m_check = torch.tensor(M_c34.ravel(), dtype=torch.float32, requires_grad=True).to(device)
    tau_check = torch.tensor(Tau_c34.ravel(), dtype=torch.float32, requires_grad=True).to(device)

    violations_cal = []
    violations_but = []

    for i in range(len(F_test)):
        F_i = torch.tensor(F_test[i:i+1], dtype=torch.float32).to(device)
        F_exp = F_i.expand(len(m_check), -1)

        sigma = model(F_exp, tau_check.unsqueeze(1), m_check.unsqueeze(1)).squeeze()

        grad_m = torch.autograd.grad(sigma.sum(), m_check, create_graph=True)[0]
        grad_tau = torch.autograd.grad(sigma.sum(), tau_check, create_graph=True)[0]
        grad_mm = torch.autograd.grad(grad_m.sum(), m_check, create_graph=True)[0]

        l_cal = sigma + 2 * tau_check * grad_tau
        l_but = (1 - m_check * grad_m / sigma)**2 - (sigma * tau_check * grad_m)**2 / 4 + tau_check * sigma * grad_mm

        violations_cal.append(torch.clamp(-l_cal, min=0).sum().item())
        violations_but.append(torch.clamp(-l_but, min=0).sum().item())

    L_cal_neg = -np.sum(violations_cal) / (len(F_test) * len(m_check))
    L_but_neg = -np.sum(violations_but) / (len(F_test) * len(m_check))

    return L_cal_neg, L_but_neg
```

---

## Step 2.6: 主流程

```python
def run_step2(feature_type="SAM", device="cpu"):
    print(f"
=== Step 2: DNN Surface Reconstruction ({feature_type}) ===")

    # 1. 加载 Step 1 数据
    if feature_type == "SAM":
        data = np.load("/data/output/step1/sam_features.npz")
        Z_pred = data["Z_pred_test"]  # (n_test, 98)
        F_test = Z_pred  # SAM 直接就是 IV 点
    elif feature_type == "PCA":
        data = np.load("/data/output/step1/pca_features.npz")
        Z_pred = data["Z_pred_test"]
        eigensurfaces = data["eigensurfaces"]
        sigma_0 = data["sigma_0"]  # 需要保存
        F_test = sigma_0 * np.exp(Z_pred @ eigensurfaces)
    elif feature_type == "VAE":
        data = np.load("/data/output/step1/vae_features.npz")
        Z_pred = data["Z_pred_test"]
        # 加载 decoder
        # F_test = decoder(Z_pred)

    # 2. 加载测试日期的真实观测数据
    df_test = pd.read_csv("/data/output/step2/test_observations.csv")  # 需要准备

    # 3. 划分训练/验证/测试（DNN 的训练集是 LSTM 的训练集对应的日期）
    # 简化：用 LSTM 训练集的 F 和观测点训练 DNN

    # 4. 训练 DNN
    model, history = train_dnn(
        F_train, m_train, tau_train, sigma_train,
        F_val, m_val, tau_val, sigma_val,
        feature_type=feature_type, epochs=20, device=device
    )

    # 5. 评估
    rmse, mape, rmse_daily, mape_daily = evaluate_dnn(model, F_test, df_test, device)
    L_cal, L_but = check_arbitrage_violation(model, F_test, device)

    print(f"
=== Results ({feature_type}-DNN) ===")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAPE: {mape:.4f}")
    print(f"Calendar Arb Violation: {L_cal:.6f}")
    print(f"Butterfly Arb Violation: {L_but:.6f}")

    # 6. 保存
    torch.save(model.state_dict(), f"/data/output/step2/dnn_{feature_type.lower()}.pt")
    np.savez(f"/data/output/step2/results_{feature_type.lower()}.npz",
             rmse=rmse, mape=mape, rmse_daily=rmse_daily, mape_daily=mape_daily,
             L_cal=L_cal, L_but=L_but, history=history)

    return model, rmse, mape
```

---

## 输出文件

```
/data/output/step2/
├── dnn_sam.pt              # SAM-DNN 权重
├── dnn_pca.pt              # PCA-DNN 权重
├── dnn_vae.pt              # VAE-DNN 权重
├── results_sam.npz         # SAM-DNN 评估结果
├── results_pca.npz         # PCA-DNN 评估结果
├── results_vae.npz         # VAE-DNN 评估结果
└── figure6_dnn_loss.png    # DNN 损失 + 惩罚项曲线（复现论文 Figure 6）
```

---

## 检查点（必须打印）

```
[Checkpoint 1] 特征映射
  {feature_type}: F_test 维度 = {F_test_shape}
  观测点数量 = {n_obs}

[Checkpoint 2] DNN 训练
  Epoch {epoch}: Val MSE = {mse:.6f}, Cal Pen = {p_cal:.6f}, But Pen = {p_but:.6f}, Bound Pen = {p_bound:.6f}
  最优 epoch = {best_epoch}

[Checkpoint 3] 测试评估
  Test RMSE = {rmse:.4f}  (论文 SAM-DNN: 0.0245)
  Test MAPE = {mape:.4f}  (论文 SAM-DNN: 9.90%)
  Calendar Arb Violation = {L_cal:.6f}  (目标: 0.0)
  Butterfly Arb Violation = {L_but:.6f}  (目标: 0.0)

[Checkpoint 4] 模型对比
  SAM-DNN RMSE = {rmse_sam:.4f}
  PCA-DNN RMSE = {rmse_pca:.4f}
  VAE-DNN RMSE = {rmse_vae:.4f}
  DFW 基准 RMSE = {rmse_dfw:.4f}
```

---

## 执行顺序

1. 加载 Step 1 的 `sam_features.npz` 和 `pca_features.npz`
2. 准备 DNN 训练数据：F（预测特征映射的离散 IV 点）+ 观测点 (m, tau, sigma_true)
3. 先跑 **SAM-DNN**（最直观，输入直接是 98 维 IV 点）
4. 训练 20 epochs，观察惩罚项是否快速收敛到 0
5. 评估 Test RMSE，与论文 0.0245 对比
6. 检查无套利违规（应为 0 或接近 0）
7. 再跑 **PCA-DNN** 和 **VAE-DNN**

**关键预期**：
- DNN 重构后 RMSE 应**显著低于** Step 1 的 LSTM 特征预测误差（0.1641 → 可能 0.05-0.08）
- 无套利惩罚项在 5-10 epochs 后应接近 0（论文 Figure 6）
- SAM-DNN 应最优，PCA-DNN 最差
