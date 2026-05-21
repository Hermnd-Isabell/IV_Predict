# Step 1: 特征提取 + LSTM 预测
# 复现 Zhang et al. (2023) Section 3.1 & 3.2

## 目标
从 Step 0 输出的每日 98 维网格数据，提取三种特征（SAM/PCA/VAE），并用 LSTM 预测 T+1 日特征。

---

## 输入

```python
# Step 0 输出
df_grid = pd.read_parquet("/data/output/step0/daily_grid_154.parquet")
# 列: trade_date, grid_idx, m, tau, iv_dfw, iv_nw
```

---

## Method 1: SAM (Sampling Approach)

```python
def extract_sam(df_grid):
    """
    Z_t = Sigma_bar_t  # 直接就是 98 维网格 IV 值
    h(Z) = Z           # 恒等映射
    """
    # 按日期 pivot 成 (n_days, 98) 矩阵
    df_pivot = df_grid.pivot(index="trade_date", columns="grid_idx", values="iv_dfw")
    Z = df_pivot.values  # shape: (n_days, 98)
    dates = df_pivot.index
    return Z, dates
```

---

## Method 2: PCA (Principal Component Analysis)

```python
from sklearn.decomposition import PCA

def extract_pca(df_grid, n_components=3):
    """
    1. 计算 log-IV 日变化 U_t = ln(sigma_bar_t) - ln(sigma_bar_{t-1})
    2. 对 {U_t} 做 PCA
    3. 取前 K=3 个主成分
    4. Z_t = (x1(t), x2(t), x3(t))
    """
    # 1. 还原每日 98 维网格
    df_pivot = df_grid.pivot(index="trade_date", columns="grid_idx", values="iv_dfw")
    Sigma = df_pivot.values  # (n_days, 98)
    dates = df_pivot.index

    # 2. log-IV
    log_Sigma = np.log(Sigma)

    # 3. 日变化 U_t
    U = np.diff(log_Sigma, axis=0)  # (n_days-1, 98)

    # 4. PCA
    pca = PCA(n_components=n_components)
    Z_pca = pca.fit_transform(U)  # (n_days-1, 3)

    # 5. 累积还原到 log-IV 水平（从第一天开始累加）
    Z_cumsum = np.cumsum(Z_pca, axis=0)  # (n_days-1, 3)
    Z = np.vstack([np.zeros((1, n_components)), Z_cumsum])  # (n_days, 3)

    # 保存特征曲面（用于后续还原）
    eigensurfaces = pca.components_  # (3, 98)
    explained_variance = pca.explained_variance_ratio_

    return Z, dates, eigensurfaces, explained_variance
```

**还原公式**：
```python
F_{T+1} = sigma_0 * exp(sum_{k=1}^3 x_k(T+1) * f_k)
```

---

## Method 3: VAE (Variational Autoencoder)

```python
import torch
import torch.nn as nn
import torch.optim as optim

class VAE(nn.Module):
    def __init__(self, input_dim=98, hidden_dim=128, latent_dim=10):
        super().__init__()
        # Encoder: 98 -> 128 -> 128 -> 128 -> (mu, sigma)各10维
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: 10 -> 128 -> 128 -> 128 -> 98
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta=1.0):
    """
    L = MSE(recon, x) + beta * KL(N(mu, sigma^2) || N(0, I))
    """
    RE = nn.functional.mse_loss(recon, x, reduction="sum")
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return RE + beta * KL


def train_vae(Z_raw, latent_dim=10, epochs=200, batch_size=128, lr=0.001):
    """
    Z_raw: (n_days, 98) numpy array
    """
    model = VAE(input_dim=98, latent_dim=latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(torch.tensor(Z_raw, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in loader:
            x = batch[0]
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {train_loss/len(dataset):.4f}")

    return model


def extract_vae(df_grid, latent_dim=10):
    """
    Z_t = mu(Y_t)  # encoder 输出的均值
    h(Z) = Decoder(Z)  # F_{T+1} = N_D(Z_hat_{T+1})
    """
    df_pivot = df_grid.pivot(index="trade_date", columns="grid_idx", values="iv_dfw")
    Z_raw = df_pivot.values  # (n_days, 98)
    dates = df_pivot.index

    # 训练 VAE
    model = train_vae(Z_raw, latent_dim=latent_dim)

    # 提取特征
    model.eval()
    with torch.no_grad():
        Z_tensor = torch.tensor(Z_raw, dtype=torch.float32)
        mu, _ = model.encode(Z_tensor)
        Z = mu.numpy()  # (n_days, latent_dim)

    return Z, dates, model
```

**超参数搜索**：试 `d ∈ {2, 5, 10, 15, 20}`，选 Test RMSE 最小的。

---

## LSTM 特征预测（三种特征共用同一架构）

### 输入构造（三尺度）

```python
def build_lstm_input(Z, window_month=22, window_week=5):
    """
    Z: (n_days, feature_dim)
    输出: (n_days, 3, feature_dim)  # 3 个时间步
    """
    n_days, feat_dim = Z.shape
    X = np.zeros((n_days, 3, feat_dim))

    for t in range(n_days):
        # Z^1: 月均线
        start_m = max(0, t - window_month + 1)
        X[t, 0, :] = Z[start_m:t+1].mean(axis=0) if t >= 0 else Z[0]

        # Z^2: 周均线
        start_w = max(0, t - window_week + 1)
        X[t, 1, :] = Z[start_w:t+1].mean(axis=0) if t >= 0 else Z[0]

        # Z^3: 当前值
        X[t, 2, :] = Z[t]

    return X
```

### LSTM 模型

```python
import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=12, output_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim or input_dim)

        # 输出激活
        self.activation = nn.ReLU()  # SAM: ReLU (IV>0)
        # PCA/VAE: Identity (系数可正可负)

    def forward(self, x):
        # x: (batch, 3, feature_dim)
        lstm_out, _ = self.lstm(x)  # (batch, 3, hidden_dim)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)
        out = self.fc(last_hidden)  # (batch, output_dim)
        return self.activation(out)
```

### 训练参数（论文 Table 2）

```python
def train_lstm(X_train, y_train, X_val, y_val, feature_type="SAM"):
    """
    feature_type: "SAM" | "PCA" | "VAE"
    """
    input_dim = X_train.shape[2]
    output_dim = y_train.shape[1]

    model = LSTMPredictor(input_dim, hidden_dim=12, output_dim=output_dim)

    # SAM 用 ReLU，PCA/VAE 用 Identity
    if feature_type in ["PCA", "VAE"]:
        model.activation = nn.Identity()

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 训练 200 epochs
    epochs = 200
    batch_size = 128

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_model = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.tensor(X_val, dtype=torch.float32))
            val_loss = criterion(val_pred, torch.tensor(y_val, dtype=torch.float32)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss/len(train_dataset):.6f}, Val Loss: {val_loss:.6f}")

    # 加载最优模型
    model.load_state_dict(best_model)
    return model
```

### 数据划分（时序划分，避免泄露）

```python
# 论文划分：Train 2015-2018.6, Val 2018.7-2018.12, Test 2019-2020
# 50ETF 适配：根据实际日期范围按比例划分
def temporal_split(X, y, dates, train_ratio=0.75, val_ratio=0.15):
    n = len(dates)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test
```

### 预测目标

```python
# y_t = Z_{t+1}  # 预测下一日特征
# 所以 X 从第 0 天到第 T-1 天，y 从第 1 天到第 T 天
y = Z[1:]         # (n_days-1, feature_dim)
X = X[:-1]        # (n_days-1, 3, feature_dim)
```

---

## 主流程

```python
def run_step1(df_grid, feature_type="SAM"):
    """
    feature_type: "SAM" | "PCA" | "VAE"
    """
    print(f"
=== Step 1: {feature_type} + LSTM ===")

    # 1. 特征提取
    if feature_type == "SAM":
        Z, dates = extract_sam(df_grid)
        eigensurfaces = None
        model_vae = None
    elif feature_type == "PCA":
        Z, dates, eigensurfaces, ev = extract_pca(df_grid, n_components=3)
        print(f"PCA 解释方差: {ev}")
        model_vae = None
    elif feature_type == "VAE":
        Z, dates, model_vae = extract_vae(df_grid, latent_dim=10)

    # 2. 构建 LSTM 输入
    X = build_lstm_input(Z)
    y = Z[1:]
    X = X[:-1]
    dates_y = dates[1:]

    # 3. 时序划分
    X_train, y_train, X_val, y_val, X_test, y_test = temporal_split(X, y, dates_y)

    # 4. 训练 LSTM
    model = train_lstm(X_train, y_train, X_val, y_val, feature_type=feature_type)

    # 5. 预测
    model.eval()
    with torch.no_grad():
        Z_pred_train = model(torch.tensor(X_train, dtype=torch.float32)).numpy()
        Z_pred_val = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
        Z_pred_test = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    # 6. 评估
    from sklearn.metrics import mean_squared_error
    rmse_train = np.sqrt(mean_squared_error(y_train, Z_pred_train))
    rmse_val = np.sqrt(mean_squared_error(y_val, Z_pred_val))
    rmse_test = np.sqrt(mean_squared_error(y_test, Z_pred_test))

    print(f"RMSE: Train={rmse_train:.6f}, Val={rmse_val:.6f}, Test={rmse_test:.6f}")

    # 7. 保存
    results = {
        "feature_type": feature_type,
        "Z": Z,
        "dates": dates,
        "Z_pred_test": Z_pred_test,
        "dates_test": dates_y[len(y_train)+len(y_val):],
        "rmse_train": rmse_train,
        "rmse_val": rmse_val,
        "rmse_test": rmse_test,
    }
    if feature_type == "PCA":
        results["eigensurfaces"] = eigensurfaces
    if feature_type == "VAE":
        results["model_vae"] = model_vae

    return results
```

---

## 输出文件

```python
# SAM
results_sam = run_step1(df_grid, "SAM")
np.savez("/data/output/step1/sam_features.npz", 
         Z=results_sam["Z"], 
         Z_pred=results_sam["Z_pred_test"],
         dates=results_sam["dates"],
         dates_test=results_sam["dates_test"])

# PCA
results_pca = run_step1(df_grid, "PCA")
np.savez("/data/output/step1/pca_features.npz",
         Z=results_pca["Z"],
         Z_pred=results_pca["Z_pred_test"],
         eigensurfaces=results_pca["eigensurfaces"],
         dates=results_pca["dates"],
         dates_test=results_pca["dates_test"])

# VAE（最耗时，可延后）
# results_vae = run_step1(df_grid, "VAE")
```

---

## 检查点（必须打印）

```
[Checkpoint 1] 特征提取
  SAM: Z 维度 = (n_days, 98)
  PCA: Z 维度 = (n_days, 3), 解释方差 = [ev1, ev2, ev3]
  VAE: Z 维度 = (n_days, d), 重构 MSE = {recon_mse}

[Checkpoint 2] LSTM 输入
  X 维度 = (n_samples, 3, feature_dim)
  y 维度 = (n_samples, feature_dim)
  训练/验证/测试比例 = {train_ratio}/{val_ratio}/{test_ratio}

[Checkpoint 3] LSTM 训练
  最优 epoch = {best_epoch}
  训练 RMSE = {rmse_train:.6f}
  验证 RMSE = {rmse_val:.6f}
  测试 RMSE = {rmse_test:.6f}

[Checkpoint 4] 输出文件
  sam_features.npz: Z={Z_sam_shape}, Z_pred={pred_sam_shape}
  pca_features.npz: Z={Z_pca_shape}, Z_pred={pred_pca_shape}
```

---

## 执行顺序

1. 加载 `daily_grid_154.parquet`
2. 先跑 **SAM**（最简单，无需训练）
3. 再跑 **PCA**（协方差分解，快）
4. 两者 LSTM 训练（200 epochs，约 10-20 分钟）
5. 保存 `sam_features.npz` 和 `pca_features.npz`
6. **VAE 可延后**：需要调参 d，训练耗时较长

**下一步**：拿到 SAM/PCA 的 LSTM 预测后，进入 Step 2（DNN 曲面重构）。
