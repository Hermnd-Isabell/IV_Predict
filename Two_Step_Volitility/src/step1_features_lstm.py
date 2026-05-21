# -*- coding: utf-8 -*-
"""
Step 1: 特征提取 + LSTM 预测
复现 Zhang et al. (2023) Section 3.1 & 3.2

输入: output/step0/daily_grid_154.parquet
输出:
  - output/step1/sam_features.npz
  - output/step1/pca_features.npz
  - output/step1/vae_features.npz（可选，延后）

三种特征方法:
  1. SAM: 直接采样 98/154 维网格 IV 值
  2. PCA: log-IV 日变化的主成分分析（K=3）
  3. VAE: 变分自编码器（latent_dim 可搜）

LSTM 共用同一架构，三尺度输入（月均线/周均线/当前值）。
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from models.predictors.lstm import LSTMPredictor

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------
# 路径配置
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_GRID = PROJECT_ROOT / "output" / "spx_step0" / "daily_grid_154.parquet"
OUTPUT_DIR = PROJECT_ROOT / "output" / "spx_step1"

BANNER_WIDTH = 50

# 训练参数（论文 Table 2）
LSTM_HIDDEN = 12
LSTM_EPOCHS = 200
LSTM_BATCH_SIZE = 128
LSTM_LR = 0.01
LSTM_WINDOW_MONTH = 22
LSTM_WINDOW_WEEK = 5

# VAE 默认参数
VAE_HIDDEN = 128
VAE_EPOCHS = 200
VAE_BATCH_SIZE = 128
VAE_LR = 0.001

# ------------------------------------------------------------------
# 数据划分（时序）
# ------------------------------------------------------------------
def temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    train_ratio: float = 0.75,
    val_ratio: float = 0.15,
) -> tuple:
    """按时间顺序划分训练/验证/测试集。当 val_ratio=0 时只返回 train/test。"""
    n = len(dates)
    n_train = int(n * train_ratio)

    if val_ratio > 0:
        n_val = int(n * val_ratio)
        return (
            X[:n_train], y[:n_train],
            X[n_train : n_train + n_val], y[n_train : n_train + n_val],
            X[n_train + n_val :], y[n_train + n_val :],
        )
    else:
        # 无验证集: 返回 train, empty, test
        empty = np.empty((0, *X.shape[1:]), dtype=X.dtype)
        empty_y = np.empty((0, *y.shape[1:]), dtype=y.dtype)
        return (
            X[:n_train], y[:n_train],
            empty, empty_y,
            X[n_train:], y[n_train:],
        )


# ------------------------------------------------------------------
# Method 1: SAM (Sampling Approach)
# ------------------------------------------------------------------
def extract_sam(df_grid: pd.DataFrame, iv_col: str = "iv_dfw") -> tuple[np.ndarray, np.ndarray]:
    """
    Z_t = Sigma_bar_t  (直接就是网格 IV 值)
    h(Z) = Z           (恒等映射)
    返回 Z: (n_days, n_grid), dates: (n_days,)
    """
    df_pivot = df_grid.pivot(index="trade_date", columns="grid_idx", values=iv_col)
    Z = df_pivot.values.astype(np.float64)
    dates = df_pivot.index.values
    return Z, dates


# ------------------------------------------------------------------
# Method 2: PCA
# ------------------------------------------------------------------
def extract_pca(
    df_grid: pd.DataFrame,
    n_components: int = 3,
    iv_col: str = "iv_dfw",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    1. 还原每日网格 Sigma
    2. 计算 log-IV 日变化 U_t
    3. 对 U_t 做 PCA，取前 K 个主成分
    4. Z_t = cumsum(x_k(t))，累积还原到 log-IV 水平

    返回: Z, dates, eigensurfaces, explained_variance_ratio
    """
    df_pivot = df_grid.pivot(index="trade_date", columns="grid_idx", values=iv_col)
    Sigma = df_pivot.values.astype(np.float64)
    dates = df_pivot.index.values

    # log-IV
    log_Sigma = np.log(Sigma)

    # 日变化 U_t = diff(log(Sigma))
    U = np.diff(log_Sigma, axis=0)  # (n_days-1, n_grid)

    # PCA
    pca = PCA(n_components=n_components)
    Z_pca = pca.fit_transform(U)  # (n_days-1, K)

    # 累积还原（从第一天开始累加）
    Z_cumsum = np.cumsum(Z_pca, axis=0)  # (n_days-1, K)
    Z = np.vstack([np.zeros((1, n_components)), Z_cumsum])  # (n_days, K)

    eigensurfaces = pca.components_  # (K, n_grid)
    ev = pca.explained_variance_ratio_

    return Z, dates, eigensurfaces, ev


def reconstruct_pca(
    Z: np.ndarray,
    eigensurfaces: np.ndarray,
    sigma_0: np.ndarray,
) -> np.ndarray:
    """
    还原公式: F_{T+1} = sigma_0 * exp(sum_{k=1}^K x_k(T+1) * f_k)

    Z: (n_days, K)
    eigensurfaces: (K, n_grid)
    sigma_0: (n_grid,)
    返回: (n_days, n_grid)
    """
    return sigma_0 * np.exp(Z @ eigensurfaces)


# ------------------------------------------------------------------
# Method 3: VAE
# ------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 10):
        super().__init__()
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

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    RE = nn.functional.mse_loss(recon, x, reduction="sum")
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return RE + beta * KL


def train_vae(
    Z_raw: np.ndarray,
    latent_dim: int = 10,
    epochs: int = VAE_EPOCHS,
    batch_size: int = VAE_BATCH_SIZE,
    lr: float = VAE_LR,
) -> VAE:
    n_samples, input_dim = Z_raw.shape
    model = VAE(input_dim=input_dim, hidden_dim=VAE_HIDDEN, latent_dim=latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(Z_raw, dtype=torch.float32)
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for (batch_x,) in loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(batch_x)
            loss = vae_loss(recon, batch_x, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if epoch % 20 == 0:
            avg_loss = train_loss / n_samples
            print(f"  VAE Epoch {epoch:3d}, Loss: {avg_loss:.4f}")

    return model


def extract_vae(
    df_grid: pd.DataFrame,
    latent_dim: int = 10,
    iv_col: str = "iv_dfw",
) -> tuple[np.ndarray, np.ndarray, VAE]:
    df_pivot = df_grid.pivot(index="trade_date", columns="grid_idx", values=iv_col)
    Z_raw = df_pivot.values.astype(np.float64)
    dates = df_pivot.index.values

    print(f"  [VAE] 训练 latent_dim={latent_dim}, input_dim={Z_raw.shape[1]}")
    model = train_vae(Z_raw, latent_dim=latent_dim)

    model.eval()
    with torch.no_grad():
        Z_tensor = torch.tensor(Z_raw, dtype=torch.float32)
        mu, _ = model.encode(Z_tensor)
        Z = mu.numpy()

    # 计算重构 MSE
    with torch.no_grad():
        recon, _, _ = model(Z_tensor)
        recon_mse = nn.functional.mse_loss(recon, Z_tensor).item()
    print(f"  [VAE] 重构 MSE: {recon_mse:.6f}")

    return Z, dates, model


# ------------------------------------------------------------------
# LSTM 输入构造（三尺度）
# ------------------------------------------------------------------
def build_lstm_input(
    Z: np.ndarray,
    window_month: int = LSTM_WINDOW_MONTH,
    window_week: int = LSTM_WINDOW_WEEK,
) -> np.ndarray:
    """
    Z: (n_days, feature_dim)
    输出 X: (n_days, 3, feature_dim)
      - X[:,0,:]: 月均线 (rolling mean)
      - X[:,1,:]: 周均线
      - X[:,2,:]: 当前值
    """
    n_days, feat_dim = Z.shape
    X = np.zeros((n_days, 3, feat_dim), dtype=np.float64)

    for t in range(n_days):
        start_m = max(0, t - window_month + 1)
        X[t, 0, :] = Z[start_m : t + 1].mean(axis=0)

        start_w = max(0, t - window_week + 1)
        X[t, 1, :] = Z[start_w : t + 1].mean(axis=0)

        X[t, 2, :] = Z[t]

    return X


# ------------------------------------------------------------------
# Step 1 训练函数
# ------------------------------------------------------------------
def train_step1(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_class: type = LSTMPredictor,
    model_kwargs: dict | None = None,
    output_activation: str = "relu",
    train_kwargs: dict | None = None,
    device: str = "cpu",
) -> nn.Module:
    """训练 Step 1 时序预测模型，带验证集早停（保存最优模型）。"""
    model_kwargs = model_kwargs or {}
    train_kwargs = train_kwargs or {
        "epochs": LSTM_EPOCHS,
        "batch_size": LSTM_BATCH_SIZE,
        "lr": LSTM_LR,
    }

    input_dim = X_train.shape[2]
    output_dim = y_train.shape[1]

    model = model_class(
        input_dim=input_dim,
        output_dim=output_dim,
        output_activation=output_activation,
        **model_kwargs,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=train_kwargs["lr"])
    criterion = nn.MSELoss()

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_kwargs["batch_size"], shuffle=True
    )

    has_val = X_val is not None and len(X_val) > 0

    best_val_loss = float("inf")
    best_state: dict | None = None
    best_epoch = 0

    epochs = train_kwargs["epochs"]
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if has_val:
            # 验证
            model.eval()
            with torch.no_grad():
                val_pred = model(torch.tensor(X_val, dtype=torch.float32, device=device))
                val_loss = criterion(
                    val_pred, torch.tensor(y_val, dtype=torch.float32, device=device)
                ).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()
                best_epoch = epoch

            if epoch % 20 == 0:
                avg_train = train_loss / len(train_dataset)
                print(f"  Epoch {epoch:3d}, Train Loss: {avg_train:.6f}, Val Loss: {val_loss:.6f}")
        else:
            # 无验证集: 每 20 epoch 打印训练 loss，最后 epoch 保存状态
            if epoch % 20 == 0:
                avg_train = train_loss / len(train_dataset)
                print(f"  Epoch {epoch:3d}, Train Loss: {avg_train:.6f}")
            if epoch == epochs - 1:
                best_state = model.state_dict().copy()
                best_epoch = epoch

    if best_state is not None:
        model.load_state_dict(best_state)

    if has_val:
        print(f"  [{model_class.__name__}] 最优 epoch: {best_epoch}, best val loss: {best_val_loss:.6f}")
    else:
        print(f"  [{model_class.__name__}] 训练完成 (无验证集), epoch: {best_epoch}")
    return model


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def run_step1(
    df_grid: pd.DataFrame,
    feature_type: str = "SAM",
    vae_latent_dim: int = 10,
    model_class: type = LSTMPredictor,
    model_kwargs: dict | None = None,
    train_kwargs: dict | None = None,
    device: str = "cpu",
    val_ratio: float = 0.15,
    iv_col: str = "iv_dfw",
) -> dict:
    """
    feature_type: "SAM" | "PCA" | "VAE"
    model_class: 预测模型类（默认 LSTMPredictor）
    """
    print(f"\n{'=' * BANNER_WIDTH}")
    print(f"Step 1: {feature_type} + {model_class.__name__}")
    print(f"{'=' * BANNER_WIDTH}")

    # ---------- 1. 特征提取 ----------
    if feature_type == "SAM":
        Z, dates = extract_sam(df_grid, iv_col=iv_col)
        eigensurfaces = None
        model_vae = None
        print(f"  [Checkpoint 1] SAM ({iv_col}): Z 维度 = {Z.shape}")
    elif feature_type == "PCA":
        Z, dates, eigensurfaces, ev = extract_pca(df_grid, n_components=3, iv_col=iv_col)
        model_vae = None
        print(f"  [Checkpoint 1] PCA ({iv_col}): Z 维度 = {Z.shape}, 解释方差 = {ev}")
    elif feature_type == "VAE":
        Z, dates, model_vae = extract_vae(df_grid, latent_dim=vae_latent_dim, iv_col=iv_col)
        eigensurfaces = None
        print(f"  [Checkpoint 1] VAE ({iv_col}): Z 维度 = {Z.shape}, latent_dim={vae_latent_dim}")
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    # ---------- 2. 构建 LSTM 输入 ----------
    X = build_lstm_input(Z)
    y = Z[1:]
    X = X[:-1]
    dates_y = dates[1:]

    print(f"  [Checkpoint 2] LSTM 输入: X={X.shape}, y={y.shape}")

    # ---------- 3. 时序划分 ----------
    X_train, y_train, X_val, y_val, X_test, y_test = temporal_split(
        X, y, dates_y, train_ratio=0.75, val_ratio=val_ratio
    )
    if val_ratio > 0:
        print(
            f"  [Checkpoint 2] 划分: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}"
        )
    else:
        print(
            f"  [Checkpoint 2] 划分: Train={len(y_train)}, Test={len(y_test)} (无验证集)"
        )

    # ---------- 4. 训练 Step 1 预测模型 ----------
    output_activation = "identity" if feature_type in ("PCA", "VAE") else "relu"
    model = train_step1(
        X_train, y_train, X_val, y_val,
        model_class=model_class,
        model_kwargs=model_kwargs,
        output_activation=output_activation,
        train_kwargs=train_kwargs,
        device=device,
    )

    # ---------- 5. 预测 ----------
    model.eval()
    with torch.no_grad():
        Z_pred_train = model(torch.tensor(X_train, dtype=torch.float32, device=device)).cpu().numpy()
        Z_pred_test = model(torch.tensor(X_test, dtype=torch.float32, device=device)).cpu().numpy()

    # ---------- 6. 评估 ----------
    rmse_train = np.sqrt(mean_squared_error(y_train, Z_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, Z_pred_test))

    if val_ratio > 0 and len(y_val) > 0:
        with torch.no_grad():
            Z_pred_val = model(torch.tensor(X_val, dtype=torch.float32, device=device)).cpu().numpy()
        rmse_val = np.sqrt(mean_squared_error(y_val, Z_pred_val))
        print(f"  [Checkpoint 3] RMSE: Train={rmse_train:.6f}, Val={rmse_val:.6f}, Test={rmse_test:.6f}")
    else:
        rmse_val = float("nan")
        print(f"  [Checkpoint 3] RMSE: Train={rmse_train:.6f}, Test={rmse_test:.6f} (无验证集)")

    # ---------- 7. 组装结果 ----------
    split_offset = len(y_train) + (len(y_val) if val_ratio > 0 else 0)
    results = {
        "feature_type": feature_type,
        "Z": Z,
        "dates": dates,
        "Z_pred_test": Z_pred_test,
        "dates_test": dates_y[split_offset:],
        "rmse_train": rmse_train,
        "rmse_val": rmse_val,
        "rmse_test": rmse_test,
    }
    if feature_type == "PCA":
        results["eigensurfaces"] = eigensurfaces
        results["explained_variance"] = ev
    if feature_type == "VAE":
        results["model_vae"] = model_vae

    return results


def save_results(results: dict, out_dir: Path) -> Path:
    """保存 npz 文件。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    ft = results["feature_type"].lower()
    path = out_dir / f"{ft}_features.npz"

    data_to_save = {
        "Z": results["Z"],
        "Z_pred": results["Z_pred_test"],
        "dates": results["dates"],
        "dates_test": results["dates_test"],
        "rmse_train": results["rmse_train"],
        "rmse_val": results["rmse_val"],
        "rmse_test": results["rmse_test"],
    }
    if "eigensurfaces" in results:
        data_to_save["eigensurfaces"] = results["eigensurfaces"]
        data_to_save["explained_variance"] = results["explained_variance"]

    np.savez_compressed(path, **data_to_save)
    print(f"  Saved: {path}")

    # VAE 模型额外保存 pt 文件
    if results.get("feature_type") == "VAE" and "model_vae" in results:
        pt_path = out_dir / f"{ft}_model.pt"
        torch.save(results["model_vae"].state_dict(), pt_path)
        print(f"  Saved: {pt_path}")

    return path


def main() -> None:
    if not INPUT_GRID.exists():
        raise FileNotFoundError(f"Step 0 输出不存在: {INPUT_GRID}")

    print(f"[Load] 读取网格数据: {INPUT_GRID}")
    df_grid = pd.read_parquet(INPUT_GRID)
    print(f"  记录数: {len(df_grid)}, 列: {list(df_grid.columns)}")

    n_grid = df_grid["grid_idx"].nunique()
    print(f"  网格维度: {n_grid}")

    # ---------- SAM ----------
    results_sam = run_step1(df_grid, "SAM")
    save_results(results_sam, OUTPUT_DIR)

    # ---------- PCA ----------
    results_pca = run_step1(df_grid, "PCA")
    save_results(results_pca, OUTPUT_DIR)

    # ---------- VAE ----------
    results_vae = run_step1(df_grid, "VAE", vae_latent_dim=10)
    save_results(results_vae, OUTPUT_DIR)

    print("\n[Done] Step 1 完成。")


if __name__ == "__main__":
    main()
