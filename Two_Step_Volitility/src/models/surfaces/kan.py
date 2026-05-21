# -*- coding: utf-8 -*-
"""KAN (Kolmogorov-Arnold Network) 曲面重构模型 (Step 2) — 高效简化版。

核心优化：用**单个共享 MLP + 索引嵌入**替代为每个连接独立建 MLP，
既保留"每个连接是可学习 1D 函数"的 KAN 直觉，又避免 Python 循环开销。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..base import BaseSurface


class KANLayer(nn.Module):
    """
    KAN 层：每个输入-输出连接是可学习的 1D 函数。

    高效实现：
      - 所有连接共享一个 MLP backbone
      - 用可学习的"连接嵌入"区分不同 (i,j) 连接
      - 输入：x_j 的值 + 连接嵌入 e_{ij}
      - 输出：f_{ij}(x_j)

    数学形式仍满足：
        y_i = Σ_j f_{ij}(x_j) + Σ_j W_{ij}·x_j
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 连接嵌入：每个 (i, j) 对应一个低维向量
        self.num_connections = in_dim * out_dim
        self.conn_embed = nn.Parameter(
            torch.randn(self.num_connections, hidden_dim) * 0.1
        )

        # 共享 MLP：输入 (1 + hidden_dim) -> 1
        # 输入 = [x_j 的值, 连接嵌入 e_{ij}]
        self.shared_mlp = nn.Sequential(
            nn.Linear(1 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 基函数（线性残差连接）
        self.base_weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (batch, in_dim)
        返回:
            (batch, out_dim)
        """
        batch_size = x.shape[0]
        device = x.device

        # 基函数输出
        base = torch.matmul(x, self.base_weight.t())  # (B, out)

        # ---- 高效 KAN 计算 ----
        # 1. 构造所有 (i,j) 对的输入：x_j 重复 num_connections 次
        # x: (B, in) -> 扩展为 (B, out, in) -> 展平为 (B * out * in, 1)
        x_expanded = x.unsqueeze(1).expand(batch_size, self.out_dim, self.in_dim)
        x_flat = x_expanded.reshape(-1, 1)  # (B * out * in, 1)

        # 2. 连接嵌入重复 batch_size 次
        # conn_embed: (out * in, hidden) -> 扩展为 (B * out * in, hidden)
        embed_expanded = self.conn_embed.unsqueeze(0).expand(
            batch_size, -1, -1
        ).reshape(-1, self.conn_embed.shape[1])

        # 3. 拼接并送入共享 MLP
        mlp_input = torch.cat([x_flat, embed_expanded], dim=1)  # (B*out*in, 1+hidden)
        kan_values = self.shared_mlp(mlp_input).squeeze(-1)  # (B*out*in,)

        # 4. reshape 回 (B, out, in) 并在 in 维度求和
        kan_out = kan_values.view(batch_size, self.out_dim, self.in_dim).sum(dim=2)
        # (B, out)

        return base + kan_out


class KANSurface(BaseSurface):
    """
    KAN 曲面重构模型。
    架构：输入 -> [KANLayer -> LayerNorm] x num_layers -> Softplus
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        kan_hidden: int = 16,
        num_layers: int = 2,
        output_activation: str = "softplus",
    ):
        super().__init__(input_dim, output_activation)

        dims = [input_dim] + [hidden_dim] * num_layers + [1]

        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(KANLayer(dims[i], dims[i + 1], kan_hidden))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))

        self.layers = nn.Sequential(*layers)

    def forward(
        self,
        F: torch.Tensor,
        tau: torch.Tensor,
        m: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([F, tau, m], dim=1)
        x = self.layers(x)
        return self.activation(x)
