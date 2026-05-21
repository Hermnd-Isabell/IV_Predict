# -*- coding: utf-8 -*-
"""ResNet 曲面重构模型 (Step 2)"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..base import BaseSurface, make_activation


class ResidualBlock(nn.Module):
    """
    ResNet 残差块：
    x -> Linear -> Activation -> Linear -> (+ x) -> Activation
    """

    def __init__(self, dim: int, activation: str = "tanh"):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = make_activation(activation)

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

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        num_blocks: int = 3,
        activation: str = "tanh",
        output_activation: str = "softplus",
    ):
        super().__init__(input_dim, output_activation)

        if hidden_dims is None:
            hidden_dims = [50, 50, 50]
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

        # 隐藏层激活函数（与残差块内部共用同一族激活）
        self.act = make_activation(activation)

    def forward(
        self,
        F_input: torch.Tensor,
        tau_input: torch.Tensor,
        m_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            F_input: (batch, n_grid) 预测的离散 IV 点
            tau_input: (batch, 1) 查询期限
            m_input: (batch, 1) 查询 moneyness
        返回:
            (batch, 1) IV 值
        """
        x = torch.cat([F_input, tau_input, m_input], dim=1)
        x = self.act(self.input_fc(x))

        for block in self.residual_blocks:
            x = block(x)

        x = self.output_fc(x)
        return self.activation(x)
