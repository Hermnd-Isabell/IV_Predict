# -*- coding: utf-8 -*-
"""MLP (原 DNN_Surface) 曲面重构模型 (Step 2)"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..base import BaseSurface


class MLPSurface(BaseSurface):
    """3 层隐藏层 MLP 曲面重构模型。

    输入: F (batch, n_grid), tau (batch, 1), m (batch, 1)
    输出: sigma (batch, 1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 50,
        output_activation: str = "softplus",
    ):
        super().__init__(input_dim, output_activation)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.hidden_activation = nn.Tanh()

    def forward(
        self,
        F_input: torch.Tensor,
        tau_input: torch.Tensor,
        m_input: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([F_input, tau_input, m_input], dim=1)
        x = self.hidden_activation(self.fc1(x))
        x = self.hidden_activation(self.fc2(x))
        x = self.hidden_activation(self.fc3(x))
        x = self.fc_out(x)
        return self.activation(x)
