# -*- coding: utf-8 -*-
"""GRU 时序预测模型 (Step 1)"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..base import BasePredictor


class GRUPredictor(BasePredictor):
    """
    GRU 时序预测模型。
    与 LSTM 接口完全一致：输入 (batch, seq_len, input_dim)，输出 (batch, output_dim)。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 12,
        num_layers: int = 1,
        dropout: float = 0.0,
        output_activation: str = "relu",
    ):
        super().__init__(input_dim, output_dim, output_activation)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (batch, seq_len, input_dim)
        返回:
            (batch, output_dim)
        """
        out, _ = self.gru(x)  # out: (batch, seq_len, hidden_dim)
        last_hidden = out[:, -1, :]  # (batch, hidden_dim)
        return self.activation(self.fc(last_hidden))
