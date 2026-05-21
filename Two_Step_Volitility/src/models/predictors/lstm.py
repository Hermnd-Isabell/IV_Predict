# -*- coding: utf-8 -*-
"""LSTM 时序预测模型 (Step 1)"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..base import BasePredictor


class LSTMPredictor(BasePredictor):
    """单层 LSTM + Linear 预测器，支持三尺度输入。

    输入: (batch, seq_len, input_dim)
    输出: (batch, output_dim)
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
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)
        out = self.fc(last_hidden)  # (batch, output_dim)
        return self.activation(out)
