# -*- coding: utf-8 -*-
"""Transformer / Informer 时序预测模型 (Step 1) — 预留框架"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..base import BasePredictor


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1), :]


class TransformerPredictor(BasePredictor):
    """Transformer Encoder 预测器 — 预留实现。

    输入: (batch, seq_len, input_dim)
    输出: (batch, output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        output_activation: str = "relu",
    ):
        super().__init__(input_dim, output_dim, output_activation)
        raise NotImplementedError("TransformerPredictor 尚未实现，请先完成具体网络结构。")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
