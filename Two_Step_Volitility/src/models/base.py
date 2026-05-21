# -*- coding: utf-8 -*-
"""
模型接口基类
定义 Step 1 预测模型和 Step 2 曲面重构模型的统一接口契约。
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


# 激活函数注册表：字符串 -> 无参构造器（既支持 nn.Module 子类，也支持 lambda）
_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "identity": nn.Identity,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
}


def make_activation(name: str) -> nn.Module:
    """根据字符串名构造一个全新的激活函数模块。

    支持的取值: ``relu``、``identity``、``tanh``、``sigmoid``、``softplus``、``leaky_relu``。
    """
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation: {name!r}. "
            f"Supported: {sorted(_ACTIVATIONS.keys())}"
        )
    return _ACTIVATIONS[name]()


class BasePredictor(nn.Module):
    """Step 1 时序预测模型基类

    所有预测模型必须满足此接口：
      - 输入: (batch, seq_len, input_dim)
      - 输出: (batch, output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        output_activation: str = "relu",
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = make_activation(output_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (batch, seq_len, input_dim)
        返回:
            (batch, output_dim)
        """
        raise NotImplementedError


class BaseSurface(nn.Module):
    """Step 2 曲面重构模型基类

    所有曲面模型必须满足此接口：
      - 输入: F (batch, n_grid), tau (batch, 1), m (batch, 1)
      - 输出: sigma (batch, 1)
    """

    def __init__(
        self,
        input_dim: int,
        output_activation: str = "softplus",
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim  # = n_grid + 2
        self.activation = make_activation(output_activation)

    def forward(
        self,
        F: torch.Tensor,
        tau: torch.Tensor,
        m: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            F:   (batch, n_grid)  预测的离散 IV 点
            tau: (batch, 1)       查询期限
            m:   (batch, 1)       查询 moneyness
        返回:
            (batch, 1) IV 值
        """
        raise NotImplementedError
