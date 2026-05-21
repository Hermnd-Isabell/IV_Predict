# -*- coding: utf-8 -*-
"""
模型注册表与工厂函数
通过字符串配置一键切换 Step 1 / Step 2 的网络架构。
"""
from __future__ import annotations

from .predictors.lstm import LSTMPredictor
from .predictors.gru import GRUPredictor
from .predictors.transformer import TransformerPredictor
from .surfaces.mlp import MLPSurface
from .surfaces.resnet import ResNetSurface
from .surfaces.kan import KANSurface

__all__ = [
    "STEP1_MODELS",
    "STEP2_MODELS",
    "get_step1_model",
    "get_step2_model",
    "LSTMPredictor",
    "GRUPredictor",
    "TransformerPredictor",
    "MLPSurface",
    "ResNetSurface",
    "KANSurface",
]

STEP1_MODELS: dict[str, type] = {
    "lstm": LSTMPredictor,
    "gru": GRUPredictor,
    "transformer": TransformerPredictor,
}

STEP2_MODELS: dict[str, type] = {
    "mlp": MLPSurface,
    "resnet": ResNetSurface,
    "kan": KANSurface,
}


def get_step1_model(model_type: str) -> type:
    """根据名称获取 Step 1 预测模型类。"""
    if model_type not in STEP1_MODELS:
        raise ValueError(
            f"Unknown Step 1 model: '{model_type}'. "
            f"Available: {list(STEP1_MODELS.keys())}"
        )
    return STEP1_MODELS[model_type]


def get_step2_model(model_type: str) -> type:
    """根据名称获取 Step 2 曲面重构模型类。"""
    if model_type not in STEP2_MODELS:
        raise ValueError(
            f"Unknown Step 2 model: '{model_type}'. "
            f"Available: {list(STEP2_MODELS.keys())}"
        )
    return STEP2_MODELS[model_type]
