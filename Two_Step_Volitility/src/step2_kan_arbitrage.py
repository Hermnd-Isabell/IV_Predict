# -*- coding: utf-8 -*-
"""
KAN + λ=1 无套利约束训练模块
基于稀疏网格、动态 lambda 调度、penalty_interval 优化
复现 prompt_kan_lambda1_retrain.md 要求
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from step2_dnn_surface import evaluate_step2, check_arbitrage_violation, DAYS_PER_YEAR

# 稀疏网格参数 (20 x 10 = 200 点，计算量降低 8 倍)
SPARSE_M_N = 20
SPARSE_TAU_N = 10


def build_sparse_grids(device="cpu"):
    """构建稀疏无套利网格"""
    m_min, m_max = np.log(0.6), np.log(2.0)
    m_grid = np.linspace(
        -(2 * abs(m_min)) ** (1 / 3),
        (2 * m_max) ** (1 / 3),
        SPARSE_M_N,
    )
    tau_grid = np.exp(
        np.linspace(
            np.log(1 / DAYS_PER_YEAR),
            np.log(730 / DAYS_PER_YEAR + 1),
            SPARSE_TAU_N,
        )
    )
    M, Tau = np.meshgrid(m_grid, tau_grid, indexing="ij")

    m_tensor = torch.tensor(M.ravel(), dtype=torch.float32, device=device)
    tau_tensor = torch.tensor(Tau.ravel(), dtype=torch.float32, device=device)

    return m_tensor, tau_tensor


def get_lambda_schedule(epoch, warmup_epochs=10, target_lambda=1.0):
    """动态 lambda 调度：预热期线性增加"""
    if warmup_epochs <= 0:
        return target_lambda
    if epoch < warmup_epochs:
        return target_lambda * (epoch / warmup_epochs)
    return target_lambda


def compute_penalties_sparse(model, F_rep, m_grid_base, tau_grid_base, device):
    """
    在稀疏网格上计算无套利惩罚（仅返回值，不 backward，用于验证）。
    """
    n_sparse = len(m_grid_base)
    F_exp = F_rep.expand(n_sparse, -1)

    # ---- 日历套利（独立 forward） ----
    m_cal = m_grid_base.clone().requires_grad_(True)
    tau_cal = tau_grid_base.clone().requires_grad_(True)
    sigma_cal = model(F_exp, tau_cal.unsqueeze(1), m_cal.unsqueeze(1)).squeeze()
    grad_tau = torch.autograd.grad(sigma_cal.sum(), tau_cal, create_graph=True)[0]
    l_cal = sigma_cal + 2 * tau_cal * grad_tau
    penalty_cal = torch.clamp(-l_cal, min=0).mean()

    # ---- 蝶式套利（独立 forward） ----
    m_but = m_grid_base.clone().requires_grad_(True)
    tau_but = tau_grid_base.clone().requires_grad_(True)
    sigma_but = model(F_exp, tau_but.unsqueeze(1), m_but.unsqueeze(1)).squeeze()
    sigma_safe = torch.clamp(sigma_but, min=1e-6)
    grad_m = torch.autograd.grad(
        sigma_but.sum(), m_but, create_graph=True, retain_graph=True
    )[0]
    grad_mm = torch.autograd.grad(grad_m.sum(), m_but, create_graph=True)[0]
    grad_m_safe = grad_m / sigma_safe
    term1 = (1 - m_but * grad_m_safe) ** 2
    term2 = (sigma_safe * tau_but * grad_m_safe) ** 2 / 4
    term3 = tau_but * sigma_safe * grad_mm
    l_but = term1 - term2 + term3
    penalty_but = torch.clamp(-l_but, min=0).mean()

    return penalty_cal, penalty_but


def compute_and_apply_penalties(model, F_rep, m_grid_base, tau_grid_base, device, lambda_val):
    """
    在稀疏网格上计算无套利惩罚，并手动将梯度应用到模型参数。

    关键：使用 torch.autograd.grad(..., create_graph=False) 分别计算
    penalty_cal 和 penalty_but 对模型参数的梯度，避免计算图冲突。
    """
    n_sparse = len(m_grid_base)
    F_exp = F_rep.expand(n_sparse, -1)

    # ---- 日历套利 ----
    m_cal = m_grid_base.clone().requires_grad_(True)
    tau_cal = tau_grid_base.clone().requires_grad_(True)
    sigma_cal = model(F_exp, tau_cal.unsqueeze(1), m_cal.unsqueeze(1)).squeeze()
    grad_tau = torch.autograd.grad(sigma_cal.sum(), tau_cal, create_graph=True)[0]
    l_cal = sigma_cal + 2 * tau_cal * grad_tau
    penalty_cal = torch.clamp(-l_cal, min=0).mean()

    # ---- 蝶式套利 ----
    m_but = m_grid_base.clone().requires_grad_(True)
    tau_but = tau_grid_base.clone().requires_grad_(True)
    sigma_but = model(F_exp, tau_but.unsqueeze(1), m_but.unsqueeze(1)).squeeze()
    sigma_safe = torch.clamp(sigma_but, min=1e-6)
    grad_m = torch.autograd.grad(
        sigma_but.sum(), m_but, create_graph=True, retain_graph=True
    )[0]
    grad_mm = torch.autograd.grad(grad_m.sum(), m_but, create_graph=True)[0]
    grad_m_safe = grad_m / sigma_safe
    term1 = (1 - m_but * grad_m_safe) ** 2
    term2 = (sigma_safe * tau_but * grad_m_safe) ** 2 / 4
    term3 = tau_but * sigma_safe * grad_mm
    l_but = term1 - term2 + term3
    penalty_but = torch.clamp(-l_but, min=0).mean()

    # 手动计算惩罚项对模型参数的梯度
    params = [p for p in model.parameters() if p.requires_grad]

    # penalty_cal 的梯度（保留计算图以便后续计算 penalty_but）
    grads_cal = torch.autograd.grad(
        penalty_cal, params, create_graph=False, retain_graph=True, allow_unused=True
    )

    # penalty_but 的梯度
    grads_but = torch.autograd.grad(
        penalty_but, params, create_graph=False, allow_unused=True
    )

    # 手动累加梯度
    for p, g_cal, g_but in zip(params, grads_cal, grads_but):
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        if g_cal is not None:
            p.grad += lambda_val * g_cal
        if g_but is not None:
            p.grad += lambda_val * g_but

    return penalty_cal.item(), penalty_but.item()


def train_kan_with_arbitrage(
    train_data,
    val_data,
    n_grid,
    model_class,
    model_kwargs,
    output_activation="softplus",
    epochs=50,
    batch_size_days=8,
    lr=0.001,
    lambda_penalty=1.0,
    warmup_epochs=10,
    penalty_interval=5,
    device="cpu",
):
    """
    带无套利约束的 KAN 训练。

    优化策略：
    1. 稀疏网格 (200 点)
    2. 动态 lambda 调度 (warmup)
    3. penalty_interval 降低计算频率
    4. 最后 5 个 epoch 恢复精确计算
    """
    model = model_class(
        input_dim=n_grid + 2,
        output_activation=output_activation,
        **model_kwargs,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 预计算稀疏网格
    m_grid, tau_grid = build_sparse_grids(device)
    print(f"[Sparse Grid] m={SPARSE_M_N}, tau={SPARSE_TAU_N}, total={len(m_grid)} points")

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    zero_pen = torch.tensor(0.0, device=device)
    last_penalties = (zero_pen, zero_pen)

    history = {
        "train_loss": [],
        "val_loss": [],
        "mse": [],
        "pen_cal": [],
        "pen_but": [],
        "pen_bound": [],
        "lambda": [],
    }

    n_train = len(train_data)

    for epoch in range(epochs):
        # 动态 lambda
        current_lambda = get_lambda_schedule(epoch, warmup_epochs, lambda_penalty)

        # 最后 5 个 epoch 精确计算 (interval=1)
        current_interval = 1 if epoch >= epochs - 5 else penalty_interval

        model.train()
        indices = torch.randperm(n_train)
        train_loss_epoch = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size_days):
            idx = indices[i : i + batch_size_days].tolist()
            batch = [train_data[j] for j in idx]

            optimizer.zero_grad()
            batch_mse_sum = 0.0

            # MSE 损失
            for item in batch:
                F_t = torch.tensor(item["F"], dtype=torch.float32, device=device).unsqueeze(0)
                m_t = torch.tensor(item["m"], dtype=torch.float32, device=device)
                tau_t = torch.tensor(item["tau"], dtype=torch.float32, device=device)
                sigma_t = torch.tensor(item["sigma"], dtype=torch.float32, device=device)

                F_exp = F_t.expand(len(m_t), -1)
                sigma_pred = model(F_exp, tau_t.unsqueeze(1), m_t.unsqueeze(1)).squeeze()
                mse = nn.functional.mse_loss(sigma_pred, sigma_t)
                batch_mse_sum = batch_mse_sum + mse

            # 无套利惩罚（稀疏网格，按 interval 计算，手动累加梯度）
            if current_lambda > 0 and (
                n_batches % current_interval == 0 or epoch >= epochs - 5
            ):
                F_rep = torch.tensor(
                    batch[0]["F"], dtype=torch.float32, device=device
                ).unsqueeze(0)
                p_cal, p_but = compute_and_apply_penalties(
                    model, F_rep, m_grid, tau_grid, device, current_lambda
                )
                last_penalties = (p_cal, p_but)

            # MSE backward（惩罚梯度已手动累加）
            total_loss = batch_mse_sum / len(batch)
            total_loss.backward()
            optimizer.step()

            train_loss_epoch += total_loss.item()
            n_batches += 1
            if n_batches % 50 == 0:
                print(f"    batch {n_batches}/{n_train // batch_size_days + 1}...", flush=True)

        # 验证（只计算 MSE + 惩罚）
        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for item in val_data:
                F_t = torch.tensor(item["F"], dtype=torch.float32, device=device).unsqueeze(0)
                m_t = torch.tensor(item["m"], dtype=torch.float32, device=device)
                tau_t = torch.tensor(item["tau"], dtype=torch.float32, device=device)
                sigma_t = torch.tensor(item["sigma"], dtype=torch.float32, device=device)
                n_obs = len(sigma_t)
                F_exp = F_t.expand(n_obs, -1)
                sigma_pred = model(F_exp, tau_t.unsqueeze(1), m_t.unsqueeze(1)).squeeze()
                val_mse += nn.functional.mse_loss(sigma_pred, sigma_t).item()

        val_pcal = 0.0
        val_pbut = 0.0
        n_pen = 0
        if current_lambda > 0:
            # 随机采样 50 天计算惩罚，避免 436 天全部计算导致验证极慢
            rng = random.Random(42 + epoch)
            sample_val = rng.sample(val_data, min(50, len(val_data)))
            for item in sample_val:
                F_t = torch.tensor(item["F"], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.enable_grad():
                    p_cal, p_but = compute_penalties_sparse(
                        model, F_t, m_grid, tau_grid, device
                    )
                val_pcal += p_cal.item()
                val_pbut += p_but.item()
            n_pen = len(sample_val)

        n_val = len(val_data)
        val_mse /= n_val
        if current_lambda > 0 and n_pen > 0:
            val_pcal /= n_pen
            val_pbut /= n_pen

        val_total = val_mse + current_lambda * (val_pcal + val_pbut)

        history["train_loss"].append(train_loss_epoch / n_batches)
        history["val_loss"].append(val_total)
        history["mse"].append(val_mse)
        history["pen_cal"].append(val_pcal)
        history["pen_but"].append(val_pbut)
        history["pen_bound"].append(0.0)
        history["lambda"].append(current_lambda)

        if val_total < best_val_loss:
            best_val_loss = val_total
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        print(
            f"  Epoch {epoch:2d}: λ={current_lambda:.3f} "
            f"Train={train_loss_epoch / n_batches:.6f} | "
            f"Val MSE={val_mse:.6f} Cal={val_pcal:.6f} But={val_pbut:.6f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"  [DNN] 最优 epoch: {best_epoch}, best val loss: {best_val_loss:.6f}")
    return model, history


def run_step2_kan_arbitrage(
    feature_type,
    data_dict,
    model_class,
    model_kwargs=None,
    output_activation="softplus",
    train_kwargs=None,
    device="cpu",
):
    """运行 KAN + 无套利约束的完整流程"""
    model_kwargs = model_kwargs or {}
    train_kwargs = train_kwargs or {
        "epochs": 50,
        "batch_size_days": 8,
        "lr": 0.001,
        "lambda_penalty": 1.0,
        "warmup_epochs": 10,
        "penalty_interval": 5,
    }

    dataset = data_dict[feature_type]
    train_data = dataset["train"]
    val_data = dataset["val"]
    test_data = dataset["test"]
    n_grid = data_dict["n_grid"]

    n_train_obs = sum(len(d["sigma"]) for d in train_data)
    n_val_obs = sum(len(d["sigma"]) for d in val_data)
    n_test_obs = sum(len(d["sigma"]) for d in test_data)

    print(f"[Checkpoint 1] 特征映射")
    print(f"  {feature_type}: n_grid = {n_grid}")
    print(f"  训练观测点: {n_train_obs} (天数: {len(train_data)})")
    print(f"  验证观测点: {n_val_obs} (天数: {len(val_data)})")
    print(f"  测试观测点: {n_test_obs} (天数: {len(test_data)})")

    print(f"\n[Checkpoint 2] KAN + Arbitrage 训练")
    print(
        f"  epochs={train_kwargs['epochs']}, "
        f"batch_days={train_kwargs['batch_size_days']}"
    )
    print(
        f"  lambda={train_kwargs['lambda_penalty']}, "
        f"warmup={train_kwargs['warmup_epochs']}"
    )
    print(
        f"  penalty_interval={train_kwargs['penalty_interval']}, "
        f"sparse_grid={SPARSE_M_N * SPARSE_TAU_N}"
    )

    model, history = train_kan_with_arbitrage(
        train_data,
        val_data,
        n_grid,
        model_class=model_class,
        model_kwargs=model_kwargs,
        output_activation=output_activation,
        device=device,
        **train_kwargs,
    )

    print(f"\n[Checkpoint 3] 测试评估")
    rmse, mape, rmse_daily, mape_daily = evaluate_step2(model, test_data, device)
    L_cal, L_but = check_arbitrage_violation(model, test_data, device)

    print(f"  Test RMSE  = {rmse:.6f}")
    print(f"  Test MAPE  = {mape:.6f}")
    print(f"  Calendar Arb Violation (L_cal) = {L_cal:.8f}")
    print(f"  Butterfly Arb Violation (L_but) = {L_but:.8f}")

    return model, history, rmse, mape, L_cal, L_but
