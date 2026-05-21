# 任务：KAN + λ=1 重新训练 —— 无套利约束下的完整验证

## 目标

在已验证成功的 KAN 架构基础上，施加无套利惩罚项（λ=1）重新训练，确保输出曲面满足日历套利和蝶式套利约束，同时通过计算优化将训练时间控制在合理范围。

## 背景与问题

当前最佳结果（λ=0）：
- Test RMSE = 0.0898（历史最佳，比 MLP 降低 18.7%）
- Calendar Arb Violation = -0.00230（存在违规）
- Butterfly Arb Violation = -0.00070（存在违规）

学术要求：论文 Zhang et al. (2023) 的核心创新是 DNN 内置无套利约束。KAN 作为替代方案，必须同样满足无套利条件。

核心挑战：KAN 的 forward 涉及大量独立 MLP，自动微分计算二阶导数的计算量是 MLP 的 5-10 倍。

---

## 计算优化策略（必须实施）

### 优化 1：稀疏化无套利网格（最关键）

论文使用 1600 点密集网格（40×40）。对于 KAN 的复杂结构，可以大幅稀疏化：

```python
# 优化后网格（200 点，计算量降低 8 倍）
m_grid_c34 = np.linspace(-(2*abs(m_min))**(1/3), (2*m_max)**(1/3), 20)
tau_grid_c34 = np.exp(np.linspace(np.log(1/365), np.log(730/365 + 1), 10))
M_c34, Tau_c34 = np.meshgrid(m_grid_c34, tau_grid_c34, indexing="ij")
# 总计 20 × 10 = 200 点
```

理论依据：无套利条件是全局光滑性约束，200 个均匀分布的点已能覆盖主要违规区域。

### 优化 2：动态 λ 调度（预热 + 渐进）

```python
def get_lambda_schedule(epoch, warmup_epochs=10, target_lambda=1.0):
    if epoch < warmup_epochs:
        return target_lambda * (epoch / warmup_epochs)
    return target_lambda
```

效果：前 10 个 epoch KAN 先学习拟合数据，之后逐步引入无套利约束。

### 优化 3：惩罚项计算频率降低（每 5 个 batch 一次）

```python
penalty_interval = 5  # 每 5 个 batch 计算一次惩罚项

for batch_idx, batch in enumerate(train_loader):
    mse_loss = compute_mse_loss(model, batch)

    if batch_idx % penalty_interval == 0:
        penalty_cal, penalty_but = compute_arbitrage_penalties(model, ...)
        last_penalties = (penalty_cal, penalty_but)
    else:
        penalty_cal, penalty_but = last_penalties

    total_loss = mse_loss + lambda_penalty * (penalty_cal + penalty_but)
```

最后 5 个 epoch 恢复为每个 batch 计算，确保精确收敛。

### 优化 4：预计算网格坐标（避免重复创建 tensor）

```python
# 在模型初始化时一次性创建
self.register_buffer('m_grid_c34', torch.tensor(M_c34.ravel()))
self.register_buffer('tau_grid_c34', torch.tensor(Tau_c34.ravel()))
```

### 优化 5：混合精度训练（如果 GPU 支持）

```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    loss = compute_loss(...)
scaler.scale(loss).backward()
```

---

## 修改后的训练流程

```python
def train_kan_with_arbitrage(train_data, val_data, n_grid, model_class, model_kwargs, train_kwargs, device="cpu"):
    model = model_class(input_dim=n_grid + 2, **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_kwargs["lr"])

    # 预计算稀疏网格
    m_min, m_max = np.log(0.6), np.log(2.0)
    m_grid = np.linspace(-(2*abs(m_min))**(1/3), (2*m_max)**(1/3), 20)
    tau_grid = np.exp(np.linspace(np.log(1/365), np.log(730/365 + 1), 10))
    M_grid, Tau_grid = np.meshgrid(m_grid, tau_grid, indexing="ij")

    m_tensor = torch.tensor(M_grid.ravel(), dtype=torch.float32, requires_grad=True).to(device)
    tau_tensor = torch.tensor(Tau_grid.ravel(), dtype=torch.float32, requires_grad=True).to(device)

    best_val_loss = float("inf")
    best_state = None
    last_penalties = (torch.tensor(0.0), torch.tensor(0.0))

    for epoch in range(train_kwargs["epochs"]):
        lambda_penalty = get_lambda_schedule(epoch, train_kwargs["warmup_epochs"], train_kwargs["lambda_penalty"])

        # 最后 5 个 epoch 精确计算
        penalty_interval = 1 if epoch >= train_kwargs["epochs"] - 5 else train_kwargs["penalty_interval"]

        model.train()
        for batch_idx, batch in enumerate(train_loader):
            F_batch, m_obs, tau_obs, sigma_obs = batch

            optimizer.zero_grad()

            # MSE 损失
            sigma_pred = model(F_batch, m_obs, tau_obs)
            mse_loss = F.mse_loss(sigma_pred, sigma_obs)

            # 无套利惩罚
            if batch_idx % penalty_interval == 0 or epoch >= train_kwargs["epochs"] - 5:
                F_exp = F_batch[0:1].expand(len(m_tensor), -1)
                sigma_grid = model(F_exp, tau_tensor.unsqueeze(1), m_tensor.unsqueeze(1)).squeeze()

                grad_m = torch.autograd.grad(sigma_grid.sum(), m_tensor, create_graph=True)[0]
                grad_tau = torch.autograd.grad(sigma_grid.sum(), tau_tensor, create_graph=True)[0]
                grad_mm = torch.autograd.grad(grad_m.sum(), m_tensor, create_graph=True)[0]

                l_cal = sigma_grid + 2 * tau_tensor * grad_tau
                l_but = (1 - m_tensor * grad_m / sigma_grid)**2 - (sigma_grid * tau_tensor * grad_m)**2 / 4 + tau_tensor * sigma_grid * grad_mm

                penalty_cal = torch.clamp(-l_cal, min=0).mean()
                penalty_but = torch.clamp(-l_but, min=0).mean()
                last_penalties = (penalty_cal, penalty_but)
            else:
                penalty_cal, penalty_but = last_penalties

            total_loss = mse_loss + lambda_penalty * (penalty_cal + penalty_but)
            total_loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_loss = compute_val_loss(model, val_data)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: lambda={lambda_penalty:.3f}, Val={val_loss:.6f}, MSE={mse_loss:.6f}, Cal={penalty_cal:.6f}, But={penalty_but:.6f}")

    model.load_state_dict(best_state)
    return model
```

---

## 预期结果

| 配置 | Test RMSE | 无套利违规 | 训练时间 |
|------|-----------|-----------|---------|
| KAN (λ=0, 50ep) | **0.0898** | -0.0023 / -0.0007 | ~8.5 小时 |
| KAN (λ=1, 50ep, 稀疏网格) | **0.095-0.105** | **≈ 0 / ≈ 0** | **~12-18 小时** |

成功标准：
- Test RMSE < 0.10（仍优于 MLP 的 0.1104）
- L_cal ≈ 0.0，L_but ≈ 0.0

---

## 检查点

```
[Checkpoint 1] 网格优化
  - 稀疏网格点数: {n_sparse} (目标 200)
  - 内存占用: {mem_mb:.1f} MB

[Checkpoint 2] 训练启动
  - lambda 预热生效 (epoch 0: 0.0, epoch 10: 1.0): ✅/❌
  - 惩罚项计算间隔: {interval}

[Checkpoint 3] 训练过程（每 5 epoch）
  Epoch {epoch}: lambda={lambda:.3f}, Val MSE={val_mse:.6f}, Cal={pen_cal:.6f}, But={pen_but:.6f}

[Checkpoint 4] 最终结果
  - Test RMSE: {rmse:.4f} (目标 < 0.10)
  - Calendar Arb: {lcal:.6f} (目标 ≈ 0)
  - Butterfly Arb: {lbut:.6f} (目标 ≈ 0)

[Checkpoint 5] 与 λ=0 对比
  - λ=0 RMSE: 0.0898
  - λ=1 RMSE: {rmse_lambda1:.4f}
  - 无套利代价: {cost:.4f}
```

---

## 故障排查

| 问题 | 原因 | 解决 |
|------|------|------|
| 训练极慢（每 epoch > 30 分钟） | 惩罚项每个 batch 都计算 | 确认 penalty_interval=5 生效 |
| GPU OOM | KAN 参数多 + 网格大 | 减小 batch_size 到 512，或进一步稀疏网格到 100 点 |
| 惩罚项不收敛 | lambda 预热太慢或网格太稀疏 | 缩短 warmup 到 5 epoch，或增加网格到 400 点 |
| RMSE > 0.12 | lambda 太强，过度约束 | 降低 lambda 到 0.5 |
| NaN 损失 | KAN 梯度爆炸 | 加 gradient clipping (max_norm=1.0) |

---

## 执行顺序

1. 修改 `compute_arbitrage_penalties`：使用稀疏网格（200 点）
2. 修改 `train_step2()`：添加动态 lambda 调度、penalty_interval=5、预计算网格
3. 创建 `config_kan_lambda1.json`：lambda_penalty=1.0, warmup_epochs=10, penalty_interval=5
4. 运行 `python main.py --config config_kan_lambda1.json`
5. 监控前 10 epoch：确认 lambda 从 0 增加到 1
6. 等待 50 epochs 完成（预计 12-18 小时）

**关键提醒**：
- 如果 10 小时后还在 epoch 20 左右，不要中断
- 如果 epoch 30 后 Val MSE 不再下降且惩罚项 > 0.01，尝试降低 lambda 到 0.5
- 保存中间 checkpoint（每 10 epoch）

**预计总耗时：12-18 小时**
