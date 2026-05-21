# -*- coding: utf-8 -*-
"""
Step 0 插值 —— 带检查点恢复的长运行包装器

解决后台 shell 超时/崩溃问题：
  - 每处理 CHECKPOINT_INTERVAL 天保存一次中间状态
  - 崩溃后重新运行可自动恢复
  - 日志直接写入文件（flush=True）
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from step0_interpolation import (
    PROJECT_ROOT,
    DATA_PATH,
    OUTPUT_DIR,
    COL_DATE,
    COL_IV,
    preprocess,
    get_tau_grid,
    M_GRID,
    MIN_OBS_PER_DAY,
    dfw_fit,
    dfw_predict,
    nw_interpolate_vec,
    mean_squared_error,
    NW_DEFAULT_BANDWIDTH,
    save_outputs,
    print_checkpoints,
)

CHECKPOINT_INTERVAL = 200  # 每 200 天保存一次
CHECKPOINT_DIR = PROJECT_ROOT / "output" / "step0_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = PROJECT_ROOT / "output" / "step0_checkpoint.log"


def log(msg: str) -> None:
    """同时打印到 stdout 和日志文件。"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def load_checkpoint() -> tuple[list[dict], list[dict]] | None:
    """加载最新的检查点。"""
    ckpt_files = sorted(CHECKPOINT_DIR.glob("checkpoint_*.npz"))
    if not ckpt_files:
        return None
    latest = ckpt_files[-1]
    log(f"[Checkpoint] 加载检查点: {latest.name}")
    data = np.load(latest, allow_pickle=True)
    results = list(data["results"])
    grid_data = list(data["grid_data"])
    log(f"[Checkpoint] 已处理 {len(results)} 天")
    return results, grid_data


def save_checkpoint(results: list[dict], grid_data: list[dict], idx: int) -> None:
    """保存检查点。"""
    path = CHECKPOINT_DIR / f"checkpoint_{idx:04d}.npz"
    np.savez(path, results=np.array(results, dtype=object), grid_data=np.array(grid_data, dtype=object))
    log(f"[Checkpoint] 保存到 {path.name} ({len(results)} 天)")


def run_step0_with_checkpoint(df: pd.DataFrame) -> tuple[pd.DataFrame, list, np.ndarray]:
    """带检查点的 Step 0 运行。"""
    tau_grid = get_tau_grid(df)

    M_mesh, Tau_mesh = np.meshgrid(M_GRID, tau_grid, indexing="ij")
    m_flat = M_mesh.ravel()
    tau_flat = Tau_mesh.ravel()

    h1_fix, h2_fix = NW_DEFAULT_BANDWIDTH

    grouped = df.groupby(COL_DATE, sort=True)
    dates = list(grouped.groups.keys())
    total = len(dates)

    # 尝试恢复
    checkpoint = load_checkpoint()
    if checkpoint:
        results, grid_data = checkpoint
        processed_dates = {r["date"] for r in results}
        start_idx = next((i for i, d in enumerate(dates) if d not in processed_dates), total)
        log(f"[Resume] 从第 {start_idx + 1}/{total} 天继续")
    else:
        results, grid_data = [], []
        start_idx = 0
        log(f"[Start] 全新开始，共 {total} 天")

    for i in range(start_idx, total):
        date = dates[i]
        df_day = grouped.get_group(date)

        if len(df_day) < MIN_OBS_PER_DAY:
            continue

        m_obs = df_day["moneyness"].values
        tau_obs = df_day["tau"].values
        iv_obs = df_day[COL_IV].values

        # DFW
        coef = dfw_fit(m_obs, tau_obs, iv_obs)
        iv_dfw_flat = dfw_predict(coef, m_flat, tau_flat)
        iv_dfw_obs = dfw_predict(coef, m_obs, tau_obs)
        rmse_dfw = np.sqrt(mean_squared_error(iv_obs, iv_dfw_obs))

        # NW (固定带宽，跳过耗时的 CV)
        iv_nw_flat = nw_interpolate_vec(m_obs, tau_obs, iv_obs, m_flat, tau_flat, h1_fix, h2_fix)
        iv_nw_obs = nw_interpolate_vec(m_obs, tau_obs, iv_obs, m_obs, tau_obs, h1_fix, h2_fix)
        rmse_nw = np.sqrt(mean_squared_error(iv_obs, iv_nw_obs))

        results.append({"date": date, "n": len(df_day), "dfw_rmse": rmse_dfw, "nw_rmse": rmse_nw, "h1": h1_fix, "h2": h2_fix})
        grid_data.append({"date": date, "iv_dfw": iv_dfw_flat, "iv_nw": iv_nw_flat})

        # 定期保存检查点
        if (i + 1) % CHECKPOINT_INTERVAL == 0 or i == total - 1:
            save_checkpoint(results, grid_data, i + 1)
            log(f"[Progress] {i + 1}/{total} ({(i + 1) / total * 100:.1f}%)  "
                f"avg_dfw={np.mean([r['dfw_rmse'] for r in results]):.4f}  "
                f"avg_nw={np.mean([r['nw_rmse'] for r in results]):.4f}")

    return pd.DataFrame(results), grid_data, tau_grid


def main():
    log("=" * 60)
    log("Step 0 插值 —— 固定带宽 NW (跳过 CV)")
    log(f"数据: {DATA_PATH}")
    log(f"NW 带宽: h1={NW_DEFAULT_BANDWIDTH[0]}, h2={NW_DEFAULT_BANDWIDTH[1]} (固定)")
    log(f"检查点间隔: 每 {CHECKPOINT_INTERVAL} 天")
    log("=" * 60)

    log("[Load] 加载数据...")
    t0 = time.time()
    df_raw = pd.read_csv(DATA_PATH)
    log(f"  原始记录: {len(df_raw):,} 条  (加载 {time.time() - t0:.1f}s)")

    t0 = time.time()
    df = preprocess(df_raw)
    log(f"[Preprocess] done ({time.time() - t0:.1f}s)")

    t0 = time.time()
    res_df, grid_data, tau_grid = run_step0_with_checkpoint(df)
    log(f"[Step 0] 完成 ({time.time() - t0:.1f}s)")

    log("[Save] 保存最终输出...")
    table1_path, grid_path = save_outputs(grid_data, res_df, tau_grid)
    log(f"  {table1_path}")
    log(f"  {grid_path}")

    print_checkpoints(df, tau_grid, res_df, grid_data)

    # 清理检查点
    for ckpt in CHECKPOINT_DIR.glob("checkpoint_*.npz"):
        ckpt.unlink()
    log("[Cleanup] 已清理临时检查点")
    log("[Done] Step 0 全部完成！")


if __name__ == "__main__":
    main()
