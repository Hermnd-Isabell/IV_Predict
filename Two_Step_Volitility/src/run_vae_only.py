# -*- coding: utf-8 -*-
"""仅运行 Step2 VAE 部分"""
from pathlib import Path
import sys
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step2_dnn_surface import prepare_data, run_step2, evaluate_dnn, OUTPUT_DIR

STEP0_GRID = Path("output/spx_step0/daily_grid_154.parquet")
STEP1_SAM = Path("output/spx_step1/sam_features.npz")
STEP1_PCA = Path("output/spx_step1/pca_features.npz")
STEP1_VAE = Path("output/spx_step1/vae_features.npz")
RAW_DATA = Path("data/raw/spx_options.csv")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log_path = OUTPUT_DIR / "vae_training.log"
log_file = open(log_path, "w", encoding="utf-8")

def log(msg):
    print(msg, flush=True)
    log_file.write(msg + "\n")
    log_file.flush()

device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"[Device] {device}")

log("[Data] Preparing data...")
data_dict = prepare_data(STEP0_GRID, RAW_DATA, STEP1_SAM, STEP1_PCA, STEP1_VAE)

log("[Train] Starting VAE-DNN training...")
model_vae, hist_vae, rmse_vae, mape_vae, Lcal_vae, Lbut_vae = run_step2("VAE", data_dict, device)
_, _, rmse_d_vae, mape_d_vae = evaluate_dnn(model_vae, data_dict["VAE"]["test"], device)

log("[Save] Saving model and results...")
torch.save(model_vae.state_dict(), OUTPUT_DIR / "dnn_vae.pt")
np.savez(
    OUTPUT_DIR / "results_vae.npz",
    rmse=rmse_vae, mape=mape_vae,
    rmse_daily=np.array(rmse_d_vae),
    mape_daily=np.array(mape_d_vae),
    L_cal=Lcal_vae, L_but=Lbut_vae,
    hist_train_loss=np.array(hist_vae["train_loss"]),
    hist_val_loss=np.array(hist_vae["val_loss"]),
    hist_mse=np.array(hist_vae["mse"]),
    hist_pen_cal=np.array(hist_vae["pen_cal"]),
    hist_pen_but=np.array(hist_vae["pen_but"]),
    hist_pen_bound=np.array(hist_vae["pen_bound"]),
)

log(f"[VAE] Test RMSE={rmse_vae:.6f}, MAPE={mape_vae:.6f}")
log(f"[Done] VAE model saved to {OUTPUT_DIR / 'dnn_vae.pt'}")
log_file.close()
