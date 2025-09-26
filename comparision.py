# evaluate_only_requested_metrics.py
import os, glob, time, math
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from monai.data import Dataset, DataLoader, decollate_batch
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ToTensord
from monai.metrics import (
    SurfaceDiceMetric,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
    MeanIoU,
)
from monai.losses import SoftclDiceLoss  # fallback if skimage not available

# Optional: true clDice metric via skeletonization
try:
    from skimage.morphology import skeletonize_3d
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

# =========================
# CONFIG
# =========================
PRED_DIR = r"C:\Users\priya\PycharmProjects\nnunet-setup\helper\comparision\compare"
LABEL_DIR = r"C:\Users\priya\PycharmProjects\nnunet-setup\helper\comparision\raw"
OUT_DIR   = r"C:\Users\priya\PycharmProjects\nnunet-setup\testing\Dataset003_CoronaryMed\evaluation_metrics_se"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 1
NUM_WORKERS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPACING_MM = (1.5, 1.5, 2.0)     # evaluation spacing (kept fixed)
SURF_TOL_MM = 1.0                # Surface Dice tolerance in mm
PROB_THRESHOLD = None            # set to 0.5 if your preds are probabilities

# =========================
# HELPERS
# =========================
def list_pairs(pred_dir: str, label_dir: str) -> List[Tuple[str, str]]:
    preds = sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz")))
    gts   = sorted(glob.glob(os.path.join(label_dir, "*.nii.gz")))
    gt_map = {os.path.basename(x): x for x in gts}
    pairs = []
    for p in preds:
        bn = os.path.basename(p)
        if bn in gt_map:
            pairs.append((p, gt_map[bn]))
    return pairs

def make_dataset(pairs: List[Tuple[str, str]]) -> Dataset:
    tx = Compose([
        LoadImaged(keys=["pred", "label"]),
        EnsureChannelFirstd(keys=["pred", "label"]),
        Orientationd(keys=["pred", "label"], axcodes="RAS"),
        Spacingd(keys=["pred", "label"], pixdim=SPACING_MM, mode=("nearest", "nearest")),
        ToTensord(keys=["pred", "label"]),
    ])
    return Dataset(data=[{"pred": p, "label": l} for p, l in pairs], transform=tx)

def to_one_hot_list(batch_tensor: torch.Tensor, as_prob: bool = False):
    """
    Converts (B,1,...) tensor to list of one-hot (2,...) tensors.
    If as_prob=True, treats input as FG probability in [0,1] and builds soft one-hot.
    If PROB_THRESHOLD is set and as_prob=False, thresholds predictions first.
    """
    out = []
    for t in decollate_batch(batch_tensor):
        t = t.float()  # (1, ...)
        if as_prob:
            fg = t.clamp(0, 1)
            bg = 1.0 - fg
            one = torch.cat([bg, fg], dim=0)  # (2, ...)
        else:
            if PROB_THRESHOLD is not None:
                t = (t > PROB_THRESHOLD).float()
            # hard one-hot from {0,1}
            bg = (t == 0).float()
            fg = (t == 1).float()
            one = torch.cat([bg, fg], dim=0)
        out.append(one)
    return out

def foreground_np(one_hot: torch.Tensor) -> np.ndarray:
    """Extract foreground (class-1) binary array as numpy uint8 from one-hot (2,...) tensor."""
    return (one_hot[1] > 0.5).cpu().numpy().astype(np.uint8)

def compute_cldice_metric(pred_fg: np.ndarray, gt_fg: np.ndarray) -> float:
    """True clDice metric using 3D skeletons (requires scikit-image)."""
    eps = 1e-8
    skel_pred = skeletonize_3d(pred_fg.astype(bool)).astype(np.uint8)
    skel_gt   = skeletonize_3d(gt_fg.astype(bool)).astype(np.uint8)
    tprec = (skel_pred & gt_fg).sum() / (skel_pred.sum() + eps)
    tsens = (skel_gt & pred_fg).sum() / (skel_gt.sum() + eps)
    return float(2.0 * tprec * tsens / (tprec + tsens + eps))

def compute_cldice_fallback_torch(pred_fg_t: torch.Tensor, gt_fg_t: torch.Tensor) -> float:
    """
    Fallback: use MONAI's SoftclDiceLoss (lower is better) and convert to score.
    pred_fg_t, gt_fg_t: tensors shaped (1, Z, Y, X) with values in {0,1}
    Returns a scalar clDice-like score in [0,1].
    """
    loss = SoftclDiceLoss(iter_=3)(pred_fg_t, gt_fg_t)  # returns mean loss
    score = 1.0 - float(loss.item())
    return max(0.0, min(1.0, score))

# =========================
# MAIN EVALUATION
# =========================
def main():
    pairs = list_pairs(PRED_DIR, LABEL_DIR)
    if not pairs:
        print("No matching pred/label pairs found.")
        return

    dataset = make_dataset(pairs)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Instantiate ONLY the requested metrics
    surf_dice = SurfaceDiceMetric(
        include_background=False,
        reduction="mean",
        class_thresholds=[SURF_TOL_MM]  # tolerance for FG
    )
    hd95 = HausdorffDistanceMetric(
        include_background=False,
        reduction="mean",
        percentile=95.0
    )
    assd = SurfaceDistanceMetric(
        include_background=False,
        reduction="mean",
        symmetric=True
    )
    miou = MeanIoU(
        include_background=False,
        reduction="mean",
    )

    rows = []
    for i, batch in enumerate(loader):
        pred = batch["pred"].to(DEVICE)   # (B,1, Z, Y, X)
        lab  = batch["label"].to(DEVICE)  # (B,1, Z, Y, X)
        case_id = os.path.basename(pairs[i][1])

        # If your predictions are probabilities, set as_prob=True
        pred_1h = to_one_hot_list(pred, as_prob=(PROB_THRESHOLD is not None))
        lab_1h  = to_one_hot_list(lab,  as_prob=False)

        # --- Compute ONLY the requested metrics ---
        # Surface Dice @ SURF_TOL_MM
        surf_dice(y_pred=pred_1h, y=lab_1h)
        sd_val = float(surf_dice.aggregate().item())
        surf_dice.reset()

        # Hausdorff (95th percentile)
        hd95(y_pred=pred_1h, y=lab_1h)
        hd95_val = float(hd95.aggregate().item())
        hd95.reset()

        # ASSD (Average Symmetric Surface Distance)
        assd(y_pred=pred_1h, y=lab_1h)
        assd_val = float(assd.aggregate().item())
        assd.reset()

        # Mean IoU
        miou(y_pred=pred_1h, y=lab_1h)
        miou_val = float(miou.aggregate().item())
        miou.reset()

        # MSE on foreground channel (voxel-wise)
        pred_fg = pred_1h[0][1].unsqueeze(0)  # (1, Z, Y, X)
        lab_fg  = lab_1h[0][1].unsqueeze(0)   # (1, Z, Y, X)
        mse_val = float(F.mse_loss(pred_fg, lab_fg).item())

        # clDice metric
        pred_fg_np = foreground_np(pred_1h[0])
        lab_fg_np  = foreground_np(lab_1h[0])
        if _HAS_SKIMAGE:
            cldice_val = compute_cldice_metric(pred_fg_np, lab_fg_np)
        else:
            # fallback using SoftclDiceLoss → score = 1 - loss
            cldice_val = compute_cldice_fallback_torch(pred_fg, lab_fg)

        rows.append({
            "Image": case_id,
            f"SurfaceDice@{SURF_TOL_MM}mm": sd_val,
            "HD95_mm": hd95_val,
            "ASSD_mm": assd_val,
            "MeanIoU": miou_val,
            "MSE": mse_val,
            "clDice": cldice_val,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "per_case_metrics.csv"), index=False)

    # Aggregate (mean/std only, as requested)
    summary = df.drop(columns=["Image"]).agg(["mean", "std"]).round(6)
    summary.to_csv(os.path.join(OUT_DIR, "summary_metrics.csv"))

    print("\nPer-case metrics saved to:", os.path.join(OUT_DIR, "per_case_metrics.csv"))
    print("Summary (mean/std) saved to :", os.path.join(OUT_DIR, "summary_metrics.csv"))
    print("\nSummary:\n", summary)

if __name__ == "__main__":
    main()
