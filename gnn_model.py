from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict
from contextlib import nullcontext
from collections import defaultdict
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_laplace,
    binary_dilation,
)
import os, yaml
from dataclasses import fields, is_dataclass, replace
from typing import get_origin, Tuple, List
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import SplineConv
from torch_cluster import knn_graph
from gnn_model.gnn_modules import load_cfg_from_yaml,gm_load,gm_predict_graph,make_graph_case,match_triplets,build_nodes,_build_train_items,process_case_infer,process_case_build_only,gnn_model,train_one_epoch

try:
    import vtk
    _HAVE_VTK = True
except Exception:
    _HAVE_VTK = False


#Config
class CFG:


    sample_percent: float = 30.0
    image_glob: str = "*.nii.gz"

    # Node sampling (from prob/seg only)
    prob_threshold: float = 0.35
    voxel_subsample_zyx: Tuple[int,int,int] = (4,4,4)
    include_shell_dilate_vox: int = 1

    # Band where outside nodes are allowed
    nms_radius_vox: Tuple[int,int,int] = (1,1,1)
    target_outside_ratio: float = 0.35  # cap outside fraction

    # Graph edges
    knn_k: int = 12
    knn_radius_mm: float = 6.0
    max_edge_len_mm: float = 8.0

    # Gap/bridge proposals
    gap_r_mm: float = 40.0
    gap_cos_min: float = 0.60
    gap_dr_mm_max: float = 10

    # Edge features (geometry-only)
    use_line_integrals: bool = True
    n_line_samples: int = 24
    vesselness_floor: float = 0.5

    # Labeling from GT connectivity
    gt_dilate_vox: int = 1
    pos_path_max_mm: float = 15.0

    # Training
    train_enable: bool = True
    neg_per_pos: int = 8
    batch_size_cases: int = 1
    max_epochs: int = 400
    base_lr: float = 1e-3
    weight_decay: float = 3e-4
    amp: bool = True
    seed: int = 100

    # Inference / export
    infer_enable: bool = True
    edge_prob_thresh: float = 0.6
    add_back_thresh: float = 0.85
    mst_lambda_len_inv: float = 0.05
    export_candidates_vtp: bool = True
    export_mst_preview_vtp: bool = True
    export_predicted_vtp: bool = True

    # Vessel belt (kept)
    belt_shells_mm: Tuple[float, ...] = (0.8, 1.6, 2.4, -0.6)
    belt_step_mm: float = 1.0
    belt_prob_max: float = 0.50
    belt_vesselness_min: float = 0.12
    belt_band_half_mm: float = 0.5
    belt_nms_vox: Tuple[int, int, int] = (1, 1, 1)
    belt_max_points: int = 20000

    # Checkpoint
    ckpt_name: str = "gnn_checkpoint.pt"

cfg = CFG()


#helpers

#main
import os, glob, time, argparse
import numpy as np
import torch
from nnUNet.nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
from nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def parse_args():
    p = argparse.ArgumentParser("GNN model")
    p.add_argument("--gnn-folder", required=True, type=str,
                   help="Folder to save GNN checkpoints and graph outputs (graph_out)")
    p.add_argument("--pred-out", required=True, type=str,
                   help="Folder to save nnU-Net predictions for GNN training(pred_for_gnn)")
    p.add_argument("--dataset-id", default=None, type=str,
                   help="nnU-Net dataset ID")
    p.add_argument("--fold", default=5, type=int,
                   help="nnU-Net fold to use")
    p.add_argument("--cfg", type=str, help="Path to cfg.yaml")
    return p.parse_args()

def _norm(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

def main():
    args = parse_args()
    cfg =CFG()
    if args.cfg:
        cfg = load_cfg_from_yaml(args.cfg, cfg)

    dataset_id = args.dataset_id
    images_dir = os.path.join(nnUNet_raw, dataset_id, "imagesTr")
    labels_dir = os.path.join(nnUNet_raw, dataset_id, "labelsTr")

    # pred dir
    pred_dir = args.pred_out
    os.makedirs(pred_dir, exist_ok=True)

    # graph_out
    gnn_base   = os.path.join(_norm(args.gnn_folder), dataset_id)
    graph_out  = os.path.join(gnn_base, "graph_out")
    ckpt_dir   = gnn_base  # save checkpoints here
    os.makedirs(graph_out, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ----- pick subset to predict -----
    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))
    stems = [os.path.basename(f)[:-7] for f in label_files]
    if not stems:
        raise RuntimeError(f"No labels in {labels_dir}")

    rng = np.random.default_rng(cfg.seed)
    k = max(1, int(round(len(stems) * (cfg.sample_percent / 100.0))))
    picked = sorted(rng.choice(stems, size=k, replace=False).tolist())
    print(f"[nnUNet] predicting {len(picked)}/{len(stems)} cases ({cfg.sample_percent:.1f}%)")

    list_of_lists = []
    for s in picked:
        mods = sorted(glob.glob(os.path.join(images_dir, f"{s}_*.nii.gz")))
        if not mods:
            p = os.path.join(images_dir, f"{s}.nii.gz")
            if os.path.isfile(p):
                mods = [p]
        if not mods:
            print(f"[skip] images missing for {s}")
            continue
        list_of_lists.append(mods)
    if not list_of_lists:
        raise RuntimeError("No images matched the selected stems.")

    # ----- nnU-Net predictor -----
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    model_folder = os.path.join(nnUNet_results, dataset_id, "nnUNetTrainer__nnUNetPlans__3d_fullres")
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        device=device, verbose=False, verbose_preprocessing=False, allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        model_folder, use_folds=(args.fold,), checkpoint_name="checkpoint_best.pth"
    )
    predictor.predict_from_files_sequential(
        list_of_lists_or_source_folder=list_of_lists,
        output_folder_or_list_of_truncated_output_files=pred_dir,
        save_probabilities=False,
        overwrite=True,
        folder_with_segs_from_prev_stage=None
    )

    # ----- GNN training as-is -----
    triplets = match_triplets(images_dir, labels_dir, pred_dir, cfg.image_glob)
    if not triplets:
        raise RuntimeError("No matched (image, GT, pred) triplets found after nnUNet prediction.")
    n = len(triplets); n_train = max(1, int(0.8 * n))
    train_list = triplets[:n_train]
    infer_list = triplets

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        cfg.amp = False

    model = None
    if cfg.train_enable:
        items = _build_train_items(train_list, cfg)
        assert len(items) > 0, "No valid training graphs. Adjust thresholds or check data."
        loader = GeoDataLoader(items, batch_size=cfg.batch_size_cases, shuffle=True)
        in_node = int(items[0].x.shape[1])
        edge_in = int(items[0].edge_attr.shape[1])
        model = gnn_model(in_node=in_node, edge_in=edge_in, hidden=96, layers=3, dropout=0.2).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

        best = 1e9
        for epoch in range(1, cfg.max_epochs + 1):
            t0 = time.time()
            tr_loss = train_one_epoch(model, loader, opt, scaler, device)
            dt = time.time() - t0
            print(f"Epoch {epoch:03d} | loss {tr_loss:.4f} | {dt:.1f}s")
            ckpt = {"model_state": model.state_dict(), "cfg": cfg.__dict__, "in_node": in_node, "edge_in": edge_in}
            torch.save(ckpt, os.path.join(ckpt_dir, cfg.ckpt_name))
            if tr_loss < best:
                best = tr_loss
                torch.save(ckpt, os.path.join(ckpt_dir, "best_" + cfg.ckpt_name))

    if cfg.infer_enable:
        if model is None:
            ckpt_path = os.path.join(ckpt_dir, "best_" + cfg.ckpt_name)
            if os.path.isfile(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
                in_node = int(ckpt.get("in_node", 5))
                edge_in = int(ckpt.get("edge_in", 6))
                model = gnn_model(in_node=in_node, edge_in=edge_in, hidden=96, layers=3, dropout=0.2).to(device)
                model.load_state_dict(ckpt["model_state"])
                model.eval()
            else:
                print("[WARN] No trained model available; skipping inference.")
                return

        for (im, _, pr) in infer_list:
            try:
                process_case_infer(im, pr, model, cfg, device)
            except Exception as e:
                print(f"[ERR][infer] {os.path.basename(pr)}: {e}")

    print("\nDone. Open the VTPs in ParaView/3D Slicer to inspect.")

if __name__ == "__main__":
    main()
