from __future__ import annotations
import os, glob, time, argparse, yaml
import numpy as np
import torch
from nnUNet.nnunetv2.paths import nnUNet_raw, nnUNet_results
from nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from gnn_model.gnn_modules import (
    ensure_inputs, _build_train_items, process_case_infer,
    gnn_model, train_one_epoch
)
from torch_geometric.loader import DataLoader as GeoDataLoader

def parse_args():
    p = argparse.ArgumentParser("GNN model")
    p.add_argument("--gnn-folder", required=True, type=str,
                   help="Folder to save GNN checkpoints and graph outputs (graph_out)")
    p.add_argument("--pred-out", required=True, type=str,
                   help="Folder to save nnU-Net predictions for GNN training(pred_for_gnn)")
    p.add_argument("--dataset-id", default=None, type=str,
                   help="nnU-Net dataset ID")
    p.add_argument("--fold", default=5, type=int,
                   help="nnU-Net fold")
    p.add_argument("--cfg", type=str, help="Path to cfg.yaml")
    return p.parse_args()

#helpers

def _path_fix(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

def _cfg_get(cfg: dict, key: str, default=None):
    return cfg[key] if key in cfg and cfg[key] is not None else default

def _expected_pred_path(pred_dir: str, stem: str) -> str:
    return os.path.join(pred_dir, f"{stem}.nii.gz")

def _missing_pred_stems(pred_dir: str, stems: list[str]) -> list[str]:
    return [s for s in stems if not os.path.isfile(_expected_pred_path(pred_dir, s))]


def main():
    args = parse_args()
    with open(args.cfg, "r") as f:
        cfg: dict = yaml.safe_load(f) or {}

    #parse args --------------------------------------------------------------------------------------------------------
    dataset_id = args.dataset_id
    pred_dir   = args.pred_out
    gnn_folder = args.gnn_folder
    if not dataset_id or not pred_dir or not gnn_folder:
        raise RuntimeError("dataset_id, pred_out or gnn_folder not defined")

    fold    = args.fold
    os.makedirs(pred_dir, exist_ok=True)
    gnn_base  = os.path.join(gnn_folder, dataset_id)
    graph_out = os.path.join(gnn_base, "graph_out")
    ckpt_dir  = gnn_base
    os.makedirs(graph_out, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    #sample Dataset ----------------------------------------------------------------------------------------------------
    images_dir = os.path.join(nnUNet_raw, dataset_id, "imagesTr")
    labels_dir = os.path.join(nnUNet_raw, dataset_id, "labelsTr")
    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))
    stems = [os.path.basename(f)[:-7] for f in label_files]
    if not stems:
        raise RuntimeError(f"No labels in {labels_dir}")

    seed = int(_cfg_get(cfg, "seed", 123))
    sample_percent = float(_cfg_get(cfg, "sample_percent", 30.0))
    rng = np.random.default_rng(seed)
    k = max(1, int(round(len(stems) * (sample_percent / 100.0))))
    picked = sorted(rng.choice(stems, size=k, replace=False).tolist())
    print(f"[nnUNet] selected {len(picked)}/{len(stems)} cases ({sample_percent:.1f}%)")

    list_of_lists = []
    picked_mods: dict[str, list[str]] = {}
    for s in picked:
        mods = sorted(glob.glob(os.path.join(images_dir, f"{s}_*.nii.gz")))
        if not mods:
            p = os.path.join(images_dir, f"{s}.nii.gz")
            if os.path.isfile(p):
                mods = [p]
        if not mods:
            print(f" images missing for {s}")
            continue
        list_of_lists.append(mods)
        picked_mods[s] = mods
    if not list_of_lists:
        raise RuntimeError("No images matched the selected stems.")

    # Predict segmentations --------------------------------------------------------------------------------------------
    skip_if_pred_exists = bool(_cfg_get(cfg, "skip_if_pred_exists", True))
    missing = _missing_pred_stems(pred_dir, list(picked_mods.keys()))
    if skip_if_pred_exists and len(missing) == 0:
        print("All selected cases already have predictions. Skipping nnU-Net inference.")
    else:
        run_lists = list_of_lists
        if skip_if_pred_exists:
            run_lists = [picked_mods[s] for s in picked if s in missing]
            print(f"[nnUNet] will predict {len(run_lists)} cases.")
        else:
            print("[nnUNet] overwriting existing predictions")

        if run_lists:
            device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
            model_folder = _cfg_get(cfg, "nnunet_model_folder")
            if model_folder:
                model_folder = _path_fix(model_folder)
            else:
                model_folder = os.path.join(
                    nnUNet_results, dataset_id, "nnUNetTrainer__nnUNetPlans__3d_fullres"
                )

            predictor = nnUNetPredictor(
                tile_step_size=float(_cfg_get(cfg, "tile_step_size", 0.5)),
                use_gaussian=bool(_cfg_get(cfg, "use_gaussian", True)),
                use_mirroring=bool(_cfg_get(cfg, "use_mirroring", True)),
                device=device, verbose=False, verbose_preprocessing=False, allow_tqdm=True
            )
            predictor.initialize_from_trained_model_folder(
                model_folder,
                use_folds=(fold,),
                checkpoint_name=str(_cfg_get(cfg, "checkpoint_name", "checkpoint_best.pth"))
            )
            predictor.predict_from_files_sequential(
                list_of_lists_or_source_folder=run_lists,
                output_folder_or_list_of_truncated_output_files=pred_dir,
                save_probabilities=bool(_cfg_get(cfg, "save_probabilities", False)),
                overwrite=bool(_cfg_get(cfg, "overwrite_preds", False)),  # default False
                folder_with_segs_from_prev_stage=None
            )
        else:
            print("[nnUNet] prediction completed")

    #Compile dataset ---------------------------------------------------------------------------------------
    image_glob = str(_cfg_get(cfg, "image_glob", "*.nii.gz"))
    triplets = ensure_inputs(images_dir, labels_dir, pred_dir, image_glob)
    if not triplets:
        raise RuntimeError("[gnn] No matched (image, GT, pred) found")
    n = len(triplets); n_train = max(1, int(0.8 * n))
    train_list = triplets[:n_train]
    infer_list = triplets

    # training and inference
    train_enable = bool(_cfg_get(cfg, "train_enable", True))
    infer_enable = bool(_cfg_get(cfg, "infer_enable", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(_cfg_get(cfg, "amp", True)) and torch.cuda.is_available()

    #training-----------------------------------------------------------------------------------------------
    model = None
    if train_enable:
        items = _build_train_items(train_list, cfg)
        assert len(items) > 0, " [gnn] No valid training graphs"
        loader = GeoDataLoader(items, batch_size=int(_cfg_get(cfg, "batch_size_cases", 2)), shuffle=True)

        in_node = int(items[0].x.shape[1])
        edge_in = int(items[0].edge_attr.shape[1])
        model = gnn_model(                                                  #instantiate GNN-model
            in_node=in_node,
            edge_in=edge_in,
            hidden=int(_cfg_get(cfg, "hidden", 96)),
            layers=int(_cfg_get(cfg, "layers", 3)),
            dropout=float(_cfg_get(cfg, "dropout", 0.2)),
        ).to(device)

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(_cfg_get(cfg, "base_lr", 1e-3)),
            weight_decay=float(_cfg_get(cfg, "weight_decay", 1e-4))
        )
        scaler = torch.amp.GradScaler("cuda", enabled=amp)

        best = float("inf")
        max_epochs = int(_cfg_get(cfg, "max_epochs", 150))
        ckpt_name = str(_cfg_get(cfg, "ckpt_name", "gnn_best.pt"))

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            tr_loss = train_one_epoch(model, loader, opt, scaler, device)
            dt = time.time() - t0
            print(f"Epoch {epoch:03d} | loss {tr_loss:.4f} | {dt:.1f}s")
            ckpt = {
                "model_state": model.state_dict(),
                "cfg": cfg,
                "in_node": in_node, "edge_in": edge_in
            }
            torch.save(ckpt, os.path.join(ckpt_dir, ckpt_name))
            if tr_loss < best:
                best = tr_loss
                torch.save(ckpt, os.path.join(ckpt_dir, "best_" + ckpt_name))

    #infer--------------------------------------------------------------------------------------------

    if infer_enable:
        if model is None:
            ckpt_name = str(_cfg_get(cfg, "ckpt_name", "gnn_best.pt"))
            ckpt_path = os.path.join(ckpt_dir, "best_" + ckpt_name)
            if os.path.isfile(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device)
                in_node = int(ckpt.get("in_node", 11))
                edge_in = int(ckpt.get("edge_in", 8))
                model = gnn_model(
                    in_node=in_node,
                    edge_in=edge_in,
                    hidden=int(_cfg_get(cfg, "hidden", 96)),
                    layers=int(_cfg_get(cfg, "layers", 3)),
                    dropout=float(_cfg_get(cfg, "dropout", 0.2)),
                ).to(device)
                model.load_state_dict(ckpt["model_state"])
                model.eval()
            else:
                print("[gnn] checkpoint not available")
                return

        for (im, _, pr) in infer_list:
            try:
                process_case_infer(img_p=im,pred_p= pr,model= model,cfg= cfg , device= device ,out_dir=graph_out)
            except Exception as e:
                print(f"[gnn][infer] {os.path.basename(pr)}: {e}")

    print("\n Training/inference completed")

if __name__ == "__main__":
    main()
