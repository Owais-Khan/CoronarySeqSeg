from __future__ import annotations
import os, glob, time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from contextlib import nullcontext
from math import radians, sin, cos
from collections import defaultdict

import argparse
import os
import sys
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_laplace,
    map_coordinates,
    binary_dilation,
)
from scipy.spatial import cKDTree
import networkx as nx
# ---- Deep Learning / GNN ----
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import SplineConv


# ---- Optional GPU kNN (torch-cluster) ----
_USE_TORCH_CLUSTER = True
try:
    if not torch.cuda.is_available():
        _USE_TORCH_CLUSTER = False
    else:
        from torch_cluster import knn_graph  # type: ignore
except Exception:
    _USE_TORCH_CLUSTER = False

# ---- Optional GPU line integrals via grid_sample ----
_USE_TORCH_SAMPLER = True

try:
    import vtk
    _HAVE_VTK = True
except Exception:
    _HAVE_VTK = False


#Config
@dataclass
class CFG:
    # I/O
    images_dir: str = r"C:\Users\priya\PycharmProjects\nnunet-setup\helper\test\images"
    labels_dir: str = r"C:\Users\priya\PycharmProjects\nnunet-setup\helper\test\labels"
    preds_dir:  str = r"C:\Users\priya\PycharmProjects\nnunet-setup\helper\test\preds"
    out_dir:    str = r"C:\Users\priya\PycharmProjects\nnunet-setup\testing\Dataset003_CoronaryMed\graphs_vtp"
    image_glob: str = "*.nii.gz"

    # Node sampling (from prob/seg only)
    prob_threshold: float = 0.35
    voxel_subsample_zyx: Tuple[int,int,int] = (4,4,4)
    include_shell_dilate_vox: int = 1

    # Tangent-guided growth around main vessels
    grow_max_mm: float = 40.0
    grow_step_mm: float = 0.7
    grow_dirs: int = 4
    grow_cone_deg: float = 15.0
    grow_early_stop_fails: int = 3  # early stop when consecutive fails

    # Band where outside nodes are allowed
    outside_band_min_mm: float = 0.8
    outside_band_max_mm: float = 8.0

    # Post-filtering for grown nodes
    nms_radius_vox: Tuple[int,int,int] = (1,1,1)
    target_outside_ratio: float = 0.35  # cap outside fraction

    # Optional uncertain belt sampling
    use_uncertain_belt: bool = True
    belt_low: float = 0.05
    belt_high: float = 0.9
    belt_step_zyx: Tuple[int,int,int] = (6,6,6)

    # Graph edges
    knn_k: int = 12
    knn_radius_mm: float = 6.0
    max_edge_len_mm: float = 8.0

    # Gap/bridge proposals
    gap_r_mm: float = 40.0
    gap_cos_min: float = 0.60
    gap_dr_mm_max: float = 10

    # Edge features
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
    max_epochs: int = 300
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

    belt_shells_mm: Tuple[float, ...] = (0.8, 1.6, 2.4, -0.6)  # +ve outside, -ve slight inside
    belt_step_mm: float = 1.0
    belt_prob_max: float = 0.50
    belt_vesselness_min: float = 0.12
    belt_band_half_mm: float = 0.5
    belt_nms_vox: Tuple[int, int, int] = (1, 1, 1)
    belt_max_points: int = 20000

    # --- Endpoint branch filling ---
    ep_seed_enable: bool = True
    ep_step_mm: float = 0.4
    ep_max_mm: float = 20.0  # reach along tangent (each direction)
    ep_per_dir: int = 24  # cap per direction per endpoint
    ep_prob_max: float = 0.50
    ep_vesselness_min: float = 0.12
    ep_edt_min_mm: float = 0.6
    ep_edt_max_mm: float = 10.0
    ep_nms_vox: Tuple[int, int, int] = (1, 1, 1)
    ep_budget: int = 20000

    # Checkpoint
    ckpt_name: str = "ecc_edgeclf.pt"

cfg = CFG()

#helpers
def _nifti_is_readable(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        nib.load(path)
        return True
    except Exception:
        try:
            r = sitk.ImageFileReader()
            r.SetFileName(path)
            r.ReadImageInformation()
            return True
        except Exception:
            return False

def read_nii_safe(path: str) -> sitk.Image:
    try:
        r = sitk.ImageFileReader()
        r.SetFileName(path)
        r.ReadImageInformation()
        size = r.GetSize()
        est = int(size[0]) * int(size[1]) * max(1,int(size[2])) * 4
        if est > 3_000_000_000:
            raise MemoryError("Large header; using nibabel fallback")
        return sitk.ReadImage(path)
    except Exception:
        img = nib.load(path)
        arr = img.get_fdata(dtype=np.float32)
        sitk_img = sitk.GetImageFromArray(np.ascontiguousarray(arr.astype(np.float32)))
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        sitk_img.SetSpacing((zooms[0], zooms[1], zooms[2]))
        sitk_img.SetOrigin((0.0,0.0,0.0))
        sitk_img.SetDirection((1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0))
        return sitk_img

def sitk_to_np(img: sitk.Image) -> np.ndarray:
    return np.ascontiguousarray(sitk.GetArrayFromImage(img))

def get_spacing_zyx(img: sitk.Image) -> np.ndarray:
    sp = np.array(img.GetSpacing(), dtype=np.float32)[::-1]
    return np.ascontiguousarray(sp)

def _mm_to_step_zyx(target_mm: float, sp_zyx: np.ndarray) -> Tuple[int,int,int]:
    return tuple(int(max(1, round(float(target_mm) / float(s)))) for s in sp_zyx)

def _in_bounds_zyx(coords: np.ndarray, shape_zyx: Tuple[int,int,int]) -> np.ndarray:
    Z,Y,X = shape_zyx
    return (coords[:,0] >= 0) & (coords[:,0] < Z) & \
           (coords[:,1] >= 0) & (coords[:,1] < Y) & \
           (coords[:,2] >= 0) & (coords[:,2] < X)



def match_triplets(images_dir: str, labels_dir: str, preds_dir: str, pattern: str) -> List[Tuple[str,str,str]]:
    def stem(p: str) -> str:
        s = os.path.basename(p)
        if s.endswith(".nii.gz"): s = s[:-7]
        if "_0000" in s: s = s.replace("_0000", "")
        return s

    imgs  = sorted(glob.glob(os.path.join(images_dir, pattern)))
    gts   = sorted(glob.glob(os.path.join(labels_dir, pattern)))
    preds = sorted(glob.glob(os.path.join(preds_dir,  pattern)))

    idx: Dict[str, List[Optional[str]]] = {}
    for p in imgs:  idx.setdefault(stem(p), [None,None,None])[0] = p
    for p in gts:   idx.setdefault(stem(p), [None,None,None])[1] = p
    for p in preds: idx.setdefault(stem(p), [None,None,None])[2] = p

    out, bad = [], []
    for k,(im,gt,pr) in idx.items():
        if not (im and gt and pr):
            bad.append((k,"missing file")); continue
        if not (_nifti_is_readable(gt) and _nifti_is_readable(pr)):
            bad.append((k,"unreadable nifti")); continue
        out.append((im,gt,pr))
    if bad:
        print(f"[match_triplets] Skipping {len(bad)} case(s):")
        for k,why in bad[:10]:
            print("  -", k, "->", why)
        if len(bad) > 10: print("  ...")
    return out



def voxel_to_phys(coords_zyx: np.ndarray, img: sitk.Image) -> np.ndarray:
    sp = np.asarray(img.GetSpacing(), dtype=np.float64)  # (x,y,z)
    org = np.asarray(img.GetOrigin(), dtype=np.float64)
    D   = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3,3)
    ijk_xyz = coords_zyx[:, ::-1].astype(np.float64) * sp
    out = (D @ ijk_xyz.T).T + org  # (N,3) XYZ mm
    return np.ascontiguousarray(out, dtype=np.float32)

def tangents_from_edt(edt_zyx: np.ndarray, coords_zyx: np.ndarray) -> np.ndarray:
    """Fast tangent estimate: negative gradient of EDT (points inward); normalize."""
    if coords_zyx.size == 0:
        return np.zeros((0,3), np.float32)
    gz, gy, gx = np.gradient(edt_zyx)  # z,y,x
    g = np.stack([gz, gy, gx], axis=-1)  # [Z,Y,X,3]
    v = g[coords_zyx[:,0], coords_zyx[:,1], coords_zyx[:,2]].astype(np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    t = -v / n
    return np.ascontiguousarray(t, dtype=np.float32)

def knn_edges(pos_mm: np.ndarray, k: int, r_mm: float, max_len_mm: float) -> np.ndarray:
    """Prefer GPU torch-cluster kNN if available; fallback to cKDTree."""
    if len(pos_mm) == 0:
        return np.zeros((2,0), np.int64)

    if _USE_TORCH_CLUSTER:
        P = torch.from_numpy(pos_mm.astype(np.float32)).to("cuda")
        kk = min(int(k)+1, len(pos_mm))  # include self; we'll unique later
        ei = knn_graph(P, k=kk, loop=False)  # [2,E], directed
        u = ei[0].detach().cpu().numpy()
        v = ei[1].detach().cpu().numpy()
        # filter by r and length
        if r_mm > 0 or max_len_mm > 0:
            d = np.linalg.norm(pos_mm[u] - pos_mm[v], axis=1)
            keep = np.ones_like(d, bool)
            if r_mm > 0: keep &= (d <= float(r_mm))
            if max_len_mm > 0: keep &= (d <= float(max_len_mm))
            u, v = u[keep], v[keep]
        a = np.minimum(u, v); b = np.maximum(u, v)
        if a.size == 0:
            return np.zeros((2,0), np.int64)
        return np.unique(np.stack([a, b], axis=0), axis=1).astype(np.int64)


    tree = cKDTree(pos_mm)
    dists, inds = tree.query(pos_mm, k=min(int(k)+1, len(pos_mm)))
    edges = set()
    for i in range(len(pos_mm)):
        js = np.atleast_1d(inds[i])[1:]
        ds = np.atleast_1d(dists[i])[1:]
        for j,dist in zip(js,ds):
            if r_mm>0 and float(dist) > float(r_mm): continue
            if max_len_mm>0 and float(dist) > float(max_len_mm): continue
            a,b = (i,int(j)) if i<int(j) else (int(j),i)
            edges.add((a,b))
    if not edges: return np.zeros((2,0), np.int64)
    return np.ascontiguousarray(np.array(sorted(edges), np.int64).T)

def _mst_endpoints(pos_mm: np.ndarray, ei: np.ndarray) -> np.ndarray:
    if ei.shape[1]==0 or pos_mm.shape[0]==0:
        return np.zeros((0,), np.int64)
    G = nx.Graph()
    for u,v in ei.T:
        w = float(np.linalg.norm(pos_mm[int(u)] - pos_mm[int(v)]))
        G.add_edge(int(u), int(v), weight=w)
    T = nx.minimum_spanning_tree(G)
    deg = dict(T.degree())
    return np.ascontiguousarray(np.array([n for n,d in deg.items() if d<=1], np.int64))

def _grid_hash_pairs(endpoints: np.ndarray, pos_mm: np.ndarray, cell: float) -> List[Tuple[int,int]]:
    """Generate candidate endpoint pairs via 3x3x3 grid hashing (fast ball approx)."""
    if len(endpoints) == 0:
        return []
    keys = np.floor(pos_mm / max(cell, 1e-6)).astype(np.int32)
    buckets: Dict[Tuple[int,int,int], List[int]] = defaultdict(list)
    for idx in endpoints:
        k = tuple(keys[int(idx)])
        buckets[k].append(int(idx))
    offsets = [(dz,dy,dx) for dz in (-1,0,1) for dy in (-1,0,1) for dx in (-1,0,1)]
    cand = set()
    for k, items in buckets.items():
        for off in offsets:
            nb = (k[0]+off[0], k[1]+off[1], k[2]+off[2])
            if nb not in buckets: continue
            neigh = buckets[nb]
            for a in items:
                for b in neigh:
                    if a>=b: continue
                    cand.add((a,b))
    return list(cand)

def add_gap_candidates(pos_mm: np.ndarray,
                       base_ei: np.ndarray,
                       tangents: np.ndarray,
                       rad_mm: np.ndarray,
                       r_gap_mm: float,
                       cos_min: float,
                       dr_mm_max: float,
                       max_len_mm: float) -> np.ndarray:
    """Propose endpoint↔endpoint gap edges using grid hashing + cheap filters first."""
    if base_ei.shape[1]==0: return base_ei
    endpoints = _mst_endpoints(pos_mm, base_ei)
    if len(endpoints)==0: return base_ei

    # coarse neighbor pairs via grid hashing (fast)
    pairs = _grid_hash_pairs(endpoints, pos_mm, cell=float(r_gap_mm))
    if not pairs:
        return base_ei

    cand = []
    for u,v in pairs:
        d = pos_mm[v]-pos_mm[u]
        L = float(np.linalg.norm(d))
        if L <= 1e-8: continue
        if (max_len_mm>0 and L>float(max_len_mm)) or (L>float(r_gap_mm)*1.5):  # quick length bounds
            continue
        if abs(float(rad_mm[u]) - float(rad_mm[v])) > float(dr_mm_max):
            continue
        dv = d / L
        cu = float(np.dot(dv, tangents[u])); cv = float(np.dot(-dv, tangents[v]))
        if 0.5*(cu+cv) < float(cos_min):
            continue
        cand.append((u,v))
    if not cand: return base_ei
    gap_ei = np.array(cand, np.int64).T
    return np.ascontiguousarray(np.unique(np.concatenate([base_ei, gap_ei], axis=1), axis=1))


# =========================
# Node sampling: helpers (prob-only)
# =========================
def prob_ridge_log(prob: np.ndarray, floor: float = 0.05) -> np.ndarray:
    """LoG on probability map to emphasize ridges (vesselness surrogate)."""
    p = np.clip(prob.astype(np.float32), 0.0, 1.0)
    resp = -gaussian_laplace(p, sigma=1.0).astype(np.float32)  # ridges positive
    resp[p < floor] = 0.0
    m = resp.max() + 1e-6
    return np.ascontiguousarray((resp / m).astype(np.float32))

def voxel_sample_coords(mask: np.ndarray, step_zyx: Tuple[int,int,int]) -> np.ndarray:
    if not mask.any(): return np.zeros((0,3), np.int64)
    Z,Y,X = mask.shape
    sz,sy,sx = [max(1,int(s)) for s in step_zyx]
    zz,yy,xx = np.meshgrid(np.arange(0,Z,sz), np.arange(0,Y,sy), np.arange(0,X,sx), indexing='ij')
    grid = np.stack([zz,yy,xx], -1).reshape(-1,3)
    keep = mask[grid[:,0], grid[:,1], grid[:,2]]
    return np.ascontiguousarray(grid[keep])

def uncertain_band_nodes(pred_np: np.ndarray, low: float, high: float, step_zyx: Tuple[int,int,int]) -> np.ndarray:
    band = (pred_np >= low) & (pred_np <= high)
    return voxel_sample_coords(band, step_zyx)

def nms_coords(coords_zyx: np.ndarray, radius_vox=(2,2,2)) -> np.ndarray:
    if len(coords_zyx) == 0: return coords_zyx
    rz, ry, rx = [max(1,int(r)) for r in radius_vox]
    key = (coords_zyx // np.array([rz,ry,rx], np.int64)).astype(np.int64)
    _, uniq_idx = np.unique(key, axis=0, return_index=True)
    return np.ascontiguousarray(coords_zyx[np.sort(uniq_idx)])

def prune_by_distance_to_pred(coords_zyx, pred_thr, img, dmin_mm, dmax_mm):
    edt_mm = distance_transform_edt(~pred_thr, sampling=img.GetSpacing()[::-1]).astype(np.float32)
    d = edt_mm[coords_zyx[:,0], coords_zyx[:,1], coords_zyx[:,2]]
    keep = (d >= float(dmin_mm)) & (d <= float(dmax_mm))
    return np.ascontiguousarray(coords_zyx[keep])

def phys_to_continuous_index(XYZ: np.ndarray, img: sitk.Image) -> np.ndarray:
    out = []
    for p in XYZ.reshape(-1,3):
        x,y,z = img.TransformPhysicalPointToContinuousIndex((float(p[0]), float(p[1]), float(p[2])))
        out.append((x,y,z))
    return np.ascontiguousarray(np.array(out, dtype=np.float64).reshape(XYZ.shape))

def probe_growth_points_prob(
    pred_np: np.ndarray,
    coords_in_zyx: np.ndarray,
    tangents_in: np.ndarray,
    img: sitk.Image,
    cfg: CFG
) -> np.ndarray:
    """Grow candidate outside nodes along ±tangent short rays; keep high LoG(prob) + low-prob points."""
    if len(coords_in_zyx)==0: return np.zeros((0,3), np.int64)
    vess = prob_ridge_log(pred_np, floor=cfg.vesselness_floor)
    sp_xyz = np.array(img.GetSpacing(), dtype=np.float64)   # (x,y,z)
    D = np.array(img.GetDirection(), dtype=np.float64).reshape(3,3)

    half_cone = radians(float(cfg.grow_cone_deg))
    n_dirs = max(2, int(cfg.grow_dirs))
    alphas = np.linspace(-half_cone, +half_cone, max(1,n_dirs//2), endpoint=True)

    picked = []
    step = float(cfg.grow_step_mm)
    steps = max(1, int(cfg.grow_max_mm/step))
    stride = max(1, len(coords_in_zyx)//2000)  # ≤ ~2000 seeds
    seed_idx = np.arange(0, len(coords_in_zyx), stride, dtype=int)
    max_fail = int(cfg.grow_early_stop_fails)

    for i in seed_idx:
        z,y,x = coords_in_zyx[i]
        X,Y,Z = img.TransformIndexToPhysicalPoint((int(x),int(y),int(z)))
        # tangent (ZYX vox) → physical XYZ
        t_vox = tangents_in[i].astype(np.float64)
        t_xyz_vox = np.array([t_vox[2], t_vox[1], t_vox[0]], dtype=np.float64)
        t_phys = (D @ (t_xyz_vox * sp_xyz)).astype(np.float64)
        n = np.linalg.norm(t_phys) + 1e-8
        t_hat = t_phys / n

        dirs = []
        for sign in (-1.0, +1.0):
            base = sign * t_hat
            a = np.array([1.0,0.0,0.0], dtype=np.float64)
            if abs(np.dot(a, base)) > 0.9: a = np.array([0.0,1.0,0.0], dtype=np.float64)
            u = np.cross(base, a); u /= (np.linalg.norm(u)+1e-8)
            if n_dirs <= 2:
                dirs.append(base)
            else:
                for ang in alphas:
                    dirs.append((base*cos(ang) + u*sin(ang)))

        for dvec in dirs:
            pts_xyz = np.array([[X,Y,Z] + (j*step)*dvec for j in range(1, steps+1)], dtype=np.float64)
            cidx = phys_to_continuous_index(pts_xyz, img)  # (N,3) (x,y,z)
            c_zyx = cidx[:, ::-1].T                        # (3,N) (z,y,x) for map_coordinates

            v_vals = map_coordinates(vess,   c_zyx, order=1, mode='nearest')     # ridge strength
            p_vals = map_coordinates(pred_np, c_zyx, order=1, mode='nearest')    # seg/prob

            fails = 0
            mask = []
            for j in range(len(v_vals)):
                keep = (p_vals[j] <= float(cfg.prob_threshold)) and (v_vals[j] >= 0.12)
                mask.append(keep)
                if keep:
                    fails = 0
                else:
                    fails += 1
                if fails >= max_fail:
                    break
            mask_out = np.array(mask, dtype=bool)
            if not np.any(mask_out):
                continue

            round_idx = np.round(c_zyx.T[:len(mask_out)]).astype(np.int64)   # (N,3)
            cand = round_idx[mask_out]                                       # (M,3)
            Zmax,Ymax,Xmax = vess.shape
            ok = (cand[:,0]>=0)&(cand[:,0]<Zmax)&(cand[:,1]>=0)&(cand[:,1]<Ymax)&(cand[:,2]>=0)&(cand[:,2]<Xmax)
            if np.any(ok):
                picked.append(cand[ok])

    if not picked:
        return np.zeros((0,3), np.int64)
    out = np.unique(np.vstack(picked), axis=0)
    return np.ascontiguousarray(out)

def belt_nodes_from_edt_and_vesselness(
    pred_np: np.ndarray,                  # float32 [Z,Y,X] in [0,1] or {0,1}
    pred_thr: np.ndarray,                 # bool [Z,Y,X]
    edt_outside_mm: np.ndarray,           # float32 [Z,Y,X] mm distance outside (0 inside)
    vess_np: np.ndarray,                  # float32 [Z,Y,X] LoG-based vesselness surrogate
    sitk_img: sitk.Image,
    shells_mm: Tuple[float, ...],
    target_step_mm: float,
    prob_max: float,
    vesselness_min: float,
    band_half_mm: float,
    nms_vox: Tuple[int,int,int],
    max_points: int
) -> np.ndarray:
    """
    Multi-shell sampling around boundary, with tight gates:
      prob <= prob_max, vesselness >= vesselness_min, and shell band |d - shell| <= band_half_mm.
    Positive shells sample outside, negative shells sample a thin inside rim.
    """
    sp_zyx = get_spacing_zyx(sitk_img)
    step_zyx = _mm_to_step_zyx(float(target_step_mm), sp_zyx)
    Z,Y,X = pred_np.shape
    picked = []

    # Inside EDT (for negative shells)
    edt_inside_mm = distance_transform_edt(pred_thr, sampling=sitk_img.GetSpacing()[::-1]).astype(np.float32)

    for sh in shells_mm:
        if sh > 0:
            band = (~pred_thr) & (np.abs(edt_outside_mm - float(sh)) <= float(band_half_mm))
        else:
            sh_abs = abs(float(sh))
            band = (pred_thr) & (np.abs(edt_inside_mm - sh_abs) <= float(band_half_mm))

        if not np.any(band):
            continue

        # Gating by probability and vesselness
        band &= (pred_np <= float(prob_max)) & (vess_np >= float(vesselness_min))
        if not np.any(band):
            continue

        coords = voxel_sample_coords(band, step_zyx)
        if coords.size:
            picked.append(coords)

    if not picked:
        return np.zeros((0,3), np.int64)

    coords = np.unique(np.vstack(picked), axis=0)
    # Tight NMS in voxels
    coords = nms_coords(coords, radius_vox=nms_vox)

    if coords.shape[0] > int(max_points):
        # Poisson-like thinning via simple striding
        idx = np.linspace(0, coords.shape[0]-1, int(max_points)).astype(int)
        coords = coords[idx]
    return np.ascontiguousarray(coords, dtype=np.int64)

def seed_endpoints_along_tangents_gated(
    coords_zyx: np.ndarray,
    tangents: np.ndarray,
    pred_np: np.ndarray,
    edt_outside_mm: np.ndarray,
    vess_np: np.ndarray,
    sitk_img: sitk.Image,
    step_mm: float,
    max_mm: float,
    per_dir: int,
    prob_max: float,
    vesselness_min: float,
    edt_min_mm: float,
    edt_max_mm: float,
    nms_vox: Tuple[int,int,int],
    budget: int
) -> np.ndarray:
    """
    From endpoints (deg<=1 in a quick local MST), sample points along ±tangent,
    then accept if gates pass: prob <= prob_max, vesselness >= vesselness_min,
    EDT(mm) in [edt_min_mm, edt_max_mm]. NMS + global budget cap.
    """
    if coords_zyx.shape[0] < 2:
        return np.zeros((0,3), np.int64)

    # Quick endpoint detection from a small kNN graph
    pos_mm = voxel_to_phys(coords_zyx, sitk_img)
    ei = knn_edges(pos_mm, k=8, r_mm=6.0, max_len_mm=12.0)
    endpoints = _mst_endpoints(pos_mm, ei)
    if endpoints.size == 0:
        return np.zeros((0,3), np.int64)

    sp_xyz = np.array(sitk_img.GetSpacing(), dtype=np.float64)
    D = np.array(sitk_img.GetDirection(), dtype=np.float64).reshape(3,3)

    n_steps = int(max_mm / max(1e-6, step_mm))
    accept = []

    for i in endpoints.tolist():
        z,y,x = coords_zyx[int(i)]
        X,Y,Z = sitk_img.TransformIndexToPhysicalPoint((int(x),int(y),int(z)))
        t_vox = tangents[int(i)].astype(np.float64)
        t_xyz_vox = np.array([t_vox[2], t_vox[1], t_vox[0]], dtype=np.float64)
        t_phys = (D @ (t_xyz_vox * sp_xyz)).astype(np.float64)
        t_hat = t_phys / (np.linalg.norm(t_phys) + 1e-8)

        for sgn in (-1.0, +1.0):
            pts_xyz = np.array([[X,Y,Z] + (j*step_mm)*sgn*t_hat for j in range(1, n_steps+1)], dtype=np.float64)
            cidx = phys_to_continuous_index(pts_xyz, sitk_img)     # (N,3) x,y,z in continuous index
            c_zyx = cidx[:, ::-1].T                                # (3,N) for map_coordinates

            p_vals = map_coordinates(pred_np,        c_zyx, order=1, mode='nearest')
            v_vals = map_coordinates(vess_np,        c_zyx, order=1, mode='nearest')
            d_vals = map_coordinates(edt_outside_mm, c_zyx, order=1, mode='nearest')

            # Gates: focus on gap corridor, yet still along vessel-like ridges
            keep = (p_vals <= float(prob_max)) & (v_vals >= float(vesselness_min)) & \
                   (d_vals >= float(edt_min_mm)) & (d_vals <= float(edt_max_mm))
            if not np.any(keep):
                continue

            rzyx = np.round(c_zyx.T[keep]).astype(np.int64)
            ok = _in_bounds_zyx(rzyx, pred_np.shape)
            if np.any(ok):
                # Cap per direction per endpoint
                sub = rzyx[ok]
                if sub.shape[0] > per_dir:
                    sub = sub[:per_dir]
                accept.append(sub)

    if not accept:
        return np.zeros((0,3), np.int64)

    out = np.unique(np.vstack(accept), axis=0)
    out = nms_coords(out, radius_vox=nms_vox)
    if out.shape[0] > int(budget):
        idx = np.linspace(0, out.shape[0]-1, int(budget)).astype(int)
        out = out[idx]
    return np.ascontiguousarray(out, dtype=np.int64)



def build_nodes(pred_np: np.ndarray, sitk_img: sitk.Image, cfg: CFG):
    pred_thr = (pred_np > float(cfg.prob_threshold))
    if cfg.include_shell_dilate_vox > 0:
        pred_thr = binary_dilation(pred_thr, iterations=int(cfg.include_shell_dilate_vox))
    coords_in = voxel_sample_coords(pred_thr, cfg.voxel_subsample_zyx)
    if coords_in.size == 0:
        resp = -gaussian_laplace(pred_np.astype(np.float32), sigma=1.0)
        k = max(1024, int(20 * (resp.size / 1e6)))
        idx = np.argsort(resp.ravel())[::-1][:k]
        coords_out = np.column_stack(np.unravel_index(idx, resp.shape)).astype(np.int64)
        sp_zyx = get_spacing_zyx(sitk_img)
        edt = distance_transform_edt(pred_thr, sampling=sitk_img.GetSpacing()[::-1]).astype(np.float32)
        rad_mm = edt[coords_out[:, 0], coords_out[:, 1], coords_out[:, 2]] * float(sp_zyx.min())
        pos_mm = voxel_to_phys(coords_out, sitk_img)
        in_pred = np.zeros((len(coords_out),), bool)
        tang = tangents_from_edt(edt, coords_out)
        return (np.ascontiguousarray(coords_out, dtype=np.int64),
                np.ascontiguousarray(in_pred, dtype=bool),
                np.ascontiguousarray(rad_mm, dtype=np.float32),
                np.ascontiguousarray(pos_mm, dtype=np.float32),
                np.ascontiguousarray(tang, dtype=np.float32))


    edt_inside  = distance_transform_edt(pred_thr,  sampling=sitk_img.GetSpacing()[::-1]).astype(np.float32)
    edt_outside = distance_transform_edt(~pred_thr, sampling=sitk_img.GetSpacing()[::-1]).astype(np.float32)
    vess_np     = prob_ridge_log(pred_np, floor=cfg.vesselness_floor)


    tang_in = tangents_from_edt(edt_inside, coords_in)

    # Grow candidates near the inside core (dir-guided)
    coords_grow = probe_growth_points_prob(
        pred_np=pred_np,
        coords_in_zyx=coords_in,
        tangents_in=tang_in,
        img=sitk_img,
        cfg=cfg
    )
    # Keep only growth points within a narrow mm band outside the seg
    coords_grow = prune_by_distance_to_pred(
        coords_grow, pred_thr, sitk_img,
        dmin_mm=cfg.outside_band_min_mm,
        dmax_mm=cfg.outside_band_max_mm
    )
    # NMS to avoid clusters
    coords_grow = nms_coords(coords_grow, radius_vox=cfg.nms_radius_vox)

    # Optional uncertain belt nodes (original)
    coords_belt = np.zeros((0, 3), np.int64)
    if cfg.use_uncertain_belt:
        coords_belt = uncertain_band_nodes(pred_np, cfg.belt_low, cfg.belt_high, cfg.belt_step_zyx)

    # NEW: vessel-belt seeding (multi-shell, gated by prob & vesselness)
    coords_belt2 = belt_nodes_from_edt_and_vesselness(
        pred_np=pred_np,
        pred_thr=pred_thr,
        edt_outside_mm=edt_outside,
        vess_np=vess_np,
        sitk_img=sitk_img,
        shells_mm=cfg.belt_shells_mm,
        target_step_mm=cfg.belt_step_mm,
        prob_max=cfg.belt_prob_max,
        vesselness_min=cfg.belt_vesselness_min,
        band_half_mm=cfg.belt_band_half_mm,
        nms_vox=cfg.belt_nms_vox,
        max_points=cfg.belt_max_points
    )

    # Union pre-augmentation
    parts = [c for c in (coords_in, coords_grow, coords_belt, coords_belt2) if c.size > 0]
    coords_pre = np.unique(np.vstack(parts), axis=0) if parts else np.zeros((0,3), np.int64)

    # NEW: endpoint branch-filling along ±tangent (gated)
    coords_end = np.zeros((0, 3), np.int64)
    if coords_pre.shape[0] >= 2 and getattr(cfg, "ep_seed_enable", True):
        # provisional tangents on the current set for seeding
        tang_pre = tangents_from_edt(edt_inside, coords_pre) if coords_pre.size else np.zeros((0,3), np.float32)
        coords_end = seed_endpoints_along_tangents_gated(
            coords_zyx=coords_pre,
            tangents=tang_pre,
            pred_np=pred_np,
            edt_outside_mm=edt_outside,
            vess_np=vess_np,
            sitk_img=sitk_img,
            step_mm=cfg.ep_step_mm,
            max_mm=cfg.ep_max_mm,
            per_dir=cfg.ep_per_dir,
            prob_max=cfg.ep_prob_max,
            vesselness_min=cfg.ep_vesselness_min,
            edt_min_mm=cfg.ep_edt_min_mm,
            edt_max_mm=cfg.ep_edt_max_mm,
            nms_vox=cfg.ep_nms_vox,
            budget=cfg.ep_budget
        )

    # Combine with endpoint seeds (still "pre" for outside ratio capping)
    if coords_end.size:
        coords_pre = np.unique(np.vstack([coords_pre, coords_end]), axis=0)

    # Outside ratio capping (apply after all seeding)
    in_pred_pre = pred_thr[coords_pre[:, 0], coords_pre[:, 1], coords_pre[:, 2]]
    coords_in2  = coords_pre[in_pred_pre]
    coords_out2 = coords_pre[~in_pred_pre]
    n_in, n_out = len(coords_in2), len(coords_out2)
    if n_in > 0 and n_out > 0:
        max_out = int(cfg.target_outside_ratio * (n_in + n_out))
        if n_out > max_out and max_out > 0:
            idx = np.linspace(0, n_out - 1, max_out).astype(int)
            coords_out2 = coords_out2[idx]

    coords = np.unique(np.vstack([coords_in2, coords_out2]), axis=0)

    # Final per-node attributes on the augmented set
    sp_zyx = get_spacing_zyx(sitk_img)
    edt_inside = distance_transform_edt(pred_thr, sampling=sitk_img.GetSpacing()[::-1]).astype(np.float32)
    rad_mm = edt_inside[coords[:, 0], coords[:, 1], coords[:, 2]] * float(sp_zyx.min())
    pos_mm = voxel_to_phys(coords, sitk_img)
    in_pred = pred_thr[coords[:, 0], coords[:, 1], coords[:, 2]]
    tang = tangents_from_edt(edt_inside, coords)

    # Optional final NMS to trim tiny clusters after augmentation
    coords = nms_coords(coords, radius_vox=cfg.nms_radius_vox)

    return (np.ascontiguousarray(coords, dtype=np.int64),
            np.ascontiguousarray(in_pred, dtype=bool),
            np.ascontiguousarray(rad_mm, dtype=np.float32),
            np.ascontiguousarray(pos_mm, dtype=np.float32),
            np.ascontiguousarray(tang, dtype=np.float32))


def affine_xyz_to_index(img: sitk.Image):
    sp = np.asarray(img.GetSpacing(), dtype=np.float64)      # (sx,sy,sz)
    org = np.asarray(img.GetOrigin(), dtype=np.float64)      # (ox,oy,oz)
    D   = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3,3)
    A   = D @ np.diag(sp)                                    # phys = A @ index + org
    Ainv = np.linalg.inv(A)
    return Ainv, org

def nodes_phys_to_index(pos_mm: np.ndarray, Ainv: np.ndarray, org: np.ndarray) -> np.ndarray:
    out = ((Ainv @ (pos_mm.astype(np.float64) - org).T).T)   # (N,3) (x,y,z)
    return np.ascontiguousarray(out, dtype=np.float64)
def line_integrals_torch(pos_idx_xyz: np.ndarray,
                         edges: np.ndarray,
                         vol_zyx: np.ndarray,
                         n: int,
                         device: Optional[torch.device] = None,
                         batch_edges: int = 4096) -> np.ndarray:
    if edges.shape[1] == 0:
        return np.zeros((0,), np.float32)

    dev = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    vol = torch.from_numpy(vol_zyx.astype(np.float32))[None, None].to(dev)
    Z, Y, X = vol_zyx.shape
    u, v = edges
    P0 = torch.from_numpy(pos_idx_xyz[u].astype(np.float32)).to(dev)
    P1 = torch.from_numpy(pos_idx_xyz[v].astype(np.float32)).to(dev)

    t = torch.linspace(0, 1, int(n), device=dev).view(n, 1, 1)  # [n,1,1]

    E = P0.shape[0]
    out = torch.empty((E,), device=dev, dtype=torch.float32)

    # process in chunks to avoid huge [E, n] buffers
    for s in range(0, E, batch_edges):
        e = min(s + batch_edges, E)
        P0b = P0[s:e]  # [Eb,3]
        P1b = P1[s:e]  # [Eb,3]
        Eb = P0b.shape[0]

        # [n,Eb,3] (xyz in index space)
        S = (1 - t) * P0b[None, :, :] + t * P1b[None, :, :]
        # normalize to [-1,1] and permute to [Eb, n, 1, 1, 3]
        z = 2 * (S[..., 2] / max(Z - 1, 1)) - 1
        y = 2 * (S[..., 1] / max(Y - 1, 1)) - 1
        x = 2 * (S[..., 0] / max(X - 1, 1)) - 1
        grid = torch.stack([x, y, z], dim=-1).permute(1, 0, 2)  # [Eb, n, 3]
        grid = grid.unsqueeze(2).unsqueeze(2)                   # [Eb, n, 1, 1, 3]  <-- FIX

        # expand vol to batch Eb
        vol_b = vol.expand(Eb, -1, -1, -1, -1)                  # [Eb,1,Z,Y,X]

        vals = F.grid_sample(
            vol_b, grid,
            mode='bilinear',
            align_corners=True
        )  # [Eb,1,n,1,1]
        vals = vals.squeeze(4).squeeze(3).squeeze(1)            # [Eb,n]
        out[s:e] = vals.mean(dim=1)

    return out.detach().cpu().numpy()

def line_integrals_batch(pos_idx_xyz: np.ndarray,
                         edges: np.ndarray,
                         vol_zyx: np.ndarray,
                         n: int,
                         batch_edges: int = 5000) -> np.ndarray:
    """
    CPU fallback for line sampling along edges using trilinear interpolation.

    Args
    ----
    pos_idx_xyz : (N,3) float64/float32
        Node positions in *index space* (x,y,z) corresponding to vol_zyx.
    edges       : (2,E) int64
        Edge index pairs (u,v).
    vol_zyx     : (Z,Y,X) float32/float64
        Volume to sample.
    n           : int
        Number of samples per edge (includes both endpoints; like torch.linspace(0,1,n)).
    batch_edges : int
        Process edges in batches to control memory.

    Returns
    -------
    means : (E,) float32
        Mean sampled value along each edge.
    """
    if edges.shape[1] == 0:
        return np.zeros((0,), dtype=np.float32)

    n = max(2, int(n))  # ensure at least endpoints
    u, v = edges
    P0 = pos_idx_xyz[u].astype(np.float64, copy=False)  # (E,3) in (x,y,z)
    P1 = pos_idx_xyz[v].astype(np.float64, copy=False)

    Z, Y, X = vol_zyx.shape
    out = np.empty((edges.shape[1],), dtype=np.float32)

    # parameter along the segment [0,1]
    t = np.linspace(0.0, 1.0, n, dtype=np.float64).reshape(n, 1, 1)  # (n,1,1)

    for s in range(0, P0.shape[0], batch_edges):
        e = min(s + batch_edges, P0.shape[0])
        P0b = P0[s:e]  # (Eb,3)
        P1b = P1[s:e]  # (Eb,3)
        Eb  = P0b.shape[0]

        # sample points S in index space (x,y,z): (n,Eb,3)
        S = (1.0 - t) * P0b[None, :, :] + t * P1b[None, :, :]

        # map_coordinates expects coords in (z,y,x) order with shape (3, M)
        # Flatten in edge-major order to later reshape back to (Eb, n)
        # S[...,0]=x, S[...,1]=y, S[...,2]=z
        z = S[..., 2].T.reshape(-1)
        y = S[..., 1].T.reshape(-1)
        x = S[..., 0].T.reshape(-1)

        coords_zyx = np.stack([z, y, x], axis=0)  # (3, Eb*n)

        vals = map_coordinates(
            vol_zyx,
            coords_zyx,
            order=1,           # trilinear
            mode='nearest',    # clamp to border
            prefilter=False
        ).astype(np.float32, copy=False)

        vals = vals.reshape(Eb, n)    # (Eb, n)
        out[s:e] = vals.mean(axis=1)

    return out


def _adaptive_integrals(pos_mm: np.ndarray, edge_index: np.ndarray, pos_idx_xyz: np.ndarray,
                        vol_zyx: np.ndarray, n_max: int) -> np.ndarray:
    """Adaptive sampling count by edge length; bucketed for efficiency."""
    if edge_index.shape[1] == 0:
        return np.zeros((0,), np.float32)
    lengths = np.linalg.norm(pos_mm[edge_index[1]] - pos_mm[edge_index[0]], axis=1)
    # e.g., ~1.5mm per sample, min 8, max n_max
    n_each = np.clip((lengths / 1.5).astype(int), 8, int(n_max))
    out = np.zeros((edge_index.shape[1],), np.float32)
    for n in np.unique(n_each):
        mask = (n_each == n)
        sub_e = edge_index[:, mask]
        if _USE_TORCH_SAMPLER and torch.cuda.is_available():
            vals = line_integrals_torch(pos_idx_xyz, sub_e, vol_zyx, int(n))
        else:
            vals = line_integrals_batch(pos_idx_xyz, sub_e, vol_zyx, int(n))
        out[mask] = vals
    return out

def edge_features(pos_mm: np.ndarray,
                  tangents: np.ndarray,
                  edge_index: np.ndarray,
                  rad_mm: Optional[np.ndarray] = None,
                  prob_zyx: Optional[np.ndarray] = None,
                  vess_zyx: Optional[np.ndarray] = None,
                  pos_idx_xyz: Optional[np.ndarray] = None,
                  n_samples: int = 24,
                  use_integrals: bool = False) -> np.ndarray:
    if edge_index.shape[1] == 0:
        fdim = 6 + (3 if use_integrals else 0)
        return np.zeros((0, fdim), dtype=np.float32)

    u, v = edge_index
    d = pos_mm[v] - pos_mm[u]
    dist = np.linalg.norm(d, axis=1, keepdims=True)
    dx = np.abs(d)
    t_u = tangents[u]; t_v = tangents[v]
    denom = (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
    cos_u = (np.sum(d * t_u, axis=1, keepdims=True) / denom)
    cos_v = (np.sum(-d * t_v, axis=1, keepdims=True) / denom)
    cos_sym = 0.5 * (cos_u + cos_v)
    dr = np.zeros_like(dist) if rad_mm is None else np.abs(rad_mm[u] - rad_mm[v]).reshape(-1, 1)

    base = np.concatenate([dist, dx, cos_sym, dr], axis=1).astype(np.float32)
    base = np.ascontiguousarray(base, dtype=np.float32)
    if not use_integrals:
        return base

    assert (prob_zyx is not None) and (vess_zyx is not None) and (pos_idx_xyz is not None)
    prob_mean = _adaptive_integrals(pos_mm, edge_index, pos_idx_xyz, prob_zyx, n_max=n_samples)
    vess_mean = _adaptive_integrals(pos_mm, edge_index, pos_idx_xyz, vess_zyx, n_max=n_samples)
    prob_cost = np.maximum(0.0, 1.0 - prob_mean).astype(np.float32)
    feat = np.concatenate([base,
                           prob_mean[:,None], vess_mean[:,None], prob_cost[:,None]], axis=1).astype(np.float32)
    return np.ascontiguousarray(feat, dtype=np.float32)


# =========================
# Labeling by GT connectivity
# =========================
def label_edges_by_gt_graph(cand_ei: np.ndarray,
                            pos_mm: np.ndarray,
                            mask_inside_gt: np.ndarray,
                            D_pos_mm: float) -> np.ndarray:
    N = len(pos_mm)
    if N == 0 or cand_ei.shape[1] == 0:
        return np.zeros((cand_ei.shape[1],), dtype=np.uint8)

    inside_nodes = np.flatnonzero(mask_inside_gt.astype(bool))
    if len(inside_nodes) < 2:
        return np.zeros((cand_ei.shape[1],), dtype=np.uint8)

    in_set = set(int(i) for i in inside_nodes.tolist())
    G = [[] for _ in range(N)]
    W = [[] for _ in range(N)]
    for u, v in cand_ei.T:
        u = int(u); v = int(v)
        if u in in_set and v in in_set:
            d = float(np.linalg.norm(pos_mm[v] - pos_mm[u]))
            G[u].append(v); W[u].append(d)
            G[v].append(u); W[v].append(d)

    import heapq
    labels = np.zeros((cand_ei.shape[1],), dtype=np.uint8)
    dist_cache: Dict[int, Dict[int, float]] = {}

    def dijkstra_trunc(src: int, cutoff: float) -> Dict[int, float]:
        if src in dist_cache: return dist_cache[src]
        dist = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > cutoff: break
            for v, w in zip(G[u], W[u]):
                nd = d + w
                if nd <= cutoff and nd < dist.get(v, 1e18):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        dist_cache[src] = dist
        return dist

    for k, (u, v) in enumerate(cand_ei.T):
        u = int(u); v = int(v)
        if (u not in in_set) or (v not in in_set):
            continue
        dmap = dijkstra_trunc(u, float(D_pos_mm))
        if dmap.get(v, 1e18) <= float(D_pos_mm):
            labels[k] = 1
    return np.ascontiguousarray(labels, dtype=np.uint8)


# =========================
# ECC model (NNConv encoder + pairwise MLP)
# =========================
class EdgeNet(nn.Module):
    def __init__(self, in_edge: int, in_ch: int, out_ch: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_edge, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, in_ch * out_ch)
        )
        self.in_ch = in_ch
        self.out_ch = out_ch
    def forward(self, edge_attr):
        return self.mlp(edge_attr)

# from torch_geometric.nn import SplineConv

class ECCEdgeClassifier(nn.Module):
    """
    Memory-safe SplineConv edge classifier (drop-in).
    - Message passing uses SplineConv with a *small* pseudo space:
        pseudo_cols = [0,1,2,3,4]  -> [dist, |dx| (3), cos_sym]
      This keeps K = kernel_size ** len(pseudo_cols) manageable.
    - Pairwise head still gets the full edge_attr to preserve all cues.
    """
    def __init__(self, in_node: int, edge_in: int,
                 hidden: int = 96, layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.dropout = float(dropout)

        # Choose which edge_attr columns to use as Spline pseudo-coordinates.
        # Your base edge_features order is:
        #   [0]=dist, [1:4]=|dx|, [4]=cos_sym, [5]=dr, [+ (6:)=integrals if enabled]
        # We'll use [0,1,2,3,4] -> 5 dims (dist, |dx| xyz, cos) for pseudo.
        self.pseudo_cols = [0, 1, 2, 3, 4] if edge_in >= 5 else list(range(edge_in))

        # Smaller kernel size to control K = kernel_size ** len(pseudo_cols)
        self.kernel_size = 3
        self.pseudo_dim = len(self.pseudo_cols)

        # SplineConv stack
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        ch_in = int(in_node)
        for _ in range(int(layers)):
            conv = SplineConv(
                in_channels=ch_in,
                out_channels=hidden,
                dim=self.pseudo_dim,          # use only the selected pseudo dims
                kernel_size=self.kernel_size,
                aggr='mean'
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden))
            ch_in = hidden

        # Pairwise classifier head (unchanged): z[u], z[v], and full edge_attr_pairs
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*hidden + edge_in, hidden), nn.ReLU(inplace=True), nn.Dropout(self.dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(inplace=True),
            nn.Linear(hidden//2, 1)
        )

    def _build_pseudo(self, gnn_edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Build Spline pseudo in [0,1]^D from selected columns of gnn_edge_attr.
        - Map cosine (column 4 in our selection) from [-1,1] -> [0,1].
        - Then min-max normalize each column to [0,1].
        """
        if gnn_edge_attr.ndim != 2:
            raise RuntimeError("Expected gnn_edge_attr of shape [E, D].")

        e = gnn_edge_attr[:, self.pseudo_cols].clone()  # [E, pseudo_dim]
        e = torch.nan_to_num(e, nan=0.0, posinf=1.0, neginf=0.0)

        # If our selection contains the cosine column (index 4 in the *full* layout),
        # figure out its position inside the sliced tensor and remap it.
        try:
            pos_cos = self.pseudo_cols.index(4)
            e[:, pos_cos] = ((e[:, pos_cos] + 1.0) * 0.5).clamp(0.0, 1.0)
        except ValueError:
            pass  # cos not present in the slice

        # Min-max per column to [0,1] (robust to degenerate ranges)
        col_min = e.amin(dim=0)
        col_max = e.amax(dim=0)
        denom = (col_max - col_min).clamp_min(1e-6)
        pseudo = ((e - col_min) / denom).clamp(0.0, 1.0)
        return pseudo

    def forward(self,
                x: torch.Tensor,                    # [N, in_node]
                gnn_edge_index: torch.Tensor,       # [2, E_msg]
                gnn_edge_attr: torch.Tensor,        # [E_msg, edge_in]
                edge_pairs: torch.Tensor,           # [2, E_cand]
                edge_attr_pairs: torch.Tensor):     # [E_cand, edge_in]
        pseudo = self._build_pseudo(gnn_edge_attr)  # [E_msg, pseudo_dim] in [0,1]

        z = x
        for conv, ln in zip(self.convs, self.norms):
            z_res = z
            z = conv(z, gnn_edge_index, pseudo)
            z = ln(z)
            z = F.relu(z)
            if z_res.shape[1] == z.shape[1]:
                z = z + z_res
            z = F.dropout(z, p=self.dropout, training=self.training)

        u, v = edge_pairs
        feats = torch.cat([z[u], z[v], edge_attr_pairs], dim=1)
        logits = self.edge_mlp(feats).squeeze(1)
        return logits



# =========================
# Build one training graph
# =========================
def _node_features(pred_np: np.ndarray,
                   coords: np.ndarray,
                   rad_mm: np.ndarray,
                   pos_mm: np.ndarray) -> np.ndarray:
    """Node features: [prob, radius_mm, pos_norm(3)] -> 5 dims."""
    probv = pred_np[coords[:,0], coords[:,1], coords[:,2]]
    pos_norm = (pos_mm - pos_mm.mean(0, keepdims=True)) / (pos_mm.std(0, keepdims=True) + 1e-6)
    x = np.concatenate([probv[:,None].astype(np.float32),
                        rad_mm[:,None].astype(np.float32),
                        pos_norm.astype(np.float32)], 1)  # [N,5]
    return np.ascontiguousarray(x, dtype=np.float32)

def make_graph_case(img_p: str, gt_p: str, pred_p: str, cfg: CFG) -> Optional[Data]:
    # Reference image is the probability image (no raw image required)
    gt   = read_nii_safe(gt_p)
    prd  = read_nii_safe(pred_p)

    gt_np  = (sitk_to_np(gt) > 0.5).astype(bool)
    if cfg.gt_dilate_vox > 0:
        gt_np = binary_dilation(gt_np, iterations=int(cfg.gt_dilate_vox))
    pred_np = sitk_to_np(prd).astype(np.float32)
    if pred_np.max() > 1.5:
        pred_np = (pred_np > 0.5).astype(np.float32)

    # Nodes (and tangents)
    coords, in_pred, rad_mm, pos_mm, tang = build_nodes(pred_np, prd, cfg)
    if len(coords) < 2:
        return None

    # Edges
    ei_knn = knn_edges(pos_mm, cfg.knn_k, cfg.knn_radius_mm, cfg.max_edge_len_mm)
    if ei_knn.shape[1] == 0:
        return None
    ei_cand = add_gap_candidates(
        pos_mm=pos_mm, base_ei=ei_knn, tangents=tang, rad_mm=rad_mm,
        r_gap_mm=cfg.gap_r_mm, cos_min=cfg.gap_cos_min, dr_mm_max=cfg.gap_dr_mm_max,
        max_len_mm=cfg.max_edge_len_mm
    )

    # Edge features (one pass for cand; index subset for gnn)
    vess_np = prob_ridge_log(pred_np, floor=cfg.vesselness_floor)
    Ainv, org = affine_xyz_to_index(prd)
    pos_idx_xyz = nodes_phys_to_index(pos_mm, Ainv, org)

    ea_cand = edge_features(
        pos_mm, tang, ei_cand, rad_mm,
        prob_zyx=pred_np, vess_zyx=vess_np, pos_idx_xyz=pos_idx_xyz,
        n_samples=cfg.n_line_samples, use_integrals=cfg.use_line_integrals
    )

    # Map gnn_ei features from cand
    if ei_cand.shape[1] > 0:
        uv = ei_cand.T
        key = { (int(a),int(b)): i for i,(a,b) in enumerate(uv) }
        gmask = np.zeros(ei_knn.shape[1], dtype=int)
        for i,(a,b) in enumerate(ei_knn.T):
            gmask[i] = key[(int(a),int(b))] if (int(a),int(b)) in key else key[(int(b),int(a))]
        ea_knn = ea_cand[gmask]
    else:
        ea_knn = np.zeros((0, ea_cand.shape[1] if ea_cand.ndim==2 else 0), dtype=np.float32)

    # Node features
    x = _node_features(pred_np, coords, rad_mm, pos_mm)

    # Labels from GT connectivity
    mask_inside_gt = (gt_np[coords[:,0], coords[:,1], coords[:,2]]).astype(bool)
    y = label_edges_by_gt_graph(ei_cand, pos_mm, mask_inside_gt, D_pos_mm=cfg.pos_path_max_mm)
    pos_mask = (y == 1); neg_mask = ~pos_mask
    if pos_mask.sum() == 0:
        return None

    pos_ei = ei_cand[:, pos_mask]
    pos_ea = ea_cand[pos_mask]
    neg_idx = np.flatnonzero(neg_mask)
    if len(neg_idx) == 0:
        return None
    rng = np.random.default_rng(cfg.seed)
    n_pos = int(pos_mask.sum())
    n_neg = min(len(neg_idx), cfg.neg_per_pos * n_pos)
    sel = rng.choice(neg_idx, size=n_neg, replace=False)
    neg_ei = ei_cand[:, sel]
    neg_ea = ea_cand[sel]

    # ---- Ensure contiguity before torch.from_numpy ----
    x              = np.ascontiguousarray(x, dtype=np.float32)
    pos_mm_c       = np.ascontiguousarray(pos_mm, dtype=np.float32)
    ei_knn_c       = np.ascontiguousarray(ei_knn, dtype=np.int64)
    ea_knn_c       = np.ascontiguousarray(ea_knn, dtype=np.float32)
    pos_ei_c       = np.ascontiguousarray(pos_ei, dtype=np.int64)
    neg_ei_c       = np.ascontiguousarray(neg_ei, dtype=np.int64)
    pos_ea_c       = np.ascontiguousarray(pos_ea, dtype=np.float32)
    neg_ea_c       = np.ascontiguousarray(neg_ea, dtype=np.float32)
    spacing_c      = np.ascontiguousarray(get_spacing_zyx(prd), dtype=np.float32)
    coords_zyx_c   = np.ascontiguousarray(coords, dtype=np.int64)

    return Data(
        x=torch.from_numpy(x),
        pos=torch.from_numpy(pos_mm_c),
        edge_index=torch.from_numpy(ei_knn_c),
        edge_attr=torch.from_numpy(ea_knn_c),
        pos_edge_index=torch.from_numpy(pos_ei_c),
        neg_edge_index=torch.from_numpy(neg_ei_c),
        pos_edge_attr=torch.from_numpy(pos_ea_c),
        neg_edge_attr=torch.from_numpy(neg_ea_c),
        spacing=torch.from_numpy(spacing_c),
        coords_zyx=torch.from_numpy(coords_zyx_c),
        meta={"img": img_p, "lab": gt_p, "pred": pred_p}
    )


# =========================
# Export VTP
# =========================
def _assemble_polylines(n_nodes: int, edge_index_np: np.ndarray) -> List[List[int]]:
    if edge_index_np.size == 0: return []
    u = edge_index_np[0].tolist(); v = edge_index_np[1].tolist()
    adj = {i: set() for i in range(n_nodes)}
    for a,b in zip(u,v):
        a=int(a); b=int(b)
        adj[a].add(b); adj[b].add(a)
    deg = {i: len(adj[i]) for i in adj}
    polylines = []
    visited = set()
    def mark(a,b):
        a,b = (a,b) if a<b else (b,a); visited.add((a,b))
    def seen(a,b):
        a,b = (a,b) if a<b else (b,a); return (a,b) in visited
    starts = [i for i,d in deg.items() if d != 2] or list(adj.keys())
    for s in starts:
        for nb in list(adj[s]):
            if seen(s, nb): continue
            line = [s, nb]; mark(s, nb)
            prev, cur = s, nb
            while deg.get(cur, 0) == 2:
                nxts = [x for x in adj[cur] if x != prev]
                if not nxts: break
                nxt = nxts[0]
                if seen(cur, nxt): break
                line.append(nxt); mark(cur, nxt)
                prev, cur = cur, nxt
            if len(line) >= 2:
                polylines.append(line)
    for a in list(adj.keys()):
        for b in list(adj[a]):
            if seen(a,b): continue
            polylines.append([a,b]); mark(a,b)
    return polylines

def export_graph_to_vtp(coords_zyx: np.ndarray, sitk_img: sitk.Image, edge_index_np: np.ndarray, out_path: str):
    if not _HAVE_VTK:
        print(f"[WARN] VTK not available; skipping VTP export: {out_path}")
        return
    points = vtk.vtkPoints()
    for z,y,x in coords_zyx:
        X,Y,Z = sitk_img.TransformIndexToPhysicalPoint((int(x),int(y),int(z)))
        points.InsertNextPoint(float(X), float(Y), float(Z))
    lines = vtk.vtkCellArray()
    for seq in _assemble_polylines(len(coords_zyx), edge_index_np):
        if len(seq) < 2: continue
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(seq))
        for i,pid in enumerate(seq):
            polyline.GetPointIds().SetId(i, int(pid))
        lines.InsertNextCell(polyline)
    poly = vtk.vtkPolyData(); poly.SetPoints(points); poly.SetLines(lines)
    wr = vtk.vtkXMLPolyDataWriter(); wr.SetFileName(out_path); wr.SetInputData(poly); wr.Write()
    print("Saved:", out_path)


# =========================
# MST preview & predicted assembly
# =========================
def assemble_mst_weighted(pos_mm: np.ndarray, ei: np.ndarray, lambda_len_inv: float) -> np.ndarray:
    if ei.shape[1]==0: return ei
    lengths = np.linalg.norm(pos_mm[ei[1]] - pos_mm[ei[0]], axis=1) + 1e-8
    w = lambda_len_inv / lengths  # prefer shorter edges
    G = nx.Graph()
    G.add_nodes_from(range(int(pos_mm.shape[0])))
    for (u,v), ww in zip(ei.T, w):
        G.add_edge(int(u), int(v), weight=float(ww))
    T = nx.minimum_spanning_tree(G)
    if T.number_of_edges()==0:
        return np.zeros((2,0), np.int64)
    mst_edges = np.array(list(T.edges()), np.int64).T
    return np.ascontiguousarray(mst_edges, dtype=np.int64)

def assemble_edges_longpaths(pos_mm: np.ndarray,
                             ei: np.ndarray,
                             probs: np.ndarray,
                             lambda_len_inv: float,
                             add_back_thresh: float) -> np.ndarray:
    if ei.shape[1] == 0:
        return ei
    lengths = np.linalg.norm(pos_mm[ei[1]] - pos_mm[ei[0]], axis=1)
    w = -np.log(np.clip(probs, 1e-6, 1.0)) + lambda_len_inv / (lengths + 1e-6)
    G = nx.Graph()
    n_nodes = int(pos_mm.shape[0])
    G.add_nodes_from(range(n_nodes))
    for (u, v), ww in zip(ei.T, w):
        G.add_edge(int(u), int(v), weight=float(ww))
    T = nx.minimum_spanning_tree(G)
    mst_edges = np.array(list(T.edges()), dtype=np.int64)
    if mst_edges.size == 0:
        kept = ei
    else:
        kept = mst_edges.T
        if add_back_thresh > 0:
            keep_mask = probs >= add_back_thresh
            kept_extra = ei[:, keep_mask]
            if kept_extra.size > 0:
                kept = np.unique(np.concatenate([kept, kept_extra], axis=1), axis=1)
    return np.ascontiguousarray(kept, dtype=np.int64)


# =========================
# Train / Inference utilities
# =========================
def _build_train_items(train_list: List[Tuple[str,str,str]], cfg: CFG) -> List[Data]:
    items: List[Data] = []
    print("Building training graphs...")
    for (im, gt, pr) in train_list:
        try:
            d = make_graph_case(im, gt, pr, cfg)
            if d is not None:
                items.append(d)
        except Exception as e:
            print(f"[ERR] Read/build failed: {os.path.basename(im)}: {e}")
        print(f"Built {len(items)}/{len(train_list)} training graphs.")
    print(f"Built {len(items)} training graphs.")
    return items

def train_one_epoch(model, loader, opt, scaler, device) -> float:
    model.train()
    total = 0.0
    amp_ctx = torch.amp.autocast('cuda', enabled=(CFG.amp and torch.cuda.is_available())) if torch.cuda.is_available() else nullcontext()
    for data in loader:
        data = data.to(device)
        opt.zero_grad(set_to_none=True)
        with amp_ctx:
            pos_logits = model(data.x, data.edge_index, data.edge_attr, data.pos_edge_index, data.pos_edge_attr)
            neg_logits = model(data.x, data.edge_index, data.edge_attr, data.neg_edge_index, data.neg_edge_attr)
            pos_labels = torch.ones_like(pos_logits)
            neg_labels = torch.zeros_like(neg_logits)
            loss_pos = F.binary_cross_entropy_with_logits(pos_logits, pos_labels)
            loss_neg = F.binary_cross_entropy_with_logits(neg_logits, neg_labels)
            loss = loss_pos + loss_neg
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += float(loss.detach().cpu())
    return total / max(1, len(loader))

@torch.no_grad()
def predict_edges(model,
                  x: torch.Tensor,
                  gnn_ei: torch.Tensor,
                  cand_ei: torch.Tensor,
                  cand_ea: torch.Tensor,
                  gnn_ea: Optional[torch.Tensor],
                  device: torch.device) -> np.ndarray:
    model.eval()
    amp_ctx = torch.amp.autocast('cuda', enabled=(CFG.amp and torch.cuda.is_available())) if torch.cuda.is_available() else nullcontext()
    with amp_ctx:
        if gnn_ea is None: gnn_ea = cand_ea
        logits = model(x, gnn_ei, gnn_ea, cand_ei, cand_ea)
        probs = torch.sigmoid(logits)
    return probs.detach().cpu().numpy()


# =========================
# Per-case: build & export (inspection)
# =========================
def process_case_build_only(img_p: str, gt_p: str, pred_p: str, cfg: CFG):
    # Use pred as reference (no raw image needed)
    name = os.path.basename(pred_p).replace(".nii.gz","").replace("_0000","")
    print(f"\n--- {name} ---")

    gt   = read_nii_safe(gt_p)
    prd  = read_nii_safe(pred_p)
    gt_np   = (sitk_to_np(gt)   > 0.5).astype(bool)
    pred_np = sitk_to_np(prd).astype(np.float32)
    if pred_np.max() > 1.5:
        pred_np = (pred_np > 0.5).astype(np.float32)

    # Nodes
    coords, in_pred, rad_mm, pos_mm, tang = build_nodes(pred_np, prd, cfg)
    if len(coords) < 2:
        print("[WARN] No nodes; skipping.")
        return

    # kNN edges + gaps
    ei_knn = knn_edges(pos_mm, cfg.knn_k, cfg.knn_radius_mm, cfg.max_edge_len_mm)
    ei_cand = add_gap_candidates(
        pos_mm=pos_mm,
        base_ei=ei_knn,
        tangents=tang,
        rad_mm=rad_mm,
        r_gap_mm=cfg.gap_r_mm,
        cos_min=cfg.gap_cos_min,
        dr_mm_max=cfg.gap_dr_mm_max,
        max_len_mm=cfg.max_edge_len_mm
    )

    # Stats
    n = len(coords)
    frac_outside = 1.0 - float(in_pred.sum())/max(1,n)
    gt_cover = float(gt_np[coords[:,0], coords[:,1], coords[:,2]].mean()) if n>0 else 0.0
    print(f"nodes={n} | in_pred={int(in_pred.sum())} ({1-frac_outside:.2%}), outside={int((~in_pred).sum())} ({frac_outside:.2%})")
    print(f"E_knn={int(ei_knn.shape[1])} | E_cand={int(ei_cand.shape[1])} | nodes_in_GT={gt_cover:.2%}")

    # Exports
    os.makedirs(cfg.out_dir, exist_ok=True)
    if cfg.export_candidates_vtp:
        out_cand = os.path.join(cfg.out_dir, f"{name}_graph_candidates.vtp")
        export_graph_to_vtp(coords, prd, ei_cand, out_cand)

    if cfg.export_mst_preview_vtp:
        ei_mst = assemble_mst_weighted(pos_mm, ei_cand, lambda_len_inv=cfg.mst_lambda_len_inv)
        out_mst = os.path.join(cfg.out_dir, f"{name}_graph_mst.vtp")
        export_graph_to_vtp(coords, prd, ei_mst, out_mst)

def save_predicted_graph_to_vtp(G: nx.Graph, out_path: str):
    import vtk

    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    # point data arrays
    arr_rad = vtk.vtkDoubleArray(); arr_rad.SetName("MaximumInscribedSphereRadius")
    arr_tan = vtk.vtkDoubleArray(); arr_tan.SetName("Tangent"); arr_tan.SetNumberOfComponents(3)

    # map node id -> vtk point id
    idmap = {}
    for i, (n, data) in enumerate(G.nodes(data=True)):
        x, y, z = map(float, data["pos_phys"])    # XYZ (mm)
        pid = pts.InsertNextPoint(x, y, z)
        idmap[n] = pid
        # radius
        r = float(data.get("radius_mm", 0.5))
        arr_rad.InsertNextValue(r)
        # tangent (optional; store zeros if missing)
        t = np.asarray(data.get("tangent", [0.0, 0.0, 0.0]), dtype=float)
        if t.size != 3: t = np.zeros(3, float)
        arr_tan.InsertNextTuple(t.tolist())

    # cell data arrays (per-edge)
    arr_prob = vtk.vtkDoubleArray(); arr_prob.SetName("EdgeProbability")
    arr_len  = vtk.vtkDoubleArray(); arr_len.SetName("LengthMM")
    arr_rmin = vtk.vtkDoubleArray(); arr_rmin.SetName("RadiusMinMM")
    arr_ravg = vtk.vtkDoubleArray(); arr_ravg.SetName("RadiusMeanMM")

    # build lines
    for u, v, ed in G.edges(data=True):
        lid = vtk.vtkLine()
        lid.GetPointIds().SetId(0, idmap[u])
        lid.GetPointIds().SetId(1, idmap[v])
        lines.InsertNextCell(lid)
        # per-edge attrs
        arr_prob.InsertNextValue(float(ed.get("edge_prob", 0.5)))
        arr_len.InsertNextValue(float(ed.get("weight", 0.0)))
        arr_rmin.InsertNextValue(float(ed.get("radius_min_mm", min(G.nodes[u]["radius_mm"], G.nodes[v]["radius_mm"]))))
        arr_ravg.InsertNextValue(float(ed.get("radius_mean_mm", 0.5*(G.nodes[u]["radius_mm"]+G.nodes[v]["radius_mm"]))))

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetLines(lines)

    # attach arrays
    poly.GetPointData().AddArray(arr_rad)
    poly.GetPointData().AddArray(arr_tan)
    poly.GetPointData().SetActiveScalars("MaximumInscribedSphereRadius")

    poly.GetCellData().AddArray(arr_prob)
    poly.GetCellData().AddArray(arr_len)
    poly.GetCellData().AddArray(arr_rmin)
    poly.GetCellData().AddArray(arr_ravg)

    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(out_path)
    w.SetInputData(poly)
    w.Write()

# =========================
# Inference per case (uses trained model)
# =========================
def process_case_infer(img_p: str, pred_p: str, model: ECCEdgeClassifier, cfg: CFG, device: torch.device):
    # Use pred as reference (no raw image needed)
    name = os.path.basename(pred_p).replace(".nii.gz","").replace("_0000","")
    prd  = read_nii_safe(pred_p)
    pred_np = sitk_to_np(prd).astype(np.float32)
    if pred_np.max() > 1.5: pred_np = (pred_np > 0.5).astype(np.float32)

    coords, in_pred, rad_mm, pos_mm, tang = build_nodes(pred_np, prd, cfg)
    if len(coords) < 2:
        print(f"[WARN] empty node set for {name}; skipping.")
        return

    gnn_ei = knn_edges(pos_mm, cfg.knn_k, cfg.knn_radius_mm, cfg.max_edge_len_mm)
    cand_ei = add_gap_candidates(pos_mm, gnn_ei, tang, rad_mm,
                                 r_gap_mm=cfg.gap_r_mm, cos_min=cfg.gap_cos_min,
                                 dr_mm_max=cfg.gap_dr_mm_max, max_len_mm=cfg.max_edge_len_mm)

    # features (cand once; subset for gnn)
    vess_np = prob_ridge_log(pred_np, floor=cfg.vesselness_floor)
    Ainv, org = affine_xyz_to_index(prd)
    pos_idx_xyz = nodes_phys_to_index(pos_mm, Ainv, org)
    cand_ea = edge_features(pos_mm, tang, cand_ei, rad_mm,
                            prob_zyx=pred_np, vess_zyx=vess_np, pos_idx_xyz=pos_idx_xyz,
                            n_samples=cfg.n_line_samples, use_integrals=cfg.use_line_integrals)
    uv = cand_ei.T
    key = { (int(a),int(b)): i for i,(a,b) in enumerate(uv) }
    gmask = np.zeros(gnn_ei.shape[1], dtype=int)
    for i,(a,b) in enumerate(gnn_ei.T):
        gmask[i] = key[(int(a),int(b))] if (int(a),int(b)) in key else key[(int(b),int(a))]
    gnn_ea = cand_ea[gmask]

    # node features
    x = _node_features(pred_np, coords, rad_mm, pos_mm)

    # ---- Ensure contiguity before torch.from_numpy ----
    x_t       = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)
    gnn_ei_t  = torch.from_numpy(np.ascontiguousarray(gnn_ei, dtype=np.int64)).to(device)
    gnn_ea_t  = torch.from_numpy(np.ascontiguousarray(gnn_ea, dtype=np.float32)).to(device)
    cand_ei_t = torch.from_numpy(np.ascontiguousarray(cand_ei, dtype=np.int64)).to(device)
    cand_ea_t = torch.from_numpy(np.ascontiguousarray(cand_ea, dtype=np.float32)).to(device)

    probs = predict_edges(model, x_t, gnn_ei_t, cand_ei_t, cand_ea_t, gnn_ea_t, device)
    keep = probs >= cfg.edge_prob_thresh
    ei_keep = cand_ei[:, keep]
    probs_keep = probs[keep]

    ei_pred = assemble_edges_longpaths(pos_mm, ei_keep, probs_keep,
                                       lambda_len_inv=cfg.mst_lambda_len_inv,
                                       add_back_thresh=cfg.add_back_thresh)

    if cfg.export_predicted_vtp and ei_pred.shape[1] > 0:
        out_vtp = os.path.join(cfg.out_dir, f"{name}_graph_predicted.vtp")
        export_graph_to_vtp(coords, prd, ei_pred, out_vtp)


# =========================
# Public API for external callers (SeqSeg, etc.)
# =========================
import copy
# --- put near other imports in gnn_model.py ---
import copy
from dataclasses import asdict

# ---------------------------
# cfg merge helper
# ---------------------------
def _merge_cfg(runtime_cfg, ckpt_cfg_dict: dict, prefer: str = "runtime"):
    """
    Merge a runtime CFG (dataclass) with a checkpoint cfg (dict).
    prefer='runtime' keeps runtime values when both present; 'ckpt' keeps ckpt values.
    """
    base = copy.deepcopy(runtime_cfg)
    if ckpt_cfg_dict is None:
        return base
    # turn dataclass -> dict, merge, then back to dataclass
    r = dict(asdict(base))
    if prefer.lower() == "runtime":
        # runtime wins: only fill missing keys from ckpt
        for k, v in ckpt_cfg_dict.items():
            if k not in r or r[k] is None:
                r[k] = v
    else:
        # ckpt wins: overwrite runtime with ckpt values when present
        for k, v in ckpt_cfg_dict.items():
            r[k] = v
    # rebuild CFG
    try:
        return CFG(**r)
    except TypeError:
        # in case of extra keys in older/newer ckpts
        known = {k: r[k] for k in r if k in CFG().__dict__}
        return CFG(**known)


# ---------------------------
# gm_load (rewritten)
# ---------------------------
def gm_load(ckpt_path: Optional[str] = None,
            device: Optional[torch.device] = None,
            *,
            runtime_cfg: Optional[CFG] = None,
            prefer_cfg: str = "runtime"   # 'runtime' or 'ckpt'
            ):
    """
    Load the trained ECC edge classifier AND return a merged cfg.

    prefer_cfg:
      - 'runtime' (default): keep the caller's CFG values when both exist; fill gaps from ckpt.
      - 'ckpt'            : take the training-time cfg values from checkpoint (overwriting runtime).
    """
    dev = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    # fall back to global cfg if no explicit runtime_cfg is provided
    base_runtime_cfg = copy.deepcopy(runtime_cfg if runtime_cfg is not None else cfg)

    ckpt_p = ckpt_path or os.path.join(base_runtime_cfg.out_dir, 'best_' + base_runtime_cfg.ckpt_name)
    if not os.path.isfile(ckpt_p):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")

    # Torch 2.6+: weights_only defaults True; we need full state dict here.
    try:
        ckpt = torch.load(ckpt_p, map_location=dev, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_p, map_location=dev)

    # Merge cfgs (ckpt may or may not contain a 'cfg' dict)
    ckpt_cfg_dict = ckpt.get('cfg', None)
    merged_cfg = _merge_cfg(base_runtime_cfg, ckpt_cfg_dict, prefer=prefer_cfg)

    # Respect model dims saved in ckpt; fall back to dims implied by merged_cfg
    default_in_node = 5
    default_edge_in = 9 if merged_cfg.use_line_integrals else 6
    in_node = int(ckpt.get('in_node', default_in_node))
    edge_in = int(ckpt.get('edge_in', default_edge_in))

    model = ECCEdgeClassifier(in_node=in_node, edge_in=edge_in, hidden=96, layers=3, dropout=0.2).to(dev)
    state_key = 'model_state' if 'model_state' in ckpt else 'model'
    model.load_state_dict(ckpt[state_key], strict=True)
    model.eval()

    # Return merged cfg so downstream code uses the same knobs as your preferred policy
    return (model, merged_cfg, dev, ckpt_p)


# ---------------------------
# gm_predict_graph (rewritten)
# ---------------------------
def gm_predict_graph(
    *,
    seg_img: sitk.Image,                 # reference image if prob_img is None
    prob_img: Optional[sitk.Image],      # soft prob or bin seg; if None, seg_img is used
    model_tuple,                         # output from gm_load()
    # Optional behavior controls (all default to using cfg values)
    node_min_spacing_mm: Optional[float] = None,   # if None -> use cfg.voxel_subsample_zyx (no adaptive spacing)
    knn_k: Optional[int] = None,
    knn_radius_mm: Optional[float] = 8,
    edge_prob_thresh: Optional[float] = 0.5,
    # Make this path configurable like process_case_infer
    cfg_overrides: Optional[dict] = None,
    use_adaptive_spacing: bool = True,  # True -> derive voxel_subsample_zyx from node_min_spacing_mm
    return_debug: bool = False           # True -> also return diagnostics dict
) -> nx.Graph | tuple[nx.Graph, dict]:
    """
    Build a graph from probability/segmentation, score edges with ECC, and return an nx.Graph.
    Uses the merged cfg from gm_load() and applies optional overrides so you can match process_case_infer.
    Node attrs: pos (zyx vox), pos_phys (XYZ mm), tangent
    Edge attrs: weight (mm), edge_prob (0..1)
    """
    model, base_cfg, device, _ = model_tuple
    local_cfg = copy.deepcopy(base_cfg)

    # Apply explicit overrides (same knobs you tweak in process_case_infer)
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            if hasattr(local_cfg, k):
                setattr(local_cfg, k, v)

    # If caller provided explicit kNN / threshold, reflect them into cfg for consistency
    if knn_k is not None:            local_cfg.knn_k = int(knn_k)
    if knn_radius_mm is not None:    local_cfg.knn_radius_mm = float(knn_radius_mm)
    if edge_prob_thresh is not None: local_cfg.edge_prob_thresh = float(edge_prob_thresh)

    # ----- volumes -----
    ref_img = prob_img if prob_img is not None else seg_img
    prob_np = sitk_to_np(ref_img).astype(np.float32)
    if prob_np.max() > 1.5:  # treat >1.5 as hard mask
        prob_np = (prob_np > 0.5).astype(np.float32)

    # ----- node density -----
    if use_adaptive_spacing and (node_min_spacing_mm is not None):
        sp_zyx = get_spacing_zyx(ref_img)  # Z,Y,X mm
        step_zyx = tuple(int(max(1, np.ceil(float(node_min_spacing_mm) / float(s)))) for s in sp_zyx)
        local_cfg.voxel_subsample_zyx = step_zyx
    # else: keep voxel_subsample_zyx from (merged) cfg — same behavior as process_case_infer

    # ----- build nodes (inside seg + grown band) -----
    coords, in_pred, rad_mm, pos_mm, tang = build_nodes(prob_np, ref_img, local_cfg)
    G = nx.Graph()
    if coords.size < 2:
        return (G, {"reason": "no_nodes"}) if return_debug else G

    # ----- edges: kNN + gap candidates -----
    gnn_ei = knn_edges(
        pos_mm,
        k=int(local_cfg.knn_k),
        r_mm=float(local_cfg.knn_radius_mm),
        max_len_mm=float(local_cfg.max_edge_len_mm)
    )
    if gnn_ei.shape[1] == 0:
        return (G, {"reason": "no_knn_edges", "N": len(coords)}) if return_debug else G

    cand_ei = add_gap_candidates(
        pos_mm=pos_mm, base_ei=gnn_ei, tangents=tang, rad_mm=rad_mm,
        r_gap_mm=float(local_cfg.gap_r_mm),
        cos_min=float(local_cfg.gap_cos_min),
        dr_mm_max=float(local_cfg.gap_dr_mm_max),
        max_len_mm=float(local_cfg.max_edge_len_mm)
    )
    if cand_ei.shape[1] == 0:
        return (G, {"reason": "no_candidate_edges", "E_knn": int(gnn_ei.shape[1])}) if return_debug else G

    # ----- features (compute once on candidates; subset for message edges) -----
    vess_np = prob_ridge_log(prob_np, floor=float(local_cfg.vesselness_floor))
    Ainv, org = affine_xyz_to_index(ref_img)
    pos_idx_xyz = nodes_phys_to_index(pos_mm, Ainv, org)

    cand_ea = edge_features(
        pos_mm, tang, cand_ei, rad_mm,
        prob_zyx=prob_np, vess_zyx=vess_np, pos_idx_xyz=pos_idx_xyz,
        n_samples=int(local_cfg.n_line_samples),
        use_integrals=bool(local_cfg.use_line_integrals)
    )
    # map gnn_ei -> cand_ea rows
    uv = cand_ei.T
    key = { (int(a),int(b)): i for i,(a,b) in enumerate(uv) }
    gmask = np.zeros(gnn_ei.shape[1], dtype=int)
    for i,(a,b) in enumerate(gnn_ei.T):
        gmask[i] = key.get((int(a),int(b)), key.get((int(b),int(a))))
    gnn_ea = cand_ea[gmask]

    # node features
    x_np = _node_features(prob_np, coords, rad_mm, pos_mm)

    # ----- predict edge probabilities -----
    x_t       = torch.from_numpy(np.ascontiguousarray(x_np,    dtype=np.float32)).to(device)
    gnn_ei_t  = torch.from_numpy(np.ascontiguousarray(gnn_ei,  dtype=np.int64)).to(device)
    gnn_ea_t  = torch.from_numpy(np.ascontiguousarray(gnn_ea,  dtype=np.float32)).to(device)
    cand_ei_t = torch.from_numpy(np.ascontiguousarray(cand_ei, dtype=np.int64)).to(device)
    cand_ea_t = torch.from_numpy(np.ascontiguousarray(cand_ea, dtype=np.float32)).to(device)

    probs = predict_edges(model, x_t, gnn_ei_t, cand_ei_t, cand_ea_t, gnn_ea_t, device)
    thr = float(local_cfg.edge_prob_thresh)
    keep = probs >= thr
    if keep.sum() == 0:
        return (G, {"reason": "no_edges_above_threshold",
                    "E_cand": int(cand_ei.shape[1]),
                    "thr": thr}) if return_debug else G

    ei_keep    = cand_ei[:, keep]
    probs_keep = probs[keep]

    # assemble long paths (MST + confident add-back) with the SAME knobs as process_case_infer
    ei_pred = assemble_edges_longpaths(
        pos_mm, ei_keep, probs_keep,
        lambda_len_inv=float(local_cfg.mst_lambda_len_inv),
        add_back_thresh=float(local_cfg.add_back_thresh)
    )

    # probability lookup for final edges
    prob_map = {tuple(sorted(map(int, e))): float(p) for e, p in zip(ei_keep.T, probs_keep)}

    # ----- build nx.Graph -----
    # ----- build nx.Graph -----
    for i, (z, y, x) in enumerate(coords):
        pxyz = ref_img.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
        r_mm = float(rad_mm[i])  # <-- radius from build_nodes (already in mm)
        G.add_node(
            int(i),
            pos=np.array([int(z), int(y), int(x)], dtype=int),  # voxel ZYX
            pos_phys=np.array(pxyz, dtype=float),  # physical XYZ (mm)
            tangent=tang[i],
            radius_mm=r_mm  # <-- ATTACHED
        )

    for u, v in ei_pred.T:
        u = int(u);
        v = int(v)
        p1 = G.nodes[u]['pos_phys'];
        p2 = G.nodes[v]['pos_phys']
        ru = float(G.nodes[u]['radius_mm']);
        rv = float(G.nodes[v]['radius_mm'])
        # conservative edge radius choices (pick one or keep both):
        r_min = min(ru, rv)
        r_mean = 0.5 * (ru + rv)

        G.add_edge(
            u, v,
            weight=float(np.linalg.norm(p1 - p2)),  # length in mm
            edge_prob=prob_map.get(tuple(sorted((u, v))), 0.5),
            radius_min_mm=r_min,  # <-- OPTIONAL
            radius_mean_mm=r_mean  # <-- OPTIONAL
        )

    if return_debug:
        dbg = {
            "N": len(coords),
            "E_knn": int(gnn_ei.shape[1]),
            "E_cand": int(cand_ei.shape[1]),
            "E_keep": int(ei_keep.shape[1]),
            "E_final": int(G.number_of_edges()),
            "thr": thr,
            "cfg_used": copy.deepcopy(local_cfg).__dict__,
        }
        return G, dbg
    return G

def _parse_args():
    p = argparse.ArgumentParser("GNN model runner")
    p.add_argument("--cfg", required=True, help="Path to YAML config file")
    return p.parse_args()
def load_cfg_from_yaml(path: str) -> CFG:
    """Minimal loader: read YAML and overlay on CFG defaults."""
    import os, yaml
    from dataclasses import fields
    from typing import get_origin

    path = os.path.abspath(os.path.expanduser(path))
    with open(path, "r") as f:
        y = yaml.safe_load(f) or {}

    cfg = CFG()  # start with defaults
    # map field name -> type once
    ftypes = {f.name: f.type for f in fields(CFG)}

    for k, v in y.items():
        if k not in ftypes:
            continue  # ignore unknown keys
        # list -> tuple if the CFG field is a Tuple[…]
        if get_origin(ftypes[k]) is tuple and isinstance(v, list):
            v = tuple(v)
        setattr(cfg, k, v)

    # expand common path fields
    for k in ("images_dir", "labels_dir", "preds_dir", "out_dir",
              "train_graph_cache_dir", "infer_graph_cache_dir"):
        v = getattr(cfg, k, "")
        if isinstance(v, str) and v:
            setattr(cfg, k, os.path.abspath(os.path.expanduser(v)))

    return cfg

# =========================
# Main
# =========================
def main():
    args = _parse_args()

    cfg_path = os.path.abspath(args.cfg)  # <-- use the VARIABLE, not the string
    if not os.path.isfile(cfg_path):
        print(f"[ERR] Config not found: {cfg_path}")
        sys.exit(1)

    # load your YAML-driven config
    cfg = load_cfg_from_yaml(cfg_path)
    os.makedirs(cfg.out_dir, exist_ok=True)

    triplets = match_triplets(cfg.images_dir, cfg.labels_dir, cfg.preds_dir, cfg.image_glob)
    if not triplets:
        raise RuntimeError("No matched (image, GT, pred) triplets found.")

    # Split 80/20 for training/inference
    n = len(triplets); n_train = max(1, int(0.8*n))
    train_list = triplets[:n_train]
    infer_list = triplets
    model = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Training

    if not torch.cuda.is_available():
        cfg.amp = False

    model = None
    if cfg.train_enable:
        items = _build_train_items(train_list, cfg)
        assert len(items) > 0, "No valid training graphs. Adjust thresholds or check data."

        loader = GeoDataLoader(items, batch_size=cfg.batch_size_cases, shuffle=True)

        in_node = int(items[0].x.shape[1])
        edge_in = int(items[0].edge_attr.shape[1])
        model = ECCEdgeClassifier(in_node=in_node, edge_in=edge_in, hidden=96, layers=3, dropout=0.2).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay)
        scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp)

        best = 1e9
        for epoch in range(1, cfg.max_epochs+1):
            t0 = time.time()
            tr_loss = train_one_epoch(model, loader, opt, scaler, device)
            dt = time.time() - t0
            print(f"Epoch {epoch:03d} | loss {tr_loss:.4f} | {dt:.1f}s")

            ckpt = {'model_state': model.state_dict(),
                    'cfg': cfg.__dict__,
                    'in_node': in_node,
                    'edge_in': edge_in}
            torch.save(ckpt, os.path.join(cfg.out_dir, cfg.ckpt_name))
            if tr_loss < best:
                best = tr_loss
                torch.save(ckpt, os.path.join(cfg.out_dir, 'best_' + cfg.ckpt_name))

    # Inference
    if cfg.infer_enable:
        if model is None:
            ckpt_path = os.path.join(cfg.out_dir, 'best_' + cfg.ckpt_name)
            if os.path.isfile(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
                in_node = int(ckpt.get('in_node', 5))
                edge_in = int(ckpt.get('edge_in', 9 if cfg.use_line_integrals else 6))
                model = ECCEdgeClassifier(in_node=in_node, edge_in=edge_in, hidden=96, layers=3, dropout=0.2).to(device)
                model.load_state_dict(ckpt['model_state'])
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
