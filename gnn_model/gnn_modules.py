from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from contextlib import nullcontext
from collections import defaultdict
import glob
import os, yaml, copy
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, gaussian_laplace, binary_dilation
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import SplineConv
from torch_cluster import knn_graph

try:
    import vtk
    _HAVE_VTK = True
except Exception:
    _HAVE_VTK = False


def _norm_path(v: str) -> str:
    v = os.path.expanduser(os.path.expandvars(str(v)))
    return os.path.abspath(v)

def _looks_like_path(key: str) -> bool:
    key = key.lower()
    return any(s in key for s in ["dir", "folder", "path", "out"])

def C(cfg: dict, key: str, default=None):
    return cfg[key] if (cfg is not None and key in cfg and cfg[key] is not None) else default

def load_cfg_from_yaml(path: str, defaults: dict | None = None) -> dict:
    """Loads YAML into a plain dict, merges optional defaults, normalizes path-like values."""
    with open(_norm_path(path), "r") as f:
        y = yaml.safe_load(f) or {}
    out = dict(defaults or {})
    for k, v in y.items():
        if isinstance(v, str) and _looks_like_path(k):
            out[k] = _norm_path(v)
        else:
            out[k] = v
    return out




from torch_cluster import radius_graph
from torch_geometric.utils import to_undirected, coalesce, remove_self_loops
from torch_geometric.nn.pool import fps, voxel_grid
from torch_scatter import scatter_mean

from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra, connected_components

def _to_torch(x: np.ndarray, device=None, dtype=None) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(x))
    if dtype is not None:
        t = t.to(dtype)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return t.to(device)

def _pair_hash(ei: torch.Tensor, N: int) -> torch.Tensor:
    # ei: [2, E]
    a = torch.minimum(ei[0], ei[1])
    b = torch.maximum(ei[0], ei[1])
    return (a.long() * N) + b.long()

@torch.no_grad()
def _map_gnn_to_cand_gpu(gnn_ei_np: np.ndarray, cand_ei_np: np.ndarray, N: int, device=None) -> np.ndarray:

    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    gnn_ei  = _to_torch(gnn_ei_np, device=device, dtype=torch.long)
    cand_ei = _to_torch(cand_ei_np, device=device, dtype=torch.long)
    # hash both
    h_gnn  = _pair_hash(gnn_ei,  N)
    h_cand = _pair_hash(cand_ei, N)

    order = torch.argsort(h_cand)
    h_sorted = h_cand[order]
    idx = torch.searchsorted(h_sorted, h_gnn)
    idx = torch.clamp(idx, 0, h_sorted.numel()-1)
    take = order[idx]
    ok = (h_cand[take] == h_gnn)
    take = torch.where(ok, take, torch.zeros_like(take))
    return take.detach().cpu().numpy()

def _edt_mm_sitk(mask_bool: np.ndarray, ref_img: sitk.Image) -> Tuple[np.ndarray, np.ndarray]:

    itk = sitk.GetImageFromArray(mask_bool.astype(np.uint8))
    itk.CopyInformation(ref_img)
    edt_in  = sitk.SignedMaurerDistanceMap(itk,      insideIsPositive=True,  squaredDistance=False, useImageSpacing=True)
    edt_out = sitk.SignedMaurerDistanceMap(1 - itk,  insideIsPositive=True,  squaredDistance=False, useImageSpacing=True)
    edt_in_np  = sitk.GetArrayFromImage(edt_in).astype(np.float32)
    edt_out_np = sitk.GetArrayFromImage(edt_out).astype(np.float32)
    return edt_in_np, edt_out_np


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
        est = int(size[0]) * int(size[1]) * max(1, int(size[2])) * 4
        if est > 3_000_000_000:
            raise MemoryError("Large header; using nibabel fallback")
        return sitk.ReadImage(path)
    except Exception:
        img = nib.load(path)
        arr = img.get_fdata(dtype=np.float32)
        sitk_img = sitk.GetImageFromArray(np.ascontiguousarray(arr.astype(np.float32)))
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        sitk_img.SetSpacing((zooms[0], zooms[1], zooms[2]))
        sitk_img.SetOrigin((0.0, 0.0, 0.0))
        sitk_img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        return sitk_img

def sitk_to_np(img: sitk.Image) -> np.ndarray:
    return np.ascontiguousarray(sitk.GetArrayFromImage(img))

def get_spacing_zyx(img: sitk.Image) -> np.ndarray:
    sp = np.array(img.GetSpacing(), dtype=np.float32)[::-1]
    return np.ascontiguousarray(sp)

def match_triplets(images_dir: str, labels_dir: str, preds_dir: str, pattern: str) -> List[Tuple[str, str, str]]:
    def stem(p: str) -> str:
        s = os.path.basename(p)
        if s.endswith(".nii.gz"):
            s = s[:-7]
        if "_0000" in s:
            s = s.replace("_0000", "")
        return s
    imgs  = sorted(glob.glob(os.path.join(images_dir, pattern)))
    gts   = sorted(glob.glob(os.path.join(labels_dir, pattern)))
    preds = sorted(glob.glob(os.path.join(preds_dir,  pattern)))
    idx: Dict[str, List[Optional[str]]] = {}
    for p in imgs:  idx.setdefault(stem(p), [None, None, None])[0] = p
    for p in gts:   idx.setdefault(stem(p), [None, None, None])[1] = p
    for p in preds: idx.setdefault(stem(p), [None, None, None])[2] = p
    out, bad = [], []
    for k, (im, gt, pr) in idx.items():
        if not (im and gt and pr):
            bad.append((k, "missing file")); continue
        if not (_nifti_is_readable(gt) and _nifti_is_readable(pr)):
            bad.append((k, "unreadable nifti")); continue
        out.append((im, gt, pr))
    if bad:
        print(f"[match_triplets] Skipping {len(bad)} case(s):")
        for k, why in bad[:10]:
            print("  -", k, "->", why)
        if len(bad) > 10: print("  ...")
    return out

def voxel_to_phys(coords_zyx: np.ndarray, img: sitk.Image) -> np.ndarray:
    sp = np.asarray(img.GetSpacing(), dtype=np.float64)
    org = np.asarray(img.GetOrigin(), dtype=np.float64)
    D   = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    ijk_xyz = coords_zyx[:, ::-1].astype(np.float64) * sp
    out = (D @ ijk_xyz.T).T + org
    return np.ascontiguousarray(out, dtype=np.float32)

def tangents_from_edt(edt_zyx: np.ndarray, coords_zyx: np.ndarray) -> np.ndarray:
    if coords_zyx.size == 0:
        return np.zeros((0, 3), np.float32)
    gz, gy, gx = np.gradient(edt_zyx)
    g = np.stack([gz, gy, gx], axis=-1)
    v = g[coords_zyx[:, 0], coords_zyx[:, 1], coords_zyx[:, 2]].astype(np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    t = -v / n
    return np.ascontiguousarray(t, dtype=np.float32)


# ------------------------
# Node & edge proposal
# ------------------------
def prob_ridge_log(prob: np.ndarray, floor: float = 0.05) -> np.ndarray:
    p = np.clip(prob.astype(np.float32), 0.0, 1.0)
    resp = -gaussian_laplace(p, sigma=1.0).astype(np.float32)
    resp[p < floor] = 0.0
    m = resp.max() + 1e-6
    return np.ascontiguousarray((resp / m).astype(np.float32))

def voxel_sample_coords(mask: np.ndarray, step_zyx: Tuple[int, int, int]) -> np.ndarray:
    if not mask.any(): return np.zeros((0, 3), np.int64)
    Z, Y, X = mask.shape
    sz, sy, sx = [max(1, int(s)) for s in step_zyx]
    zz, yy, xx = np.meshgrid(np.arange(0, Z, sz),
                             np.arange(0, Y, sy),
                             np.arange(0, X, sx), indexing='ij')
    grid = np.stack([zz, yy, xx], -1).reshape(-1, 3)
    keep = mask[grid[:, 0], grid[:, 1], grid[:, 2]]
    return np.ascontiguousarray(grid[keep])

def nms_coords(coords_zyx: np.ndarray, radius_vox=(2, 2, 2)) -> np.ndarray:
    if len(coords_zyx) == 0: return coords_zyx
    rz, ry, rx = [max(1, int(r)) for r in radius_vox]
    key = (coords_zyx // np.array([rz, ry, rx], np.int64)).astype(np.int64)
    _, uniq_idx = np.unique(key, axis=0, return_index=True)
    return np.ascontiguousarray(coords_zyx[np.sort(uniq_idx)])

def _mm_to_step_zyx(target_mm: float, sp_zyx: np.ndarray) -> Tuple[int, int, int]:
    return tuple(int(max(1, round(float(target_mm) / float(s)))) for s in sp_zyx)

def boundary_nodes_from_edt(
    pred_np: np.ndarray,
    pred_thr: np.ndarray,
    edt_outside_mm: np.ndarray,
    vess_np: np.ndarray,
    sitk_img: sitk.Image,
    shells_mm: Tuple[float, ...],
    target_step_mm: float,
    prob_max: float,
    vesselness_min: float,
    band_half_mm: float,
    nms_vox: Tuple[int, int, int],
    max_points: int
) -> np.ndarray:
    sp_zyx = get_spacing_zyx(sitk_img)
    step_zyx = _mm_to_step_zyx(float(target_step_mm), sp_zyx)

    edt_inside_mm, _ = _edt_mm_sitk(pred_thr.astype(bool), sitk_img)

    picked = []
    for sh in shells_mm:
        if sh > 0:
            band = (~pred_thr) & (np.abs(edt_outside_mm - float(sh)) <= float(band_half_mm))
        else:
            sh_abs = abs(float(sh))
            band = (pred_thr) & (np.abs(edt_inside_mm - sh_abs) <= float(band_half_mm))
        if not np.any(band):
            continue
        band &= (pred_np <= float(prob_max)) & (vess_np >= float(vesselness_min))
        if not np.any(band):
            continue
        coords = voxel_sample_coords(band, step_zyx)
        if coords.size:
            picked.append(coords)
    if not picked:
        return np.zeros((0, 3), np.int64)

    coords = np.unique(np.vstack(picked), axis=0)
    coords = nms_coords(coords, radius_vox=nms_vox)
    if coords.shape[0] > int(max_points):
        idx = np.linspace(0, coords.shape[0] - 1, int(max_points)).astype(int)
        coords = coords[idx]
    return np.ascontiguousarray(coords, dtype=np.int64)


def build_nodes(pred_np: np.ndarray, sitk_img: sitk.Image, cfg: dict):
    prob_threshold           = float(C(cfg, "prob_threshold", 0.5))
    include_shell_dilate_vox = int(C(cfg, "include_shell_dilate_vox", 0))
    voxel_subsample_zyx      = tuple(C(cfg, "voxel_subsample_zyx", (2, 2, 2)))
    vesselness_floor         = float(C(cfg, "vesselness_floor", 0.05))

    boundary_shells_mm      = tuple(C(cfg, "boundary_shells_mm", ()))  # () = disabled
    boundary_step_mm        = float(C(cfg, "boundary_step_mm", 1.0))
    boundary_prob_max       = float(C(cfg, "boundary_prob_max", 0.6))
    boundary_vess_min       = float(C(cfg, "boundary_vesselness_min", 0.1))
    boundary_band_half_mm   = float(C(cfg, "boundary_band_half_mm", 0.5))
    boundary_nms_vox        = tuple(C(cfg, "boundary_nms_vox", (2, 2, 2)))
    boundary_max_points     = int(C(cfg, "boundary_max_points", 20000))
    target_out_ratio    = float(C(cfg, "target_outside_ratio", 0.4))
    nms_radius_vox      = tuple(C(cfg, "nms_radius_vox", (2, 2, 2)))

    # Optional fast downsampling controls (disabled by default)
    fps_ratio           = C(cfg, "fps_ratio", None)
    voxel_grid_mm       = C(cfg, "voxel_grid_mm", None)

    pred_thr = (pred_np > prob_threshold)
    if include_shell_dilate_vox > 0:
        pred_thr = binary_dilation(pred_thr, iterations=include_shell_dilate_vox)

    coords_in = voxel_sample_coords(pred_thr, voxel_subsample_zyx)
    if coords_in.size == 0:
        resp = -gaussian_laplace(pred_np.astype(np.float32), sigma=1.0)
        k = max(1024, int(20 * (resp.size / 1e6)))
        idx = np.argsort(resp.ravel())[::-1][:k]
        coords_out = np.column_stack(np.unravel_index(idx, resp.shape)).astype(np.int64)
        sp_zyx = get_spacing_zyx(sitk_img)
        edt_inside_mm, _ = _edt_mm_sitk(pred_thr.astype(bool), sitk_img)
        rad_mm = edt_inside_mm[coords_out[:, 0], coords_out[:, 1], coords_out[:, 2]]
        pos_mm = voxel_to_phys(coords_out, sitk_img)
        in_pred = np.zeros((len(coords_out),), bool)
        tang = tangents_from_edt(edt_inside_mm, coords_out)
        return (np.ascontiguousarray(coords_out, dtype=np.int64),
                np.ascontiguousarray(in_pred, dtype=bool),
                np.ascontiguousarray(rad_mm, dtype=np.float32),
                np.ascontiguousarray(pos_mm, dtype=np.float32),
                np.ascontiguousarray(tang, dtype=np.float32))

    # Fast EDTs in mm (multi-threaded)
    edt_inside_mm, edt_outside_mm = _edt_mm_sitk(pred_thr.astype(bool), sitk_img)
    vess_np = prob_ridge_log(pred_np, floor=vesselness_floor)

    coords_boundary = np.zeros((0, 3), np.int64)
    if len(boundary_shells_mm) > 0:
        coords_boundary = boundary_nodes_from_edt(
            pred_np=pred_np, pred_thr=pred_thr, edt_outside_mm=edt_outside_mm,
            vess_np=vess_np, sitk_img=sitk_img, shells_mm=boundary_shells_mm,
            target_step_mm=boundary_step_mm, prob_max=boundary_prob_max,
            vesselness_min=boundary_vess_min, band_half_mm=boundary_band_half_mm,
            nms_vox=boundary_nms_vox, max_points=boundary_max_points
        )

    parts = [c for c in (coords_in, coords_boundary) if c.size > 0]
    coords_pre = np.unique(np.vstack(parts), axis=0) if parts else np.zeros((0, 3), np.int64)

    in_pred_pre = pred_thr[coords_pre[:, 0], coords_pre[:, 1], coords_pre[:, 2]]
    coords_in2  = coords_pre[in_pred_pre]
    coords_out2 = coords_pre[~in_pred_pre]
    n_in, n_out = len(coords_in2), len(coords_out2)
    if n_in > 0 and n_out > 0:
        max_out = int(target_out_ratio * (n_in + n_out))
        if n_out > max_out and max_out > 0:
            idx = np.linspace(0, n_out - 1, max_out).astype(int)
            coords_out2 = coords_out2[idx]
    coords = np.unique(np.vstack([coords_in2, coords_out2]), axis=0)

    sp_zyx = get_spacing_zyx(sitk_img)
    rad_mm = edt_inside_mm[coords[:, 0], coords[:, 1], coords[:, 2]]  # already in mm
    pos_mm = voxel_to_phys(coords, sitk_img)
    in_pred = pred_thr[coords[:, 0], coords[:, 1], coords[:, 2]]
    tang = tangents_from_edt(edt_inside_mm, coords)

    coords = nms_coords(coords, radius_vox=nms_radius_vox)

    # ---- Optional fast downsampling on GPU (kept OFF unless cfg specifies) ----
    if voxel_grid_mm is not None or fps_ratio is not None:
        P = _to_torch(pos_mm.astype(np.float32))
        keep_idx_np = None

        if voxel_grid_mm is not None:
            size = torch.tensor([float(voxel_grid_mm[2]), float(voxel_grid_mm[1]), float(voxel_grid_mm[0])], device=P.device)
            cluster = voxel_grid(P, size=size)  # returns cluster id per point
            # use cluster means for positions, and pick reps for the rest
            reps = torch.ops.torch_sparse.unique(cluster)[0] if hasattr(torch.ops, "torch_sparse") else torch.unique(cluster, sorted=True)
            keep_idx_np = reps.detach().cpu().numpy()
        elif fps_ratio is not None:
            ratio = float(fps_ratio)
            keep = fps(P, ratio=ratio, random_start=False)
            keep_idx_np = keep.detach().cpu().numpy()

        if keep_idx_np is not None and keep_idx_np.size > 0:
            coords = coords[keep_idx_np]
            in_pred = in_pred[keep_idx_np]
            rad_mm = rad_mm[keep_idx_np]
            pos_mm = pos_mm[keep_idx_np]
            tang   = tang[keep_idx_np]

    return (np.ascontiguousarray(coords, dtype=np.int64),
            np.ascontiguousarray(in_pred, dtype=bool),
            np.ascontiguousarray(rad_mm, dtype=np.float32),
            np.ascontiguousarray(pos_mm, dtype=np.float32),
            np.ascontiguousarray(tang, dtype=np.float32))



def knn_edges(pos_mm: np.ndarray, r_mm: float, max_len_mm: float) -> np.ndarray:
    if len(pos_mm) == 0:
        return np.zeros((2, 0), np.int64)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    P = _to_torch(pos_mm.astype(np.float32), device=device)

    ei = radius_graph(P, r=float(r_mm), loop=False, max_num_neighbors=256)
    ei, _ = remove_self_loops(ei)
    ei = to_undirected(ei)
    ei = coalesce(ei)  # dedup + sort

    if float(max_len_mm) > 0:
        u, v = ei
        d = torch.linalg.norm(P[v] - P[u], dim=1)
        keep = d <= float(max_len_mm)
        ei = ei[:, keep]

    if ei.numel() == 0:
        return np.zeros((2, 0), np.int64)
    return ei.detach().cpu().numpy().astype(np.int64)


def _mst_endpoints(pos_mm: np.ndarray, ei: np.ndarray) -> np.ndarray:
    if ei.shape[1] == 0 or pos_mm.shape[0] == 0:
        return np.zeros((0,), np.int64)
    G = nx.Graph()
    for u, v in ei.T:
        w = float(np.linalg.norm(pos_mm[int(u)] - pos_mm[int(v)]))
        G.add_edge(int(u), int(v), weight=w)
    T = nx.minimum_spanning_tree(G)
    deg = dict(T.degree())
    return np.ascontiguousarray(np.array([n for n, d in deg.items() if d <= 1], np.int64))

def _grid_hash_pairs(endpoints: np.ndarray, pos_mm: np.ndarray, cell: float) -> List[Tuple[int, int]]:
    if len(endpoints) == 0:
        return []
    keys = np.floor(pos_mm / max(cell, 1e-6)).astype(np.int32)
    buckets: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for idx in endpoints:
        k = tuple(keys[int(idx)])
        buckets[k].append(int(idx))
    offsets = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    cand = set()
    for k, items in buckets.items():
        for off in offsets:
            nb = (k[0] + off[0], k[1] + off[1], k[2] + off[2])
            if nb not in buckets: continue
            neigh = buckets[nb]
            for a in items:
                for b in neigh:
                    if a >= b: continue
                    cand.add((a, b))
    return list(cand)

def add_gap_candidates(pos_mm: np.ndarray,
                       base_ei: np.ndarray,
                       tangents: np.ndarray,
                       rad_mm: np.ndarray,
                       r_gap_mm: float,
                       cos_min: float,
                       dr_mm_max: float,
                       max_len_mm: float) -> np.ndarray:
    if base_ei.shape[1] == 0: return base_ei
    endpoints = _mst_endpoints(pos_mm, base_ei)
    if len(endpoints) == 0: return base_ei
    pairs = _grid_hash_pairs(endpoints, pos_mm, cell=float(r_gap_mm))
    if not pairs:
        return base_ei
    cand = []
    for u, v in pairs:
        d = pos_mm[v] - pos_mm[u]
        L = float(np.linalg.norm(d))
        if L <= 1e-8: continue
        if (max_len_mm > 0 and L > float(max_len_mm)) or (L > float(r_gap_mm) * 1.5):
            continue
        if abs(float(rad_mm[u]) - float(rad_mm[v])) > float(dr_mm_max):
            continue
        dv = d / L
        cu = float(np.dot(dv, tangents[u])); cv = float(np.dot(-dv, tangents[v]))
        if 0.5 * (cu + cv) < float(cos_min):
            continue
        cand.append((u, v))
    if not cand: return base_ei
    gap_ei = np.array(cand, np.int64).T
    return np.ascontiguousarray(np.unique(np.concatenate([base_ei, gap_ei], axis=1), axis=1))

def edge_features(pos_mm: np.ndarray,
                  tangents: np.ndarray,
                  edge_index: np.ndarray,
                  rad_mm: Optional[np.ndarray] = None,
                  use_integrals: bool = False) -> np.ndarray:
    if edge_index.shape[1] == 0:
        fdim = 6 + (3 if use_integrals else 0)
        return np.zeros((0, fdim), dtype=np.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    P  = _to_torch(pos_mm.astype(np.float32), device=device)
    T  = _to_torch(tangents.astype(np.float32), device=device)
    R  = _to_torch(rad_mm.astype(np.float32), device=device) if rad_mm is not None else None
    ei = _to_torch(edge_index.astype(np.int64), device=device)

    u, v = ei
    d    = P[v] - P[u]
    dist = torch.linalg.norm(d, dim=1, keepdim=True)
    dx   = d.abs()
    cosu = (d * T[u]).sum(-1, keepdim=True) / (dist + 1e-8)
    cosv = ((-d) * T[v]).sum(-1, keepdim=True) / (dist + 1e-8)
    cos  = 0.5 * (cosu + cosv)
    dr   = torch.zeros_like(dist) if R is None else (R[u] - R[v]).abs().unsqueeze(1)

    base = torch.cat([dist, dx, cos, dr], dim=1).contiguous()

    # Placeholder for integrals: keep same width if requested
    if use_integrals:
        integ = torch.zeros((base.size(0), 3), device=base.device, dtype=base.dtype)
        base = torch.cat([base, integ], dim=1)

    return base.detach().cpu().numpy().astype(np.float32)


def _node_features(pred_np: np.ndarray,
                   coords: np.ndarray,
                   rad_mm: np.ndarray,
                   pos_mm: np.ndarray) -> np.ndarray:
    probv = pred_np[coords[:, 0], coords[:, 1], coords[:, 2]]
    pos_norm = (pos_mm - pos_mm.mean(0, keepdims=True)) / (pos_mm.std(0, keepdims=True) + 1e-6)
    x = np.concatenate([probv[:, None].astype(np.float32),
                        rad_mm[:, None].astype(np.float32),
                        pos_norm.astype(np.float32)], 1)
    return np.ascontiguousarray(x, dtype=np.float32)


# ------------------------
# Supervision from GT
# ------------------------
def label_edges_by_gt_graph(cand_ei: np.ndarray,
                            pos_mm: np.ndarray,
                            mask_inside_gt: np.ndarray,
                            D_pos_mm: float) -> np.ndarray:
    N = len(pos_mm)
    E = cand_ei.shape[1]
    if N == 0 or E == 0:
        return np.zeros((E,), dtype=np.uint8)

    inside_nodes = np.flatnonzero(mask_inside_gt.astype(bool))
    if inside_nodes.size < 2:
        return np.zeros((E,), dtype=np.uint8)

    u, v = cand_ei
    in_u = mask_inside_gt[u]
    in_v = mask_inside_gt[v]
    both_inside = in_u & in_v
    if not np.any(both_inside):
        return np.zeros((E,), dtype=np.uint8)

    # Build sparse graph only over both-inside edges
    uu   = u[both_inside].astype(np.int64)
    vv   = v[both_inside].astype(np.int64)
    w    = np.linalg.norm(pos_mm[vv] - pos_mm[uu], axis=1).astype(np.float64)
    W    = coo_matrix((w, (uu, vv)), shape=(N, N))
    W    = (W + W.T).tocsr()

    # Truncated multi-source Dijkstra from all inside nodes
    D = dijkstra(W, directed=False, indices=inside_nodes, limit=float(D_pos_mm))
    # map node -> row in D
    row_of = -np.ones(N, dtype=np.int64)
    row_of[inside_nodes] = np.arange(inside_nodes.size, dtype=np.int64)

    labels = np.zeros((E,), dtype=np.uint8)
    sel_idx = np.flatnonzero(both_inside)
    rows = row_of[u[sel_idx]]
    cols = v[sel_idx]
    dist_uv = D[rows, cols]
    labels[sel_idx] = (dist_uv <= float(D_pos_mm)).astype(np.uint8)
    return labels



# ------------------------
# GNN model (PyTorch class stays)
# ------------------------
class gnn_model(nn.Module):
    def __init__(self, in_node: int, edge_in: int,
                 hidden: int = 96, layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.dropout = float(dropout)
        self.pseudo_cols = [0, 1, 2, 3, 4] if edge_in >= 5 else list(range(edge_in))
        self.kernel_size = 3
        self.pseudo_dim = len(self.pseudo_cols)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        ch_in = int(in_node)
        for _ in range(int(layers)):
            conv = SplineConv(
                in_channels=ch_in,
                out_channels=hidden,
                dim=self.pseudo_dim,
                kernel_size=self.kernel_size,
                aggr='mean'
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden))
            ch_in = hidden
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden + edge_in, hidden), nn.ReLU(inplace=True), nn.Dropout(self.dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1)
        )

    def _build_pseudo(self, gnn_edge_attr: torch.Tensor) -> torch.Tensor:
        if gnn_edge_attr.ndim != 2:
            raise RuntimeError("Expected gnn_edge_attr of shape [E, D].")
        e = gnn_edge_attr[:, self.pseudo_cols].clone()
        e = torch.nan_to_num(e, nan=0.0, posinf=1.0, neginf=0.0)
        try:
            pos_cos = self.pseudo_cols.index(4)
            e[:, pos_cos] = ((e[:, pos_cos] + 1.0) * 0.5).clamp(0.0, 1.0)
        except ValueError:
            pass
        col_min = e.amin(dim=0)
        col_max = e.amax(dim=0)
        denom = (col_max - col_min).clamp_min(1e-6)
        pseudo = ((e - col_min) / denom).clamp(0.0, 1.0)
        return pseudo

    def forward(self,
                x: torch.Tensor,
                gnn_edge_index: torch.Tensor,
                gnn_edge_attr: torch.Tensor,
                edge_pairs: torch.Tensor,
                edge_attr_pairs: torch.Tensor):
        pseudo = self._build_pseudo(gnn_edge_attr)
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


# ------------------------
# Graph building per case
# ------------------------
def make_graph_case(img_p: str, gt_p: str, pred_p: str, cfg: dict) -> Optional[Data]:
    gt   = read_nii_safe(gt_p)
    prd  = read_nii_safe(pred_p)

    gt_np  = (sitk_to_np(gt) > 0.5).astype(bool)
    if int(C(cfg, "gt_dilate_vox", 0)) > 0:
        gt_np = binary_dilation(gt_np, iterations=int(C(cfg, "gt_dilate_vox", 0)))

    pred_np = sitk_to_np(prd).astype(np.float32)
    if pred_np.max() > 1.5:
        pred_np = (pred_np > 0.5).astype(np.float32)

    coords, in_pred, rad_mm, pos_mm, tang = build_nodes(pred_np, prd, cfg)
    if len(coords) < 2:
        return None

    ei_knn = knn_edges(
        pos_mm,
        r_mm=float(C(cfg, "knn_radius_mm", 8.0)),
        max_len_mm=float(C(cfg, "max_edge_len_mm", 15.0))
    )
    if ei_knn.shape[1] == 0:
        return None
    ei_cand = add_gap_candidates(
        pos_mm=pos_mm, base_ei=ei_knn, tangents=tang, rad_mm=rad_mm,
        r_gap_mm=float(C(cfg, "gap_r_mm", 8.0)),
        cos_min=float(C(cfg, "gap_cos_min", 0.5)),
        dr_mm_max=float(C(cfg, "gap_dr_mm_max", 1.0)),
        max_len_mm=float(C(cfg, "max_edge_len_mm", 15.0))
    )

    ea_cand = edge_features(pos_mm, tang, ei_cand, rad_mm, use_integrals=False)

    if ei_cand.shape[1] > 0:
        N = int(pos_mm.shape[0])
        gmask = _map_gnn_to_cand_gpu(ei_knn, ei_cand, N=N)
        ea_knn = ea_cand[gmask]
    else:
        ea_knn = np.zeros((0, ea_cand.shape[1] if ea_cand.ndim == 2 else 0), dtype=np.float32)

    x = _node_features(pred_np, coords, rad_mm, pos_mm)

    mask_inside_gt = (gt_np[coords[:, 0], coords[:, 1], coords[:, 2]]).astype(bool)
    y = label_edges_by_gt_graph(ei_cand, pos_mm, mask_inside_gt,
                                D_pos_mm=float(C(cfg, "pos_path_max_mm", 10.0)))
    pos_mask = (y == 1); neg_mask = ~pos_mask
    if pos_mask.sum() == 0:
        return None

    pos_ei = ei_cand[:, pos_mask]
    pos_ea = ea_cand[pos_mask]
    neg_idx = np.flatnonzero(neg_mask)
    if len(neg_idx) == 0:
        return None
    rng = np.random.default_rng(int(C(cfg, "seed", 123)))
    n_pos = int(pos_mask.sum())
    npp   = int(C(cfg, "neg_per_pos", 4))
    n_neg = min(len(neg_idx), npp * n_pos)
    sel = rng.choice(neg_idx, size=n_neg, replace=False)
    neg_ei = ei_cand[:, sel]
    neg_ea = ea_cand[sel]

    # contiguous
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


# ------------------------
# Export helpers
# ------------------------
def _assemble_polylines(n_nodes: int, edge_index_np: np.ndarray) -> List[List[int]]:
    if edge_index_np.size == 0: return []
    u = edge_index_np[0].tolist(); v = edge_index_np[1].tolist()
    adj = {i: set() for i in range(n_nodes)}
    for a, b in zip(u, v):
        a = int(a); b = int(b)
        adj[a].add(b); adj[b].add(a)
    deg = {i: len(adj[i]) for i in adj}
    polylines = []
    visited = set()
    def mark(a, b):
        a, b = (a, b) if a < b else (b, a); visited.add((a, b))
    def seen(a, b):
        a, b = (a, b) if a < b else (b, a); return (a, b) in visited
    starts = [i for i, d in deg.items() if d != 2] or list(adj.keys())
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
            if seen(a, b): continue
            polylines.append([a, b]); mark(a, b)
    return polylines

def export_graph_to_vtp(coords_zyx: np.ndarray, sitk_img: sitk.Image, edge_index_np: np.ndarray, out_path: str):
    if not _HAVE_VTK:
        print(f"[WARN] VTK not available; skipping VTP export: {out_path}")
        return
    points = vtk.vtkPoints()
    for z, y, x in coords_zyx:
        X, Y, Z = sitk_img.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
        points.InsertNextPoint(float(X), float(Y), float(Z))
    lines = vtk.vtkCellArray()
    for seq in _assemble_polylines(len(coords_zyx), edge_index_np):
        if len(seq) < 2: continue
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(seq))
        for i, pid in enumerate(seq):
            polyline.GetPointIds().SetId(i, int(pid))
        lines.InsertNextCell(polyline)
    poly = vtk.vtkPolyData(); poly.SetPoints(points); poly.SetLines(lines)
    wr = vtk.vtkXMLPolyDataWriter(); wr.SetFileName(out_path); wr.SetInputData(poly); wr.Write()
    print("Saved:", out_path)


# ------------------------
# Edge assembly
# ------------------------
def assemble_mst_weighted(pos_mm: np.ndarray, ei: np.ndarray, lambda_len_inv: float) -> np.ndarray:
    if ei.shape[1] == 0:
        return ei
    u, v = ei
    length = np.linalg.norm(pos_mm[v] - pos_mm[u], axis=1) + 1e-8
    w = (float(lambda_len_inv) / length).astype(np.float64)
    N = int(pos_mm.shape[0])
    W = coo_matrix((w, (u, v)), shape=(N, N))
    W = (W + W.T) * 0.5
    T = minimum_spanning_tree(W.tocsr()).tocoo()
    if T.nnz == 0:
        return np.zeros((2, 0), np.int64)
    # undirected edge list from MST
    mst_u = T.row.astype(np.int64)
    mst_v = T.col.astype(np.int64)
    mst = np.stack([mst_u, mst_v], axis=0)
    # unique undirected
    a = np.minimum(mst[0], mst[1])
    b = np.maximum(mst[0], mst[1])
    E = np.unique(np.stack([a, b], axis=0), axis=1)
    return np.ascontiguousarray(E, dtype=np.int64)


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



def _build_train_items(train_list: List[Tuple[str, str, str]], cfg: dict) -> List[Data]:
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


from contextlib import nullcontext
import torch
import torch.nn.functional as F
from torch_geometric import transforms as T

def _build_inbuilt_transform(cfg: dict | None):
    tr = []
    tr.append(T.ToUndirected())
    tr.append(T.NormalizeFeatures())
    return T.Compose(tr) if len(tr) > 0 else None


def train_one_epoch(model, loader, opt, scaler, device, cfg: dict | None = None) -> float:
    model.train()
    total = 0.0
    amp_flag = bool(C(cfg or {}, "amp", True)) and torch.cuda.is_available()
    amp_ctx  = torch.amp.autocast('cuda', enabled=amp_flag) if torch.cuda.is_available() else nullcontext()

    tg_tr = _build_inbuilt_transform(cfg)

    for data in loader:
        data = data.to(device)
        opt.zero_grad(set_to_none=True)
        data_aug = tg_tr(data) if tg_tr is not None else data

        with amp_ctx:
            pos_logits = model(
                data_aug.x, data_aug.edge_index, data_aug.edge_attr,
                data.pos_edge_index, data.pos_edge_attr   # <- supervision tensors unchanged
            )
            neg_logits = model(
                data_aug.x, data_aug.edge_index, data_aug.edge_attr,
                data.neg_edge_index, data.neg_edge_attr
            )

            pos_labels = torch.ones_like(pos_logits)
            neg_labels = torch.zeros_like(neg_logits)
            loss_pos   = F.binary_cross_entropy_with_logits(pos_logits, pos_labels)
            loss_neg   = F.binary_cross_entropy_with_logits(neg_logits, neg_labels)
            loss       = loss_pos + loss_neg

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += float(loss.detach().cpu())

    return total / max(1, len(loader))

@torch.no_grad()
def predict_edges(model: nn.Module,
                  x: torch.Tensor,
                  gnn_ei: torch.Tensor,
                  cand_ei: torch.Tensor,
                  cand_ea: torch.Tensor,
                  gnn_ea: Optional[torch.Tensor],
                  device: torch.device,
                  cfg: dict | None = None) -> np.ndarray:
    model.eval()
    amp_flag = bool(C(cfg or {}, "amp", True)) and torch.cuda.is_available()
    amp_ctx = torch.amp.autocast('cuda', enabled=amp_flag) if torch.cuda.is_available() else nullcontext()
    with amp_ctx:
        if gnn_ea is None: gnn_ea = cand_ea
        logits = model(x, gnn_ei, gnn_ea, cand_ei, cand_ea)
        probs = torch.sigmoid(logits)
    return probs.detach().cpu().numpy()


# ------------------------
# Per-case utilities
# ------------------------
def process_case_build_only(img_p: str, gt_p: str, pred_p: str, cfg: dict):
    name = os.path.basename(pred_p).replace(".nii.gz", "").replace("_0000", "")
    print(f"\n--- {name} ---")

    gt   = read_nii_safe(gt_p)
    prd  = read_nii_safe(pred_p)
    gt_np   = (sitk_to_np(gt)   > 0.5).astype(bool)
    pred_np = sitk_to_np(prd).astype(np.float32)
    if pred_np.max() > 1.5:
        pred_np = (pred_np > 0.5).astype(np.float32)

    coords, in_pred, rad_mm, pos_mm, tang = build_nodes(pred_np, prd, cfg)
    if len(coords) < 2:
        print("[WARN] No nodes; skipping.")
        return

    ei_knn = knn_edges(pos_mm,
                       float(C(cfg, "knn_radius_mm", 8.0)),
                       float(C(cfg, "max_edge_len_mm", 15.0)))
    ei_cand = add_gap_candidates(
        pos_mm=pos_mm,
        base_ei=ei_knn,
        tangents=tang,
        rad_mm=rad_mm,
        r_gap_mm=float(C(cfg, "gap_r_mm", 8.0)),
        cos_min=float(C(cfg, "gap_cos_min", 0.5)),
        dr_mm_max=float(C(cfg, "gap_dr_mm_max", 1.0)),
        max_len_mm=float(C(cfg, "max_edge_len_mm", 15.0))
    )

    n = len(coords)
    frac_outside = 1.0 - float(in_pred.sum()) / max(1, n)
    gt_cover = float(gt_np[coords[:, 0], coords[:, 1], coords[:, 2]].mean()) if n > 0 else 0.0
    print(f"nodes={n} | in_pred={int(in_pred.sum())} ({1-frac_outside:.2%}), outside={int((~in_pred).sum())} ({frac_outside:.2%})")
    print(f"E_knn={int(ei_knn.shape[1])} | E_cand={int(ei_cand.shape[1])} | nodes_in_GT={gt_cover:.2%}")

    out_dir = C(cfg, "out_dir", "graph_out")
    os.makedirs(out_dir, exist_ok=True)
    if bool(C(cfg, "export_candidates_vtp", False)):
        out_cand = os.path.join(out_dir, f"{name}_graph_candidates.vtp")
        export_graph_to_vtp(coords, prd, ei_cand, out_cand)

    if bool(C(cfg, "export_mst_preview_vtp", False)):
        ei_mst = assemble_mst_weighted(pos_mm, ei_cand, lambda_len_inv=float(C(cfg, "mst_lambda_len_inv", 0.1)))
        out_mst = os.path.join(out_dir, f"{name}_graph_mst.vtp")
        export_graph_to_vtp(coords, prd, ei_mst, out_mst)


def process_case_infer(img_p: str, pred_p: str, model: nn.Module, cfg: dict, device: torch.device):
    name = os.path.basename(pred_p).replace(".nii.gz", "").replace("_0000", "")
    prd  = read_nii_safe(pred_p)
    pred_np = sitk_to_np(prd).astype(np.float32)
    if pred_np.max() > 1.5: pred_np = (pred_np > 0.5).astype(np.float32)

    coords, in_pred, rad_mm, pos_mm, tang = build_nodes(pred_np, prd, cfg)
    if len(coords) < 2:
        print(f"[WARN] empty node set for {name}; skipping.")
        return

    gnn_ei = knn_edges(pos_mm,
                       float(C(cfg, "knn_radius_mm", 8.0)),
                       float(C(cfg, "max_edge_len_mm", 15.0)))
    cand_ei = add_gap_candidates(pos_mm, gnn_ei, tang, rad_mm,
                                 r_gap_mm=float(C(cfg, "gap_r_mm", 8.0)),
                                 cos_min=float(C(cfg, "gap_cos_min", 0.5)),
                                 dr_mm_max=float(C(cfg, "gap_dr_mm_max", 1.0)),
                                 max_len_mm=float(C(cfg, "max_edge_len_mm", 15.0)))

    cand_ea = edge_features(pos_mm, tang, cand_ei, rad_mm, use_integrals=False)
    N = int(pos_mm.shape[0])
    gmask = _map_gnn_to_cand_gpu(gnn_ei, cand_ei, N=N)
    gnn_ea = cand_ea[gmask]

    x = _node_features(pred_np, coords, rad_mm, pos_mm)

    x_t       = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)
    gnn_ei_t  = torch.from_numpy(np.ascontiguousarray(gnn_ei, dtype=np.int64)).to(device)
    gnn_ea_t  = torch.from_numpy(np.ascontiguousarray(gnn_ea, dtype=np.float32)).to(device)
    cand_ei_t = torch.from_numpy(np.ascontiguousarray(cand_ei, dtype=np.int64)).to(device)
    cand_ea_t = torch.from_numpy(np.ascontiguousarray(cand_ea, dtype=np.float32)).to(device)

    probs = predict_edges(model, x_t, gnn_ei_t, cand_ei_t, cand_ea_t, gnn_ea_t, device, cfg)
    keep = probs >= float(C(cfg, "edge_prob_thresh", 0.5))
    ei_keep = cand_ei[:, keep]
    probs_keep = probs[keep]

    ei_pred = assemble_edges_longpaths(
        pos_mm, ei_keep, probs_keep,
        lambda_len_inv=float(C(cfg, "mst_lambda_len_inv", 0.1)),
        add_back_thresh=float(C(cfg, "add_back_thresh", 0.9))
    )

    out_dir = C(cfg, "out_dir", "graph_out")
    if bool(C(cfg, "export_predicted_vtp", False)) and ei_pred.shape[1] > 0:
        os.makedirs(out_dir, exist_ok=True)
        out_vtp = os.path.join(out_dir, f"{name}_graph_predicted.vtp")
        export_graph_to_vtp(coords, prd, ei_pred, out_vtp)


def save_predicted_graph_to_vtp(G: nx.Graph, out_path: str):
    import vtk

    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    arr_rad = vtk.vtkDoubleArray(); arr_rad.SetName("MaximumInscribedSphereRadius")
    arr_tan = vtk.vtkDoubleArray(); arr_tan.SetName("Tangent"); arr_tan.SetNumberOfComponents(3)

    idmap = {}
    for i, (n, data) in enumerate(G.nodes(data=True)):
        x, y, z = map(float, data["pos_phys"])
        pid = pts.InsertNextPoint(x, y, z)
        idmap[n] = pid
        r = float(data.get("radius_mm", 0.5))
        arr_rad.InsertNextValue(r)
        t = np.asarray(data.get("tangent", [0.0, 0.0, 0.0]), dtype=float)
        if t.size != 3: t = np.zeros(3, float)
        arr_tan.InsertNextTuple(t.tolist())

    arr_prob = vtk.vtkDoubleArray(); arr_prob.SetName("EdgeProbability")
    arr_len  = vtk.vtkDoubleArray(); arr_len.SetName("LengthMM")
    arr_rmin = vtk.vtkDoubleArray(); arr_rmin.SetName("RadiusMinMM")
    arr_ravg = vtk.vtkDoubleArray(); arr_ravg.SetName("RadiusMeanMM")

    for u, v, ed in G.edges(data=True):
        lid = vtk.vtkLine()
        lid.GetPointIds().SetId(0, idmap[u])
        lid.GetPointIds().SetId(1, idmap[v])
        lines.InsertNextCell(lid)
        arr_prob.InsertNextValue(float(ed.get("edge_prob", 0.5)))
        arr_len.InsertNextValue(float(ed.get("weight", 0.0)))
        arr_rmin.InsertNextValue(float(ed.get("radius_min_mm", min(G.nodes[u]["radius_mm"], G.nodes[v]["radius_mm"]))))
        arr_ravg.InsertNextValue(float(ed.get("radius_mean_mm", 0.5*(G.nodes[u]["radius_mm"]+G.nodes[v]["radius_mm"]))))

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetLines(lines)
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


# ------------------------
# GNN load / predict (dict cfg)
# ------------------------
def gm_load(ckpt_path: Optional[str] = None,
            device: Optional[torch.device] = None,
            *, cfg: dict | None = None):
    dev = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    base_cfg = copy.deepcopy(cfg or {})
    out_dir = C(base_cfg, "out_dir", ".")
    ckpt_name = C(base_cfg, "ckpt_name", "gnn_best.pt")
    ckpt_p = ckpt_path or os.path.join(_norm_path(out_dir), 'best_' + ckpt_name)
    if not os.path.isfile(ckpt_p):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")
    try:
        ckpt = torch.load(ckpt_p, map_location=dev, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_p, map_location=dev)
    default_in_node = 5
    default_edge_in = 6
    in_node = int(ckpt.get('in_node', default_in_node))
    edge_in = int(ckpt.get('edge_in', default_edge_in))
    model = gnn_model(in_node=in_node, edge_in=edge_in,
                      hidden=int(C(base_cfg, "hidden", 96)),
                      layers=int(C(base_cfg, "layers", 3)),
                      dropout=float(C(base_cfg, "dropout", 0.2))).to(dev)
    state_key = 'model_state' if 'model_state' in ckpt else 'model'
    model.load_state_dict(ckpt[state_key], strict=True)
    model.eval()
    return (model, base_cfg, dev, ckpt_p)

def gm_predict_graph(*,
    seg_img: sitk.Image,
    prob_img: Optional[sitk.Image],
    model_tuple,
    node_min_spacing_mm: Optional[float] = None,
    knn_k: Optional[int] = None,
    knn_radius_mm: Optional[float] = 8,
    edge_prob_thresh: Optional[float] = 0.5,
    cfg: Optional[dict] = None,
    use_adaptive_spacing: bool = True,
    return_debug: bool = False
) -> nx.Graph | tuple[nx.Graph, dict]:
    """
    Build a graph in physical XYZ with per-node radius and per-edge probabilities.
    Returns G or (G, dbg) if return_debug=True.
    """
    model, base_cfg, device, _ = model_tuple
    local_cfg = copy.deepcopy(base_cfg)
    if cfg:
        local_cfg.update(cfg)

    if knn_k is not None:            local_cfg["knn_k"] = int(knn_k)
    if knn_radius_mm is not None:    local_cfg["knn_radius_mm"] = float(knn_radius_mm)
    if edge_prob_thresh is not None: local_cfg["edge_prob_thresh"] = float(edge_prob_thresh)

    # reference image for spacing/origin/direction
    ref_img = prob_img if prob_img is not None else seg_img

    # prediction array (normalize if mask)
    prob_np = sitk_to_np(ref_img).astype(np.float32)
    if prob_np.max() > 1.5:
        prob_np = (prob_np > 0.5).astype(np.float32)

    # adaptive node spacing in voxels if requested
    if use_adaptive_spacing and (node_min_spacing_mm is not None):
        sp_zyx = get_spacing_zyx(ref_img)
        step_zyx = tuple(int(max(1, np.ceil(float(node_min_spacing_mm) / float(s)))) for s in sp_zyx)
        local_cfg["voxel_subsample_zyx"] = step_zyx

    # node proposal (coords are Z,Y,X indices)
    coords, in_pred, rad_mm, pos_mm, tang = build_nodes(prob_np, ref_img, local_cfg)
    G = nx.Graph()
    if coords.size < 2:
        dbg = {"reason": "no_nodes"}
        return (G, dbg) if return_debug else G

    # KNN edges (candidate neighborhood)
    gnn_ei = knn_edges(
        pos_mm,
        r_mm=float(C(local_cfg, "knn_radius_mm", 8.0)),
        max_len_mm=float(C(local_cfg, "max_edge_len_mm", 15.0))
    )
    if gnn_ei.shape[1] == 0:
        dbg = {"reason": "no_knn_edges", "N": int(len(coords))}
        return (G, dbg) if return_debug else G

    # add "gap" candidates guided by tangents/radius
    cand_ei = add_gap_candidates(
        pos_mm=pos_mm, base_ei=gnn_ei, tangents=tang, rad_mm=rad_mm,
        r_gap_mm=float(C(local_cfg, "gap_r_mm", 8.0)),
        cos_min=float(C(local_cfg, "gap_cos_min", 0.5)),
        dr_mm_max=float(C(local_cfg, "gap_dr_mm_max", 1.0)),
        max_len_mm=float(C(local_cfg, "max_edge_len_mm", 15.0))
    )
    if cand_ei.shape[1] == 0:
        dbg = {"reason": "no_candidate_edges", "E_knn": int(gnn_ei.shape[1])}
        return (G, dbg) if return_debug else G

    cand_ea = edge_features(pos_mm, tang, cand_ei, rad_mm, use_integrals=False)

    # align gnn_ea to cand ordering (for residual head, if used)
    uv = cand_ei.T
    key = {(int(a), int(b)): i for i, (a, b) in enumerate(uv)}
    gmask = np.zeros(gnn_ei.shape[1], dtype=int)
    for i, (a, b) in enumerate(gnn_ei.T):
        gmask[i] = key.get((int(a), int(b)), key.get((int(b), int(a))))
    gnn_ea = cand_ea[gmask]

    # node features
    x_np = _node_features(prob_np, coords, rad_mm, pos_mm)  # provided by your gnn codebase

    # to torch
    x_t       = torch.from_numpy(np.ascontiguousarray(x_np,    dtype=np.float32)).to(device)
    gnn_ei_t  = torch.from_numpy(np.ascontiguousarray(gnn_ei,  dtype=np.int64)).to(device)
    gnn_ea_t  = torch.from_numpy(np.ascontiguousarray(gnn_ea,  dtype=np.float32)).to(device)
    cand_ei_t = torch.from_numpy(np.ascontiguousarray(cand_ei, dtype=np.int64)).to(device)
    cand_ea_t = torch.from_numpy(np.ascontiguousarray(cand_ea, dtype=np.float32)).to(device)

    # predict edge probabilities for the candidate edges
    probs = predict_edges(model, x_t, gnn_ei_t, cand_ei_t, cand_ea_t, gnn_ea_t, device, local_cfg)
    thr = float(C(local_cfg, "edge_prob_thresh", 0.5))
    keep = probs >= thr
    if keep.sum() == 0:
        dbg = {"reason": "no_edges_above_threshold", "E_cand": int(cand_ei.shape[1]), "thr": thr}
        return (G, dbg) if return_debug else G

    ei_keep    = cand_ei[:, keep]
    probs_keep = probs[keep]

    # optionally wire longpaths / MST-style assembly
    ei_pred = assemble_edges_longpaths(
        pos_mm, ei_keep, probs_keep,
        lambda_len_inv=float(C(local_cfg, "mst_lambda_len_inv", 0.1)),
        add_back_thresh=float(C(local_cfg, "add_back_thresh", 0.9))
    )
    prob_map = {tuple(sorted(map(int, e))): float(p) for e, p in zip(ei_keep.T, probs_keep)}

    # ---- build graph in PHYSICAL XYZ ----
    # coords[i] is (z,y,x) indices → convert to (x,y,z) physical
    for i, (z, y, x) in enumerate(coords):
        pxyz = ref_img.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))  # (X,Y,Z) mm ✅
        r_mm = float(rad_mm[i])
        G.add_node(
            int(i),
            pos=np.array([int(z), int(y), int(x)], dtype=int),  # keep index (ZYX) if needed
            pos_phys=np.array(pxyz, dtype=float),               # PHYSICAL XYZ
            tangent=tang[i],
            radius_mm=r_mm,
            MaximumInscribedSphereRadius=r_mm
        )

    for u, v in ei_pred.T:
        u = int(u); v = int(v)
        p1 = np.asarray(G.nodes[u]['pos_phys'], float)
        p2 = np.asarray(G.nodes[v]['pos_phys'], float)
        ru = float(G.nodes[u]['radius_mm']); rv = float(G.nodes[v]['radius_mm'])
        r_min  = min(ru, rv)
        r_mean = 0.5 * (ru + rv)
        G.add_edge(
            u, v,
            length_mm=float(np.linalg.norm(p1 - p2)),
            edge_prob=prob_map.get(tuple(sorted((u, v))), 0.5),
            radius_min_mm=r_min,
            radius_mean_mm=r_mean
        )

    if return_debug:
        dbg = {
            "N": G.number_of_nodes(),
            "E_final": G.number_of_edges(),
            "thr": thr,
            "cfg_used": copy.deepcopy(local_cfg),
        }
        return G, dbg
    return G
