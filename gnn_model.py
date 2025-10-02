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
    sp = np.asarray(img.GetSpacing(), dtype=np.float64)
    org = np.asarray(img.GetOrigin(), dtype=np.float64)
    D   = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3,3)
    ijk_xyz = coords_zyx[:, ::-1].astype(np.float64) * sp
    out = (D @ ijk_xyz.T).T + org
    return np.ascontiguousarray(out, dtype=np.float32)

def tangents_from_edt(edt_zyx: np.ndarray, coords_zyx: np.ndarray) -> np.ndarray:
    if coords_zyx.size == 0:
        return np.zeros((0,3), np.float32)
    gz, gy, gx = np.gradient(edt_zyx)
    g = np.stack([gz, gy, gx], axis=-1)
    v = g[coords_zyx[:,0], coords_zyx[:,1], coords_zyx[:,2]].astype(np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    t = -v / n
    return np.ascontiguousarray(t, dtype=np.float32)

def _parse_args():
    p = argparse.ArgumentParser("GNN model runner")
    p.add_argument("--cfg", required=False, help="Path to YAML config file")
    return p.parse_args()

def _coerce(value, anno):
    origin = get_origin(anno)
    if origin is tuple or origin is Tuple:
        return tuple(value) if isinstance(value, (list, tuple)) else (value,)
    if origin is list or origin is List:
        return list(value) if isinstance(value, (list, tuple)) else [value]
    if anno in (int, float, bool, str):
        return anno(value)
    return value
def _norm_path(v: str) -> str:
    v = os.path.expanduser(os.path.expandvars(str(v)))
    return os.path.abspath(v)

def _looks_like_path(key: str) -> bool:
    key = key.lower()
    return any(s in key for s in ["dir", "folder", "path", "out"])

def load_cfg_from_yaml(path: str, base_cfg) -> "CFG":
    with open(os.path.abspath(os.path.expanduser(path)), "r") as f:
        y = yaml.safe_load(f) or {}
    kv = {}
    fmeta = {f.name: f.type for f in fields(base_cfg)}
    for k, v in y.items():
        if k not in fmeta:
            print(f"[cfg] ignoring unknown key: {k}")
            continue
        vv = _coerce(v, fmeta[k])
        if isinstance(vv, str) and _looks_like_path(k):
            vv = _norm_path(vv)
        kv[k] = vv
    return replace(base_cfg, **kv)

#Graph build helpers

def knn_edges(pos_mm: np.ndarray, k: int, r_mm: float, max_len_mm: float) -> np.ndarray:
    if len(pos_mm) == 0:
        return np.zeros((2,0), np.int64)
    P = torch.from_numpy(pos_mm.astype(np.float32)).to("cuda")
    kk = min(int(k)+1, len(pos_mm))
    ei = knn_graph(P, k=kk, loop=False)
    u = ei[0].detach().cpu().numpy()
    v = ei[1].detach().cpu().numpy()
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
    if base_ei.shape[1]==0: return base_ei
    endpoints = _mst_endpoints(pos_mm, base_ei)
    if len(endpoints)==0: return base_ei
    pairs = _grid_hash_pairs(endpoints, pos_mm, cell=float(r_gap_mm))
    if not pairs:
        return base_ei
    cand = []
    for u,v in pairs:
        d = pos_mm[v]-pos_mm[u]
        L = float(np.linalg.norm(d))
        if L <= 1e-8: continue
        if (max_len_mm>0 and L>float(max_len_mm)) or (L>float(r_gap_mm)*1.5):
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


def prob_ridge_log(prob: np.ndarray, floor: float = 0.05) -> np.ndarray:
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

def nms_coords(coords_zyx: np.ndarray, radius_vox=(2,2,2)) -> np.ndarray:
    if len(coords_zyx) == 0: return coords_zyx
    rz, ry, rx = [max(1,int(r)) for r in radius_vox]
    key = (coords_zyx // np.array([rz,ry,rx], np.int64)).astype(np.int64)
    _, uniq_idx = np.unique(key, axis=0, return_index=True)
    return np.ascontiguousarray(coords_zyx[np.sort(uniq_idx)])

def _mm_to_step_zyx(target_mm: float, sp_zyx: np.ndarray) -> Tuple[int,int,int]:
    return tuple(int(max(1, round(float(target_mm) / float(s)))) for s in sp_zyx)


#Graph build helpers

def belt_nodes_from_edt(
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
    nms_vox: Tuple[int,int,int],
    max_points: int
) -> np.ndarray:
    sp_zyx = get_spacing_zyx(sitk_img)
    step_zyx = _mm_to_step_zyx(float(target_step_mm), sp_zyx)
    picked = []
    edt_inside_mm = distance_transform_edt(pred_thr, sampling=sitk_img.GetSpacing()[::-1]).astype(np.float32)
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
        return np.zeros((0,3), np.int64)
    coords = np.unique(np.vstack(picked), axis=0)
    coords = nms_coords(coords, radius_vox=nms_vox)
    if coords.shape[0] > int(max_points):
        idx = np.linspace(0, coords.shape[0]-1, int(max_points)).astype(int)
        coords = coords[idx]
    return np.ascontiguousarray(coords, dtype=np.int64)

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

    # Precomputing
    edt_inside  = distance_transform_edt(pred_thr,  sampling=sitk_img.GetSpacing()[::-1]).astype(np.float32)
    edt_outside = distance_transform_edt(~pred_thr, sampling=sitk_img.GetSpacing()[::-1]).astype(np.float32)
    vess_np     = prob_ridge_log(pred_np, floor=cfg.vesselness_floor)


    coords_belt = belt_nodes_from_edt(
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

    parts = [c for c in (coords_in, coords_belt) if c.size > 0]
    coords_pre = np.unique(np.vstack(parts), axis=0) if parts else np.zeros((0,3), np.int64)


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

    sp_zyx = get_spacing_zyx(sitk_img)
    rad_mm = edt_inside[coords[:, 0], coords[:, 1], coords[:, 2]] * float(sp_zyx.min())
    pos_mm = voxel_to_phys(coords, sitk_img)
    in_pred = pred_thr[coords[:, 0], coords[:, 1], coords[:, 2]]
    tang = tangents_from_edt(edt_inside, coords)

    coords = nms_coords(coords, radius_vox=cfg.nms_radius_vox)

    return (np.ascontiguousarray(coords, dtype=np.int64),
            np.ascontiguousarray(in_pred, dtype=bool),
            np.ascontiguousarray(rad_mm, dtype=np.float32),
            np.ascontiguousarray(pos_mm, dtype=np.float32),
            np.ascontiguousarray(tang, dtype=np.float32))


#Features
def edge_features(pos_mm: np.ndarray,
                  tangents: np.ndarray,
                  edge_index: np.ndarray,
                  rad_mm: Optional[np.ndarray] = None,
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
    return np.ascontiguousarray(base, dtype=np.float32)

def _node_features(pred_np: np.ndarray,
                   coords: np.ndarray,
                   rad_mm: np.ndarray,
                   pos_mm: np.ndarray) -> np.ndarray:
    probv = pred_np[coords[:,0], coords[:,1], coords[:,2]]
    pos_norm = (pos_mm - pos_mm.mean(0, keepdims=True)) / (pos_mm.std(0, keepdims=True) + 1e-6)
    x = np.concatenate([probv[:,None].astype(np.float32),
                        rad_mm[:,None].astype(np.float32),
                        pos_norm.astype(np.float32)], 1)
    return np.ascontiguousarray(x, dtype=np.float32)


#supervision using Ground Truth
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


#GNN model
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
            nn.Linear(2*hidden + edge_in, hidden), nn.ReLU(inplace=True), nn.Dropout(self.dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(inplace=True),
            nn.Linear(hidden//2, 1)
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


#graph building
def make_graph_case(img_p: str, gt_p: str, pred_p: str, cfg: CFG) -> Optional[Data]:
    gt   = read_nii_safe(gt_p)
    prd  = read_nii_safe(pred_p)

    gt_np  = (sitk_to_np(gt) > 0.5).astype(bool)
    if cfg.gt_dilate_vox > 0:
        gt_np = binary_dilation(gt_np, iterations=int(cfg.gt_dilate_vox))
    pred_np = sitk_to_np(prd).astype(np.float32)
    if pred_np.max() > 1.5:
        pred_np = (pred_np > 0.5).astype(np.float32)

    coords, in_pred, rad_mm, pos_mm, tang = build_nodes(pred_np, prd, cfg)
    if len(coords) < 2:
        return None

    ei_knn = knn_edges(pos_mm, cfg.knn_k, cfg.knn_radius_mm, cfg.max_edge_len_mm)
    if ei_knn.shape[1] == 0:
        return None
    ei_cand = add_gap_candidates(
        pos_mm=pos_mm, base_ei=ei_knn, tangents=tang, rad_mm=rad_mm,
        r_gap_mm=cfg.gap_r_mm, cos_min=cfg.gap_cos_min, dr_mm_max=cfg.gap_dr_mm_max,
        max_len_mm=cfg.max_edge_len_mm
    )

    # geometry-only features
    ea_cand = edge_features(pos_mm, tang, ei_cand, rad_mm, use_integrals=False)

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

    x = _node_features(pred_np, coords, rad_mm, pos_mm)

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


#Export helpers
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

#MST assisted
def assemble_mst_weighted(pos_mm: np.ndarray, ei: np.ndarray, lambda_len_inv: float) -> np.ndarray:
    if ei.shape[1]==0: return ei
    lengths = np.linalg.norm(pos_mm[ei[1]] - pos_mm[ei[0]], axis=1) + 1e-8
    w = lambda_len_inv / lengths
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


#Training
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
    name = os.path.basename(pred_p).replace(".nii.gz","").replace("_0000","")
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

    n = len(coords)
    frac_outside = 1.0 - float(in_pred.sum())/max(1,n)
    gt_cover = float(gt_np[coords[:,0], coords[:,1], coords[:,2]].mean()) if n>0 else 0.0
    print(f"nodes={n} | in_pred={int(in_pred.sum())} ({1-frac_outside:.2%}), outside={int((~in_pred).sum())} ({frac_outside:.2%})")
    print(f"E_knn={int(ei_knn.shape[1])} | E_cand={int(ei_cand.shape[1])} | nodes_in_GT={gt_cover:.2%}")

    os.makedirs(cfg.out_dir, exist_ok=True)
    if cfg.export_candidates_vtp:
        out_cand = os.path.join(cfg.out_dir, f"{name}_graph_candidates.vtp")
        export_graph_to_vtp(coords, prd, ei_cand, out_cand)

    if cfg.export_mst_preview_vtp:
        ei_mst = assemble_mst_weighted(pos_mm, ei_cand, lambda_len_inv=cfg.mst_lambda_len_inv)
        out_mst = os.path.join(cfg.out_dir, f"{name}_graph_mst.vtp")
        export_graph_to_vtp(coords, prd, ei_mst, out_mst)


#inference

def process_case_infer(img_p: str, pred_p: str, model: gnn_model, cfg: CFG, device: torch.device):
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

    cand_ea = edge_features(pos_mm, tang, cand_ei, rad_mm, use_integrals=False)
    uv = cand_ei.T
    key = { (int(a),int(b)): i for i,(a,b) in enumerate(uv) }
    gmask = np.zeros(gnn_ei.shape[1], dtype=int)
    for i,(a,b) in enumerate(gnn_ei.T):
        gmask[i] = key.get((int(a),int(b)), key.get((int(b),int(a))))
    gnn_ea = cand_ea[gmask]

    x = _node_features(pred_np, coords, rad_mm, pos_mm)

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


import copy

def _merge_cfg(runtime_cfg, ckpt_cfg_dict: dict, prefer: str = "runtime"):
    base = copy.deepcopy(runtime_cfg)
    if ckpt_cfg_dict is None:
        return base
    r = dict(asdict(base))
    if prefer.lower() == "runtime":
        for k, v in ckpt_cfg_dict.items():
            if k not in r or r[k] is None:
                r[k] = v
    else:
        for k, v in ckpt_cfg_dict.items():
            r[k] = v
    try:
        return CFG(**r)
    except TypeError:
        known = {k: r[k] for k in r if k in CFG().__dict__}
        return CFG(**known)

def gm_load(ckpt_path: Optional[str] = None,
            device: Optional[torch.device] = None,
            *,
            runtime_cfg: Optional[CFG] = None,
            prefer_cfg: str = "runtime"):
    dev = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    base_runtime_cfg = copy.deepcopy(runtime_cfg if runtime_cfg is not None else cfg)
    ckpt_p = ckpt_path or os.path.join(base_runtime_cfg.out_dir, 'best_' + base_runtime_cfg.ckpt_name)
    if not os.path.isfile(ckpt_p):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")
    try:
        ckpt = torch.load(ckpt_p, map_location=dev, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_p, map_location=dev)
    ckpt_cfg_dict = ckpt.get('cfg', None)
    merged_cfg = _merge_cfg(base_runtime_cfg, ckpt_cfg_dict, prefer=prefer_cfg)
    default_in_node = 5
    default_edge_in = 6  # geometry-only
    in_node = int(ckpt.get('in_node', default_in_node))
    edge_in = int(ckpt.get('edge_in', default_edge_in))
    model = gnn_model(in_node=in_node, edge_in=edge_in, hidden=96, layers=3, dropout=0.2).to(dev)
    state_key = 'model_state' if 'model_state' in ckpt else 'model'
    model.load_state_dict(ckpt[state_key], strict=True)
    model.eval()
    return (model, merged_cfg, dev, ckpt_p)

def gm_predict_graph(
    *,
    seg_img: sitk.Image,
    prob_img: Optional[sitk.Image],
    model_tuple,
    node_min_spacing_mm: Optional[float] = None,
    knn_k: Optional[int] = None,
    knn_radius_mm: Optional[float] = 8,
    edge_prob_thresh: Optional[float] = 0.5,
    cfg_overrides: Optional[dict] = None,
    use_adaptive_spacing: bool = True,
    return_debug: bool = False
) -> nx.Graph | tuple[nx.Graph, dict]:
    model, base_cfg, device, _ = model_tuple
    local_cfg = copy.deepcopy(base_cfg)
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            if hasattr(local_cfg, k):
                setattr(local_cfg, k, v)
    if knn_k is not None:            local_cfg.knn_k = int(knn_k)
    if knn_radius_mm is not None:    local_cfg.knn_radius_mm = float(knn_radius_mm)
    if edge_prob_thresh is not None: local_cfg.edge_prob_thresh = float(edge_prob_thresh)

    ref_img = prob_img if prob_img is not None else seg_img
    prob_np = sitk_to_np(ref_img).astype(np.float32)
    if prob_np.max() > 1.5:
        prob_np = (prob_np > 0.5).astype(np.float32)

    if use_adaptive_spacing and (node_min_spacing_mm is not None):
        sp_zyx = get_spacing_zyx(ref_img)
        step_zyx = tuple(int(max(1, np.ceil(float(node_min_spacing_mm) / float(s)))) for s in sp_zyx)
        local_cfg.voxel_subsample_zyx = step_zyx

    coords, in_pred, rad_mm, pos_mm, tang = build_nodes(prob_np, ref_img, local_cfg)
    G = nx.Graph()
    if coords.size < 2:
        return (G, {"reason": "no_nodes"}) if return_debug else G

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

    cand_ea = edge_features(pos_mm, tang, cand_ei, rad_mm, use_integrals=False)
    uv = cand_ei.T
    key = { (int(a),int(b)): i for i,(a,b) in enumerate(uv) }
    gmask = np.zeros(gnn_ei.shape[1], dtype=int)
    for i,(a,b) in enumerate(gnn_ei.T):
        gmask[i] = key.get((int(a),int(b)), key.get((int(b),int(a))))
    gnn_ea = cand_ea[gmask]

    x_np = _node_features(prob_np, coords, rad_mm, pos_mm)

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

    ei_pred = assemble_edges_longpaths(
        pos_mm, ei_keep, probs_keep,
        lambda_len_inv=float(local_cfg.mst_lambda_len_inv),
        add_back_thresh=float(local_cfg.add_back_thresh)
    )

    prob_map = {tuple(sorted(map(int, e))): float(p) for e, p in zip(ei_keep.T, probs_keep)}

    for i, (z, y, x) in enumerate(coords):
        pxyz = ref_img.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
        r_mm = float(rad_mm[i])
        G.add_node(
            int(i),
            pos=np.array([int(z), int(y), int(x)], dtype=int),
            pos_phys=np.array(pxyz, dtype=float),
            tangent=tang[i],
            radius_mm=r_mm
        )

    for u, v in ei_pred.T:
        u = int(u); v = int(v)
        p1 = G.nodes[u]['pos_phys']; p2 = G.nodes[v]['pos_phys']
        ru = float(G.nodes[u]['radius_mm']); rv = float(G.nodes[v]['radius_mm'])
        r_min = min(ru, rv); r_mean = 0.5 * (ru + rv)
        G.add_edge(
            u, v,
            weight=float(np.linalg.norm(p1 - p2)),
            edge_prob=prob_map.get(tuple(sorted((u, v))), 0.5),
            radius_min_mm=r_min,
            radius_mean_mm=r_mean
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
