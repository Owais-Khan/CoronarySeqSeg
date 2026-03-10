from typing import List, Tuple, Optional, Dict, Any
from contextlib import nullcontext
from collections import defaultdict
import glob
import os
import copy

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from torch_geometric.nn import SplineConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import (
    distance_transform_edt,
    uniform_filter,
    gaussian_laplace,
    generate_binary_structure,
)
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
import networkx as nx
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import SplineConv
from torch_geometric.utils import (
    to_undirected,
    coalesce,
    remove_self_loops,
    get_laplacian,
)
from torch_cluster import knn_graph
from typing import Optional

try:
    import vtk
    _HAVE_VTK = True
except Exception:
    _HAVE_VTK = False


#helpers --------------------------------------------------------------------------------------------
def _cfg_get(cfg: dict, key: str, default=None):
    return cfg[key] if (cfg is not None and key in cfg and cfg[key] is not None) else default

def _to_torch(x: np.ndarray,
              device: Optional[torch.device] = None,
              dtype: Optional[torch.dtype] = None) -> torch.Tensor:

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not x.flags['C_CONTIGUOUS']:
        x = np.ascontiguousarray(x)

    t = torch.as_tensor(x)
    if dtype is not None or t.device != device:
        t = t.to(device=device, dtype=dtype)
    return t

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
            raise MemoryError("memory error")
        return sitk.ReadImage(path)
    except Exception:
        img = nib.load(path)
        arr = img.get_fdata(dtype=np.float32)
        sitk_img = sitk.GetImageFromArray(np.ascontiguousarray(arr.astype(np.float32)))
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        sitk_img.SetSpacing((zooms[0], zooms[1], zooms[2]))
        sitk_img.SetOrigin((0.0, 0.0, 0.0))
        sitk_img.SetDirection((1.0, 0.0, 0.0,
                               0.0, 1.0, 0.0,
                               0.0, 0.0, 1.0))
        return sitk_img

def sitk_to_np(img: sitk.Image) -> np.ndarray:
    return np.ascontiguousarray(sitk.GetArrayFromImage(img))

def get_spacing_zyx(img: sitk.Image) -> np.ndarray:                                 #image spacing
    sp = np.array(img.GetSpacing(), dtype=np.float32)[::-1]
    return np.ascontiguousarray(sp)

def _standardize(arr: np.ndarray) -> np.ndarray:
    m = arr.mean(axis=0, keepdims=True)
    s = arr.std(axis=0, keepdims=True) + 1e-6
    return (arr - m) / s


def _normalize_pos(pos_mm: np.ndarray) -> np.ndarray:
    if pos_mm.shape[0] == 0:
        return np.zeros_like(pos_mm, dtype=np.float32)

    pos = pos_mm.astype(np.float32)
    center = pos.mean(axis=0, keepdims=True)
    pos_centered = pos - center
    std = pos_centered.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    pos_norm = pos_centered / std
    return np.ascontiguousarray(pos_norm, dtype=np.float32)


def voxel_to_phys(coords_zyx: np.ndarray, img: sitk.Image) -> np.ndarray:           #Transforms z,y,x to x,y,z
    sp = np.asarray(img.GetSpacing(), dtype=np.float64)
    org = np.asarray(img.GetOrigin(), dtype=np.float64)
    D   = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3, 3)

    ijk = coords_zyx[:, ::-1].astype(np.float64)
    ijk_xyz = ijk * sp[None, :]
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


def return_edt_in_and_out(mask_bool: np.ndarray,
                 ref_img: sitk.Image) -> Tuple[np.ndarray, np.ndarray]:
    itk = sitk.GetImageFromArray(mask_bool.astype(np.uint8))
    itk.CopyInformation(ref_img)
    edt_in = sitk.SignedMaurerDistanceMap(
        itk
    )
    edt_out = sitk.SignedMaurerDistanceMap(
        1 - itk
    )
    edt_in_np  = sitk.GetArrayFromImage(edt_in).astype(np.float32)
    edt_out_np = sitk.GetArrayFromImage(edt_out).astype(np.float32)
    return edt_in_np, edt_out_np

def ensure_inputs(images_dir: str, labels_dir: str, preds_dir: str,
                   pattern: str) -> List[Tuple[str, str, str]]:
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
    out, miss = [], []
    for k, (im, gt, pr) in idx.items():
        if not (im and gt and pr):
            miss.append((k, "missing file")); continue
        if not (_nifti_is_readable(gt) and _nifti_is_readable(pr)):
            miss.append((k, "unreadable")); continue
        out.append((im, gt, pr))
    if miss:
        print(f"[ensure_inputs] Skipping {len(miss)} case(s):")
        for k, why in miss[:10]:
            print("  -", k, "->", why)
        if len(miss) > 10:
            print("  ...")
    return out


def _mm_to_step_zyx(target_mm: float, sp_zyx: np.ndarray) -> Tuple[int, int, int]:
    return tuple(int(max(1, round(float(target_mm) / float(s)))) for s in sp_zyx)

def voxel_sample_coords(mask: np.ndarray, step_zyx: Tuple[int, int, int]) -> np.ndarray:
    if not mask.any():
        return np.zeros((0, 3), np.int64)
    Z, Y, X = mask.shape
    sz, sy, sx = [max(1, int(s)) for s in step_zyx]
    zz, yy, xx = np.meshgrid(np.arange(0, Z, sz),
                             np.arange(0, Y, sy),
                             np.arange(0, X, sx), indexing='ij')
    grid = np.stack([zz, yy, xx], -1).reshape(-1, 3)
    keep = mask[grid[:, 0], grid[:, 1], grid[:, 2]]
    return np.ascontiguousarray(grid[keep])

def nms_coords(coords_zyx: np.ndarray, radius_vox=(2, 2, 2)) -> np.ndarray:
    if len(coords_zyx) == 0:
        return coords_zyx
    rz, ry, rx = [max(1, int(r)) for r in radius_vox]
    key = (coords_zyx // np.array([rz, ry, rx], np.int64)).astype(np.int64)
    _, uniq_idx = np.unique(key, axis=0, return_index=True)
    return np.ascontiguousarray(coords_zyx[np.sort(uniq_idx)])


#GNN Architecture ---------------------------------------------------------------------------------
class gnn_model(nn.Module):
    def __init__(
        self,
        in_node: int,
        edge_in: int,
        hidden: int = 96,
        layers: int = 3,
        dropout: float = 0.2,
        kernel_size: int = 7,
        pseudo_cols: Optional[List[int]] = None,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.edge_in = int(edge_in)

        if pseudo_cols is None:
            pseudo_cols = [0, 1, 2, 3]
        self.pseudo_cols = pseudo_cols
        self.pseudo_dim = len(self.pseudo_cols)
        self.kernel_size = int(kernel_size)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        ch_in = int(in_node)
        for _ in range(int(layers)):                                #SplineConv Stack
            conv = SplineConv(
                in_channels=ch_in,
                out_channels=hidden,
                dim=self.pseudo_dim,
                kernel_size=self.kernel_size,
                aggr="mean",
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden))
            ch_in = hidden

        self.node_head = nn.Linear(hidden, 1)          #Node prediction

        self.edge_head = nn.Sequential(                           #Edge classification
            nn.Linear(2 * hidden + self.edge_in, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(hidden, 1),
        )

    def _build_pseudo(self, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_attr.ndim != 2:
            raise RuntimeError("Expected edge_attr of shape [E, D].")
        e = edge_attr[:, self.pseudo_cols].clone()
        e = torch.nan_to_num(e, nan=0.0, posinf=1.0, neginf=0.0)

        col_min = e.amin(dim=0)
        col_max = e.amax(dim=0)
        denom = (col_max - col_min).clamp_min(1e-6)
        pseudo = ((e - col_min) / denom).clamp(0.0, 1.0)
        return pseudo

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ):
        pseudo = self._build_pseudo(edge_attr)

        z = x
        for conv, ln in zip(self.convs, self.norms):
            z_res = z
            z = conv(z, edge_index, pseudo)
            z = ln(z)
            z = F.relu(z)
            if z.shape == z_res.shape:
                z = z + z_res
            z = F.dropout(z, p=self.dropout, training=self.training)

        node_logits = self.node_head(z).squeeze(1)
        u, v = edge_index
        h_u = z[u]
        h_v = z[v]
        edge_feat = torch.cat([h_u, h_v, edge_attr], dim=1)
        edge_logits = self.edge_head(edge_feat).squeeze(1)

        return node_logits, edge_logits


def gap_bridging(
    pos_mm: np.ndarray,
    base_ei: np.ndarray,
    tangents: np.ndarray,
    rad_mm: np.ndarray,
    r_gap_mm: float,
    cos_min: float,
    dr_mm_max: float,
    max_len_mm: float,
) -> np.ndarray:
    if base_ei.shape[1] == 0:
        return base_ei

    N = pos_mm.shape[0]
    if N < 2:
        return base_ei

    G = nx.Graph()
    for (u, v) in base_ei.T:
        u = int(u); v = int(v)
        p1 = pos_mm[u]; p2 = pos_mm[v]
        L = float(np.linalg.norm(p2 - p1))
        G.add_edge(u, v, length=L)

    if G.number_of_edges() == 0:
        return base_ei

    T = nx.minimum_spanning_tree(G)                            #MST
    endpoints = [n for n, d in T.degree() if d == 1]
    if len(endpoints) == 0:
        return base_ei

    endpoints = np.asarray(endpoints, dtype=int)                #identify endpoint
    pts_end = pos_mm[endpoints]

    tree = cKDTree(pts_end)
    idx_pairs = tree.query_pairs(r=float(r_gap_mm))
    if not idx_pairs:
        return base_ei

    base_set = set((int(min(u, v)), int(max(u, v))) for u, v in base_ei.T)
    cand_edges = []

    for i, j in idx_pairs:
        u = int(endpoints[i])
        v = int(endpoints[j])
        if u == v:
            continue

        p_u = pos_mm[u]
        p_v = pos_mm[v]
        d = p_v - p_u
        L = float(np.linalg.norm(d))
        if L <= 1e-8:
            continue
        if (max_len_mm > 0 and L > float(max_len_mm)) or (L > float(r_gap_mm) * 1.5):
            continue
        if abs(float(rad_mm[u]) - float(rad_mm[v])) > float(dr_mm_max):
            continue

        dv = d / L
        cu = float(np.dot(dv, tangents[u]))
        cv = float(np.dot(-dv, tangents[v]))
        cmean = 0.5 * (cu + cv)
        if cmean < float(cos_min):
            continue

        key = (min(u, v), max(u, v))
        if key in base_set:
            continue

        cand_edges.append((u, v))
        base_set.add(key)

    if not cand_edges:
        return base_ei

    gap_ei = np.array(cand_edges, np.int64).T
    ei_all = np.concatenate([base_ei, gap_ei], axis=1)
    ei_all = np.ascontiguousarray(np.unique(ei_all, axis=1))

    return ei_all


def prob_ridge_log(prob: np.ndarray, floor: float = 0.05) -> np.ndarray:
    p = np.clip(prob.astype(np.float32), 0.0, 1.0)
    resp = -gaussian_laplace(p, sigma=1.0).astype(np.float32)  # ridges positive
    resp[p < floor] = 0.0
    m = resp.max() + 1e-6
    return np.ascontiguousarray((resp / m).astype(np.float32))


def edge_features(
    pos_mm: np.ndarray,
    tangents: np.ndarray,
    edge_index: np.ndarray,
    rad_mm: Optional[np.ndarray] = None,
    is_skel: Optional[np.ndarray] = None,
    degree: Optional[np.ndarray] = None,
) -> np.ndarray:
    if edge_index.shape[1] == 0:
        return np.zeros((0, 8), np.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    P  = _to_torch(pos_mm.astype(np.float32), device=device)
    T  = _to_torch(tangents.astype(np.float32), device=device)
    R  = _to_torch(rad_mm.astype(np.float32), device=device) if rad_mm is not None else None
    ei = _to_torch(edge_index.astype(np.int64), device=device)

    u, v = ei

    d    = P[v] - P[u]
    dist = torch.linalg.norm(d, dim=1, keepdim=True)

    cosu = (d * T[u]).sum(-1, keepdim=True) / (dist + 1e-8)
    cosv = ((-d) * T[v]).sum(-1, keepdim=True) / (dist + 1e-8)
    cos  = 0.5 * (cosu + cosv)

    if R is not None:
        rad_u = R[u]
        rad_v = R[v]
        dr = (rad_u - rad_v).abs().unsqueeze(1)
        rad_mean = (0.5 * (rad_u + rad_v)).unsqueeze(1)
    else:
        dr = torch.zeros_like(dist)
        rad_mean = torch.zeros_like(dist)

    feat_list = [dist, cos, dr, rad_mean]

    if is_skel is not None:
        S = _to_torch(is_skel.astype(np.float32), device=device)
        skel_u = S[u].unsqueeze(1)
        skel_v = S[v].unsqueeze(1)
        skel_pair = skel_u * skel_v
        skel_any  = ((skel_u + skel_v) > 0).float()
        feat_list += [skel_pair, skel_any]

    if degree is not None:
        D = _to_torch(degree.astype(np.float32), device=device)
        deg_u = D[u].unsqueeze(1)
        deg_v = D[v].unsqueeze(1)

        is_end_u = (deg_u == 1).float()
        is_end_v = (deg_v == 1).float()
        end_pair = is_end_u * is_end_v

        any_junc = ((deg_u >= 3) | (deg_v >= 3)).float()
        feat_list += [end_pair, any_junc]

    while len(feat_list) < 8:
        feat_list.append(torch.zeros_like(dist))

    base = torch.cat(feat_list, dim=1).contiguous()
    return base.detach().cpu().numpy().astype(np.float32)


def build_nodes_from_seg(
    seg_np: np.ndarray,
    ref_img: sitk.Image,
    cfg: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    def _empty():
        return (
            np.zeros((0, 3), np.int64),
            np.zeros((0,), np.float32),
            np.zeros((0, 3), np.float32),
            np.zeros((0, 3), np.float32),
            np.zeros((0,), np.bool_),
        )

    seg_np = seg_np.astype(np.float32)
    prob_thr = float(_cfg_get(cfg, "prob_threshold", 0.5))
    seg_bin = seg_np > prob_thr

    if not seg_bin.any():
        return _empty()

    edt_inside_mm, _ = return_edt_in_and_out(seg_bin.astype(bool), ref_img)
    sp_zyx = get_spacing_zyx(ref_img)

    step_mm = float(_cfg_get(cfg, "node_step_mm", 1.5))
    step_zyx = _mm_to_step_zyx(step_mm, sp_zyx)

    skel_step_mm = float(_cfg_get(cfg, "skel_step_mm", step_mm))
    skel_step_zyx = _mm_to_step_zyx(skel_step_mm, sp_zyx)

    coords_core = voxel_sample_coords(seg_bin, step_zyx)
    if coords_core.size == 0:
        coords_core = np.zeros((0, 3), dtype=np.int64)
    else:
        coords_core = coords_core.astype(np.int64)

    coords_skel = np.zeros((0, 3), dtype=np.int64)
    if bool(_cfg_get(cfg, "use_skel_nodes", True)):
        skel = skeletonize(seg_bin.astype(np.uint8))
        if skel.any():
            coords_skel = voxel_sample_coords(skel > 0, skel_step_zyx)
            if coords_skel.size == 0:
                coords_skel = np.zeros((0, 3), dtype=np.int64)
            else:
                coords_skel = coords_skel.astype(np.int64)

    if coords_core.size == 0 and coords_skel.size == 0:
        return _empty()

    coords = np.vstack([coords_core, coords_skel])
    is_skel = np.zeros(coords.shape[0], dtype=bool)
    is_skel[coords_core.shape[0]:] = True

    coords_u, inv = np.unique(coords, axis=0, return_inverse=True)
    is_skel_u = np.zeros(coords_u.shape[0], dtype=bool)
    np.logical_or.at(is_skel_u, inv, is_skel)
    coords = coords_u
    is_skel = is_skel_u

    max_nodes = int(_cfg_get(cfg, "max_nodes", 150000))
    if coords.shape[0] > max_nodes:
        skel_idx = np.nonzero(is_skel)[0]
        core_idx = np.nonzero(~is_skel)[0]

        if skel_idx.size >= max_nodes:
            sel_skel = np.random.choice(skel_idx, max_nodes, replace=False)
            keep_idx = np.sort(sel_skel)
        else:
            n_core_needed = max_nodes - skel_idx.size
            sel_core = core_idx[:n_core_needed]
            keep_idx = np.sort(np.concatenate([skel_idx, sel_core]))

        coords = coords[keep_idx]
        is_skel = is_skel[keep_idx]

    rad_mm = edt_inside_mm[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)
    pos_mm = voxel_to_phys(coords, ref_img).astype(np.float32)
    tangents = tangents_from_edt(edt_inside_mm, coords).astype(np.float32)

    r_min_abs = float(_cfg_get(cfg, "node_min_radius_mm", 0.3))
    r_max_abs = float(_cfg_get(cfg, "node_max_radius_mm", 1e6))

    q = _cfg_get(cfg, "node_radius_quantile", None)
    if q is not None:
        q = float(q)
        if 0.0 < q < 1.0 and np.any(rad_mm > 0):
            r_q = float(np.quantile(rad_mm[rad_mm > 0], q))
            r_thr_min = max(r_min_abs, r_q)
        else:
            r_thr_min = r_min_abs
    else:
        r_thr_min = r_min_abs

    keep = np.ones_like(rad_mm, dtype=bool)
    core_mask = ~is_skel
    keep[core_mask] = (rad_mm[core_mask] >= r_thr_min) & (rad_mm[core_mask] <= r_max_abs)

    if keep.sum() == 0:
        keep = np.ones_like(rad_mm, dtype=bool)

    coords   = coords[keep]
    rad_mm   = rad_mm[keep]
    pos_mm   = pos_mm[keep]
    tangents = tangents[keep]
    is_skel  = is_skel[keep]

    if coords.shape[0] < 2:
        return _empty()

    return (
        np.ascontiguousarray(coords, dtype=np.int64),
        np.ascontiguousarray(rad_mm, dtype=np.float32),
        np.ascontiguousarray(pos_mm, dtype=np.float32),
        np.ascontiguousarray(tangents, dtype=np.float32),
        np.ascontiguousarray(is_skel, dtype=np.bool_),
    )


def build_edges(
    pos_mm: np.ndarray,
    tangents: np.ndarray,
    rad_mm: np.ndarray,
    cfg: dict,
    is_skel: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    N = pos_mm.shape[0]
    if N < 2:
        return (
            np.zeros((2, 0), np.int64),
            np.zeros((0, 8), np.float32),
            np.zeros((N,), np.int64),
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    P = _to_torch(pos_mm.astype(np.float32), device=device)

    max_L = float(_cfg_get(cfg, "max_edge_len_mm", 12.0))

    ei = knn_graph(P, k=int(_cfg_get(cfg, "knn_k", 24)))
    ei, _ = remove_self_loops(ei)
    ei = to_undirected(ei)
    ei = coalesce(ei)

    if max_L > 0:
        u, v = ei
        d = torch.linalg.norm(P[v] - P[u], dim=1)
        keep = d <= max_L
        ei = ei[:, keep]

    ei_np = ei.detach().cpu().numpy().astype(np.int64)

    if bool(_cfg_get(cfg, "use_gap_edges", True)):
        ei_np = gap_bridging(
            pos_mm=pos_mm,
            base_ei=ei_np,
            tangents=tangents,
            rad_mm=rad_mm,
            r_gap_mm=float(_cfg_get(cfg, "gap_radius_mm", 8.0)),
            cos_min=float(_cfg_get(cfg, "gap_cos_min", 0.3)),
            dr_mm_max=float(_cfg_get(cfg, "gap_dr_mm_max", 1.5)),
            max_len_mm=float(_cfg_get(cfg, "gap_max_len_mm", 20.0)),
        )

    if ei_np.shape[1] > 0:
        deg = np.bincount(ei_np.reshape(-1), minlength=N).astype(np.int64)
    else:
        deg = np.zeros((N,), np.int64)

    ea = edge_features(pos_mm, tangents, ei_np, rad_mm, is_skel, deg)
    return ei_np, ea, deg



def _node_features(
    prob_np: np.ndarray,
    coords: np.ndarray,
    rad_mm: np.ndarray,
    pos_mm: np.ndarray,
    tangents: np.ndarray,
    edt_inside_mm: np.ndarray,
    ridge_np: np.ndarray,
    is_skel: np.ndarray,
    degree: Optional[np.ndarray] = None,
) -> np.ndarray:


    prob_vals  = prob_np[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)
    ridge_vals = ridge_np[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)

    edt_radius_mm = rad_mm.astype(np.float32)

    node_type = is_skel.astype(np.float32).reshape(-1, 1)

    if degree is None:
        deg = np.ones(coords.shape[0], dtype=np.float32)
    else:
        deg = degree.astype(np.float32)
    deg_norm = (deg / (deg.max() + 1e-6)).reshape(-1, 1)

    scalars = np.stack([prob_vals, edt_radius_mm, ridge_vals], axis=1).astype(np.float32)
    scalars = np.concatenate([scalars, node_type, deg_norm], axis=1)
    scalars = _standardize(scalars)

    tang = tangents.astype(np.float32)
    pos_norm = _normalize_pos(pos_mm)

    x = np.concatenate([scalars, tang, pos_norm], axis=1)
    return np.ascontiguousarray(x, dtype=np.float32)


def build_gt_tree_from_volume(
    gt_img: sitk.Image,
    gt_np: np.ndarray,
    cfg: dict
) -> Tuple[np.ndarray, nx.Graph]:
    gt_bin = gt_np > 0.5
    if not gt_bin.any():
        return np.zeros((0, 3), np.float32), nx.Graph()

    skel = skeletonize(gt_bin.astype(np.uint8))
    coords = np.argwhere(skel > 0)
    if coords.shape[0] == 0:
        return np.zeros((0, 3), np.float32), nx.Graph()

    edt_inside_mm, _ = return_edt_in_and_out(gt_bin.astype(bool), gt_img)
    rad_mm = edt_inside_mm[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)

    pos_mm = voxel_to_phys(coords, gt_img).astype(np.float32)
    N = pos_mm.shape[0]

    max_edge_len_mm = float(_cfg_get(cfg, "gt_max_edge_len_mm", 2.5))
    knn_k          = int(_cfg_get(cfg, "gt_knn_k", 8))

    tree = cKDTree(pos_mm)
    k = min(knn_k + 1, N)

    dists, idxs = tree.query(pos_mm, k=k)

    G = nx.Graph()
    for i in range(N):
        G.add_node(
            i,
            pos_phys=pos_mm[i],
            radius_mm=float(rad_mm[i]),
        )

    for i in range(N):
        for j, d in zip(idxs[i][1:], dists[i][1:]):
            if d <= 0.0:
                continue
            if d > max_edge_len_mm:
                continue
            u, v = int(i), int(j)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if G.has_edge(u, v):
                continue
            G.add_edge(u, v, length=float(d), edge_prob=1.0)

    if G.number_of_edges() == 0:
        return pos_mm, G

    if bool(_cfg_get(cfg, "gt_enforce_tree", True)):
        H = nx.Graph()
        H.add_nodes_from(G.nodes(data=True))
        for comp in nx.connected_components(G):
            sub = G.subgraph(comp).copy()
            if sub.number_of_edges() == 0:
                continue
            T = nx.minimum_spanning_tree(sub, weight="length")
            H.add_edges_from(T.edges(data=True))
        G = H

    return pos_mm, G


def node_mapping_to_gt(
    pos_mm: np.ndarray,
    gt_pos_mm: np.ndarray,
    gt_graph: nx.Graph,
    cfg: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = pos_mm.shape[0]
    inside_mask = np.zeros((N,), bool)
    gt_idx_map  = -np.ones((N,), np.int64)
    dist_to_gt  = np.full((N,), np.inf, np.float32)

    if N == 0 or gt_pos_mm.shape[0] == 0 or gt_graph.number_of_nodes() == 0:
        return inside_mask, gt_idx_map, dist_to_gt

    R_loose = float(_cfg_get(cfg, "node_label_radius_loose_mm", 1.8))

    tree = cKDTree(gt_pos_mm)
    dist, idx = tree.query(pos_mm, k=1)
    dist = dist.astype(np.float32)

    valid = np.zeros(gt_pos_mm.shape[0], dtype=bool)
    if gt_graph.number_of_nodes() > 0:
        valid_nodes = np.asarray(list(gt_graph.nodes()), dtype=int)
        valid[valid_nodes] = True

    mapped_valid = valid[idx]
    inside = (dist <= R_loose) & mapped_valid

    inside_mask[:] = inside
    dist_to_gt[:]  = dist
    gt_idx_map[inside] = idx[inside].astype(np.int64)

    return inside_mask, gt_idx_map, dist_to_gt


def node_labels_from_gt_tree(
    dist_to_gt_mm: np.ndarray,
    inside_mask: np.ndarray,
    cfg: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    N = dist_to_gt_mm.shape[0]
    dist = dist_to_gt_mm.astype(np.float32)
    inside = inside_mask.astype(bool)

    R_hard = float(_cfg_get(cfg, "node_R_hard_mm", 1))
    R_soft = float(_cfg_get(cfg, "node_R_soft_mm", 5.0))

    w_hard = float(_cfg_get(cfg, "node_w_hard", 1.0))
    w_soft = float(_cfg_get(cfg, "node_w_soft", 0.5))
    w_neg  = float(_cfg_get(cfg, "node_w_neg", 0.3))

    node_y = np.zeros((N,), np.float32)
    node_w = np.full((N,), w_neg, np.float32)

    valid = inside & np.isfinite(dist)

    hard_pos = valid & (dist <= R_hard)
    soft_pos = valid & (dist > R_hard) & (dist <= R_soft)

    node_y[hard_pos] = 1.0
    node_y[soft_pos] = 1.0

    node_w[hard_pos] = w_hard
    node_w[soft_pos] = w_soft

    return (
        np.ascontiguousarray(node_y, dtype=np.float32),
        np.ascontiguousarray(node_w, dtype=np.float32),
    )


def edge_labels_from_gt_tree(
    edge_index: np.ndarray,
    pos_mm: np.ndarray,
    inside_mask: np.ndarray,
    gt_idx_map: np.ndarray,
    dist_to_gt_mm: np.ndarray,
    gt_graph: nx.Graph,
    cfg: dict,
) -> Tuple[np.ndarray, np.ndarray]:

    E = edge_index.shape[1]
    y_edge = np.zeros((E,), np.float32)
    edge_w = np.zeros((E,), np.float32)

    if E == 0 or gt_graph.number_of_nodes() == 0:
        return y_edge, edge_w

    R_pos = float(_cfg_get(cfg, "edge_label_R_pos_mm", 1.0))
    R_max = float(_cfg_get(cfg, "edge_label_R_max_mm", 5.0))
    D_pos_mm = float(_cfg_get(cfg, "edge_gt_D_pos_mm", 10.0))

    dist_to_gt_mm = dist_to_gt_mm.astype(np.float32)
    inside = inside_mask.astype(bool)
    N = dist_to_gt_mm.shape[0]

    zone = np.zeros((N,), np.int64)

    close_mask = inside & (dist_to_gt_mm <= R_pos)
    near_mask  = inside & (dist_to_gt_mm > R_pos) & (dist_to_gt_mm <= R_max)
    far_mask   = dist_to_gt_mm > R_max

    zone[close_mask] = 2
    zone[near_mask]  = 1
    zone[far_mask]   = 0

    w_pos_core  = float(_cfg_get(cfg, "edge_w_pos_core", 1.0))
    w_neg_core  = float(_cfg_get(cfg, "edge_w_neg_core", 1.0))
    w_neg_near  = float(_cfg_get(cfg, "edge_w_neg_near", 0.5))
    w_neg_bg    = float(_cfg_get(cfg, "edge_w_neg_bg", 0.2))

    u = edge_index[0].astype(np.int64)
    v = edge_index[1].astype(np.int64)

    dist_cache: Dict[int, Dict[int, float]] = {}

    def get_dist_map(src_gt: int) -> Dict[int, float]:
        if src_gt in dist_cache:
            return dist_cache[src_gt]
        dmap = nx.single_source_dijkstra_path_length(
            gt_graph, src_gt, weight="length"
        )
        dist_cache[src_gt] = dmap
        return dmap

    BIG = 1e8

    for k in range(E):
        uu = int(u[k])
        vv = int(v[k])

        zu = int(zone[uu])
        zv = int(zone[vv])

        y = 0.0
        w = w_neg_bg

        if zu == 0 or zv == 0:
            y_edge[k] = y
            edge_w[k] = w
            continue

        if zu == 2 and zv == 2:
            gu = int(gt_idx_map[uu])
            gv = int(gt_idx_map[vv])

            if gu >= 0 and gv >= 0:
                dmap = get_dist_map(gu)
                d_gt = float(dmap.get(gv, BIG))

                if d_gt < BIG and d_gt <= D_pos_mm:
                    y = 1.0
                    w = w_pos_core
                else:
                    y = 0.0
                    w = w_neg_core
            else:
                y = 0.0
                w = w_neg_near

        elif (zu == 1 or zv == 1):
            y = 0.0
            w = w_neg_near

        y_edge[k] = y
        edge_w[k] = w

    return (
        np.ascontiguousarray(y_edge, dtype=np.float32),
        np.ascontiguousarray(edge_w, dtype=np.float32),
    )


def compute_centerlines_from_graph(
    G_in: nx.Graph,
    cfg: dict,
    prune_tiny_branches: bool = True,
) -> Tuple[nx.Graph, List[np.ndarray]]:

    if G_in.number_of_nodes() == 0 or G_in.number_of_edges() == 0:
        return G_in.copy(), []

    G = G_in.copy()

    p_min = float(_cfg_get(cfg, "cl_edge_prob_min", 0.35))
    L_max = float(_cfg_get(cfg, "cl_edge_len_max_mm", 30.0))

    core_edges = []
    for u, v, d in G.edges(data=True):
        p = float(d.get("edge_prob", 1.0))
        L = float(d.get("length", 0.0))
        if p >= p_min and (L == 0.0 or L <= L_max):
            core_edges.append((u, v))

    if not core_edges:
        core_edges = list(G.edges())

    G_core = G.edge_subgraph(core_edges).copy()
    for n in G.nodes():
        if n not in G_core:
            G_core.add_node(n, **G.nodes[n])

    eps = 1e-6
    G_cl = nx.Graph()
    G_cl.add_nodes_from(G_core.nodes(data=True))

    for comp in nx.connected_components(G_core):
        sub = G_core.subgraph(comp).copy()
        if sub.number_of_edges() == 0:
            continue

        for u, v, d in sub.edges(data=True):
            p = float(d.get("edge_prob", 1.0))
            L = float(d.get("length", 0.0))
            if L <= 0.0:
                p1 = np.asarray(sub.nodes[u]["pos_phys"], dtype=np.float32)
                p2 = np.asarray(sub.nodes[v]["pos_phys"], dtype=np.float32)
                L = float(np.linalg.norm(p2 - p1))
            r_u = float(sub.nodes[u].get("radius_mm", 0.5))
            r_v = float(sub.nodes[v].get("radius_mm", 0.5))
            r = 0.5 * (r_u + r_v)
            score = p * (r + eps) / (L + eps)
            d["score"] = score
            d.setdefault("length", L)

        T = sub
        for u, v, d in T.edges(data=True):
            G_cl.add_edge(u, v, **d)

    if G_cl.number_of_edges() == 0:
        G_cl = G_core

    if prune_tiny_branches:
        min_branch_len_mm = float(_cfg_get(cfg, "cl_min_branch_len_mm", 3.0))

        protected_nodes = set(nx.articulation_points(G_cl))
        protected_edges = set()
        for u, v in nx.bridges(G_cl):
            a, b = (u, v) if u <= v else (v, u)
            protected_edges.add((a, b))

        def _branch_geom_length(path_nodes: List[int]) -> float:
            if len(path_nodes) < 2:
                return 0.0
            acc = 0.0
            for i in range(len(path_nodes) - 1):
                u = path_nodes[i]
                v = path_nodes[i + 1]
                L = float(G_cl[u][v].get("length", 0.0))
                if L <= 0.0:
                    p1 = np.asarray(G_cl.nodes[u]["pos_phys"], dtype=np.float32)
                    p2 = np.asarray(G_cl.nodes[v]["pos_phys"], dtype=np.float32)
                    L = float(np.linalg.norm(p2 - p1))
                acc += L
            return acc

        changed = True
        while changed:
            changed = False
            leaves = [n for n, d in G_cl.degree() if d == 1]
            if not leaves:
                break
            for leaf in leaves:
                if leaf not in G_cl:
                    continue
                if leaf in protected_nodes:
                    continue
                path = [leaf]
                cur = leaf
                prev = None
                while True:
                    nbrs = [x for x in G_cl.neighbors(cur) if x != prev]
                    if not nbrs:
                        break
                    nxt = nbrs[0]
                    edge_key = (min(cur, nxt), max(cur, nxt))
                    if edge_key in protected_edges:
                        path.append(nxt)
                        break
                    path.append(nxt)
                    prev, cur = cur, nxt
                    if G_cl.degree(cur) != 2:
                        break
                Ltwig = _branch_geom_length(path)
                if Ltwig < min_branch_len_mm:
                    keep_last = path[-1]
                    for n in path[:-1]:
                        if n == keep_last:
                            continue
                        if n in protected_nodes:
                            continue
                        if n in G_cl:
                            G_cl.remove_node(n)
                            changed = True

    polylines: List[np.ndarray] = []
    for comp in nx.connected_components(G_cl):
        sub = G_cl.subgraph(comp)
        if sub.number_of_nodes() < 2:
            continue
        deg = dict(sub.degree())
        endpoints = [n for n, d in deg.items() if d != 2]
        if not endpoints:
            endpoints = [next(iter(sub.nodes()))]

        visited_edges = set()

        def _mark(u, v):
            a, b = (u, v) if u <= v else (v, u)
            visited_edges.add((a, b))

        def _seen(u, v):
            a, b = (u, v) if u <= v else (v, u)
            return (a, b) in visited_edges

        for s in endpoints:
            for nb in list(sub.neighbors(s)):
                if _seen(s, nb):
                    continue
                line_nodes = [s, nb]
                _mark(s, nb)
                prev, cur = s, nb
                while deg.get(cur, 0) == 2:
                    nbrs = [x for x in sub.neighbors(cur) if x != prev]
                    if not nbrs:
                        break
                    nxt = nbrs[0]
                    if _seen(cur, nxt):
                        break
                    line_nodes.append(nxt)
                    _mark(cur, nxt)
                    prev, cur = cur, nxt
                if len(line_nodes) >= 2:
                    coords = np.stack(
                        [np.asarray(sub.nodes[n]["pos_phys"], dtype=np.float32)
                         for n in line_nodes],
                        axis=0,
                    )
                    polylines.append(coords)

        for u, v in sub.edges():
            if _seen(u, v):
                continue
            coords = np.stack(
                [
                    np.asarray(sub.nodes[u]["pos_phys"], dtype=np.float32),
                    np.asarray(sub.nodes[v]["pos_phys"], dtype=np.float32),
                ],
                axis=0,
            )
            polylines.append(coords)
            _mark(u, v)

    return G_cl, polylines


def attach_edge_metrics_mm_from_phys(
    G: nx.Graph,
    prob_key: str = "edge_prob",
    pos_phys_key: str = "pos_phys",
    length_key: str = "length_mm",
    cost_key: str = "length_cost",
    prob_exp: float = 1.5,
) -> None:

    eps = 1e-6
    for u, v, d in G.edges(data=True):
        p1 = np.asarray(G.nodes[u][pos_phys_key], dtype=np.float32)
        p2 = np.asarray(G.nodes[v][pos_phys_key], dtype=np.float32)
        L = float(np.linalg.norm(p2 - p1))
        d[length_key] = L
        prob = float(d.get(prob_key, 1.0))
        prob = max(prob, eps)
        d[cost_key] = float(L / (prob ** prob_exp + eps))


def prune_short_or_lowprob_spurs(
    G: nx.Graph,
    *,
    length_key: str = "length_mm",
    prob_key: str = "edge_prob",
    Lspur_min_mm: float = 2.0,
    prob_min: float = 0.15,
) -> None:
    changed = True
    while changed:
        changed = False
        leaves = [n for n, d in G.degree() if d == 1]
        for leaf in leaves:
            if leaf not in G:
                continue
            nbrs = list(G.neighbors(leaf))
            if not nbrs:
                continue
            nb = nbrs[0]
            ed = G.get_edge_data(leaf, nb, default={})
            L = float(ed.get(length_key, 0.0))
            p = float(ed.get(prob_key, 1.0))
            if (L < Lspur_min_mm) or (p < prob_min):
                G.remove_node(leaf)
                changed = True


def _seed_from_core_radius(
    G: nx.Graph,
    radius_key: str = "radius_mm",
) -> Optional[int]:
    if G.number_of_nodes() == 0:
        return None
    best = None
    best_r = -1.0
    for n, d in G.nodes(data=True):
        r = float(d.get(radius_key, 0.0))
        if r > best_r:
            best_r = r
            best = n
    return best


def _endpoints(G: nx.Graph) -> List[int]:
    return [n for n, d in G.degree() if d == 1]


def _dedupe_by_mm(
    G: nx.Graph,
    nodes: List[int],
    *,
    min_sep_mm: float = 5.0,
    pos_phys_key: str = "pos_phys",
) -> List[int]:
    out: List[int] = []
    pts: List[np.ndarray] = []
    min_sep_sq = float(min_sep_mm) ** 2

    for n in nodes:
        p = np.asarray(G.nodes[n][pos_phys_key], dtype=np.float32)
        keep = True
        for q in pts:
            if np.sum((p - q) ** 2) < min_sep_sq:
                keep = False
                break
        if keep:
            out.append(n)
            pts.append(p)
    return out


def select_seed_and_targets_from_features(
        Gc: nx.Graph,
        *,
        max_targets: int = 150,
        prob_exp: float = 1.5,
        Lspur_min_mm: float = 2.0,
        prob_min: float = 0.15,
        min_sep_mm: float = 5.0,
        length_key: str = 'length_mm',
        cost_key: str = 'length_cost',
        pos_phys_key: str = 'pos_phys'
) -> Tuple[Optional[int], List[int]]:
    if Gc.number_of_nodes() == 0:
        return None, []

    H = Gc.copy()
    attach_edge_metrics_mm_from_phys(
        H,
        prob_key='edge_prob',
        pos_phys_key=pos_phys_key,
        cost_key=cost_key,
        length_key=length_key,
        prob_exp=prob_exp
    )
    prune_short_or_lowprob_spurs(
        H,
        length_key=length_key,
        prob_key='edge_prob',
        Lspur_min_mm=Lspur_min_mm,
        prob_min=prob_min
    )

    if H.number_of_nodes() == 0:
        return None, []

    seed = _seed_from_core_radius(H, radius_key='radius_mm')
    if seed is None:
        return None, []

    eps = _endpoints(H)
    dist = nx.single_source_dijkstra_path_length(H, seed, weight=cost_key)
    ranked = sorted(
        (eps if eps else [n for n in H.nodes() if n != seed]),
        key=lambda n: dist.get(n, -np.inf),
        reverse=True
    )
    ranked = _dedupe_by_mm(H, ranked, min_sep_mm=min_sep_mm, pos_phys_key=pos_phys_key)
    if max_targets and max_targets > 0:
        ranked = ranked[:max_targets]
    return seed, ranked


def assemble_two_seed_target_trees(
    G_in: nx.Graph,
    cfg: dict,
    *,
    length_key: str = "length_mm",
    cost_key: str = "length_cost",
    pos_phys_key: str = "pos_phys",
) -> Tuple[nx.Graph, nx.Graph, list, list]:

    if G_in.number_of_nodes() == 0:
        return G_in, nx.Graph(), [], []

    comps = list(nx.connected_components(G_in))
    if len(comps) == 0:
        return G_in, nx.Graph(), [], []

    comps_sorted = sorted(comps, key=len, reverse=True)
    selected_comps = comps_sorted[:2]

    trees_raw: List[nx.Graph] = []
    seed_list = []
    targets_list = []

    for comp in selected_comps:
        sub = G_in.subgraph(comp).copy()
        if sub.number_of_nodes() == 0:
            continue

        for u, v, d in sub.edges(data=True):
            L = float(d.get("length", 0.0))
            if L <= 0.0:
                p1 = np.asarray(sub.nodes[u][pos_phys_key], dtype=np.float32)
                p2 = np.asarray(sub.nodes[v][pos_phys_key], dtype=np.float32)
                L = float(np.linalg.norm(p2 - p1))
            d.setdefault(length_key, L)
            d.setdefault(cost_key, L)
            d.setdefault("edge_prob", float(d.get("edge_prob", 1.0)))

        attach_edge_metrics_mm_from_phys(
            sub,
            prob_key="edge_prob",
            pos_phys_key=pos_phys_key,
            cost_key=cost_key,
            length_key=length_key,
            prob_exp=float(_cfg_get(cfg, "infer_prob_exp", 1.5)),
        )

        seed, targets = select_seed_and_targets_from_features(
            sub,
            max_targets=int(_cfg_get(cfg, "infer_max_targets", 150)),
            prob_exp=float(_cfg_get(cfg, "infer_prob_exp", 1.5)),
            Lspur_min_mm=float(_cfg_get(cfg, "infer_Lspur_min_mm", 6.0)),
            prob_min=float(_cfg_get(cfg, "infer_prob_min", 0.15)),
            min_sep_mm=float(_cfg_get(cfg, "infer_min_sep_mm", 9.0)),
            length_key=length_key,
            cost_key=cost_key,
            pos_phys_key=pos_phys_key,
        )
        seed_list.append(seed)
        targets_list.append(targets)

        if seed is None or not targets:
            trees_raw.append(sub)
            continue

        G_tree = nx.Graph()
        for n, d in sub.nodes(data=True):
            G_tree.add_node(n, **d)

        for t in targets:
            if t not in sub:
                continue
            try:
                path = nx.shortest_path(sub, seed, t, weight=cost_key)
            except nx.NetworkXNoPath:
                continue
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if G_tree.has_edge(u, v):
                    continue
                ed = sub.get_edge_data(u, v, default={})
                G_tree.add_edge(u, v, **ed)

        if G_tree.number_of_nodes() == 0:
            G_tree = sub

        trees_raw.append(G_tree)

    while len(trees_raw) < 2:
        trees_raw.append(nx.Graph())

    G0 = trees_raw[0]
    G1 = trees_raw[1]

    return G0, G1, seed_list, targets_list


def make_graph_case_refine(img_p: str,
                           gt_p: str,
                           pred_p: str,
                           cfg: dict) -> Optional[Data]:

    gt_img   = read_nii_safe(gt_p)
    pred_img = read_nii_safe(pred_p)

    gt_np   = (sitk_to_np(gt_img) > 0.5).astype(np.uint8)
    pred_np = sitk_to_np(pred_img).astype(np.float32)
    if pred_np.max() > 1.5:
        pred_np = (pred_np > 0.5).astype(np.float32)

    gt_pos_mm, gt_graph_raw = build_gt_tree_from_volume(gt_img, gt_np, cfg)
    if gt_graph_raw.number_of_nodes() == 0:
        return None

    gt_graph_core = gt_graph_raw
    if gt_graph_core.number_of_nodes() == 0:
        return None

    # --- nodes from prediction (NO belt nodes) ---
    coords, rad_mm, pos_mm, tangents, is_skel = build_nodes_from_seg(
        seg_np=pred_np,
        ref_img=pred_img,
        cfg=cfg,
    )

    if coords.shape[0] < 2:
        return None

    edge_index, edge_attr, degree = build_edges(pos_mm, tangents, rad_mm, cfg, is_skel=is_skel)
    if edge_index.shape[1] == 0:
        return None

    inside_mask, gt_idx_map, dist_to_gt_mm = node_mapping_to_gt(
        pos_mm,
        gt_pos_mm,
        gt_graph_core,
        cfg,
    )
    if inside_mask.sum() == 0:
        return None

    y_edge, edge_weight = edge_labels_from_gt_tree(
        edge_index,
        pos_mm,
        inside_mask,
        gt_idx_map,
        dist_to_gt_mm,
        gt_graph_core,
        cfg,
    )
    if (y_edge > 0.5).sum() == 0:
        return None

    node_y_np, node_w_np = node_labels_from_gt_tree(
        dist_to_gt_mm=dist_to_gt_mm,
        inside_mask=inside_mask,
        cfg=cfg,
    )
    if (node_y_np > 0.5).sum() == 0:
        return None

    prob_thr = float(_cfg_get(cfg, "prob_threshold", 0.5))
    seg_bin = (pred_np > prob_thr)
    edt_inside_mm, _ = return_edt_in_and_out(seg_bin.astype(bool), pred_img)
    ridge_np = prob_ridge_log(pred_np)

    x_np = _node_features(
        pred_np,
        coords,
        rad_mm,
        pos_mm,
        tangents,
        edt_inside_mm,
        ridge_np,
        is_skel,
        degree,
    )

    data = Data(
        x=torch.from_numpy(x_np).float(),
        edge_index=torch.from_numpy(edge_index).long(),
        edge_attr=torch.from_numpy(edge_attr).float(),
        edge_y=torch.from_numpy(y_edge.astype(np.float32)),
        edge_w=torch.from_numpy(edge_weight.astype(np.float32)),
        node_y=torch.from_numpy(node_y_np.astype(np.float32)),
        node_w=torch.from_numpy(node_w_np.astype(np.float32)),
        pos=torch.from_numpy(pos_mm).float(),
        coords_zyx=torch.from_numpy(coords).long(),
        spacing=torch.from_numpy(get_spacing_zyx(pred_img)).float(),
        dist_to_gt=torch.from_numpy(dist_to_gt_mm.astype(np.float32)),
        meta={"img": img_p, "lab": gt_p, "pred": pred_p},
    )
    return data


def _build_train_items(train_list: List[Tuple[str, str, str]],
                       cfg: dict) -> List[Data]:

    items: List[Data] = []
    print("Building GNN training graphs (GT-centerline tree supervision)...")
    for i, (im, gt, pr) in enumerate(train_list):
        try:
            d = make_graph_case_refine(im, gt, pr, cfg)
            if d is not None:
                items.append(d)
        except Exception as e:
            print(f"[ERR] Graph build failed for {os.path.basename(im)}: {e}")
        print(f"  [{i+1}/{len(train_list)}] built {len(items)} graphs so far.")
    print(f"Built {len(items)} graphs total.")
    return items


def _sample_edges_for_loss(
    edge_logits: torch.Tensor,
    edge_labels: torch.Tensor,
    edge_weights: torch.Tensor,
    neg_per_pos: int = 3,
):
    y = edge_labels

    pos_idx = (y > 0.5).nonzero(as_tuple=True)[0]
    neg_idx = (y <= 0.5).nonzero(as_tuple=True)[0]

    if pos_idx.numel() == 0 or neg_per_pos <= 0:
        return edge_logits, edge_labels, edge_weights

    n_pos = pos_idx.numel()
    n_neg = min(neg_idx.numel(), int(neg_per_pos * n_pos))

    if n_neg > 0:
        perm = torch.randperm(neg_idx.numel(), device=edge_logits.device)[:n_neg]
        sel_idx = torch.cat([pos_idx, neg_idx[perm]], dim=0)
    else:
        sel_idx = pos_idx

    edge_logits_sel = edge_logits[sel_idx]
    edge_labels_sel = edge_labels[sel_idx]
    edge_weights_sel = edge_weights[sel_idx]

    return edge_logits_sel, edge_labels_sel, edge_weights_sel


def train_one_epoch(
    model: nn.Module,
    loader: GeoDataLoader,
    opt: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    cfg: Optional[dict] = None,
) -> float:
    model.train()
    cfg = cfg or {}
    total_loss = 0.0

    amp_flag = bool(_cfg_get(cfg, "amp", True)) and torch.cuda.is_available()
    amp_ctx = torch.amp.autocast("cuda", enabled=amp_flag) if torch.cuda.is_available() else nullcontext()

    lambda_edge = float(_cfg_get(cfg, "lambda_edge", 1.0))
    lambda_node = float(_cfg_get(cfg, "lambda_node", 0.0))
    lambda_lap  = float(_cfg_get(cfg, "lambda_lap", 0.0))

    neg_per_pos = int(_cfg_get(cfg, "neg_per_pos", 3))

    edge_bce_sum = 0.0
    edge_cnt     = 0
    edge_tp = edge_fp = edge_fn = 0

    for data in loader:
        data = data.to(device)
        opt.zero_grad(set_to_none=True)

        with amp_ctx:
            node_logits, edge_logits = model(data.x, data.edge_index, data.edge_attr)

            y_edge_all = data.edge_y.float()
            w_edge_all = data.edge_w.float()

            valid = w_edge_all > 0
            if valid.sum() == 0:
                continue

            edge_logits_v = edge_logits[valid]
            y_edge_v      = y_edge_all[valid]
            w_edge_v      = w_edge_all[valid]

            edge_logits_s, y_edge_s, w_edge_s = _sample_edges_for_loss(
                edge_logits_v,
                y_edge_v,
                w_edge_v,
                neg_per_pos=neg_per_pos,
            )

            with torch.no_grad():
                pos_mask = y_edge_s > 0.5
                n_pos = pos_mask.sum()
                n_neg = (~pos_mask).sum()
                if n_pos > 0:
                    pos_weight = (n_neg.float() / n_pos.float()).clamp(0.5, 10.0)
                else:
                    pos_weight = torch.tensor(1.0, device=device)

            bce_raw = F.binary_cross_entropy_with_logits(
                edge_logits_s,
                y_edge_s,
                reduction="none",
                pos_weight=pos_weight,
            )

            edge_loss_raw = bce_raw * w_edge_s
            denom_edge = w_edge_s.sum().clamp_min(1.0)
            edge_loss = edge_loss_raw.sum() / denom_edge

            node_loss = torch.zeros((), device=device)
            if lambda_node > 0.0 and hasattr(data, "node_y"):
                y_node = data.node_y.float()
                if hasattr(data, "node_w"):
                    w_node = data.node_w.float().clamp_min(0.0)
                else:
                    w_node = torch.ones_like(y_node)

                valid_n = w_node > 0
                if valid_n.any():
                    node_logits_v = node_logits[valid_n]
                    y_node_v      = y_node[valid_n]
                    w_node_v      = w_node[valid_n]

                    node_bce = F.binary_cross_entropy_with_logits(
                        node_logits_v,
                        y_node_v,
                        reduction="none",
                    )
                    node_loss_raw = node_bce * w_node_v
                    denom_node = w_node_v.sum().clamp_min(1.0)
                    node_loss = node_loss_raw.sum() / denom_node

            if lambda_lap > 0.0 and data.edge_index.size(1) > 0:
                L_edge_index, L_edge_weight = get_laplacian(
                    data.edge_index, normalization="sym", num_nodes=data.num_nodes
                )
                L_edge_index = L_edge_index.to(device)
                L_edge_weight = L_edge_weight.to(device)
                diff = node_logits[L_edge_index[0]] - node_logits[L_edge_index[1]]
                lap_smooth = (L_edge_weight * diff.pow(2)).mean()
            else:
                lap_smooth = torch.zeros((), device=device)

            loss = (
                lambda_edge * edge_loss
                + lambda_node * node_loss
                + lambda_lap * lap_smooth
            )

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        total_loss += float(loss.detach().cpu())
        edge_cnt += 1

        with torch.no_grad():
            mask_eval = w_edge_all > 0
            if mask_eval.sum() > 0:
                logits_eval = edge_logits[mask_eval]
                labels_eval = y_edge_all[mask_eval]

                edge_bce_sum += float(
                    F.binary_cross_entropy_with_logits(
                        logits_eval,
                        labels_eval,
                        reduction="mean",
                    ).cpu()
                )

                edge_pred_all = torch.sigmoid(logits_eval)
                edge_true_bin = (labels_eval > 0.5).float()
                edge_pred_bin = (edge_pred_all > 0.5).float()

                edge_tp += ((edge_pred_bin == 1) & (edge_true_bin == 1)).sum().item()
                edge_fp += ((edge_pred_bin == 1) & (edge_true_bin == 0)).sum().item()
                edge_fn += ((edge_pred_bin == 0) & (edge_true_bin == 1)).sum().item()

    avg_edge_bce = edge_bce_sum / max(edge_cnt, 1)
    edge_prec = edge_tp / max(edge_tp + edge_fp, 1)
    edge_rec  = edge_tp / max(edge_tp + edge_fn, 1)

    print(
        f"   train EDGE bce={avg_edge_bce:.4f}  "
        f"precision={edge_prec:.3f} recall={edge_rec:.3f}"
    )
    return total_loss / max(1, edge_cnt)


def _assemble_polylines_from_graph(G: nx.Graph) -> List[List[int]]:
    if G.number_of_edges() == 0:
        return []

    deg = dict(G.degree())
    endpoints = [n for n, d in deg.items() if d != 2]
    if not endpoints:
        endpoints = [next(iter(G.nodes()))]

    visited_edges = set()

    def _mark(u, v):
        a, b = (u, v) if u <= v else (v, u)
        visited_edges.add((a, b))

    def _seen(u, v):
        a, b = (u, v) if u <= v else (v, u)
        return (a, b) in visited_edges

    polylines: List[List[int]] = []

    for s in endpoints:
        for nb in list(G.neighbors(s)):
            if _seen(s, nb):
                continue
            line_nodes = [s, nb]
            _mark(s, nb)
            prev, cur = s, nb
            while deg.get(cur, 0) == 2:
                nbrs = [x for x in G.neighbors(cur) if x != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                if _seen(cur, nxt):
                    break
                line_nodes.append(nxt)
                _mark(cur, nxt)
                prev, cur = cur, nxt
            if len(line_nodes) >= 2:
                polylines.append(line_nodes)

    for u, v in G.edges():
        if _seen(u, v):
            continue
        polylines.append([u, v])
        _mark(u, v)

    return polylines


def export_centerline_graph_to_vtp(G: nx.Graph,
                                   out_path: str):
    if not _HAVE_VTK:
        print(f"[WARN] VTK not available; skipping VTP export: {out_path}")
        return

    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    arr_rad = vtk.vtkDoubleArray(); arr_rad.SetName("RadiusMM")
    arr_prob = vtk.vtkDoubleArray(); arr_prob.SetName("NodeProb")

    idmap: Dict[int, int] = {}
    for nid, data in G.nodes(data=True):
        x, y, z = map(float, data["pos_phys"])
        pid = pts.InsertNextPoint(x, y, z)
        idmap[int(nid)] = pid
        arr_rad.InsertNextValue(float(data.get("radius_mm", 0.5)))
        arr_prob.InsertNextValue(float(data.get("node_prob", 0.5)))

    poly_node_seqs = _assemble_polylines_from_graph(G)

    for seq in poly_node_seqs:
        if len(seq) < 2:
            continue
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(seq))
        for i, nid in enumerate(seq):
            polyline.GetPointIds().SetId(i, idmap[int(nid)])
        lines.InsertNextCell(polyline)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetLines(lines)
    poly.GetPointData().AddArray(arr_rad)
    poly.GetPointData().AddArray(arr_prob)
    poly.GetPointData().SetActiveScalars("RadiusMM")

    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(out_path)
    w.SetInputData(poly)
    w.Write()
    print("Saved centerline VTP:", out_path)


def smooth_centerline_graph(
    G_in: nx.Graph,
    n_iter: int = 10,
    alpha: float = 0.5,
) -> nx.Graph:
    if G_in.number_of_nodes() == 0:
        return G_in

    G = G_in.copy()
    if n_iter <= 0 or alpha <= 0.0:
        return G

    for _ in range(n_iter):
        updates = {}

        for n, data in G.nodes(data=True):
            deg = G.degree[n]
            if deg != 2:
                continue

            nbrs = list(G.neighbors(n))
            if len(nbrs) != 2:
                continue

            p  = np.asarray(data["pos_phys"], dtype=np.float32)
            p0 = np.asarray(G.nodes[nbrs[0]]["pos_phys"], dtype=np.float32)
            p1 = np.asarray(G.nodes[nbrs[1]]["pos_phys"], dtype=np.float32)

            p_avg = 0.5 * (p0 + p1)
            p_new = (1.0 - alpha) * p + alpha * p_avg
            updates[n] = p_new.astype(np.float32)

        for n, p_new in updates.items():
            G.nodes[n]["pos_phys"] = p_new

    return G


@torch.no_grad()
def gm_predict_graph(
    *,
    seg_img: sitk.Image,
    prob_img: Optional[sitk.Image],
    model: nn.Module,
    node_min_spacing_mm: Optional[float] = None,
    knn_radius_mm: Optional[float] = None,
    edge_prob_thresh: Optional[float] = 0.5,
    cfg: Optional[dict] = None,
    device: Optional[torch.device] = None,
    use_adaptive_spacing: bool = True,
    return_debug: bool = False,
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    local_cfg = copy.deepcopy(cfg or {})

    if use_adaptive_spacing and (node_min_spacing_mm is not None):
        local_cfg["node_step_mm"] = float(node_min_spacing_mm)

    if knn_radius_mm is not None:
        local_cfg["radius_graph_mm"] = float(knn_radius_mm)

    ref_img = prob_img if prob_img is not None else seg_img
    prob_np = sitk_to_np(ref_img).astype(np.float32)
    if prob_np.max() > 1.5:
        prob_np = (prob_np > 0.5).astype(np.float32)

    prob_thr = float(_cfg_get(local_cfg, "prob_threshold", 0.5))
    seg_bin = prob_np > prob_thr

    G_empty0 = nx.Graph()
    G_empty1 = nx.Graph()
    if not seg_bin.any():
        dbg = {"reason": "empty_seg"}
        if return_debug:
            return G_empty0, G_empty1, [], [], dbg
        return G_empty0, G_empty1, [], []

    edt_inside_mm, _ = return_edt_in_and_out(seg_bin.astype(bool), ref_img)
    ridge_np = prob_ridge_log(prob_np)

    coords, rad_mm, pos_mm, tangents, is_skel = build_nodes_from_seg(prob_np, ref_img, local_cfg)

    if coords.shape[0] < 2:
        dbg = {"reason": "no_nodes", "N": int(coords.shape[0])}
        if return_debug:
            return G_empty0, G_empty1, [], [], dbg
        return G_empty0, G_empty1, [], []

    edge_index, edge_attr, degree = build_edges(pos_mm, tangents, rad_mm, local_cfg, is_skel=is_skel)
    if edge_index.shape[1] == 0:
        dbg = {"reason": "no_edges", "N": int(coords.shape[0])}
        if return_debug:
            return G_empty0, G_empty1, [], [], dbg
        return G_empty0, G_empty1, [], []

    x_np = _node_features(
        prob_np,
        coords,
        rad_mm,
        pos_mm,
        tangents,
        edt_inside_mm,
        ridge_np,
        is_skel,
        degree,
    )

    x_t  = torch.from_numpy(x_np).to(device)
    ei_t = torch.from_numpy(edge_index).to(device)
    ea_t = torch.from_numpy(edge_attr).to(device)

    amp_flag = bool(_cfg_get(local_cfg, "amp", True)) and torch.cuda.is_available()
    amp_ctx = torch.amp.autocast('cuda', enabled=amp_flag) if torch.cuda.is_available() else nullcontext()

    with amp_ctx:
        node_logits, edge_logits = model(x_t, ei_t, ea_t)
        edge_probs = torch.sigmoid(edge_logits).cpu().numpy()

    edge_thr = float(_cfg_get(local_cfg, "edge_prob_thresh", edge_prob_thresh))

    keep_edges = edge_probs >= edge_thr
    ei_keep = edge_index[:, keep_edges]
    if ei_keep.shape[1] == 0:
        dbg = {"reason": "no_edges_after_prune",
               "edge_thr": edge_thr,
               "N_edges": int(edge_index.shape[1])}
        if return_debug:
            return G_empty0, G_empty1, [], [], dbg
        return G_empty0, G_empty1, [], []

    idx_map = np.full(coords.shape[0], -1, int)
    keep_idx = np.unique(ei_keep.reshape(-1))
    idx_map[keep_idx] = np.arange(keep_idx.size, dtype=int)
    ei_remap = np.vstack([idx_map[ei_keep[0]], idx_map[ei_keep[1]]])

    edge_probs_keep = edge_probs[keep_edges]

    G_pruned = nx.Graph()
    for new_id, old_id in enumerate(keep_idx):
        G_pruned.add_node(
            int(new_id),
            pos_phys=pos_mm[old_id].astype(np.float32),
            radius_mm=float(rad_mm[old_id]),
            node_prob=0.5,
            tangent=tangents[old_id].astype(np.float32),
        )

    for (u, v), p in zip(ei_remap.T, edge_probs_keep):
        u = int(u); v = int(v)
        p1 = G_pruned.nodes[u]['pos_phys']
        p2 = G_pruned.nodes[v]['pos_phys']
        L = float(np.linalg.norm(p2 - p1))
        G_pruned.add_edge(
            u, v,
            length=L,
            edge_prob=float(p),
        )

    if G_pruned.number_of_nodes() == 0:
        dbg = {"reason": "empty_pruned_graph"}
        if return_debug:
            return G_empty0, G_empty1, [], [], dbg
        return G_empty0, G_empty1, [], []

    G_cl, polylines = compute_centerlines_from_graph(
        G_pruned,
        local_cfg,
        prune_tiny_branches=False,
    )

    if G_cl.number_of_nodes() == 0:
        dbg = {"reason": "empty_centerline_graph_after_assembly"}
        if return_debug:
            return G_empty0, G_empty1, [], [], dbg
        return G_empty0, G_empty1, [], []

    G0, G1, seed_list, targets_list = assemble_two_seed_target_trees(
        G_cl,
        local_cfg,
        length_key="length_mm",
        cost_key="length_cost",
        pos_phys_key="pos_phys",
    )

    dbg = {
        "reason": "ok",
        "N_raw_nodes": int(coords.shape[0]),
        "N_pruned_nodes": int(G_pruned.number_of_nodes()),
        "N_cl_nodes": int(G_cl.number_of_nodes()),
        "N_G0_nodes": int(G0.number_of_nodes()) if G0 is not None else 0,
        "N_G1_nodes": int(G1.number_of_nodes()) if G1 is not None else 0,
    }

    if return_debug:
        return G0, G1, seed_list, targets_list, dbg
    return G0, G1, seed_list, targets_list


def process_case_infer(img_p: str,
                       pred_p: str,
                       model: nn.Module,
                       cfg: dict,
                       device: torch.device,
                       out_dir: str):
    name = os.path.basename(pred_p).replace(".nii.gz", "").replace("_0000", "")
    pred_img = read_nii_safe(pred_p)

    G0, G1, seed_list, targets_list, dbg = gm_predict_graph(
        seg_img=pred_img,
        prob_img=pred_img,
        model=model,
        cfg=cfg,
        device=device,
        node_min_spacing_mm=_cfg_get(cfg, "node_step_mm", 1.5),
        knn_radius_mm=_cfg_get(cfg, "radius_graph_mm", 8.0),
        edge_prob_thresh=_cfg_get(cfg, "edge_prob_thresh", 0.5),
        use_adaptive_spacing=True,
        return_debug=True,
    )

    print(f"[infer] {name}: G0 nodes={G0.number_of_nodes()} edges={G0.number_of_edges()}")
    print(f"[infer] {name}: G1 nodes={G1.number_of_nodes()} edges={G1.number_of_edges()}")
    print(f"       debug: {dbg}")

    out_dir = _cfg_get(cfg, "out_dir", os.path.join(os.path.dirname(pred_p), "graph_out"))
    os.makedirs(out_dir, exist_ok=True)

    if bool(_cfg_get(cfg, "export_predicted_vtp", True)) and G0.number_of_nodes() > 1:
        if bool(_cfg_get(cfg, "smooth_centerline_for_vtp", True)):
            n_iter  = int(_cfg_get(cfg, "smooth_centerline_iters", 8))
            alpha   = float(_cfg_get(cfg, "smooth_centerline_alpha", 0.5))
            G_vis0 = smooth_centerline_graph(G0, n_iter=n_iter, alpha=alpha)
        else:
            G_vis0 = G0
        out_vtp = os.path.join(out_dir, f"{name}_predicted_comp0.vtp")
        export_centerline_graph_to_vtp(G_vis0, out_vtp)

    if bool(_cfg_get(cfg, "export_predicted_vtp", True)) and G1.number_of_nodes() > 1:
        if bool(_cfg_get(cfg, "smooth_centerline_for_vtp", True)):
            n_iter  = int(_cfg_get(cfg, "smooth_centerline_iters", 8))
            alpha   = float(_cfg_get(cfg, "smooth_centerline_alpha", 0.5))
            G_vis1 = smooth_centerline_graph(G1, n_iter=n_iter, alpha=alpha)
        else:
            G_vis1 = G1
        out_vtp = os.path.join(out_dir, f"{name}_predicted_comp1.vtp")
        export_centerline_graph_to_vtp(G_vis1, out_vtp)

    return G0, G1, seed_list, targets_list


def gm_load(
    ckpt_path: str,
    cfg: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"[gm_load] checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    in_node = int(ckpt.get("in_node", 11))
    edge_in = int(ckpt.get("edge_in", 8))

    cfg = cfg or {}
    model = gnn_model(
        in_node=in_node,
        edge_in=edge_in,
        hidden=cfg.get("hidden", 48),
        layers=cfg.get("layers", 4),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def save_predicted_graph_to_vtp(G: nx.Graph,
                                out_path: str):
    export_centerline_graph_to_vtp(G, out_path)
