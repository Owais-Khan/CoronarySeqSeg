import os
import time
from typing import List, Tuple, Optional, Dict, Set, Iterable


import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt as edt
from scipy.ndimage import gaussian_filter, map_coordinates

from skimage.measure import marching_cubes

from SeqSeg.seqseg.modules.sitk_functions import (
    import_image, extract_volume, copy_settings, remove_other_vessels, check_seg_border
)
from SeqSeg.seqseg.modules.nnunet import initialize_predictor
from SeqSeg.seqseg.modules.assembly import Segmentation
from skimage.morphology import skeletonize

from seqseg_modules_modified.assembly import VesselTree
import numpy as np
import networkx as nx
from typing import Dict, Set, Tuple, List, Optional
from scipy.spatial import cKDTree as KDTree



# ----------------- helpers -----------------
def _to_np(p) -> np.ndarray:
    return np.asarray(p, dtype=float)


def _phys_tuple(p):
    a = np.asarray(p, dtype=np.float64).reshape(3)
    return (float(a[0]), float(a[1]), float(a[2]))


def _unit(v) -> np.ndarray:
    v = _to_np(v)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else np.array([1.0, 0.0, 0.0], float)


def _build_kdtree(G: nx.Graph):
    pts, ids = [], []
    for n, d in G.nodes(data=True):
        p = d.get('point', None)
        if p is not None:
            pts.append(_to_np(p))
            ids.append(n)
    if not pts:
        return None, []
    return KDTree(np.vstack(pts)), ids


def _make_index(G: nx.Graph) -> Dict[str, object]:
    kd, ids = _build_kdtree(G)
    return {"G": G, "kd": kd, "ids": ids}


def _nearest_node(kdt, ids, point):
    if kdt is None:
        return None
    _, idx = kdt.query(_to_np(point), k=1)
    return ids[int(idx)]


def ensure_node_probability(
    G: nx.Graph,
    node_prob_key: str = "node_prob",
    edge_prob_key: str = "edge_prob",
    default: float = 0.1
) -> None:
    for n in G.nodes():
        if node_prob_key in G.nodes[n]:
            continue
        neigh = list(G.neighbors(n))
        if not neigh:
            G.nodes[n][node_prob_key] = float(default)
            continue
        vals = []
        for u in neigh:
            e = G.edges[n, u]
            vals.append(float(e.get(edge_prob_key, e.get("prob", default))))
        G.nodes[n][node_prob_key] = float(np.mean(vals)) if vals else float(default)


def smooth_node_probability(
    G: nx.Graph,
    key_in: str = "node_prob",
    key_out: str = "node_prob_smooth",
    beta: float = 0.5,
    iters: int = 1
) -> None:
    p = {n: float(G.nodes[n].get(key_in, 0.5)) for n in G.nodes()}
    for _ in range(int(iters)):
        p_new = {}
        for n in G.nodes():
            neigh = list(G.neighbors(n))
            if neigh:
                m = float(np.mean([p[u] for u in neigh]))
                p_new[n] = (1.0 - beta) * p[n] + beta * m
            else:
                p_new[n] = p[n]
        p = p_new
    for n, v in p.items():
        G.nodes[n][key_out] = float(v)


def _mk_step_dict(old_point, old_radius, new_point, new_radius, tangent, angle_change: Optional[float] = None) -> dict:
    d = {
        'old point': _to_np(old_point),
        'point': _to_np(new_point),
        'old radius': float(old_radius),
        'radius': float(new_radius),
        'tangent': _to_np(tangent),
        'chances': 0,
        'seg_file': None,
        'img_file': None,
        'surf_file': None,
        'cent_file': None,
        'prob_predicted_vessel': None,
        'point_pd': None,
        'surface': None,
        'centerline': None,
        'is_inside': False,
        'time': None,
        'dice': None,
    }
    if angle_change is not None:
        d['angle change'] = float(angle_change)
    return d


def edt_for_pred(
    Gcov: nx.Graph,
    roi_img: sitk.Image,
    *,
    min_seeds: int = 2
) -> Tuple[Optional[sitk.Image], Optional[List[int]]]:
    if Gcov is None or Gcov.number_of_nodes() == 0 or roi_img is None:
        return None, None

    sz = np.array(roi_img.GetSize(), int)  # (x,y,z)
    if np.any(sz == 0):
        return None, None
    Z, Y, X = int(sz[2]), int(sz[1]), int(sz[0])

    seeds_mask = np.zeros((Z, Y, X), dtype=np.uint8)
    seeds_lab = np.zeros((Z, Y, X), dtype=np.int32)
    lut_ids: List[int] = []

    used_vox: Set[Tuple[int, int, int]] = set()
    for nid, d in Gcov.nodes(data=True):
        p = d.get('point', None)
        if p is None:
            continue
        ci = np.array(roi_img.TransformPhysicalPointToContinuousIndex(_phys_tuple(p)), float)
        vx, vy, vz = int(round(ci[0])), int(round(ci[1])), int(round(ci[2]))
        if not (0 <= vx < X and 0 <= vy < Y and 0 <= vz < Z):
            continue
        key = (vz, vy, vx)
        if key in used_vox:
            continue
        used_vox.add(key)
        lut_ids.append(int(nid))
        label_val = len(lut_ids)  # 1..K
        seeds_mask[vz, vy, vx] = 1
        seeds_lab[vz, vy, vx] = label_val

    if len(lut_ids) < min_seeds:
        return None, None

    inv = (seeds_mask == 0)
    _, inds = edt(inv, return_distances=True, return_indices=True)
    iz, iy, ix = inds[0], inds[1], inds[2]
    labels = seeds_lab[iz, iy, ix].astype(np.int32)

    lab_img = sitk.GetImageFromArray(labels)
    lab_img.SetSpacing(roi_img.GetSpacing())
    lab_img.SetOrigin(roi_img.GetOrigin())
    lab_img.SetDirection(roi_img.GetDirection())
    return lab_img, lut_ids


def vor_cell_id_at_point(vor_label_img: Optional[sitk.Image],
                         vor_lut_ids: Optional[List[int]],
                         p_phys: np.ndarray) -> Optional[int]:
    if vor_label_img is None or not vor_lut_ids:
        return None
    ci = np.array(vor_label_img.TransformPhysicalPointToContinuousIndex(_phys_tuple(p_phys)), float)
    X, Y, Z = vor_label_img.GetSize()
    ix, iy, iz = int(round(ci[0])), int(round(ci[1])), int(round(ci[2]))
    if not (0 <= ix < X and 0 <= iy < Y and 0 <= iz < Z):
        return None
    arr = sitk.GetArrayFromImage(vor_label_img)  # (z,y,x)
    lab = int(arr[iz, iy, ix])
    if lab <= 0:
        return None
    return int(vor_lut_ids[lab - 1])


def roi_cell_index(Gcov: nx.Graph,
                   vor_label_img: sitk.Image,
                   vor_lut_ids: List[int]) -> Dict[int, Set[int]]:
    arr = sitk.GetArrayFromImage(vor_label_img)  # (z,y,x)
    Z, Y, X = arr.shape
    cell2nodes: Dict[int, Set[int]] = {}
    for nid, d in Gcov.nodes(data=True):
        p = d.get("point", None)
        if p is None:
            continue
        ci = np.array(vor_label_img.TransformPhysicalPointToContinuousIndex(_phys_tuple(p)), float)
        ix, iy, iz = int(round(ci[0])), int(round(ci[1])), int(round(ci[2]))
        if not (0 <= iz < Z and 0 <= iy < Y and 0 <= ix < X):
            continue
        lab = int(arr[iz, iy, ix])
        if lab <= 0:
            continue
        cell_id = int(vor_lut_ids[lab - 1])
        cell2nodes.setdefault(cell_id, set()).add(int(nid))
    return cell2nodes


def roi_cell_adjacency(vor_label_img: sitk.Image,
                       vor_lut_ids: List[int]) -> Set[Tuple[int, int]]:
    arr = sitk.GetArrayFromImage(vor_label_img)
    Z, Y, X = arr.shape
    edges: Set[Tuple[int, int]] = set()

    def add_edge(a, b):
        if a <= 0 or b <= 0 or a == b:
            return
        ca = vor_lut_ids[a - 1]; cb = vor_lut_ids[b - 1]
        e = (min(ca, cb), max(ca, cb))
        edges.add(e)

    if X > 1:
        diff = arr[:, :, 1:] != arr[:, :, :-1]
        zz, yy, xx = np.where(diff)
        for z, y, x in zip(zz, yy, xx):
            add_edge(int(arr[z, y, x]), int(arr[z, y, x + 1]))
    if Y > 1:
        diff = arr[:, 1:, :] != arr[:, :-1, :]
        zz, yy, xx = np.where(diff)
        for z, y, x in zip(zz, yy, xx):
            add_edge(int(arr[z, y, x]), int(arr[z, y + 1, x]))
    if Z > 1:
        diff = arr[1:, :, :] != arr[:-1, :, :]
        zz, yy, xx = np.where(diff)
        for z, y, x in zip(zz, yy, xx):
            add_edge(int(arr[z, y, x]), int(arr[z + 1, y, x]))
    return edges


def _phys_to_idx(img: sitk.Image, p_phys: np.ndarray) -> np.ndarray:
    ci = np.array(img.TransformPhysicalPointToContinuousIndex(_phys_tuple(p_phys)), float)
    sz = np.array(img.GetSize(), float)
    return np.clip(ci, 0, sz - 1)


def _line_integral_prob_sitk(prob_img: sitk.Image,
                             start_phys: np.ndarray,
                             dir_unit_phys: np.ndarray,
                             ray_len_mm: float,
                             step_mm: float,
                             tube_sigma_mm: float) -> float:

    if prob_img is None:
        return 0.0

    # --- prep & smoothing (σ in *voxels*) ---
    arr = sitk.GetArrayFromImage(prob_img).astype(np.float32)
    spacing_xyz = np.asarray(prob_img.GetSpacing(), dtype=float)
    sigma_vox_zyx = (float(tube_sigma_mm) / np.maximum(spacing_xyz, 1e-12))[::-1]
    arr_s = gaussian_filter(arr, sigma=sigma_vox_zyx, mode='nearest')


    origin = np.asarray(prob_img.GetOrigin(), dtype=float)
    M = np.asarray(prob_img.GetDirection(), dtype=float).reshape(3, 3)
    spacing = spacing_xyz


    n = max(1, int(np.ceil(float(ray_len_mm) / max(float(step_mm), 1e-6))))
    p0 = np.asarray(start_phys, dtype=float)
    d  = np.asarray(dir_unit_phys, dtype=float)
    d /= (np.linalg.norm(d) + 1e-12)

    t = (np.arange(1, n + 1, dtype=float) * float(step_mm))[:, None]
    q_phys = p0[None, :] + t * d[None, :]

    rel = q_phys - origin[None, :]
    idx_xyz = (rel @ M) / np.maximum(spacing, 1e-12)


    size_xyz = np.asarray(prob_img.GetSize(), dtype=float)
    idx_xyz = np.clip(idx_xyz, 0.0, size_xyz - 1.000001)

    coords_zyx = np.stack([idx_xyz[:, 2], idx_xyz[:, 1], idx_xyz[:, 0]], axis=0)
    samples = map_coordinates(arr_s, coords_zyx, order=1, mode='nearest')
    return float(np.mean(samples))


def _snap_phys_inside(image: sitk.Image, p_xyz: np.ndarray) -> np.ndarray:
    ci = np.array(image.TransformPhysicalPointToContinuousIndex(_phys_tuple(p_xyz)))
    sz = np.array(image.GetSize(), int)
    ci = np.clip(ci, 0, sz - 1)
    return np.array(image.TransformContinuousIndexToPhysicalPoint(tuple(ci.tolist())))



def map_to_image(
    center_phys,
    box_radius_mm: float,
    volume_size_ratio: float,
    *,
    image: sitk.Image,
    min_res: int = 5,
    require_odd: bool = True,
) -> Tuple[list[int], list[int], bool]:
    if image is None:
        raise ValueError("map_to_image requires SimpleITK image.")

    ci = np.array(image.TransformPhysicalPointToContinuousIndex(_phys_tuple(center_phys)))
    img_size = np.array(list(image.GetSize()), dtype=int)
    spacing = np.array(list(image.GetSpacing()), dtype=float)

    if not np.all(np.isfinite(ci)):
        ci = (img_size - 1) / 2.0

    L_mm = max(1e-3, float(volume_size_ratio) * float(box_radius_mm))
    size_vox = np.ceil(L_mm / np.maximum(spacing, 1e-12)).astype(int)

    size_vox = np.maximum(size_vox, int(min_res))
    if require_odd:
        size_vox = size_vox + (size_vox % 2 == 0)

    if np.any(size_vox > img_size):
        size_vox = np.minimum(size_vox, img_size)

    start = np.floor(ci - 0.5 * size_vox).astype(int)
    start_clamped = np.maximum(0, np.minimum(start, img_size - size_vox))
    border = bool(np.any(start_clamped != start) or np.any(size_vox == img_size))

    size_vox = np.maximum(size_vox, 1)
    return start_clamped.tolist(), size_vox.tolist(), border


def vt_init_coverage(vt: VesselTree, G_for_cov: nx.Graph):
    ids, pts, radii = [], [], []
    for n, d in G_for_cov.nodes(data=True):
        p = d.get("point", None)
        if p is None:
            continue
        ids.append(int(n))
        pts.append(_to_np(p))
        radii.append(float(d.get("radius", d.get("MaximumInscribedSphereRadius", 1.0))))

    vt._cov_ids = ids
    vt._cov_pts = np.vstack(pts) if pts else np.zeros((0, 3), float)
    vt._cov_r   = np.asarray(radii, dtype=float) if radii else np.zeros((0,), float)
    vt._cov_kdt = KDTree(vt._cov_pts) if len(vt._cov_pts) else None
    vt._cov_r_max = float(np.max(vt._cov_r)) if vt._cov_r.size else 1.0

    vt._cov_mask = np.zeros(len(vt._cov_ids), dtype=bool)
    vt._cov_id2idx = {nid: i for i, nid in enumerate(vt._cov_ids)}

    vt.node_traversed = set()
    vt.node_not_traversed = set(vt._cov_ids)

    deg = dict(G_for_cov.degree())
    mdeg = float(max(1, max(deg.values()) if len(deg) else 1))
    vt._centrality = {n: (deg.get(n, 0) / mdeg) for n in G_for_cov.nodes()}


def vt_mark_covered_by_segment_ball(
    vt: VesselTree,
    G_for_cov: nx.Graph,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    radius_scale: float = 1.15,
    sample_stride_mm: float = 1.5,
    local_radius_probe_mm: float = 6.0,
) -> int:
    if vt._cov_kdt is None or len(vt._cov_ids) == 0:
        return 0

    p0 = _to_np(p0); p1 = _to_np(p1)
    ab = p1 - p0
    L = float(np.linalg.norm(ab))
    if L < 1e-6:
        _, j = vt._cov_kdt.query(p0, k=1)
        j = int(j)
        if not vt._cov_mask[j]:
            vt._cov_mask[j] = True
            nid = vt._cov_ids[j]
            vt.node_traversed.add(nid)
            vt.node_not_traversed.discard(nid)
            return 1
        return 0

    idx0 = vt._cov_kdt.query_ball_point(p0, r=local_radius_probe_mm) or []
    idx1 = vt._cov_kdt.query_ball_point(p1, r=local_radius_probe_mm) or []
    loc = np.unique(np.asarray(idx0 + idx1, dtype=int)) if (idx0 or idx1) else None
    if loc is not None and loc.size > 0:
        r_local = float(np.median(vt._cov_r[loc]))
    else:
        r_local = float(np.median(vt._cov_r)) if vt._cov_r.size else 1.0

    rcap = radius_scale * max(min(r_local, 5.0), 0.5)

    n_samp = max(1, int(np.ceil(L / max(sample_stride_mm, 1e-6))))
    centers = p0 + (np.linspace(0.0, 1.0, n_samp + 1)[:, None] * ab)

    cand_idx: set[int] = set()
    for c in centers:
        idxs = vt._cov_kdt.query_ball_point(c, r=rcap)
        if isinstance(idxs, int):
            idxs = [idxs]
        cand_idx.update(map(int, idxs))
    if not cand_idx:
        return 0

    idx_arr = np.fromiter(cand_idx, dtype=int)
    P = vt._cov_pts[idx_arr]
    R = vt._cov_r[idx_arr]

    ab2 = float(np.dot(ab, ab))
    t = np.clip(((P - p0) @ ab) / ab2, 0.0, 1.0)
    proj = p0 + t[:, None] * ab
    dist = np.linalg.norm(P - proj, axis=1)
    r_allow = radius_scale * np.maximum(R, 1e-6)

    ok = dist <= r_allow
    if not np.any(ok):
        return 0

    new_idx = idx_arr[ok & (~vt._cov_mask[idx_arr])]
    if new_idx.size == 0:
        return 0

    vt._cov_mask[new_idx] = True
    for j in new_idx:
        nid = vt._cov_ids[int(j)]
        vt.node_traversed.add(nid)
        vt.node_not_traversed.discard(nid)

    return int(new_idx.size)


def vt_coverage_ratio(vt: VesselTree) -> float:
    if vt._cov_mask is None or vt._cov_mask.size == 0:
        return 0.0
    denom = max(1, vt._cov_mask.size)
    return float(vt._cov_mask.sum()) / float(denom)


def _prefer_uncovered_order(arr_pt: np.ndarray, arr_rad: np.ndarray, vt: VesselTree):
    """Order candidates: (uncovered-first, then larger radius)."""
    if arr_pt is None or arr_pt.size == 0 or vt._cov_kdt is None:
        return np.arange(0)

    pts = np.atleast_2d(arr_pt)
    _, idxs = vt._cov_kdt.query(pts, k=1)
    idxs = np.atleast_1d(idxs).astype(int)

    is_covered = vt._cov_mask[idxs]
    penalty = is_covered.astype(int)

    return np.lexsort((-np.asarray(arr_rad, float), penalty))

def _to_np(p):
    return np.asarray(p, float)

def _collect_points(G: nx.Graph):
    ids, pts = [], []
    for n, d in G.nodes(data=True):
        p = d.get('point')
        if p is None:
            continue
        ids.append(int(n))
        pts.append(_to_np(p))
    if not ids:
        return np.asarray([], int), np.zeros((0,3), float)
    return np.asarray(ids, int), np.vstack(pts)

def _ensure_edge_length(G: nx.Graph, weight: str = "length_mm") -> str:
    need = False
    for _, _, d in G.edges(data=True):
        if weight not in d:
            need = True; break
    if not need:
        return weight
    for u, v, d in G.edges(data=True):
        pu = _to_np(G.nodes[u].get('point')) if 'point' in G.nodes[u] else None
        pv = _to_np(G.nodes[v].get('point')) if 'point' in G.nodes[v] else None
        if pu is None or pv is None:
            d[weight] = float(d.get('weight', 1.0))  # fallback
        else:
            d[weight] = float(np.linalg.norm(pu - pv))
    return weight

def build_cross_voronoi_nx(
    Gcent: nx.Graph,
    Gvol: nx.Graph,
    *,
    k: int = 1,
    weight: str = "length_mm",
) -> Tuple[Dict[int, int], Dict[int, Set[int]]]:

    cent_ids, cent_pts = _collect_points(Gcent)
    vol_ids,  vol_pts  = _collect_points(Gvol)
    if cent_ids.size == 0 or vol_ids.size == 0:
        return {}, {}

    k = max(1, min(int(k), vol_ids.size))
    kdt_vol = KDTree(vol_pts)
    anchor_nodes: List[int] = []
    anchor_owner: List[int] = []
    for i, cid in enumerate(cent_ids):
        _, ii = kdt_vol.query(cent_pts[i], k=k)
        ii = np.atleast_1d(ii).astype(int)
        for j in ii:
            anchor_nodes.append(int(vol_ids[j]))
            anchor_owner.append(int(cid))

    uniq_anchor_nodes: List[int] = []
    uniq_anchor_owner: List[int] = []
    seen = set()
    for a, own in zip(anchor_nodes, anchor_owner):
        if a in seen:
            continue
        seen.add(a)
        uniq_anchor_nodes.append(a)
        uniq_anchor_owner.append(own)
    anchor_nodes = uniq_anchor_nodes
    anchor_owner = uniq_anchor_owner

    weight_attr = _ensure_edge_length(Gvol, weight=weight)
    vor_cells = nx.voronoi_cells(Gvol, anchor_nodes, weight=weight_attr)

    anchor_to_owner = {int(a): int(o) for a, o in zip(anchor_nodes, anchor_owner)}
    assign_v2c: Dict[int, int] = {}
    cell_c2v: Dict[int, Set[int]] = {int(c): set() for c in cent_ids.tolist()}

    covered: Set[int] = set()

    for a_seed, nodeset in vor_cells.items():
        cid = anchor_to_owner[int(a_seed)]
        for n in nodeset:
            assign_v2c[int(n)] = cid
            cell_c2v[cid].add(int(n))
            covered.add(int(n))

    return assign_v2c, cell_c2v


def init_cell_cov_structs(Gvol: nx.Graph, cell_c2v: Dict[int, Set[int]]):

    cell_pts, cell_ids, cell_mask, cell_kdt = {}, {}, {}, {}
    for cid, nodes in cell_c2v.items():
        nodes_sorted = sorted(nodes)
        P, keep_ids = [], []
        for n in nodes_sorted:
            d = Gvol.nodes.get(n, {})
            p = d.get('point')
            if p is None:
                continue
            P.append(np.asarray(p, float))
            keep_ids.append(int(n))

        if len(P) == 0:
            cell_pts[cid] = np.zeros((0, 3), float)
            cell_ids[cid] = np.zeros((0,), int)
            cell_mask[cid] = np.zeros((0,), bool)
            cell_kdt[cid] = None
        else:
            P = np.vstack(P)
            ids = np.asarray(keep_ids, int)
            cell_pts[cid]  = P
            cell_ids[cid]  = ids
            cell_mask[cid] = np.zeros(len(ids), dtype=bool)
            cell_kdt[cid]  = KDTree(P)
    return cell_pts, cell_ids, cell_mask, cell_kdt


def mark_cell_cover(
    cid: int,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    cell_pts: Dict[int, np.ndarray],
    cell_ids: Dict[int, np.ndarray],
    cell_mask: Dict[int, np.ndarray],
    cell_kdt: Dict[int, Optional[KDTree]],
    Gvol: nx.Graph,
    radius_scale: float = 1.15,
    sample_stride_mm: float = 1.5,
    local_radius_probe_mm: float = 6.0,
) -> int:
    """Marks Gvol nodes inside this cell as covered if they lie within a tube around segment p0->p1."""
    cid = int(cid)
    if cid not in cell_kdt or cell_kdt[cid] is None or cell_pts[cid].size == 0:
        return 0

    P    = cell_pts[cid]
    kdt  = cell_kdt[cid]
    mask = cell_mask[cid]

    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    ab = p1 - p0
    L  = float(np.linalg.norm(ab))

    if L < 1e-6:
        _, j = kdt.query(p0, k=1)
        j = int(j)
        if not mask[j]:
            mask[j] = True
            return 1
        return 0

    # local radius estimate
    idx0 = kdt.query_ball_point(p0, r=local_radius_probe_mm) or []
    idx1 = kdt.query_ball_point(p1, r=local_radius_probe_mm) or []
    loc  = np.unique(np.asarray(idx0 + idx1, dtype=int)) if (idx0 or idx1) else np.array([], int)
    if loc.size:
        rloc = []
        ids = cell_ids[cid]
        for j in loc:
            nid = int(ids[int(j)])
            d = Gvol.nodes[nid]
            rloc.append(float(d.get('radius', d.get('MaximumInscribedSphereRadius', 1.0))))
        r_local = float(np.median(rloc)) if rloc else 1.0
    else:
        r_local = 1.0

    rcap    = radius_scale * max(min(r_local, 5.0), 0.5)
    n_samp  = max(1, int(np.ceil(L / max(sample_stride_mm, 1e-6))))
    centers = p0 + (np.linspace(0.0, 1.0, n_samp + 1)[:, None] * ab)

    cand_idx: Set[int] = set()
    for c in centers:
        idxs = kdt.query_ball_point(c, r=rcap)
        if isinstance(idxs, int):
            idxs = [idxs]
        cand_idx.update(map(int, idxs))
    if not cand_idx:
        return 0

    idx_arr = np.fromiter(cand_idx, dtype=int)
    Q = P[idx_arr]
    ab2 = float(np.dot(ab, ab))
    t = np.clip(((Q - p0) @ ab) / ab2, 0.0, 1.0)
    proj = p0 + t[:, None] * ab
    dist = np.linalg.norm(Q - proj, axis=1)

    ok = dist <= rcap
    newly_idx = idx_arr[ok & (~mask[idx_arr])]
    if newly_idx.size == 0:
        return 0

    mask[newly_idx] = True
    return int(newly_idx.size)


def update_last_cov_cent_anchor(vt, point_mm: np.ndarray):
    if getattr(vt, "_cov_kdt", None) is None or vt._cov_pts is None or vt._cov_mask is None:
        return
    _, j = vt._cov_kdt.query(np.asarray(point_mm, float), k=1)
    j = int(j)
    if 0 <= j < vt._cov_mask.size and bool(vt._cov_mask[j]):
        vt._last_cov_cent_id = int(vt._cov_ids[j])


def get_next_points_for_nodes(
    prob_img: Optional[sitk.Image],
    curr_mm: np.ndarray,
    prev_mm: np.ndarray,
    radius_mm: float,
    *,
    candidate_node_ids: Iterable[int],
    graph_vol: nx.Graph,
    vol_index: Dict[str, object],
    cfg: Dict,
    k_best: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    ANGLE_ALLOW_DEG    = float(cfg.get('ANGLE_ALLOW_DEG', 175.0))
    RAY_LEN_MM         = float(cfg.get('RAY_LEN_MM', 20.0))
    TUBE_SIGMA_MM      = float(cfg.get('TUBE_SIGMA_MM', 0.3))
    W_IMG              = float(cfg.get('W_IMG', 0.6))
    W_EDGE             = float(cfg.get('W_EDGE', 0.7))
    W_NODE_CL          = float(cfg.get('W_NODE_CL', 0.3))
    W_ANGLE            = float(cfg.get('W_ANGLE', 0.2))
    PROB_GAMMA         = float(cfg.get('PROB_GAMMA', 0.5))
    MIN_RADIUS         = float(cfg.get('MIN_RADIUS', 0.0))
    ADD_RADIUS         = float(cfg.get('ADD_RADIUS', 0.0))

    if vol_index.get("kd") is None:
        return (np.empty((0,3)),) * 4

    curr = np.asarray(curr_mm, float); prev = np.asarray(prev_mm, float)
    old_vec = curr - prev
    old_vec /= (np.linalg.norm(old_vec) + 1e-12)
    cos_thr = float(np.cos(np.radians(ANGLE_ALLOW_DEG)))
    nV_curr = _nearest_node(vol_index["kd"], vol_index["ids"], curr)

    cand_pts, cand_r, cand_ang = [], [], []
    raw_img, p_edge_list, p_nc_list, vor_list = [], [], [], []

    for nid in candidate_node_ids:
        dnode = graph_vol.nodes.get(nid)
        if not dnode or 'point' not in dnode:
            continue
        nxt_p = np.asarray(dnode['point'], float)
        du = nxt_p - curr
        nrm = float(np.linalg.norm(du))
        if nrm < 1e-9:
            continue
        du /= nrm
        dotv = float(np.clip(np.dot(old_vec, du), -1.0, 1.0))
        if dotv < cos_thr:
            continue

        # node/edge priors on Gvol
        p_edge = 0.0
        nV_next = int(nid)
        if (nV_curr is not None) and graph_vol.has_edge(nV_curr, nV_next):
            e = graph_vol.edges[nV_curr, nV_next]
            p_edge = float(e.get('edge_prob', e.get('prob', 0.5)))
            radius_mm = float(e.get('radius_mm'))


        # image evidence
        step_len = max(0.1, min(nrm, RAY_LEN_MM))
        s_img = _line_integral_prob_sitk(prob_img, start_phys=curr, dir_unit_phys=du,
                                         ray_len_mm=RAY_LEN_MM, step_mm=step_len, tube_sigma_mm=TUBE_SIGMA_MM)

        # radius fusion
        r_eff = float(radius_mm)
        rv = float(dnode.get('radius', dnode.get('MaximumInscribedSphereRadius', r_eff)))
        r_eff = 0.7 * r_eff + 0.3 * rv
        p_comb = float(np.clip(W_EDGE * p_edge, 0.0, 1.0))
        r_eff *= (1.0 + PROB_GAMMA * (p_comb - 0.5))
        r_eff = (2.0/3.0) * float(radius_mm) + (1.0/3.0) * r_eff
        r_eff = max(r_eff, MIN_RADIUS)

        cand_pts.append(nxt_p)
        cand_r.append(r_eff + ADD_RADIUS)
        cand_ang.append(np.degrees(np.arccos(dotv)))
        raw_img.append(s_img); p_edge_list.append(p_edge)

    if not cand_pts:
        return (np.empty((0,3)),) * 4

    cand_pts = np.vstack(cand_pts)
    cand_r   = np.asarray(cand_r, float)
    cand_ang = np.asarray(cand_ang, float)
    raw_img  = np.asarray(raw_img, float)
    p_edge_a = np.asarray(p_edge_list, float)
    p_nc_a   = np.asarray(p_nc_list, float)

    img_n = raw_img / (np.max(raw_img) + 1e-8) if np.max(raw_img) > 0 else np.zeros_like(raw_img)
    ang_n = cand_ang / max(1e-6, ANGLE_ALLOW_DEG)
    base = (W_IMG * img_n + W_EDGE * p_edge_a + W_NODE_CL * p_nc_a + - W_ANGLE * ang_n)
    base *= cand_r

    order = np.argsort(-base)
    if k_best is not None:
        order = order[:int(k_best)]
    return cand_pts[order], cand_r[order], cand_ang[order], base[order]

def _append_seed_step(vt: VesselTree, seed_pt: np.ndarray, seed_r: float, tangent_hint: Optional[np.ndarray] = None):
    if tangent_hint is None:
        if len(vt.steps) > 0:
            tangent_hint = _unit(vt.steps[-1]['tangent'])
        else:
            tangent_hint = np.array([1.0, 0.0, 0.0], float)
    step0 = _mk_step_dict(seed_pt, seed_r, seed_pt, seed_r, _unit(tangent_hint), angle_change=0.0)
    vt.steps.append(step0)


def _targets_remaining(vt: VesselTree) -> List[int]:
    ids = set(getattr(vt, "target_ids", []) or [])
    return [t for t in ids if t not in vt.node_traversed]


def _pick_reseed_node(
    vt: VesselTree,
    Gcov: nx.Graph,
    current_point: np.ndarray,
    *,
    prefer: str = "farthest",
    min_sep_mm: float = 1,
    alpha_prob: float = 0.4,
    beta_radius: float = 0.8,
    gamma_dist: float = 0.4,
    min_degree: int = 2,
    targets_first: bool = False,
) -> Optional[np.ndarray]:

    if vt._cov_kdt is None or not vt.node_not_traversed:
        return None

    targets_rem = set(_targets_remaining(vt))
    pool = list(vt.node_not_traversed)
    if targets_first and len(targets_rem) > 0:
        pool = [nid for nid in pool if nid in targets_rem]
    if len(pool) == 0:
        pool = list(vt.node_not_traversed)

    deg = dict(Gcov.degree())
    pool_deg = [nid for nid in pool if deg.get(nid, 0) >= int(min_degree)]
    if len(pool_deg) > 0:
        pool = pool_deg
    if len(pool) == 0:
        return None

    sep_ok = np.ones(len(pool), dtype=bool)
    if len(vt.node_traversed) > 0:
        pts_tr = [_to_np(Gcov.nodes[n]['point']) for n in vt.node_traversed if 'point' in Gcov.nodes[n]]
        if len(pts_tr) > 0:
            kdt_tr = KDTree(np.vstack(pts_tr))
            for i, nid in enumerate(pool):
                p = _to_np(Gcov.nodes[nid].get('point', [0, 0, 0]))
                dmin, _ = kdt_tr.query(p, k=1)
                if dmin < float(min_sep_mm):
                    sep_ok[i] = False

    q = _to_np(current_point)
    scores, cand_pts = [], []
    for i, nid in enumerate(pool):
        if not sep_ok[i]:
            scores.append(-1e9); cand_pts.append(None); continue
        dnode = Gcov.nodes[nid]
        if 'point' not in dnode:
            scores.append(-1e9); cand_pts.append(None); continue
        p = _to_np(dnode['point'])
        pr = float(dnode.get("node_prob_smooth", dnode.get('node_prob', 0.5)))
        r = float(dnode.get('radius', dnode.get('MaximumInscribedSphereRadius', 1.0)))
        cent = float(getattr(vt, "_centrality", {}).get(nid, 0.0))
        dist = float(np.linalg.norm(p - q))
        sc = 2.0 * cent + alpha_prob * pr + beta_radius * r - gamma_dist * dist
        scores.append(sc); cand_pts.append(p)

    if max(scores) <= -1e8:
        return None

    order = np.argsort(-np.asarray(scores))
    if prefer in ("nearest", "farthest"):
        topk = order[:min(64, len(order))]
        topk = sorted(
            topk,
            key=lambda i: np.linalg.norm(cand_pts[i] - q) if cand_pts[i] is not None else (1e9),
            reverse=(prefer == "farthest"),
        )
        order = np.array(topk + [j for j in order if j not in topk])

    for i in order:
        p = cand_pts[int(i)]
        if p is not None:
            return p
    return None


def _qa_surface_from_crop(prob_or_mask_img: sitk.Image,
                          level: float = 0.5,
                          write_path_npz: Optional[str] = None):
    arr = sitk.GetArrayFromImage(prob_or_mask_img).astype(np.float32)  # (z,y,x)
    arr = np.clip(arr, 0, 1)
    if arr.max() <= 0:
        return None
    verts, faces, _, _ = marching_cubes(arr, level=level, spacing=prob_or_mask_img.GetSpacing()[::-1])
    verts_mm = verts[:, ::-1].copy()  # (x,y,z) mm
    if write_path_npz:
        np.savez_compressed(write_path_npz, verts=verts_mm, faces=faces.astype(np.int32))
    return verts_mm, faces.astype(np.int32)


def _build_graph_voronoi(G: nx.Graph, seeds: Optional[Iterable[int]] = None, weight: Optional[str] = None):
    if seeds is None:
        deg = dict(G.degree())
        if not deg:
            seeds = list(G.nodes())
        else:
            N = G.number_of_nodes()
            K = max(1, min(int(np.ceil(np.sqrt(N))), N))
            seeds = [n for n, _ in sorted(deg.items(), key=lambda kv: kv[1], reverse=True)[:K]]

    cells = nx.voronoi_cells(G, seeds, weight=weight)
    assignment = {}
    for s, nodeset in cells.items():
        for n in nodeset:
            assignment[n] = s

    cell_adj = set()
    for u, v in G.edges():
        su = assignment.get(u, None)
        sv = assignment.get(v, None)
        if su is None or sv is None or su == sv:
            continue
        a, b = (su, sv) if su < sv else (sv, su)
        cell_adj.add((a, b))

    cell2nodes = {s: set(ns) for s, ns in cells.items()}
    return assignment, cell2nodes, cell_adj

def coverage_for_cell(cid: int, cell_mask: Dict[int, np.ndarray]) -> float:
    arr = cell_mask.get(int(cid))
    if arr is None or arr.size == 0:
        return 0.0
    return float(arr.sum()) / float(max(1, arr.size))



def _phys_to_cont_index(img: sitk.Image, P_mm: np.ndarray) -> np.ndarray:
    O = np.asarray(img.GetOrigin(), float)
    S = np.asarray(img.GetSpacing(), float)
    M = np.asarray(img.GetDirection(), float).reshape(3,3)
    return ( (P_mm - O) @ M ) / np.maximum(S, 1e-12)

def _cont_index_to_phys(img: sitk.Image, I_xyz: np.ndarray) -> np.ndarray:
    O = np.asarray(img.GetOrigin(), float)
    S = np.asarray(img.GetSpacing(), float)
    M = np.asarray(img.GetDirection(), float).reshape(3,3)
    return O + (I_xyz * S) @ M.T
def _local_tangent_pca(pts_mm: np.ndarray) -> np.ndarray:
    if pts_mm.shape[0] < 2:
        return np.array([1.0,0.0,0.0], float)
    X = pts_mm - pts_mm.mean(0, keepdims=True)
    C = (X.T @ X) / max(1, X.shape[0]-1)
    w, V = np.linalg.eigh(C)
    v = V[:, np.argmax(w)]
    n = np.linalg.norm(v)
    return v / (n + 1e-12)
def get_next_points_from_crop_skeleton(
    prob_img: Optional[sitk.Image],
    curr_mm: np.ndarray,
    prev_mm: np.ndarray,
    radius_mm: float,
    *,
    cent_index: Dict[str, object],
    vol_index: Dict[str, object],
    cfg: Dict,
    k_best: Optional[int] = None,
    edt_centerline_mm: Optional[np.ndarray] = None,
    geom_img: Optional[sitk.Image] = None,
):

    import numpy as np
    from scipy.ndimage import (
        gaussian_filter, map_coordinates,
        binary_opening, binary_closing, generate_binary_structure
    )

    if prob_img is None or vol_index.get("kd") is None:
        return (np.empty((0,3)),) * 4

    # ---- cfg (weights & knobs) ----
    BIN_THR            = float(cfg.get('BIN_THR', 0.5))
    OPEN_RADIUS_VOX    = int(cfg.get('OPEN_RADIUS_VOX', 1))
    CLOSE_RADIUS_VOX   = int(cfg.get('CLOSE_RADIUS_VOX', 0))
    SKEL_SUBSAMPLE_MM  = float(cfg.get('SKEL_SUBSAMPLE_MM', 1.5))
    MAX_CAND_TOTAL     = int(cfg.get('MAX_CAND_TOTAL', 512))

    ANGLE_ALLOW_DEG    = float(cfg.get('ANGLE_ALLOW_DEG', 145.0))
    RAY_LEN_MM         = float(cfg.get('RAY_LEN_MM', 16.0))
    TUBE_SIGMA_MM      = float(cfg.get('TUBE_SIGMA_MM', 0.3))

    # linear weights (all positive, ANGLE is subtracted)
    W_IMG              = float(cfg.get('W_IMG', 0.6))
    W_EDGE             = float(cfg.get('W_EDGE', 0.6))
    W_ANGLE            = float(cfg.get('W_ANGLE', 0.2))
    W_NEAR_CL          = float(cfg.get('W_NEAR_CL', 0.4))


    # edge/path prior
    EDGE_KHOP_MAX      = int(cfg.get('EDGE_KHOP_MAX', 4))
    EDGE_FWD_COS_THR   = float(cfg.get('EDGE_FWD_COS_THR', 0.3))
    EDGE_PROB_KEY      = 'edge_prob'
    EDGE_FALLBACK_PROB = float(cfg.get('EDGE_FALLBACK_PROB', 0.1))
    # radius handling
    MIN_RADIUS         = float(cfg.get('MIN_RADIUS', 0.0))
    ADD_RADIUS         = float(cfg.get('ADD_RADIUS', 0.0))

    # score gate
    SCORE_MIN          = float(cfg.get('SCORE_MIN', 0.35))

    # ---- small helpers ----
    def _unit(v):
        v = np.asarray(v, float)
        n = float(np.linalg.norm(v))
        return v / (n + 1e-12)

    def _cont_index_to_phys(img: sitk.Image, idx_xyz: np.ndarray) -> np.ndarray:
        out = []
        for i in range(idx_xyz.shape[0]):
            out.append(np.asarray(
                img.TransformContinuousIndexToPhysicalPoint(tuple(map(float, idx_xyz[i]))), float))
        return np.vstack(out) if out else np.zeros((0,3), float)

    def _edge_prior_directional(curr_pt, cand_pt):
        G = vol_index["G"]; kd = vol_index["kd"]; ids = vol_index["ids"]
        if kd is None or G is None:
            return 0.0
        curr_pt = np.asarray(curr_pt, float)
        cand_pt = np.asarray(cand_pt, float)
        cand_dir = _unit(cand_pt - curr_pt)

        nV_curr = _nearest_node(kd, ids, curr_pt)
        nV_next = _nearest_node(kd, ids, cand_pt)
        if nV_curr is None or nV_next is None:
            return 0.0

        def eprob(u, v):
            e = G.edges[u, v]
            return float(e.get(EDGE_PROB_KEY, e.get("prob", EDGE_FALLBACK_PROB)))

        # direct edge?
        if G.has_edge(nV_curr, nV_next):
            return eprob(nV_curr, nV_next)

        from collections import deque
        q = deque()
        q.append((nV_curr, 0, 1.0))
        seen_depth = {nV_curr: 0}
        best = 0.0

        while q:
            u, d, p_u = q.popleft()
            if d >= EDGE_KHOP_MAX:
                continue
            pu = np.asarray(G.nodes[u].get("point", curr_pt), float)
            for v in G.neighbors(u):
                if v == u:
                    continue
                pv = np.asarray(G.nodes[v].get("point", pu), float)
                edge_dir = pv - pu
                en = float(np.linalg.norm(edge_dir))
                if en < 1e-9:
                    continue
                edge_dir /= en
                if float(np.dot(edge_dir, cand_dir)) < EDGE_FWD_COS_THR:
                    continue

                p_uv = eprob(u, v)
                p_v = 0.5 * (p_u + p_uv)
                if p_v <= best:
                    continue

                if v == nV_next:
                    best = max(best, p_v)
                    continue

                nd = d + 1
                if v not in seen_depth or nd < seen_depth[v]:
                    seen_depth[v] = nd
                    q.append((v, nd, p_v))
        return float(best)

    def _mean_prob_along(p0, p1, arr_s, spacing, origin, M, size_xyz):
        p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
        v = p1 - p0
        L = float(np.linalg.norm(v))
        if L < 1e-9:
            return 0.0
        v /= L
        n = max(1, int(np.ceil(min(L, RAY_LEN_MM) / max(0.5, 0.25*radius_mm))))
        t = (np.arange(1, n+1, dtype=float) * (min(L, RAY_LEN_MM) / n))[:, None]
        Q = p0[None,:] + t * v[None,:]
        rel = Q - origin[None,:]
        idx_xyz = (rel @ M) / np.maximum(spacing, 1e-12)
        idx_xyz = np.clip(idx_xyz, 0.0, size_xyz - 1.000001)
        coords_zyx = np.stack([idx_xyz[:,2], idx_xyz[:,1], idx_xyz[:,0]], axis=0)
        vals = map_coordinates(arr_s, coords_zyx, order=1, mode='nearest')
        return float(np.mean(vals))

    def _centerline_dist_mm(P_mm: np.ndarray) -> np.ndarray:
        if edt_centerline_mm is None or geom_img is None:
            if cent_index and cent_index.get("kd") is not None:
                d, _ = cent_index["kd"].query(P_mm, k=1)
                return np.asarray(d, float)
            return np.zeros(P_mm.shape[0], float)
        origin_g = np.asarray(geom_img.GetOrigin(), float)
        M_g = np.asarray(geom_img.GetDirection(), float).reshape(3,3)
        sp_g = np.asarray(geom_img.GetSpacing(), float)
        size_xyz_g = np.asarray(geom_img.GetSize(), float)
        rel = P_mm - origin_g[None,:]
        idx_xyz = (rel @ M_g) / np.maximum(sp_g, 1e-12)
        idx_xyz = np.clip(idx_xyz, 0.0, size_xyz_g - 1.000001)
        coords_zyx = np.stack([idx_xyz[:,2], idx_xyz[:,1], idx_xyz[:,0]], axis=0)
        vals = map_coordinates(edt_centerline_mm, coords_zyx, order=1, mode='nearest')
        return np.asarray(vals, float)

    # ---- binarize & clean ----
    arr = sitk.GetArrayFromImage(prob_img).astype(np.float32)
    if arr.size == 0 or float(arr.max()) <= 0:
        return (np.empty((0,3)),) * 4
    mask = (arr >= BIN_THR)
    if OPEN_RADIUS_VOX > 0:
        st = generate_binary_structure(3, 1)
        mask = binary_opening(mask, structure=st, iterations=OPEN_RADIUS_VOX)
    if CLOSE_RADIUS_VOX > 0:
        st = generate_binary_structure(3, 1)
        mask = binary_closing(mask, structure=st, iterations=CLOSE_RADIUS_VOX)
    if not mask.any():
        return (np.empty((0,3)),) * 4

    skel = skeletonize(mask)
    if not skel.any():
        return (np.empty((0,3)),) * 4

    # ---- subsample skeleton to physical points ----
    spacing = np.asarray(prob_img.GetSpacing(), float)
    stride_vox = max(1, int(round(SKEL_SUBSAMPLE_MM / float(np.median(spacing)))))
    zz, yy, xx = np.where(skel)
    if zz.size == 0:
        return (np.empty((0,3)),) * 4
    idx_keep = np.arange(0, zz.size, stride_vox, dtype=int)
    zz, yy, xx = zz[idx_keep], yy[idx_keep], xx[idx_keep]
    pts_idx = np.stack([xx, yy, zz], axis=1).astype(float)
    pts_mm  = _cont_index_to_phys(prob_img, pts_idx)
    if pts_mm.shape[0] > MAX_CAND_TOTAL:
        sel = np.linspace(0, pts_mm.shape[0]-1, MAX_CAND_TOTAL).astype(int)
        pts_mm = pts_mm[sel]

    sigma_vox_zyx = (TUBE_SIGMA_MM / np.maximum(spacing, 1e-12))[::-1]
    arr_s = gaussian_filter(arr, sigma=sigma_vox_zyx, mode='nearest')
    origin = np.asarray(prob_img.GetOrigin(), float)
    M = np.asarray(prob_img.GetDirection(), float).reshape(3,3)
    size_xyz = np.asarray(prob_img.GetSize(), float)

    curr = np.asarray(curr_mm, float)
    prev = np.asarray(prev_mm, float)
    old = _unit(curr - prev)
    cos_thr = float(np.cos(np.radians(ANGLE_ALLOW_DEG)))


    cand_pts, cand_r, cand_ang, scores = [], [], [], []
    d_vol_all, _ = vol_index["kd"].query(pts_mm, k=1)
    d_cl_all = _centerline_dist_mm(pts_mm)
    def _norm01(x):
        xmax = float(np.max(x)) if np.size(x) else 0.0
        return (x / (xmax + 1e-8)) if xmax > 0 else np.zeros_like(x)

    img_evid = np.array([_mean_prob_along(curr, p, arr_s, spacing, origin, M, size_xyz) for p in pts_mm], float)
    img_n = _norm01(img_evid)


    p_edge_a = np.array([_edge_prior_directional(curr, p) for p in pts_mm], float)

    # angle penalty
    ang_deg_all = []
    r_eff_all   = []
    for p in pts_mm:
        du = _unit(p - curr)
        dotv = float(np.clip(np.dot(old, du), -1.0, 1.0))
        if dotv < cos_thr:
            ang_deg_all.append(180.0)
        else:
            ang_deg_all.append(float(np.degrees(np.arccos(dotv))))
        # radius fusion
        r_eff = float(radius_mm)
        nV_next = _nearest_node(vol_index["kd"], vol_index["ids"], p)
        if nV_next is not None:
            rv = float(vol_index["G"].nodes[nV_next].get('radius',
                   vol_index["G"].nodes[nV_next].get('MaximumInscribedSphereRadius', r_eff)))
            r_eff = 0.7 * r_eff + 0.3 * rv
        r_eff_all.append(max(r_eff, MIN_RADIUS) + ADD_RADIUS)

    ang_deg_all = np.asarray(ang_deg_all, float)
    r_eff_all   = np.asarray(r_eff_all, float)

    f_near_cl  = np.clip(1.0 / d_cl_all,0,1)
    ang_n = ang_deg_all / max(1e-6, ANGLE_ALLOW_DEG)
    ang_n = np.clip(ang_n, 0.0, 1.0)


    # linear score
    S = (W_IMG * img_n
         + W_EDGE * p_edge_a
         + W_NEAR_CL * f_near_cl
         - W_ANGLE * ang_n)

    # forward-angle gate
    keep = ang_deg_all <= ANGLE_ALLOW_DEG
    keep &= (S >= SCORE_MIN)

    if not np.any(keep):
        return (np.empty((0,3)),) * 4

    cand_pts = pts_mm[keep]
    cand_r   = r_eff_all[keep]
    cand_ang = ang_deg_all[keep]
    S        = S[keep]

    order = np.argsort(-S)
    if k_best is not None:
        order = order[:int(k_best)]

    return cand_pts[order], cand_r[order], cand_ang[order], S[order]


def trace_centerline(
    output_folder: str,
    image_file: str,
    case: str,
    model_folder: str,
    fold: int,
    *,
    graph: nx.Graph,
    seed_node: int,
    target_nodes: List[int],
    centerline_graph: nx.Graph,
    max_steps_per_component: int = 500,
    global_config: Optional[dict] = None,
    unit: str = 'cm',
    scale: float = 1.0,
    seg_file: Optional[str] = None,
    start_seg: Optional[sitk.Image] = None,
    write_samples: bool = False,
):
    assert global_config is not None, "global_config (YAML dict) is required"

    if write_samples:
        os.makedirs(os.path.join(output_folder, "qa_meshes"), exist_ok=True)

    # units
    scale_unit = 0.1 if unit == 'cm' else 1.0

    # config
    cfg = dict(global_config)
    SEGMENTATION = bool(cfg.get('SEGMENTATION', False))
    DEBUG = bool(cfg.get('DEBUG', False))
    DEBUG_STEP = int(cfg.get('DEBUG_STEP', 0))

    VOLUME_SIZE_RATIO = float(cfg.get('VOLUME_SIZE_RATIO', 2.0))
    MAGN_RADIUS = float(cfg.get('MAGN_RADIUS', 0.5))
    ADD_RADIUS = float(cfg.get('ADD_RADIUS', 1)) * scale_unit
    MIN_RES = int(cfg.get('MIN_RES', 8))

    # Voronoi mode (soft preference only)
    VOR_ENABLE = bool(cfg.get('VOR_ENABLE', True))

    # auto re-seed / stall handling (global)
    STALL_MIN_NEW_COVER = int(cfg.get('STALL_MIN_NEW_COVER', 3))
    STALL_MAX_STEPS     = int(cfg.get('STALL_MAX_STEPS', 7))
    RESEED_MIN_SEP_MM   = float(cfg.get('RESEED_MIN_SEP_MM', 6))
    RESEED_POLICY       = cfg.get('RESEED_POLICY', 'farthest')
    RESEED_ALPHA_PROB   = float(cfg.get('RESEED_ALPHA_PROB', 0.4))
    RESEED_BETA_RADIUS  = float(cfg.get('RESEED_BETA_RADIUS', 0.8))
    RESEED_GAMMA_DIST   = float(cfg.get('RESEED_GAMMA_DIST', 0.4))
    RESEED_MIN_DEGREE   = int(cfg.get('RESEED_MIN_DEGREE', 2))
    RESEED_TARGETS_FIRST = bool(cfg.get('RESEED_TARGETS_FIRST', False))

    # coverage stop
    COVERAGE_STOP = float(cfg.get('COVERAGE_STOP', 0.99))
    STOP_WHEN_ALL_TARGETS = bool(cfg.get('STOP_WHEN_ALL_TARGETS', False))
    MAX_RESEED_NO_GAIN = int(cfg.get('MAX_RESEED_NO_GAIN', 50))

    # ---- new cfg knobs (with defaults) ----
    CROSS_VOR_K = int(cfg.get('CROSS_VOR_K', 8))
    CELL_STALL_MIN_NEW = int(cfg.get('CELL_STALL_MIN_NEW', 5))
    CELL_STALL_MAX_STEPS = int(cfg.get('CELL_STALL_MAX_STEPS', 15))
    CELL_COVERAGE_STOP = float(cfg.get('CELL_COVERAGE_STOP', 0.95))

    # image IO
    if SEGMENTATION and seg_file:
        print("\nUsing provided segmentation file for tracing:", seg_file)
        reader_im, origin_im, size_im, spacing_im = import_image(seg_file)
        image_file_effective = seg_file
    else:
        print(f"Reading image for tracing: {image_file}, scale: {scale}")
        reader_im, origin_im, size_im, spacing_im = import_image(image_file)
        image_file_effective = image_file

    try:
        geom_img = sitk.ReadImage(image_file_effective)
    except Exception:
        geom_img = sitk.Image(int(size_im[0]), int(size_im[1]), int(size_im[2]), sitk.sitkUInt8)
        geom_img.SetOrigin(tuple(map(float, origin_im)))
        geom_img.SetSpacing(tuple(map(float, spacing_im)))
        geom_img.SetDirection(tuple([1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]))

    dir_mat = np.array(geom_img.GetDirection(), float).reshape(3, 3)
    det_dir = float(np.linalg.det(dir_mat))
    print(
        f"[DBG reader_im] size={list(map(np.int64, size_im))}, spacing={list(map(np.float64, spacing_im))}, "
        f"origin={list(map(np.float64, origin_im))}, dir_row0={[round(v, 3) for v in dir_mat[0]]} det(dir)={det_dir:.6f}"
    )

    ensure_node_probability(graph, node_prob_key="node_prob", edge_prob_key="edge_prob")
    smooth_node_probability(graph, key_in="node_prob", key_out="node_prob_smooth", beta=0.4, iters=1)

    # predictor
    predictor = initialize_predictor(model_folder, fold)
    print('predictor initialized')

    #vesseltree

    vt = VesselTree(
        case=case,
        image_file=image_file_effective,
        seed_id=int(seed_node),
        target_ids=list(map(int, target_nodes)),
        graph=graph,
        centerline_pd=None,
        centerline_graph=centerline_graph,
    )
    vt.geom_img = geom_img

    Gcov = centerline_graph
    vt_init_coverage(vt, Gcov)

    # Global graph-Voronoi
    vt._gv_assign = vt._gv_cell2nodes = vt._gv_cell_adj = None
    try:
        assign, c2n, cadj = _build_graph_voronoi(Gcov, seeds=None, weight=None)
        vt._gv_assign, vt._gv_cell2nodes, vt._gv_cell_adj = assign, c2n, cadj
    except Exception:
        vt._gv_assign = vt._gv_cell2nodes = vt._gv_cell_adj = None

    # Global assembly
    assembly_segs = Segmentation(
        case,
        image_file_effective,
        weighted=bool(cfg.get('WEIGHTED_ASSEMBLY', False)),
        weight_type=cfg.get('WEIGHT_TYPE', 'radius'),
        start_seg=start_seg
    )

    # Candidate KD indices
    cent_index = _make_index(centerline_graph)
    vol_index = _make_index(graph)


    assign_v2c, cell_c2v = build_cross_voronoi_nx(centerline_graph, graph, k=CROSS_VOR_K)
    cell_pts, cell_ids, cell_mask, cell_kdt = init_cell_cov_structs(graph, cell_c2v)

    vt._last_cov_cent_id = int(seed_node)
    vt._cell_no_gain_streak = 0

    list_surfaces: List = []
    list_points: List[np.ndarray] = []
    list_inside_points: List[np.ndarray] = []
    list_centerlines: List[np.ndarray] = []

    no_cover_streak = 0
    reseed_no_gain_streak = 0

    # Seed step
    seed_pt = _to_np(Gcov.nodes[int(seed_node)]['point'])
    seed_r = float(Gcov.nodes[int(seed_node)].get('radius', Gcov.nodes[int(seed_node)].get('MaximumInscribedSphereRadius', 1.0)))
    vt.steps = []
    _append_seed_step(vt, seed_pt, seed_r, tangent_hint=np.array([1.0, 0.0, 0.0]))
    list_points.append(seed_pt)

    current_cell_id: Optional[int] = None
    cell_streak_count: int = 0

    i = 1
    while i <= max_steps_per_component:
        step = vt.steps[-1]
        curr_pt = step['point']
        old_pt = step['old point']
        curr_rad = step['radius']

        if DEBUG and i >= DEBUG_STEP:
            import pdb; pdb.set_trace()

        print(f"\n[DBG step] i={i} curr_r={curr_rad:.3f} curr_pt={_to_np(curr_pt)} old_pt={_to_np(old_pt)}")


        try:
            idx_clamped, size_clamped, border_flag = map_to_image(
                center_phys=curr_pt,
                box_radius_mm=(curr_rad + ADD_RADIUS) * MAGN_RADIUS,
                volume_size_ratio=VOLUME_SIZE_RATIO,
                image=geom_img,
                min_res=MIN_RES,
            )
            print(f"[DBG roi] index={tuple(idx_clamped)}, size={tuple(size_clamped)}, border={border_flag}")
            cropped_vol = extract_volume(reader_im, idx_clamped, size_clamped)

            # Build probability image for this crop
            if predictor is not None:
                spacing_pred_vec = (np.asarray(geom_img.GetSpacing(), float) * float(scale)).tolist()[::-1]
                img_np = sitk.GetArrayFromImage(cropped_vol)[None].astype('float32')
                t0 = time.time()
                pred = predictor.predict_single_npy_array(img_np, {'spacing': spacing_pred_vec}, None, None, True)
                prob_arr = np.clip(pred[1][1], 0, 1).astype(np.float32)
                prob_prediction = sitk.GetImageFromArray(prob_arr)
                pred_img = sitk.GetImageFromArray((pred[0] > 0).astype(np.uint8))
                pred_img = copy_settings(pred_img, cropped_vol)
                prob_prediction = copy_settings(prob_prediction, cropped_vol)
                print(f"[DBG pred] forward_time={time.time() - t0:.3f}s")
            else:
                arr = sitk.GetArrayFromImage(cropped_vol).astype(np.float32)
                amax = float(arr.max()); amin = float(arr.min())
                if amax > amin:
                    arr = (arr - amin) / (amax - amin)
                else:
                    arr = np.zeros_like(arr, dtype=np.float32)
                prob_prediction = copy_settings(sitk.GetImageFromArray(arr), cropped_vol)
                pred_img = cropped_vol

            # prune islands
            try:
                seed_vox = (np.rint(np.array(size_clamped) / 2).astype(int)).tolist()
                pred_img = remove_other_vessels(pred_img, seed_vox)
            except Exception as e:
                print(f"[WARN prune] remove_other_vessels failed: {e}")

            # QA surface (optional)
            if write_samples:
                out_npz = os.path.join(output_folder, "qa_meshes", f"qa_mesh_step_{case}_{i:05d}.npz")
                qa = _qa_surface_from_crop(prob_prediction, level=0.5, write_path_npz=out_npz)
                if qa is not None:
                    list_surfaces.append(qa)

            # global assembly
            assembly_segs.add_segmentation(
                prob_prediction, idx_clamped, size_clamped,
                (1.0 / max(curr_rad, 1e-3)) ** 2 if cfg.get('WEIGHTED_ASSEMBLY', False) else curr_rad
            )

            # quick inside/outside heuristic at current point
            try:
                ci = _phys_to_idx(prob_prediction, curr_pt)
                vx, vy, vz = [int(round(v)) for v in ci]
                arrp = sitk.GetArrayFromImage(prob_prediction)
                if 0 <= vz < arrp.shape[0] and 0 <= vy < arrp.shape[1] and 0 <= vx < arrp.shape[2]:
                    if float(arrp[vz, vy, vx]) >= 0.5:
                        list_inside_points.append(curr_pt)
            except Exception:
                pass

            if check_seg_border(size_clamped, idx_clamped, pred_img, size_im):
                print("[DBG stop] segmentation crop touched global border → stop component")
                break

        except Exception as e:
            msg = str(e).lower()
            if 'invalid index' in msg:
                print("[DBG] invalid index during crop/predict; proceeding to decision.")
                prob_prediction = None
                pred_img = None
            else:
                print(f"[ERR step {i}] crop/predict: {e}")
                prob_prediction = None
                pred_img = None

        # --- Build Voronoi for this ROI (if enabled) ---
        vor_ctx = None
        cell2nodes: Dict[int, Set[int]] = {}
        vor_adj_edges: Set[Tuple[int, int]] = set()
        if VOR_ENABLE and pred_img is not None:
            try:
                vor_label_img, vor_lut_ids = edt_for_pred(Gcov, pred_img)
                if vor_label_img is not None and vor_lut_ids:
                    cell2nodes = roi_cell_index(Gcov, vor_label_img, vor_lut_ids)
                    for cid, nodes in cell2nodes.items():
                        vt_cell = cell2nodes.get(cid, set())
                        vt_cell.update(nodes)
                    vor_adj_edges = roi_cell_adjacency(vor_label_img, vor_lut_ids)
                    vor_ctx = {"label_img": vor_label_img, "lut_ids": vor_lut_ids, "allowed_cells": None}
            except Exception as e:
                print(f"[WARN] Voronoi per-ROI failed: {e}")
                vor_ctx = None

        # Identify current Voronoi cell (if available)
        if VOR_ENABLE and vor_ctx is not None and vor_ctx.get("label_img") is not None and vor_ctx.get("lut_ids"):
            cid_now = vor_cell_id_at_point(vor_ctx["label_img"], vor_ctx["lut_ids"], curr_pt)
            if cid_now is not None and cid_now != current_cell_id:
                current_cell_id = cid_now
                cell_streak_count = 0
            # soft leash: same or adjacent
            allowed = set()
            if current_cell_id is not None:
                allowed.add(current_cell_id)
                for a, b in vor_adj_edges:
                    if a == current_cell_id: allowed.add(b)
                    elif b == current_cell_id: allowed.add(a)
            if allowed:
                vor_ctx["allowed_cells"] = allowed

        # ---- Decide next move ----
        try:
            arr_pt, arr_rad, arr_ang, arr_gscore = get_next_points_from_crop_skeleton(
                prob_img=prob_prediction,
                curr_mm=curr_pt,
                prev_mm=old_pt,
                radius_mm=curr_rad,
                cent_index=cent_index,
                vol_index=vol_index,
                cfg=cfg,
                k_best=20,
            )

            # if no local candidates → try CELL-RESTRICTED rescue first
            if arr_pt is None or arr_pt.size == 0:
                print("[DBG graph] no local candidates; trying cell-restricted next-point.")
                nodes_in_cell = cell_c2v.get(int(vt._last_cov_cent_id), set())
                if nodes_in_cell:
                    cr_pt, cr_r, cr_ang, cr_sc = get_next_points_for_nodes(
                        prob_img=prob_prediction,
                        curr_mm=curr_pt, prev_mm=old_pt, radius_mm=curr_rad,
                        candidate_node_ids=nodes_in_cell,
                        graph_vol=graph, vol_index=vol_index,
                        cfg=cfg, k_best=CROSS_VOR_K
                    )
                else:
                    cr_pt = np.empty((0, CROSS_VOR_K))

                if cr_pt is not None and cr_pt.size > 0:
                    nxt_p = _snap_phys_inside(geom_img, cr_pt[0])
                    nxt_r = float(cr_r[0])
                    tangent = _unit(nxt_p - curr_pt)
                    vt.steps.append(
                        _mk_step_dict(curr_pt, curr_rad, nxt_p, nxt_r, tangent, angle_change=float(cr_ang[0])))
                    list_points.append(nxt_p)

                    # update global coverage
                    newly = vt_mark_covered_by_segment_ball(vt, Gcov, curr_pt, nxt_p, radius_scale=2)
                    cov = vt_coverage_ratio(vt)
                    print(
                        f"[COVER] cell-rescue step +{newly}, total={vt._cov_mask.sum()}/{vt._cov_mask.size} ({100 * cov:.1f}%)")

                    if newly > 0:
                        update_last_cov_cent_anchor(vt, nxt_p)
                        no_cover_streak = 0
                        reseed_no_gain_streak = 0
                    else:
                        no_cover_streak += 1
                        reseed_no_gain_streak += 1

                    # also update cell coverage
                    cid = int(vt._last_cov_cent_id)
                    newly_cell = mark_cell_cover(cid, curr_pt, nxt_p,
                                                 cell_pts=cell_pts, cell_ids=cell_ids,
                                                 cell_mask=cell_mask, cell_kdt=cell_kdt, Gvol=graph)
                    cov_cell = coverage_for_cell(cid, cell_mask)

                    print(
                        f"[COVER cell {cid}] +{newly_cell}, {cell_mask[cid].sum()}/{cell_mask[cid].size} ({100 * cov_cell:.1f}%)")

                    if newly_cell < CELL_STALL_MIN_NEW:
                        vt._cell_no_gain_streak += 1
                    else:
                        vt._cell_no_gain_streak = 0

                    i += 1
                    continue

                # if cell rescue failed → fall back to GLOBAL reseed
                print("[DBG graph] cell rescue failed; trying GLOBAL reseed.")
                jump_pt = _pick_reseed_node(
                    vt, Gcov, curr_pt,
                    prefer=RESEED_POLICY,
                    min_sep_mm=RESEED_MIN_SEP_MM,
                    alpha_prob=RESEED_ALPHA_PROB,
                    beta_radius=RESEED_BETA_RADIUS,
                    gamma_dist=RESEED_GAMMA_DIST,
                    min_degree=RESEED_MIN_DEGREE,
                    targets_first=RESEED_TARGETS_FIRST,
                )
                if jump_pt is None:
                    print("[DBG re-seed] no valid re-seed found; stopping.")
                    break
                jump_pt = _snap_phys_inside(geom_img, jump_pt)
                _append_seed_step(vt, jump_pt, curr_rad, tangent_hint=step['tangent'])
                list_points.append(jump_pt)
                newly = vt_mark_covered_by_segment_ball(vt, Gcov, curr_pt, jump_pt, radius_scale=2)
                if newly > 0:
                    update_last_cov_cent_anchor(vt, jump_pt)
                i += 1
                continue

            # --- Normal step along best candidate ---
            print(
                f"[DBG graph] candidates k={len(arr_rad)}, "
                f"best_radius={float(arr_rad[0]):.3f}, best_angle={float(arr_ang[0]):.1f}"
            )
            nxt_p = _snap_phys_inside(geom_img, arr_pt[0])
            nxt_r = float(arr_rad[0])
            tangent = _unit(nxt_p - curr_pt)
            next_step = _mk_step_dict(curr_pt, curr_rad, nxt_p, nxt_r, tangent, angle_change=float(arr_ang[0]))
            vt.steps.append(next_step)
            list_points.append(nxt_p)

            newly = vt_mark_covered_by_segment_ball(vt, Gcov, curr_pt, nxt_p, radius_scale=2)
            cov = vt_coverage_ratio(vt)
            targets_left = len(_targets_remaining(vt))
            print(f"[COVER] step +{newly}, total={vt._cov_mask.sum()}/{vt._cov_mask.size} ({100*cov:.1f}%), targets_left={targets_left}")

            if newly < STALL_MIN_NEW_COVER:
                no_cover_streak += 1
                print(f"[STALL] low new coverage ({newly}); streak={no_cover_streak}/{STALL_MAX_STEPS}")
            else:
                no_cover_streak = 0
                reseed_no_gain_streak = 0

            # update last covered centerline anchor if any gain
            if newly > 0:
                update_last_cov_cent_anchor(vt, nxt_p)

            # cell coverage
            cid = int(vt._last_cov_cent_id)
            newly_cell = mark_cell_cover(cid, curr_pt, nxt_p,
                                         cell_pts=cell_pts, cell_ids=cell_ids,
                                         cell_mask=cell_mask, cell_kdt=cell_kdt, Gvol=graph)
            cov_cell = coverage_for_cell(cid, cell_mask)

            print(f"[COVER cell {cid}] +{newly_cell}, {cell_mask[cid].sum()}/{cell_mask[cid].size} ({100*cov_cell:.1f}%)")

            if newly_cell < CELL_STALL_MIN_NEW:
                vt._cell_no_gain_streak += 1
            else:
                vt._cell_no_gain_streak = 0

            # re-seed on stall (global)
            if no_cover_streak >= STALL_MAX_STEPS:
                print("[STALL] triggering auto GLOBAL re-seed due to coverage stall.")
                jump_pt = _pick_reseed_node(
                    vt, Gcov, nxt_p,
                    prefer=RESEED_POLICY,
                    min_sep_mm=RESEED_MIN_SEP_MM,
                    alpha_prob=RESEED_ALPHA_PROB,
                    beta_radius=RESEED_BETA_RADIUS,
                    gamma_dist=RESEED_GAMMA_DIST,
                    min_degree=RESEED_MIN_DEGREE,
                    targets_first=RESEED_TARGETS_FIRST,
                )
                if jump_pt is not None:
                    jump_pt = _snap_phys_inside(geom_img, jump_pt)
                    _append_seed_step(vt, jump_pt, nxt_r, tangent_hint=tangent)
                    list_points.append(jump_pt)

                    newly2 = vt_mark_covered_by_segment_ball(vt, Gcov, nxt_p, jump_pt, radius_scale=2)
                    cov2 = vt_coverage_ratio(vt)
                    print(f"[COVER] stall re-seed +{newly2}, total={vt._cov_mask.sum()}/{vt._cov_mask.size} ({100*cov2:.1f}%)")

                    if newly2 <= 0:
                        reseed_no_gain_streak += 1
                    else:
                        reseed_no_gain_streak = 0

                    if reseed_no_gain_streak >= MAX_RESEED_NO_GAIN:
                        print(f"[STOP] {reseed_no_gain_streak} consecutive re-seeds with no coverage; stopping component.")
                        break

                    no_cover_streak = 0
                    i += 1
                    continue

            # escalate to next
            if cov_cell >= CELL_COVERAGE_STOP or vt._cell_no_gain_streak >= CELL_STALL_MAX_STEPS:
                print("[CELL GATE] escalating to GLOBAL reseed.")
                jump_pt = _pick_reseed_node(
                    vt, Gcov, nxt_p,
                    prefer=RESEED_POLICY,
                    min_sep_mm=RESEED_MIN_SEP_MM,
                    alpha_prob=RESEED_ALPHA_PROB,
                    beta_radius=RESEED_BETA_RADIUS,
                    gamma_dist=RESEED_GAMMA_DIST,
                    min_degree=RESEED_MIN_DEGREE,
                    targets_first=RESEED_TARGETS_FIRST,
                )
                if jump_pt is None:
                    print("[DBG re-seed] no valid re-seed found; stopping component.")
                    break
                jump_pt = _snap_phys_inside(geom_img, jump_pt)
                _append_seed_step(vt, jump_pt, nxt_r, tangent_hint=tangent)
                list_points.append(jump_pt)
                vt._cell_no_gain_streak = 0
                newly2 = vt_mark_covered_by_segment_ball(vt, Gcov, nxt_p, jump_pt, radius_scale=2)
                if newly2 > 0:
                    update_last_cov_cent_anchor(vt, jump_pt)
                i += 1
                continue

            # stop conditions
            if STOP_WHEN_ALL_TARGETS and len(_targets_remaining(vt)) == 0:
                print("[STOP] all targets reached; stopping component.")
                break
            if cov >= COVERAGE_STOP or vt._cov_mask.sum() >= vt._cov_mask.size:
                print(f"[STOP] coverage {100*cov:.1f}% ≥ {100*COVERAGE_STOP:.0f}% or all nodes covered; stopping.")
                break

            # voronoi cell streak
            if VOR_ENABLE and vor_ctx is not None and vor_ctx.get("label_img") is not None and vor_ctx.get("lut_ids"):
                new_cell = vor_cell_id_at_point(vor_ctx["label_img"], vor_ctx["lut_ids"], nxt_p)
                if new_cell == current_cell_id:
                    cell_streak_count += 1
                else:
                    current_cell_id = new_cell
                    cell_streak_count = 0

            i += 1

        except Exception as e:
            print(f"[ERR step {i}] decision: {e}")
            try:
                tangent = _unit(_to_np(curr_pt) - _to_np(old_pt))
                nxt_p = _snap_phys_inside(geom_img, _to_np(curr_pt) + max(curr_rad, 0.5) * tangent)
                vt.steps.append(_mk_step_dict(curr_pt, curr_rad, nxt_p, curr_rad, tangent, angle_change=0.0))
                list_points.append(nxt_p)
                newly = vt_mark_covered_by_segment_ball(vt, Gcov, curr_pt, nxt_p, radius_scale=2)
                cov = vt_coverage_ratio(vt)
                print(f"[COVER] nudge +{newly}, total={vt._cov_mask.sum()}/{vt._cov_mask.size} ({100*cov:.1f}%)")
                if (STOP_WHEN_ALL_TARGETS and len(_targets_remaining(vt)) == 0) or cov >= COVERAGE_STOP:
                    print("[STOP] targets reached or coverage limit hit after nudge; stopping.")
                    break
                if newly > 0:
                    reseed_no_gain_streak = 0
                i += 1
            except Exception:
                break

    if len(vt.steps) > 0:
        poly = np.vstack([_to_np(s['point']) for s in vt.steps])
        list_centerlines.append(poly)

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return (
        list_centerlines,
        list_surfaces,
        list_points,
        list_inside_points,
        assembly_segs,
        vt,
        i,
    )
