
import os, time
from typing import List, Tuple, Optional, Dict, Set
from scipy.interpolate import RegularGridInterpolator

import numpy as np
import networkx as nx
import SimpleITK as sitk
import skfmm

from scipy.spatial import cKDTree as KDTree
from scipy.ndimage import (
    gaussian_filter, map_coordinates, distance_transform_edt,
    binary_opening, binary_closing, generate_binary_structure
)
from skimage.morphology import skeletonize
from skimage.measure import marching_cubes

# --- your existing utils (unchanged) ---
from SeqSeg.seqseg.modules.sitk_functions import (
    import_image, extract_volume, copy_settings,
    remove_other_vessels, check_seg_border
)
from SeqSeg.seqseg.modules.nnunet import initialize_predictor
from SeqSeg.seqseg.modules.assembly import Segmentation
from seqseg_modules_modified.assembly import VesselTree


# ======================== Small helpers ========================
def _to_np(p) -> np.ndarray:
    return np.asarray(p, dtype=float)

def _unit(v) -> np.ndarray:
    v = _to_np(v); n = float(np.linalg.norm(v))
    return v / (n + 1e-12)

def _phys_tuple(p):
    a = _to_np(p).reshape(3)
    return (float(a[0]), float(a[1]), float(a[2]))

def _build_kdtree(G: nx.Graph):
    pts, ids = [], []
    for n, d in G.nodes(data=True):
        p = d.get('pos_phys', d.get('point'))
        if p is None: continue
        pts.append(_to_np(p)); ids.append(int(n))
    if not pts: return None, []
    return KDTree(np.vstack(pts)), ids

def _make_index(G: nx.Graph) -> Dict[str, object]:
    kd, ids = _build_kdtree(G)
    return {"G": G, "kd": kd, "ids": ids}

def _nearest_node(kdt: Optional[KDTree], ids: List[int], point) -> Optional[int]:
    if kdt is None: return None
    _, idx = kdt.query(_to_np(point), k=1)
    return int(ids[int(idx)])

def _snap_phys_inside(image: sitk.Image, p_xyz: np.ndarray) -> np.ndarray:
    ci = np.array(image.TransformPhysicalPointToContinuousIndex(_phys_tuple(p_xyz)))
    sz = np.array(image.GetSize(), int)
    ci = np.clip(ci, 0, sz - 1)
    return np.array(image.TransformContinuousIndexToPhysicalPoint(tuple(ci.tolist())))

def map_to_image(center_phys, box_radius_mm: float, volume_size_ratio: float, *,
                 image: sitk.Image, min_res: int = 8, require_odd: bool = True) -> Tuple[list[int], list[int], bool]:
    ci = np.array(image.TransformPhysicalPointToContinuousIndex(_phys_tuple(center_phys)))
    img_size = np.array(list(image.GetSize()), dtype=int)
    spacing  = np.array(list(image.GetSpacing()), dtype=float)

    if not np.all(np.isfinite(ci)):
        ci = (img_size - 1) / 2.0

    L_mm = max(1e-3, float(volume_size_ratio) * float(box_radius_mm))
    size_vox = np.ceil(L_mm / np.maximum(spacing, 1e-12)).astype(int)
    size_vox = np.maximum(size_vox, int(min_res))
    if require_odd: size_vox = size_vox + (size_vox % 2 == 0)
    if np.any(size_vox > img_size): size_vox = np.minimum(size_vox, img_size)

    start = np.floor(ci - 0.5 * size_vox).astype(int)
    start_clamped = np.maximum(0, np.minimum(start, img_size - size_vox))
    border = bool(np.any(start_clamped != start) or np.any(size_vox == img_size))
    size_vox = np.maximum(size_vox, 1)
    return start_clamped.tolist(), size_vox.tolist(), border


# =================== Coverage (Gcent only) ===================
def vt_init_coverage(vt: VesselTree, Gcent: nx.Graph):
    ids, pts, radii = [], [], []
    for n, d in Gcent.nodes(data=True):
        p = d.get("pos_phys", d.get("point"))
        if p is None: continue
        ids.append(int(n)); pts.append(_to_np(p))
        radii.append(float(d.get("radius_mm", d.get("MaximumInscribedSphereRadius", 0.1))))
    vt._cov_ids = ids
    vt._cov_pts = np.vstack(pts) if pts else np.zeros((0, 3), float)
    vt._cov_r   = np.asarray(radii, float) if radii else np.zeros((0,), float)
    vt._cov_kdt = KDTree(vt._cov_pts) if len(vt._cov_pts) else None
    vt._cov_mask = np.zeros(len(vt._cov_ids), dtype=bool)
    vt._cov_id2idx = {nid: i for i, nid in enumerate(vt._cov_ids)}
    vt.node_traversed = set()
    vt.node_not_traversed = set(vt._cov_ids)

def vt_mark_covered_by_segment_ball(
    vt: VesselTree,
    Gcent: nx.Graph,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    radius_scale: float = 1.5,
    sample_stride_mm: float = 1.5,
) -> int:
    if vt._cov_kdt is None or len(vt._cov_ids) == 0: return 0
    p0 = _to_np(p0); p1 = _to_np(p1)
    ab = p1 - p0; L = float(np.linalg.norm(ab))
    if L < 1e-6:
        _, j = vt._cov_kdt.query(p0, k=1); j = int(j)
        if not vt._cov_mask[j]:
            vt._cov_mask[j] = True
            nid = vt._cov_ids[j]
            vt.node_traversed.add(nid); vt.node_not_traversed.discard(nid)
            return 1
        return 0

    # local radius estimate
    idx0 = vt._cov_kdt.query_ball_point(p0, r=6.0) or []
    idx1 = vt._cov_kdt.query_ball_point(p1, r=6.0) or []
    loc  = np.unique(np.asarray(idx0 + idx1, dtype=int)) if (idx0 or idx1) else np.array([], int)
    if loc.size:
        r_local = float(np.median(vt._cov_r[loc]))
    else:
        r_local = float(np.median(vt._cov_r)) if vt._cov_r.size else 1.0
    rcap = radius_scale * max(min(r_local, 5.0), 0.5)

    n_samp  = max(1, int(np.ceil(L / max(sample_stride_mm, 1e-6))))
    centers = p0 + (np.linspace(0.0, 1.0, n_samp + 1)[:, None] * ab)

    cand_idx: Set[int] = set()
    for c in centers:
        idxs = vt._cov_kdt.query_ball_point(c, r=rcap)
        if isinstance(idxs, int): idxs = [idxs]
        cand_idx.update(map(int, idxs))
    if not cand_idx: return 0

    idx_arr = np.fromiter(cand_idx, dtype=int)
    P = vt._cov_pts[idx_arr]
    ab2 = float(np.dot(ab, ab))
    t = np.clip(((P - p0) @ ab) / ab2, 0.0, 1.0)
    proj = p0 + t[:, None] * ab
    dist = np.linalg.norm(P - proj, axis=1)

    ok = dist <= rcap
    new_idx = idx_arr[ok & (~vt._cov_mask[idx_arr])]
    if new_idx.size == 0: return 0

    vt._cov_mask[new_idx] = True
    for j in new_idx:
        nid = vt._cov_ids[int(j)]
        vt.node_traversed.add(nid); vt.node_not_traversed.discard(nid)
    return int(new_idx.size)

def vt_coverage_ratio(vt: VesselTree) -> float:
    if vt._cov_mask is None or vt._cov_mask.size == 0: return 0.0
    return float(vt._cov_mask.sum()) / float(max(1, vt._cov_mask.size))


# ==================== Gcent backtrack polyline ====================
def _ensure_edge_length(G: nx.Graph, key: str = "length_mm") -> str:
    need = any(key not in d for _, _, d in G.edges(data=True))
    if not need: return key
    for u, v, d in G.edges(data=True):
        pu = _to_np(G.nodes[u].get('pos_phys', G.nodes[u].get('point')))
        pv = _to_np(G.nodes[v].get('pos_phys', G.nodes[v].get('point')))
        d[key] = float(np.linalg.norm(pu - pv)) if pu is not None and pv is not None else float(d.get('weight', 1.0))
    return key



# need fixing
def backtrack_gcent_poly(curr_pt: np.ndarray,
                         nxt_pt: np.ndarray,
                         cent_index: Dict[str, object],
                         Gcent: nx.Graph,
                         weight_key: str = "length_mm") -> Optional[np.ndarray]:
    kd, ids = cent_index.get("kd"), cent_index.get("ids")
    if kd is None or not ids: return None
    u = _nearest_node(kd, ids, curr_pt)
    v = _nearest_node(kd, ids, nxt_pt)
    if u is None or v is None or u == v:
        return np.vstack([_to_np(curr_pt), _to_np(nxt_pt)])

    w = _ensure_edge_length(Gcent, weight_key)
    try:
        path = nx.shortest_path(Gcent, source=u, target=v, weight=w)
    except Exception:
        return np.vstack([_to_np(curr_pt), _to_np(nxt_pt)])

    pts = [_to_np(curr_pt)]
    for nid in path:
        p = Gcent.nodes[nid].get('pos_phys', Gcent.nodes[nid].get('point'))
        if p is not None:
            pts.append(_to_np(p))
    pts.append(_to_np(nxt_pt))
    out = [pts[0]]
    for k in range(1, len(pts)):
        if np.linalg.norm(pts[k] - out[-1]) > 1e-6:
            out.append(pts[k])
    return np.vstack(out) if len(out) else None


# ==================== Candidate: Medial-axis crop ====================
# Rewritten module: fixes centerline backtracking, ensures connected/smooth centerlines,
# uses skfmm/backtracking + smoothing for local crop-derived picks (cent),
# uses Dijkstra shortest-path on Gvol for graph-derived picks (gvol).
#
# Most of the original file is preserved; only the implementations of:
#   - get_next_points_from_crop_centerline
#   - get_next_points_from_gvol
#
# were replaced to correct axis ordering, produce connected paths, and smooth results.
#
# Note: this file assumes the same imports, helpers and environment as your original code:
# numpy as np, networkx as nx, vtk, sitk, skfmm, RegularGridInterpolator, KDTree, etc.
# It reuses your helper functions: _to_np, _unit, _nearest_node, _snap_phys_inside, _make_index, vt_mark_covered_by_segment_ball, vt_coverage_ratio, etc.

import numpy as np
import networkx as nx
import vtk
import skfmm
from typing import List, Tuple, Optional, Dict
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator
from skimage.morphology import skeletonize
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure

# --- Keep your other helper imports and declarations above this file content ---
# For brevity, we assume helpers like _to_np, _unit, _nearest_node, _snap_phys_inside, _make_index are present in scope.

# Small utility: simple spline smoothing along polyline (keeps endpoints)
def _smooth_polyline_spline_keep_ends(pts: np.ndarray, resample_step_mm: float = 0.5, s: Optional[float] = None, k: int = 3) -> np.ndarray:
    pts = np.asarray(pts, float)
    if pts.shape[0] < 3:
        return pts.copy()
    # parameterize by arc length
    diffs = np.diff(pts, axis=0)
    seglen = np.sqrt((diffs ** 2).sum(axis=1))
    t = np.concatenate(([0.0], np.cumsum(seglen)))
    if t[-1] <= 0:
        return pts.copy()
    t /= t[-1]
    try:
        tck, _ = splprep(pts.T, u=t, s=s if s is not None else max(1e-3, 0.01 * len(pts)), k=min(k, max(1, pts.shape[0]-1)))
    except Exception:
        return pts.copy()
    L = seglen.sum()
    m = max(int(np.ceil(L / resample_step_mm)), 2)
    u_new = np.linspace(0.0, 1.0, m)
    out = np.vstack(splev(u_new, tck)).T
    # enforce exact endpoints
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out

# Utility: reorder gradient sampled on (z,y,x) axes back into physical (x,y,z)
def _grad_sample_to_phys(interp_grad, sample_idx):
    # interp_grad: list of 3 interpolators in order [dT/dz, dT/dy, dT/dx]
    # sample_idx: array-like [z, y, x] (matching T array axes)
    gz = float(interp_grad[0](sample_idx))
    gy = float(interp_grad[1](sample_idx))
    gx = float(interp_grad[2](sample_idx))
    return np.array([gx, gy, gz], dtype=float)

# Simplified backtrack + unified get_next_points implementation
# Requires: numpy as np, skfmm, scipy.ndimage.distance_transform_edt, scipy.interpolate.RegularGridInterpolator,
#           scipy.interpolate.splprep/splev (for smoothing), networkx as nx, KDTree available in scope,
#           and your helpers: _to_np, _unit, _nearest_node, _snap_phys_inside

import numpy as np
import skfmm
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.interpolate import RegularGridInterpolator, splprep, splev
import networkx as nx

def _resample_spline_keep_ends(poly: np.ndarray, step_mm: float = 0.5, smooth: float = 0.0):
    poly = np.asarray(poly, float)
    if poly.shape[0] < 3:
        return poly.copy()
    # parameterize by arc length
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    if s[-1] <= 0:
        return poly.copy()
    t = s / s[-1]
    k = min(3, max(1, poly.shape[0]-1))
    try:
        tck, _ = splprep(poly.T, u=t, s=float(smooth), k=k)
        npts = max(2, int(np.ceil(s[-1] / max(1e-6, float(step_mm)))))
        u_new = np.linspace(0.0, 1.0, npts)
        out = np.vstack(splev(u_new, tck)).T
        out[0] = poly[0]; out[-1] = poly[-1]
        return out
    except Exception:
        return poly.copy()

# -------------------------
# Simplified backtrack helper used by the crop-based candidate function.
# (A minimal, robust wrapper around skfmm travel_time + gradient backtrack.)
# -------------------------
def simple_backtrack_fmm(prob_img: 'sitk.Image',
                         start_phys: np.ndarray,
                         end_phys: np.ndarray,
                         *,
                         bin_thr: float = 0.5,
                         leak_frac: float = 0.05,
                         step_mm: float = 0.5,
                         max_iters: int = 2048,
                         debug: bool = False) -> Optional[np.ndarray]:
    import SimpleITK as sitk
    import skfmm
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    from scipy.interpolate import RegularGridInterpolator
    import numpy as np

    if prob_img is None:
        return None

    # ensure sitk image
    if isinstance(prob_img, str):
        prob_img = sitk.ReadImage(prob_img)

    arr = sitk.GetArrayFromImage(prob_img).astype(np.float32)  # (z,y,x)
    if arr.size == 0:
        return None

    sp_xyz = np.asarray(prob_img.GetSpacing(), float)  # (x,y,z)
    sp_zyx = sp_xyz[::-1]  # (z,y,x)

    mask = arr >= float(bin_thr)
    if not mask.any():
        return None

    # EDT in mm (array axis order)
    try:
        edt = distance_transform_edt(mask, sampling=tuple(sp_zyx))
    except Exception:
        return None

    # speed field (inside vessels larger, outside small leak)
    inside_med = float(np.median(edt[mask])) if mask.any() else float(np.max(edt))
    outside_speed = max(1e-6, inside_med * float(leak_frac))
    speed = edt + 1e-6
    speed[~mask] = outside_speed
    speed = gaussian_filter(speed, sigma=1.0)

    # phi seed at start_phys (SITK cont idx in x,y,z -> convert to z,y,x for array)
    try:
        s_idx_xyz = np.asarray(prob_img.TransformPhysicalPointToContinuousIndex(tuple(map(float, start_phys))), float)
    except Exception:
        return None
    s_idx_zyx = s_idx_xyz[::-1]
    s_idx_round = np.clip(np.rint(s_idx_zyx).astype(int), 0, np.array(arr.shape) - 1)
    phi = np.ones_like(speed, dtype=float)
    phi[s_idx_round[0], s_idx_round[1], s_idx_round[2]] = -1.0

    # travel time
    try:
        T = skfmm.travel_time(phi, speed=speed, dx=tuple(sp_zyx), order=2)
    except Exception:
        return None

    # gradients (dT/dz, dT/dy, dT/dx)
    grads = np.gradient(T, *tuple(sp_zyx))
    grid_axes = [np.arange(n) for n in T.shape]
    interp_grad = [RegularGridInterpolator(grid_axes, g, bounds_error=False, fill_value=0.0) for g in grads]
    interp_T = RegularGridInterpolator(grid_axes, T, bounds_error=False, fill_value=np.nan)

    # start continuous index for end_phys
    try:
        e_idx_xyz = np.asarray(prob_img.TransformPhysicalPointToContinuousIndex(tuple(map(float, end_phys))), float)
    except Exception:
        return None
    cur = e_idx_xyz[::-1].astype(float)  # z,y,x

    # helper: continuous index (z,y,x) -> physical (x,y,z)
    def phys_from_zyx(zyx):
        x_idx, y_idx, z_idx = float(zyx[2]), float(zyx[1]), float(zyx[0])
        try:
            return np.asarray(prob_img.TransformContinuousIndexToPhysicalPoint((x_idx, y_idx, z_idx)), float)
        except Exception:
            origin = np.asarray(prob_img.GetOrigin(), float)
            direction = np.asarray(prob_img.GetDirection(), float).reshape(3, 3)
            sp = np.asarray(prob_img.GetSpacing(), float)
            idx_xyz = np.array([x_idx, y_idx, z_idx], float)
            return (direction @ (idx_xyz * sp)) + origin

    mean_sp = float(np.mean(sp_xyz))
    step_idx = float(step_mm) / max(1e-9, float(np.mean(sp_zyx)))  # approximate index units

    path_phys = [phys_from_zyx(cur)]

    for _it in range(int(max_iters)):
        # stop if physically near seed
        if np.linalg.norm(path_phys[-1] - np.asarray(start_phys, float)) <= mean_sp:
            path_phys.append(np.asarray(start_phys, float))
            break

        sample = np.array([cur[0], cur[1], cur[2]], float)
        try:
            gz = float(interp_grad[0](sample))
            gy = float(interp_grad[1](sample))
            gx = float(interp_grad[2](sample))
        except Exception:
            break

        # order to physical axis (x,y,z)
        grad_phys = np.array([gx, gy, gz], float)
        ng = np.linalg.norm(grad_phys)
        if not np.isfinite(ng) or ng < 1e-12:
            break
        dir_phys = -grad_phys / ng

        # convert dir_phys (x,y,z) to index-space delta (z,y,x)
        delta_idx_xyz = (dir_phys / sp_xyz)  # x_idx, y_idx, z_idx
        delta_idx_zyx = np.array([delta_idx_xyz[2], delta_idx_xyz[1], delta_idx_xyz[0]], float)
        nrm_idx = np.linalg.norm(delta_idx_zyx)
        if nrm_idx < 1e-12:
            break
        nxt = cur + (delta_idx_zyx / nrm_idx) * step_idx

        # bounds check
        if np.any(nxt < -1.0) or np.any(nxt > (np.array(arr.shape) + 1.0)):
            break

        # downhill check
        try:
            Tcur = float(interp_T(cur))
            Tnxt = float(interp_T(nxt))
        except Exception:
            break
        if (not np.isfinite(Tnxt)) or (Tnxt >= Tcur - 1e-9):
            break

        cur = nxt
        path_phys.append(phys_from_zyx(cur))

    if len(path_phys) <= 1:
        return None

    # return poly in order curr -> ... -> start (consistent with upstream expectation)
    poly = np.vstack(path_phys)
    return poly

# ---------------------------------------------------------------------
# get_next_points_from_crop_centerline (simplified backtracking)
# ---------------------------------------------------------------------
def get_next_points_from_crop_centerline(
    prob_img: Optional['sitk.Image'],
    curr_mm: np.ndarray,
    prev_mm: np.ndarray,
    radius_mm: float,
    *,
    cent_index: Dict[str, object],
    vol_index: Dict[str, object],
    cfg: Dict,
    k_best: int = 20,
    prev_picked: str = 'Gcent'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (pts, radii, ang_deg, score, centerline_poly)
    - pts: Nx3 candidate points in physical mm (ordered by score desc)
    - centerline_poly: connected polyline (curr -> candidate) for top candidate (or empty)
    Simplified and defensive. Uses simple_backtrack_fmm for polyline (skips heavy backtrack if prev_picked == 'global reseed').
    """

    import numpy as np
    import SimpleITK as sitk
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import KDTree
    from skimage.morphology import skeletonize

    # guards
    if prob_img is None:
        return (np.empty((0, 3)), np.empty((0,), float), np.empty((0,), float), np.empty((0,), float), np.empty((0, 3)))

    # config
    BIN_THR = float(cfg.get('BIN_THR', 0.5))
    ANGLE_ALLOW_DEG = float(cfg.get('ANGLE_ALLOW_DEG', 150.0))
    SCORE_MIN = float(cfg.get('SCORE_MIN', 0.25))
    MA_MIN_SEP_MM = float(cfg.get('MA_MIN_SEP_MM', 0.0))
    MA_HOPS = int(cfg.get('MA_HOPS', 2))
    MIN_STRIDE_MM = float(cfg.get('MIN_STRIDE_MM', max(0.6, 0.35 * radius_mm)))
    MAX_STRIDE_MM = float(cfg.get('MAX_STRIDE_MM', max(2.5, 2.0 * radius_mm)))
    W_NEAR_CL = float(cfg.get('W_NEAR_CL', 0.6))
    W_RADIUS  = float(cfg.get('W_RADIUS', 0.4))

    try:
        arr = sitk.GetArrayFromImage(prob_img).astype(np.float32)  # (z,y,x)
    except Exception:
        return (np.empty((0, 3)),) * 4 + (np.empty((0, 3)),)

    if arr.size == 0 or float(arr.max()) <= 0:
        return (np.empty((0, 3)),) * 4 + (np.empty((0, 3)),)

    mask = arr >= BIN_THR
    if not mask.any():
        return (np.empty((0, 3)),) * 4 + (np.empty((0, 3)),)

    spacing_xyz = np.asarray(prob_img.GetSpacing(), float)  # (x,y,z)
    spacing_zyx = spacing_xyz[::-1]

    # EDT (mm) in array axis order (z,y,x)
    edt_mm = distance_transform_edt(mask, sampling=tuple(spacing_zyx))

    # skeleton of mask -> candidate voxels
    try:
        sk = skeletonize(mask.astype(bool))
    except Exception:
        # fallback to local maxima of edt if skeletonize not available
        sk = edt_mm == ndi.maximum_filter(edt_mm, size=3)  # requires import scipy.ndimage as ndi
    zs, ys, xs = np.where(sk)
    if zs.size == 0:
        return (np.empty((0, 3)),) * 4 + (np.empty((0, 3)),)

    # build list of physical points for skeleton voxels
    sk_phys = []
    sk_rad = []
    for z, y, x in zip(zs, ys, xs):
        # TransformContinuousIndexToPhysicalPoint expects (x_idx, y_idx, z_idx)
        try:
            phys = np.asarray(prob_img.TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z))), float)
        except Exception:
            # fallback linear transform
            origin = np.asarray(prob_img.GetOrigin(), float)
            direction = np.asarray(prob_img.GetDirection(), float).reshape(3, 3)
            phys = (direction @ (np.array([float(x), float(y), float(z)]) * spacing_xyz)) + origin
        sk_phys.append(phys)
        sk_rad.append(float(edt_mm[int(z), int(y), int(x)]))
    sk_phys = np.vstack(sk_phys)
    sk_rad = np.asarray(sk_rad, float)

    # quick KD for skeleton points
    kd_skel = KDTree(sk_phys)

    curr = _to_np(curr_mm)
    prev = _to_np(prev_mm)
    old_dir = _unit(curr - prev)
    cos_thr = float(np.cos(np.radians(ANGLE_ALLOW_DEG)))

    # choose candidate skeleton nodes within reasonable neighborhood (by hops converted to mm radius)
    # approximate hop->mm radius = 2*radius_mm*MA_HOPS
    search_r = max(2.0 * radius_mm * max(1, MA_HOPS), 4.0)
    idxs = kd_skel.query_ball_point(curr, r=search_r)
    if not idxs:
        # fallback: nearest skeleton point
        _, nearest_idx = kd_skel.query(curr, k=1)
        idxs = [int(nearest_idx)]

    cand_pts = []
    cand_rad = []
    cand_ang = []

    kept = []
    for ii in idxs:
        p = sk_phys[int(ii)]
        r = float(sk_rad[int(ii)])
        v = p - curr
        L = np.linalg.norm(v)
        if L < 1e-8:
            continue
        if L < MIN_STRIDE_MM or L > MAX_STRIDE_MM:
            continue
        diru = v / L
        dot = float(np.clip(np.dot(old_dir, diru), -1.0, 1.0))
        if dot < cos_thr:
            continue
        ang = float(np.degrees(np.arccos(dot)))
        # ensure separation among kept candidates
        ok = True
        for q in kept:
            if np.linalg.norm(q - p) < MA_MIN_SEP_MM:
                ok = False
                break
        if not ok:
            continue
        kept.append(p.copy())
        cand_pts.append(p)
        cand_rad.append(r)
        cand_ang.append(ang)

    if not cand_pts:
        return (np.empty((0, 3)),) * 4 + (np.empty((0, 3)),)

    P = np.vstack(cand_pts)
    R = np.asarray(cand_rad, float)
    A = np.asarray(cand_ang, float)

    # near-centerline factor: distance to Gcent kdtree (if available)
    if cent_index.get("kd") is not None:
        try:
            d2cl, _ = cent_index["kd"].query(P, k=1)
            inv = 1.0 / (np.asarray(d2cl, float) + 1e-6)
            Fcl = inv / (inv.max() + 1e-8)
        except Exception:
            Fcl = np.zeros((P.shape[0],), float)
    else:
        Fcl = np.zeros((P.shape[0],), float)

    # normalize radius to [0,1]
    Rn = (R - R.min()) / max(1e-8, (R.max() - R.min()))
    # angle penalty (smaller angle -> better)
    An = 1.0 - (A / max(ANGLE_ALLOW_DEG, 1e-6))

    scores = (W_NEAR_CL * Fcl) + (W_RADIUS * Rn)

    # apply score threshold
    keep_mask = scores >= SCORE_MIN
    if not np.any(keep_mask):
        # relax threshold once
        keep_mask = scores >= (SCORE_MIN * 0.5)
        if not np.any(keep_mask):
            return (np.empty((0, 3)),) * 4 + (np.empty((0, 3)),)

    idx_order = np.argsort(-scores[keep_mask])[:int(k_best)]
    kept_indices = np.where(keep_mask)[0][idx_order]

    Pk = P[kept_indices]
    Rk = R[kept_indices]
    Ak = A[kept_indices]
    Sk = scores[kept_indices]

    # produce backtracked centerline for top candidate unless we were globally reseeded last iteration
    centerline_poly = np.empty((0, 3), float)
    if prev_picked != 'global reseed':
        try:
            top_pt = Pk[0]
            poly = simple_backtrack_fmm(prob_img, start_phys=curr, end_phys=top_pt,
                                        bin_thr=BIN_THR,
                                        leak_frac=float(cfg.get('BACK_LEAK_FRAC', 0.05)),
                                        step_mm=float(cfg.get('BACK_STEP_MM', max(0.25, 0.5 * radius_mm))),
                                        max_iters=int(cfg.get('BACK_MAX_ITERS', 2048)))
            if poly is not None and poly.shape[0] >= 2:
                # ensure order curr -> candidate
                d0 = np.linalg.norm(poly[0] - curr)
                d1 = np.linalg.norm(poly[-1] - curr)
                if d0 > d1:
                    poly = poly[::-1]
                centerline_poly = _smooth_polyline_spline_keep_ends(poly, resample_step_mm=float(cfg.get('CENTERLINE_RESAMPLE_MM', 0.5)), s=float(cfg.get('CENTERLINE_SMOOTH', 0.0)))
        except Exception:
            centerline_poly = np.empty((0,3), float)

    return Pk, Rk, Ak, Sk, centerline_poly


# ---------------------------------------------------------------------
# get_next_points_from_gvol (simplified backtracking via graph shortest-path)
# ---------------------------------------------------------------------
def get_next_points_from_gvol(
    prob_img: Optional['sitk.Image'],
    curr_mm: np.ndarray,
    prev_mm: np.ndarray,
    radius_mm: float,
    *,
    vol_index: Dict[str, object],
    cfg: Dict,
    k_best: int = 12,
    prev_picked: str = 'gvol'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (pts, radii, ang_deg, score, centerline_poly)
    - pts: Nx3 candidate points from Gvol neighbors (ordered by score desc)
    - centerline_poly: connected polyline along Gvol (curr -> candidate) for top candidate
    """

    import numpy as np
    import networkx as nx

    G = vol_index.get("G")
    kd = vol_index.get("kd")
    ids = vol_index.get("ids")

    if G is None or kd is None or not ids:
        return (np.empty((0, 3)),) * 4 + (np.empty((0, 3)),)

    curr = _to_np(curr_mm)
    prev = _to_np(prev_mm)
    old_dir = _unit(curr - prev)

    ANGLE_ALLOW_DEG = float(cfg.get('ANGLE_ALLOW_DEG', 150.0))
    cos_thr = float(np.cos(np.radians(ANGLE_ALLOW_DEG)))
    EDGE_MINP = float(cfg.get('GVOL_MIN_EDGE_PROB', 0.25))
    TWO_HOP = int(cfg.get('GVOL_TWO_HOP', 1))

    W_EDGE = float(cfg.get('W_EDGE', 0.6))
    W_RADIUS = float(cfg.get('W_RADIUS', 0.4))
    MIN_STRIDE_MM = float(cfg.get('MIN_STRIDE_MM', max(0.6, 0.35 * radius_mm)))
    MAX_STRIDE_MM = float(cfg.get('MAX_STRIDE_MM', max(2.5, 2.0 * radius_mm)))

    n_curr = _nearest_node(kd, ids, curr)
    if n_curr is None:
        return (np.empty((0, 3)),) * 4 + (np.empty((0, 3)),)

    frontier = set(G.neighbors(int(n_curr)))
    if TWO_HOP:
        for u in list(frontier):
            frontier.update(G.neighbors(u))

    cand_pts = []
    cand_r = []
    cand_ang = []
    cand_score = []
    cand_nid = []

    for n_next in frontier:
        if not G.has_edge(int(n_curr), int(n_next)):
            continue
        e = G.edges[int(n_curr), int(n_next)]
        p_edge = float(e.get('edge_prob', e.get('prob', 0.0)))
        if p_edge < EDGE_MINP:
            continue
        node = G.nodes[int(n_next)]
        p = node.get('pos_phys', node.get('point'))
        if p is None:
            continue
        p = _to_np(p)
        step_len = np.linalg.norm(p - curr)
        if step_len < MIN_STRIDE_MM or step_len > MAX_STRIDE_MM:
            continue
        diru = _unit(p - curr)
        dotv = float(np.clip(np.dot(old_dir, diru), -1.0, 1.0))
        if dotv < cos_thr:
            continue
        ang_deg = float(np.degrees(np.arccos(dotv)))
        rv = float(node.get('radius_mm', node.get('MaximumInscribedSphereRadius', 0.1)))

        # score combining edge prob and radius
        cand_pts.append(p)
        cand_r.append(rv)
        cand_ang.append(ang_deg)
        cand_score.append((W_EDGE * p_edge) + (W_RADIUS * rv))
        cand_nid.append(int(n_next))

    if not cand_pts:
        return (np.empty((0, 3)),) * 4 + (np.empty((0, 3)),)

    P = np.vstack(cand_pts)
    R = np.asarray(cand_r, float)
    A = np.asarray(cand_ang, float)
    S = np.asarray(cand_score, float)

    # order and pick top K
    order = np.argsort(-S)[:int(k_best)]
    P_ordered = P[order]
    R_ordered = R[order]
    A_ordered = A[order]
    S_ordered = S[order]
    nids_ordered = np.array(cand_nid)[order]

    # build centerline for top candidate using graph shortest path (unless prev_picked == 'global reseed')
    centerline_poly = np.empty((0, 3), float)
    if prev_picked != 'global reseed':
        try:
            top_nid = int(nids_ordered[0])
            if int(n_curr) == top_nid:
                centerline_poly = np.array([_to_np(G.nodes[int(n_curr)].get('pos_phys', G.nodes[int(n_curr)].get('point')))])
            else:
                # ensure length attribute exists
                weight_key = _ensure_edge_length(G, "length_mm")
                try:
                    node_path = nx.shortest_path(G, int(n_curr), top_nid, weight=weight_key)
                except Exception:
                    if nx.has_path(G, int(n_curr), top_nid):
                        node_path = nx.shortest_path(G, int(n_curr), top_nid)
                    else:
                        node_path = []

                if node_path and len(node_path) >= 2:
                    path_pts = np.array([_to_np(G.nodes[int(n)].get('pos_phys', G.nodes[int(n)].get('point'))) for n in node_path], float)
                    # ensure ordering from curr->candidate
                    if np.linalg.norm(path_pts[0] - curr) > np.linalg.norm(path_pts[-1] - curr):
                        path_pts = path_pts[::-1]
                    centerline_poly = _smooth_polyline_spline_keep_ends(path_pts, resample_step_mm=float(cfg.get('CENTERLINE_RESAMPLE_MM', 0.5)), s=float(cfg.get('CENTERLINE_SMOOTH', 0.0)))
        except Exception:
            centerline_poly = np.empty((0, 3), float)

    return P_ordered, R_ordered, A_ordered, S_ordered, centerline_poly



def _targets_remaining(vt: VesselTree) -> List[int]:
    ids = set(getattr(vt, "target_ids", []) or [])
    return [t for t in ids if t not in vt.node_traversed]

def _pick_reseed_node(
    vt: VesselTree,
    Gcent: nx.Graph,
    current_point: np.ndarray,
    *,
    prefer: str = "farthest",
    min_sep_mm: float = 6.0,
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
    if len(pool) == 0: pool = list(vt.node_not_traversed)

    deg = dict(Gcent.degree())
    pool = [nid for nid in pool if deg.get(nid, 0) >= int(min_degree)]
    if len(pool) == 0: return None

    # prune near already-traversed by min_sep
    sep_ok = np.ones(len(pool), dtype=bool)
    if len(vt.node_traversed) > 0:
        pts_tr = [_to_np(Gcent.nodes[n].get('pos_phys', Gcent.nodes[n].get('point'))) for n in vt.node_traversed]
        if len(pts_tr) > 0:
            kdt_tr = KDTree(np.vstack(pts_tr))
            for i, nid in enumerate(pool):
                p = _to_np(Gcent.nodes[nid].get('pos_phys', Gcent.nodes[nid].get('point')))
                dmin, _ = kdt_tr.query(p, k=1)
                if dmin < float(min_sep_mm): sep_ok[i] = False

    q = _to_np(current_point)
    scores, cand_pts = [], []
    for i, nid in enumerate(pool):
        if not sep_ok[i]:
            scores.append(-1e9); cand_pts.append(None); continue
        dnode = Gcent.nodes[nid]
        p = _to_np(dnode.get('pos_phys', dnode.get('point')))
        pr = float(dnode.get("node_prob_smooth", dnode.get('node_prob', 0.1)))
        r  = float(dnode.get('radius_mm', dnode.get('MaximumInscribedSphereRadius', 0.1)))
        dist = float(np.linalg.norm(p - q))
        sc = alpha_prob * pr + beta_radius * r - gamma_dist * dist
        scores.append(sc); cand_pts.append(p)

    if int(max(scores)) <= -1e8: return None
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
        if p is not None: return p
    return None



# --- Helper 2: quick polyline→tube surface (verts, faces) ---------------------
def _polyline_to_tube_np(poly_xyz: np.ndarray, radius_mm: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    import vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy
    poly_xyz = np.asarray(poly_xyz, float)
    if poly_xyz.shape[0] < 2: return np.empty((0,3), float), np.empty((0,3), int)
    pts = vtk.vtkPoints()
    for p in poly_xyz: pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    line = vtk.vtkPolyLine(); line.GetPointIds().SetNumberOfIds(poly_xyz.shape[0])
    for i in range(poly_xyz.shape[0]): line.GetPointIds().SetId(i, i)
    cells = vtk.vtkCellArray(); cells.InsertNextCell(line)
    pd = vtk.vtkPolyData(); pd.SetPoints(pts); pd.SetLines(cells)
    tube = vtk.vtkTubeFilter(); tube.SetInputData(pd); tube.SetNumberOfSides(12); tube.SetRadius(float(radius_mm)); tube.CappingOn(); tube.Update()
    surf = tube.GetOutput()
    if surf.GetNumberOfPoints() == 0 or surf.GetNumberOfPolys() == 0:
        return np.empty((0,3), float), np.empty((0,3), int)
    V = vtk_to_numpy(surf.GetPoints().GetData()).astype(np.float64)
    faces = surf.GetPolys()
    ids = vtk.vtkIdList(); F = []
    faces.InitTraversal()
    while faces.GetNextCell(ids):
        if ids.GetNumberOfIds() == 3:
            F.append([ids.GetId(0), ids.GetId(1), ids.GetId(2)])
    return V, (np.asarray(F, dtype=np.int64) if len(F) else np.empty((0,3), int))
def trace_centerline(
    output_folder: str,
    image_file: str,
    case: str,
    model_folder: str,
    fold: int,
    *,
    graph: nx.Graph,                 # Gvol
    seed_node: int,
    target_nodes: List[int],
    centerline_graph: nx.Graph,      # Gcent
    max_steps_per_component: int = 500,
    global_config: Optional[dict] = None,
    unit: str = 'cm',
    scale: float = 1.0,
    seg_file: Optional[sitk.Image] = None,
    start_seg: Optional[sitk.Image] = None,
    write_samples: bool = False,
):
    """
    Simplified, robust trace_centerline function.
    Keeps previous behaviour: predict crop, assemble segs, pick next point via cent/gvol,
    perform reseed on failure, append optional surface/inside samples, maintain vt coverage.
    """

    assert global_config is not None, "global_config (YAML dict) is required"
    cfg = dict(global_config)

    # ------------------------------ NEW: helpers ------------------------------
    def _extract_inside_points_mm(prob_img: sitk.Image,
                                  thresh: float = 0.5,
                                  stride: int = 4) -> np.ndarray:
        arr = sitk.GetArrayFromImage(prob_img)  # (z,y,x)
        mask = arr >= float(thresh)
        if stride > 1:
            mask[::stride, :, :] = mask[::stride, :, :]
            mask[:, ::stride, :] = mask[:, ::stride, :]
            mask[:, :, ::stride] = mask[:, :, ::stride]
        idx_zyx = np.argwhere(mask)
        if idx_zyx.size == 0:
            return np.empty((0, 3), dtype=float)

        # Convert to (x,y,z) index order
        idx_xyz = idx_zyx[:, [2, 1, 0]].astype(float)

        origin = np.asarray(prob_img.GetOrigin(), float)          # (x,y,z)
        spacing = np.asarray(prob_img.GetSpacing(), float)        # (x,y,z)
        direction = np.asarray(prob_img.GetDirection(), float)    # len=9
        D = direction.reshape(3, 3)

        scaled = idx_xyz * spacing  # broadcast
        phys = (D @ scaled.T).T + origin
        return phys

    def _surface_from_prob(prob_img: sitk.Image,
                           iso: float = 0.5) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            from skimage import measure
        except Exception:
            return None

        arr = sitk.GetArrayFromImage(prob_img).astype(np.float32)  # (z,y,x)
        sp = np.array(prob_img.GetSpacing(), float)                 # (x,y,z)
        sp_zyx = sp[::-1]                                          # (z,y,x)

        try:
            verts_zyx, faces, _, _ = measure.marching_cubes(arr, level=float(iso), spacing=tuple(sp_zyx))
        except Exception:
            return None

        verts_xyz = verts_zyx[:, [2, 1, 0]]
        origin = np.asarray(prob_img.GetOrigin(), float)
        direction = np.asarray(prob_img.GetDirection(), float).reshape(3, 3)
        verts_mm = (direction @ verts_xyz.T).T + origin

        faces = faces.astype(np.int32)
        return verts_mm.astype(np.float32), faces
    # -------------------------------------------------------------------------

    # io & geometry
    scale_unit = 0.1 if unit == 'cm' else 1.0
    SEGMENTATION = bool(cfg.get('SEGMENTATION', False))
    VOLUME_SIZE_RATIO = float(cfg.get('VOLUME_SIZE_RATIO', 2.0))
    MAGN_RADIUS = float(cfg.get('MAGN_RADIUS', 0.5))
    ADD_RADIUS = float(cfg.get('ADD_RADIUS', 1)) * scale_unit
    MIN_RES = int(cfg.get('MIN_RES', 8))

    STALL_MAX_STEPS     = int(cfg.get('STALL_MAX_STEPS', 7))
    RESEED_MIN_SEP_MM   = float(cfg.get('RESEED_MIN_SEP_MM', 6))
    RESEED_POLICY       = cfg.get('RESEED_POLICY', 'farthest')
    RESEED_ALPHA_PROB   = float(cfg.get('RESEED_ALPHA_PROB', 0.4))
    RESEED_BETA_RADIUS  = float(cfg.get('RESEED_BETA_RADIUS', 0.8))
    RESEED_GAMMA_DIST   = float(cfg.get('RESEED_GAMMA_DIST', 0.4))
    RESEED_MIN_DEGREE   = int(cfg.get('RESEED_MIN_DEGREE', 2))
    RESEED_TARGETS_FIRST = bool(cfg.get('RESEED_TARGETS_FIRST', False))
    COVERAGE_STOP       = float(cfg.get('COVERAGE_STOP', 0.99))
    STOP_WHEN_ALL_TARGETS = bool(cfg.get('STOP_WHEN_ALL_TARGETS', False))

    # image IO
    if SEGMENTATION and isinstance(seg_file, str):
        reader_im, origin_im, size_im, spacing_im = import_image(seg_file)
        image_file_effective = seg_file
    else:
        reader_im, origin_im, size_im, spacing_im = import_image(image_file)
        image_file_effective = image_file

    try:
        geom_img = sitk.ReadImage(image_file_effective)
    except Exception:
        geom_img = sitk.Image(int(size_im[0]), int(size_im[1]), int(size_im[2]), sitk.sitkUInt8)
        geom_img.SetOrigin(tuple(map(float, origin_im)))
        geom_img.SetSpacing(tuple(map(float, spacing_im)))
        geom_img.SetDirection(tuple([1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]))

    # predictor
    predictor = initialize_predictor(model_folder, fold)
    print('predictor initialized')

    # coverage tracking on Gcent
    def densify_graph(G, step_mm=0.8):
        H = nx.Graph()
        for n, d in G.nodes(data=True):
            dd = dict(d)
            if 'pos_phys' in dd and 'point' not in dd:
                dd['point'] = np.asarray(dd['pos_phys'], float)
            elif 'point' in dd and 'pos_phys' not in dd:
                dd['pos_phys'] = np.asarray(dd['point'], float)
            H.add_node(n, **dd)

        new_id = max(G.nodes, default=-1) + 1
        for u, v in G.edges():
            pu = _to_np(G.nodes[u].get('pos_phys', G.nodes[u].get('point')))
            pv = _to_np(G.nodes[v].get('pos_phys', G.nodes[v].get('point')))
            L = float(np.linalg.norm(pv - pu))
            k = max(1, int(np.ceil(L / step_mm)))

            last = u
            for t in np.linspace(0, 1, k + 1)[1:-1]:
                p = (1 - t) * pu + t * pv
                r_u = float(G.nodes[u].get('radius_mm', G.nodes[u].get('MaximumInscribedSphereRadius', 0.1)))
                r_v = float(G.nodes[v].get('radius_mm', G.nodes[v].get('MaximumInscribedSphereRadius', 0.1)))
                r_new = 0.5 * (r_u + r_v)
                H.add_node(
                    new_id,
                    pos_phys=p, point=p,
                    radius_mm=r_new, MaximumInscribedSphereRadius=r_new
                )
                H.add_edge(last, new_id, **G.edges[u, v])
                last = new_id
                new_id += 1
            H.add_edge(last, v, **G.edges[u, v])
        return H

    Gcent = densify_graph(centerline_graph, 0.4)

    # data structures
    vt = VesselTree(
        case=case,
        image_file=image_file_effective,
        seed_id=int(seed_node),
        target_ids=list(map(int, target_nodes)),
        graph=graph,
        centerline_pd=None,
        centerline_graph=Gcent,
    )
    vt.geom_img = geom_img

    vt_init_coverage(vt, Gcent)

    # assembly
    assembly_segs = Segmentation(
        case,
        image_file_effective,
        weighted=bool(cfg.get('WEIGHTED_ASSEMBLY', False)),
        weight_type=cfg.get('WEIGHT_TYPE', 'radius'),
        start_seg=start_seg
    )

    # KD indices
    cent_index = _make_index(Gcent)
    vol_index  = _make_index(graph)

    # outputs
    list_surfaces: List = []
    list_points: List[np.ndarray] = []
    list_inside_points: List[np.ndarray] = []
    list_centerlines: List[np.ndarray] = []

    # seed step
    seed_pt = _to_np(graph.nodes[int(seed_node)].get('pos_phys', graph.nodes[int(seed_node)].get('point')))
    seed_r = float(graph.nodes[int(seed_node)].get('radius_mm',
               graph.nodes[int(seed_node)].get('MaximumInscribedSphereRadius', 0.1)))
    vt.steps = [{'old point': seed_pt, 'point': seed_pt, 'old radius': seed_r, 'radius': seed_r,
                 'tangent': np.array([1.0, 0.0, 0.0], float), 'angle change': 0.0}]
    list_points.append(seed_pt)

    # ---------- NEW: track what produced the last committed point ----------
    last_pick_type = 'seed'   # possible values: 'seed', 'cent', 'gvol', 'global reseed'
    # ---------------------------------------------------------------------

    no_cover_streak = 0
    i = 1
    while i <= max_steps_per_component:

        step = vt.steps[-1]
        curr_pt = _to_np(step['point'])
        old_pt  = _to_np(step['old point'])
        curr_rad = float(step['radius'])
        prev_picked = last_pick_type

        # ROI crop + UNet prob
        prob_prediction = None
        try:
            idx_clamped, size_clamped, border_flag = map_to_image(
                center_phys=curr_pt,
                box_radius_mm=(curr_rad + float(cfg.get('ADD_RADIUS', 1.0))) * float(cfg.get('MAGN_RADIUS', 0.5)),
                volume_size_ratio=float(cfg.get('VOLUME_SIZE_RATIO', 2.0)),
                image=geom_img,
                min_res=int(cfg.get('MIN_RES', 8)),
            )
            cropped_vol = extract_volume(*import_image(image_file_effective), idx_clamped, size_clamped)  # reader_im not reused
        except Exception:
            # fallback reader (reader_im) path
            reader_im, _, _, _ = import_image(image_file_effective)
            idx_clamped, size_clamped, border_flag = map_to_image(
                center_phys=curr_pt,
                box_radius_mm=(curr_rad + float(cfg.get('ADD_RADIUS', 1.0))) * float(cfg.get('MAGN_RADIUS', 0.5)),
                volume_size_ratio=float(cfg.get('VOLUME_SIZE_RATIO', 2.0)),
                image=geom_img,
                min_res=int(cfg.get('MIN_RES', 8)),
            )
            cropped_vol = extract_volume(reader_im, idx_clamped, size_clamped)

        try:
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
                print(f" forward_time={time.time() - t0:.3f}s")
            else:
                print('[err] check nnUNet initialization details')
                break

            try:
                seed_vox = (np.rint(np.array(size_clamped) / 2).astype(int)).tolist()
                pred_img = remove_other_vessels(pred_img, seed_vox)
            except Exception:
                pass

            assembly_segs.add_segmentation(
                prob_prediction, idx_clamped, size_clamped, curr_rad
            )

            # ------------------------- NEW: append samples -------------------------
            if write_samples and prob_prediction is not None:
                inside_pts_mm = _extract_inside_points_mm(
                    prob_prediction,
                    thresh=float(cfg.get('INSIDE_THRESH', 0.5)),
                    stride=int(cfg.get('INSIDE_SAMPLE_STRIDE', 4))
                )
                if inside_pts_mm.size > 0:
                    list_inside_points.append(inside_pts_mm)

                surf = _surface_from_prob(
                    prob_prediction,
                    iso=0.5
                )
                if surf is not None:
                    list_surfaces.append(surf)
            # ----------------------------------------------------------------------

            if check_seg_border(size_clamped, idx_clamped, pred_img, np.array(geom_img.GetSize(), int)):
                print(" segmentation crop touched global border stop component")
                break

        except Exception as e:
            print(f"[ERR crop/predict] {e}")

        # ============ Decision: centerline → Gvol → reseed ============
        picked_from = None
        try:
            cl_pts, cl_rad, cl_ang, cl_sc, centerline = get_next_points_from_crop_centerline(
                prob_img=prob_prediction, curr_mm=curr_pt, prev_mm=old_pt, radius_mm=curr_rad,
                cent_index=cent_index, vol_index=vol_index, cfg=cfg, k_best=20, prev_picked=prev_picked
            )

            if centerline is not None and getattr(centerline, "size", 0) > 0 and last_pick_type != 'global reseed':
                try:
                    poly_cl = np.asarray(centerline, dtype=float)
                    if poly_cl.shape[0] >= 2:
                        d0 = np.linalg.norm(poly_cl[0] - curr_pt)
                        d1 = np.linalg.norm(poly_cl[-1] - curr_pt)
                        if d0 < d1:
                            poly_ordered = poly_cl
                        else:
                            poly_ordered = poly_cl[::-1]
                        if poly_ordered.shape[0] >= 2:
                            list_centerlines.append(poly_ordered)
                except Exception:
                    pass

            if cl_pts is not None and cl_pts.size > 0:
                nxt_p = _snap_phys_inside(geom_img, cl_pts[0])
                nxt_r = float(cl_rad[0])
                picked_from = 'cent'
                print(f'point selected from {picked_from}', 'score :', float(cl_sc[0]))
            else:
                gv_pts, gv_rad, gv_ang, gv_sc, centerline = get_next_points_from_gvol(
                    prob_img=prob_prediction, curr_mm=curr_pt, prev_mm=old_pt, radius_mm=curr_rad,
                    vol_index=vol_index, cfg=cfg, k_best=int(cfg.get('GVOL_KBEST', 12)), prev_picked=prev_picked
                )

                if centerline is not None and getattr(centerline, "size", 0) > 0 and last_pick_type != 'global reseed':
                    try:
                        poly_cl = np.asarray(centerline, dtype=float)
                        if poly_cl.shape[0] >= 2:
                            d0 = np.linalg.norm(poly_cl[0] - curr_pt)
                            d1 = np.linalg.norm(poly_cl[-1] - curr_pt)
                            if d0 < d1:
                                poly_ordered = poly_cl
                            else:
                                poly_ordered = poly_cl[::-1]
                            if poly_ordered.shape[0] >= 2:
                                list_centerlines.append(poly_ordered)
                    except Exception:
                        pass

                if gv_pts is not None and gv_pts.size > 0:
                    nxt_p = _snap_phys_inside(geom_img, gv_pts[0])
                    nxt_r = float(gv_rad[0])
                    picked_from = 'gvol'
                    print(f'point selected from {picked_from},score', gv_sc)

            if picked_from is None:
                print("[DBG] centerline + Gvol failed GLOBAL reseed on Gcent")
                jump_pt = _pick_reseed_node(
                    vt, Gcent, curr_pt,
                    prefer=RESEED_POLICY, min_sep_mm=RESEED_MIN_SEP_MM,
                    alpha_prob=RESEED_ALPHA_PROB, beta_radius=RESEED_BETA_RADIUS,
                    gamma_dist=RESEED_GAMMA_DIST, min_degree=RESEED_MIN_DEGREE,
                    targets_first=RESEED_TARGETS_FIRST,
                )
                if jump_pt is None:
                    print("[DBG re-seed] no valid re-seed found; stopping.")
                    break
                jump_pt = _snap_phys_inside(geom_img, jump_pt)

                vt.steps.append({'old point': jump_pt, 'point': jump_pt,
                                 'old radius': curr_rad, 'radius': curr_rad,
                                 'tangent': step['tangent'], 'angle change': 0.0})
                list_points.append(jump_pt)

                vt_mark_covered_by_segment_ball(vt, Gcent, curr_pt, jump_pt, radius_scale=2)
                no_cover_streak = 0
                last_pick_type = 'global reseed'
                i += 1
                continue

            vt.steps.append({'old point': curr_pt, 'point': nxt_p,
                             'old radius': curr_rad, 'radius': nxt_r,
                             'tangent': _unit(nxt_p - curr_pt), 'angle change': 0.0})
            list_points.append(nxt_p)

            if picked_from is not None:
                last_pick_type = picked_from

            newly = vt_mark_covered_by_segment_ball(vt, Gcent, curr_pt, nxt_p, radius_scale=2)
            cov   = vt_coverage_ratio(vt)
            if newly > 0:
                no_cover_streak = 0
            else:
                no_cover_streak += 1

            if no_cover_streak >= STALL_MAX_STEPS:
                print("[STALL] triggering auto GLOBAL re-seed due to coverage stall.")
                jump_pt = _pick_reseed_node(
                    vt, Gcent, curr_pt,
                    prefer=RESEED_POLICY, min_sep_mm=RESEED_MIN_SEP_MM,
                    alpha_prob=RESEED_ALPHA_PROB, beta_radius=RESEED_BETA_RADIUS,
                    gamma_dist=RESEED_GAMMA_DIST, min_degree=RESEED_MIN_DEGREE,
                    targets_first=RESEED_TARGETS_FIRST,
                )
                if jump_pt is None:
                    print("[DBG re-seed] no valid re-seed found; stopping.")
                    break
                jump_pt = _snap_phys_inside(geom_img, jump_pt)
                vt.steps.append({'old point': jump_pt, 'point': jump_pt,
                                 'old radius': curr_rad, 'radius': curr_rad,
                                 'tangent': step['tangent'], 'angle change': 0.0})
                list_points.append(jump_pt)
                vt_mark_covered_by_segment_ball(vt, Gcent, curr_pt, jump_pt, radius_scale=2)
                no_cover_streak = 0
                last_pick_type = 'global reseed'
                i += 1
                continue

            if STOP_WHEN_ALL_TARGETS and len(_targets_remaining(vt)) == 0:
                print("[STOP] all targets reached; stopping component.")
                break
            if cov >= COVERAGE_STOP or vt._cov_mask.sum() >= vt._cov_mask.size:
                print(f"[STOP] coverage {100*cov:.1f}% ≥ {100*COVERAGE_STOP:.0f}% or all nodes covered; stopping.")
                break

            print(picked_from,
                  {'old point': curr_pt, 'point': nxt_p,
                   'old radius': curr_rad, 'radius': nxt_r,
                   'tangent': step['tangent'], 'angle change': 0.0},
                  f'coverage {100*cov:.1f}')
            i += 1
            continue

        except Exception as e:
            print(f"[ERR decision step {i}] {e}")
            break  # no nudge fallback

    # append raw steps polyline once
    if len(vt.steps) > 0:
        poly = np.vstack([_to_np(s['point']) for s in vt.steps])
        if poly.shape[0] >= 2:
            list_centerlines.append(poly)

    # tidy
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception:
        pass

    return (
        list_centerlines,   # polylines
        list_surfaces,      # (verts_mm, faces) if write_samples True
        list_points,        # committed points
        list_inside_points, # points with prob>=0.5 (sampled)
        assembly_segs,
        vt,
        i,
    )
