from __future__ import annotations
import os, time, copy
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
import networkx as nx
import SimpleITK as sitk
import skfmm
from scipy.spatial import cKDTree as KDTree
from scipy.ndimage import (
    gaussian_filter,
    map_coordinates,
    distance_transform_edt,
)
from skimage import measure
from skimage.morphology import skeletonize
from SeqSeg.seqseg.modules.sitk_functions import (
    import_image,
    extract_volume,
    copy_settings,
    remove_other_vessels,
    check_seg_border,
)
from SeqSeg.seqseg.modules.nnunet import initialize_predictor
from SeqSeg.seqseg.modules.assembly import Segmentation
from seqseg_modules_modified.assembly import VesselTree


def largest_cc(img: sitk.Image, background_value=0) -> sitk.Image:
    relabeled = sitk.RelabelComponent(
        sitk.ConnectedComponent(img != background_value),
        sortByObjectSize=True
    )
    return sitk.Cast(relabeled == 1, img.GetPixelID())

#helpers
def _to_np(p) -> np.ndarray:
    return np.asarray(p, dtype=float)

def _unit(v) -> np.ndarray:
    v = _to_np(v)
    n = float(np.linalg.norm(v))
    return v / (n + 1e-12)


def _phys_tuple(p):
    a = _to_np(p).reshape(3)
    return float(a[0]), float(a[1]), float(a[2])


def _build_kdtree(G: nx.Graph):
    pts, ids = [], []
    for n, d in G.nodes(data=True):
        p = d.get("pos_phys", d.get("point"))
        if p is None:
            continue
        p = _to_np(p)
        pts.append(p)
        ids.append(int(n))
        d["pos_phys"] = p
        d["point"] = p
    if not pts:
        return None, []
    return KDTree(np.vstack(pts)), ids

def _make_index(G: nx.Graph) -> Dict[str, object]:
    kd, ids = _build_kdtree(G)
    return {"G": G, "kd": kd, "ids": ids}

def _nearest_node(kdt: Optional[KDTree], ids: List[int], point) -> Optional[int]:
    if kdt is None or not ids:
        return None
    _, idx = kdt.query(_to_np(point), k=1)
    return int(ids[int(idx)])

def _norm_vec(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    x_min, x_max = x.min(), x.max()
    if x_max > x_min:
        return (x - x_min) / (x_max - x_min)
    else:
        return np.ones_like(x)

def _smooth_polyline_spline_keep_ends(
    pts: np.ndarray,
    resample_step_mm: float = 0.5,
    s: float = 0.0,
    k: int = 3,
) -> np.ndarray:
    ''' optional smoothing to append crop centerlines'''
    from scipy.interpolate import splprep, splev
    pts = np.asarray(pts, float)
    if pts.shape[0] < 3:
        return pts.copy()

    diffs = np.diff(pts, axis=0)
    seglen = np.sqrt((diffs**2).sum(axis=1))
    t = np.concatenate(([0.0], np.cumsum(seglen)))
    if t[-1] <= 0:
        return pts.copy()
    t /= t[-1]
    k = min(k, max(1, pts.shape[0] - 1))
    try:
        tck, _ = splprep(pts.T, u=t, s=float(s), k=k)
    except Exception:
        return pts.copy()
    L = seglen.sum()
    m = max(int(np.ceil(L / max(1e-6, float(resample_step_mm)))), 2)
    u_new = np.linspace(0.0, 1.0, m)
    out = np.vstack(splev(u_new, tck)).T
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out

def map_to_image(
    center_phys,
    box_radius_mm: float,
    volume_size_ratio: float,
    *,
    image: sitk.Image,
    min_res: int = 8,
    require_odd: bool = True,
) -> Tuple[List[int], List[int], bool]:
    '''Crop extraction'''
    ci = np.array(image.TransformPhysicalPointToContinuousIndex(_phys_tuple(center_phys)))
    img_size = np.array(list(image.GetSize()), dtype=int)
    spacing = np.array(list(image.GetSpacing()), dtype=float)
    if not np.all(np.isfinite(ci)):
        ci = (img_size - 1) / 2.0
    L = max(1e-3, float(volume_size_ratio) * float(box_radius_mm))   #crop Length(L)
    crop_size = np.ceil(L / np.maximum(spacing, 1e-12)).astype(int)
    crop_size = np.maximum(crop_size, int(min_res))
    if require_odd:
        crop_size = crop_size + (crop_size % 2 == 0)
    if np.any(crop_size > img_size):
        crop_size = np.minimum(crop_size, img_size)
    start = np.floor(ci - 0.5 * crop_size).astype(int)
    start_clamped = np.maximum(0, np.minimum(start, img_size - crop_size))
    border_touched = bool(np.any(start_clamped != start) or np.any(crop_size == img_size))
    crop_size = np.maximum(crop_size, 1)
    return start_clamped.tolist(), crop_size.tolist(), border_touched


#coverage tracking
def vt_init_coverage(vt: VesselTree, Gcent: nx.Graph):
    '''Tracking coverage on global centerline'''
    ids = list(Gcent.nodes)
    pts = []
    for nid in ids:
        d = Gcent.nodes[nid]
        p = d.get("pos_phys", d.get("point"))
        pts.append(_to_np(p))
    vt._cov_ids = ids
    vt._cov_pts = np.vstack(pts).astype(float) if pts else np.zeros((0, 3), float)
    vt._cov_kdt = KDTree(vt._cov_pts) if len(vt._cov_pts) else None
    vt._cov_mask = np.zeros(len(ids), dtype=bool)
    vt._cov_id2idx = {nid: i for i, nid in enumerate(ids)}
    vt.node_traversed = set()
    vt.node_not_traversed = set(ids)


def vt_mark_covered_from_poly(
    vt: VesselTree,
    poly: np.ndarray,
    radius_mm: float,
    radius_scale: float = 3.0,
) -> int:
    if vt._cov_kdt is None or poly.size == 0:
        return 0
    rcap = radius_scale * max(radius_mm, 0.5)
    idx_lists = vt._cov_kdt.query_ball_point(poly, r=rcap)
    cand_idx: Set[int] = set()
    for idx in idx_lists:
        if isinstance(idx, int):
            cand_idx.add(int(idx))
        else:
            cand_idx.update(map(int, idx))
    if not cand_idx:
        return 0
    idx_arr = np.fromiter(cand_idx, dtype=int)
    new_idx = idx_arr[~vt._cov_mask[idx_arr]]
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
    return float(vt._cov_mask.sum()) / float(vt._cov_mask.size)


#backtracking
def backtrack(
    prob_img: sitk.Image,
    start_phys: np.ndarray,
    end_phys: np.ndarray,
    *,
    bin_thr: float = 0.5,
    leak_frac: float = 0.05,
    step_mm: float = 0.5,
    max_iters: int = 2048,
) -> Optional[np.ndarray]:
    if prob_img is None:
        return None

    arr = sitk.GetArrayFromImage(prob_img).astype(np.float32)
    if arr.size == 0:
        return None

    sp_xyz = np.asarray(prob_img.GetSpacing(), float)
    sp_zyx = sp_xyz[::-1]
    mask = arr >= float(bin_thr)
    if not mask.any():
        return None
    try:
        edt = distance_transform_edt(mask, sampling=tuple(sp_zyx))
    except Exception:
        return None

    inside_med = float(np.median(edt[mask])) if mask.any() else float(np.max(edt))
    outside_speed = max(1e-6, inside_med * float(leak_frac))
    speed = edt + 1e-6
    speed[~mask] = outside_speed
    speed = gaussian_filter(speed, sigma=1.0)
    try:
        s_idx_xyz = np.asarray(
            prob_img.TransformPhysicalPointToContinuousIndex(tuple(map(float, start_phys))),
            float,
        )
    except Exception:
        return None
    s_idx_zyx = s_idx_xyz[::-1]
    s_idx_round = np.clip(np.rint(s_idx_zyx).astype(int), 0, np.array(arr.shape) - 1)

    phi = np.ones_like(speed, dtype=float)
    phi[s_idx_round[0], s_idx_round[1], s_idx_round[2]] = -1.0
    try:
        T = skfmm.travel_time(phi, speed=speed, dx=tuple(sp_zyx), order=2)
    except Exception:
        return None
    grads = np.gradient(T, *tuple(sp_zyx))
    def _interp_T(idx_zyx):
        z, y, x = idx_zyx
        return float(
            map_coordinates(T, np.array([[z], [y], [x]]), order=1, mode="nearest")[0]
        )
    try:
        e_idx_xyz = np.asarray(
            prob_img.TransformPhysicalPointToContinuousIndex(tuple(map(float, end_phys))),
            float,
        )
    except Exception:
        return None
    cur = e_idx_xyz[::-1].astype(float)  # z,y,x
    def phys_from_zyx(zyx):
        x_idx, y_idx, z_idx = float(zyx[2]), float(zyx[1]), float(zyx[0])
        return np.asarray(
            prob_img.TransformContinuousIndexToPhysicalPoint((x_idx, y_idx, z_idx)),
            float,
        )

    mean_sp = float(np.mean(sp_xyz))
    step_idx = float(step_mm) / max(1e-9, float(np.mean(sp_zyx)))
    path_phys = [phys_from_zyx(cur)]

    for _ in range(int(max_iters)):
        if np.linalg.norm(path_phys[-1] - np.asarray(start_phys, float)) <= mean_sp:
            path_phys.append(np.asarray(start_phys, float))
            break

        sample = np.array([cur[0], cur[1], cur[2]], float)
        gz = map_coordinates(grads[0], sample[:, None], order=1, mode="nearest")[0]
        gy = map_coordinates(grads[1], sample[:, None], order=1, mode="nearest")[0]
        gx = map_coordinates(grads[2], sample[:, None], order=1, mode="nearest")[0]
        grad_phys = np.array([gx, gy, gz], float)

        ng = np.linalg.norm(grad_phys)
        if not np.isfinite(ng) or ng < 1e-12:
            break
        dir_phys = -grad_phys / ng

        delta_idx_xyz = dir_phys / sp_xyz
        delta_idx_zyx = np.array(
            [delta_idx_xyz[2], delta_idx_xyz[1], delta_idx_xyz[0]], float
        )
        nrm_idx = np.linalg.norm(delta_idx_zyx)
        if nrm_idx < 1e-12:
            break
        nxt = cur + (delta_idx_zyx / nrm_idx) * step_idx

        if np.any(nxt < -1.0) or np.any(nxt > (np.array(arr.shape) + 1.0)):
            break

        Tcur = _interp_T(cur)
        Tnxt = _interp_T(nxt)
        if (not np.isfinite(Tnxt)) or (Tnxt >= Tcur - 1e-9):
            break

        cur = nxt
        path_phys.append(phys_from_zyx(cur))

    if len(path_phys) <= 1:
        return None
    return np.vstack(path_phys)


def get_next_points_from_crop_centerline(
    prob_img: Optional[sitk.Image],
    curr_mm: np.ndarray,
    prev_mm: np.ndarray,
    radius_mm: float,
    vt,
    *,
    cent_index: Dict[str, object],
    cfg: Dict,
    k_best: int = 5,
    vol_index: Optional[Dict[str, object]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    empty5 = (
        np.empty((0, 3)),
        np.empty((0,), float),
        np.empty((0,), float),
        np.empty((0,), float),
        np.empty((0, 3), float),
    )

    if prob_img is None:
        return empty5

    try:
        arr = sitk.GetArrayFromImage(prob_img).astype(np.float32)  # (z,y,x)
    except Exception:
        return empty5

    if arr.size == 0 or float(arr.max()) <= 0:
        return empty5

    # --------------------------- config ---------------------------
    BIN_THR         = float(cfg.get('BIN_THR', 0.5))
    ANGLE_ALLOW_DEG = float(cfg.get('ANGLE_ALLOW_DEG', 150.0))
    cos_thr         = float(np.cos(np.radians(ANGLE_ALLOW_DEG)))

    MIN_STRIDE_MM         = float(cfg.get('MIN_STRIDE_MM', max(0.6, 0.35 * radius_mm)))
    MAX_STRIDE_MM         = float(cfg.get('MAX_STRIDE_MM', max(3.0, 2.0 * radius_mm)))
    MAX_CENTERLINE_LEN_MM = float(cfg.get('MAX_CENTERLINE_LEN_MM', max(15.0, 6.0 * radius_mm)))

    LEAK_FRAC   = float(cfg.get('FMM_LEAK_FRAC', 0.05))

    W_R   = float(cfg.get('W_R', 0.5))
    W_CL  = float(cfg.get('W_CL', 0.1))
    W_LEN = float(cfg.get('W_LEN', 0.5))

    ENDPOINT_SCORE_MIN = float(cfg.get('ENDPOINT_SCORE_MIN', 0.0))
    SIGMA_FRAC         = float(cfg.get('CL_SIGMA_FRAC', 0.6))
    mask_bin = arr >= BIN_THR
    if not mask_bin.any():
        return empty5

    spacing_xyz = np.asarray(prob_img.GetSpacing(), float)
    spacing_zyx = spacing_xyz[::-1]
    origin      = np.asarray(prob_img.GetOrigin(), float)
    direction   = np.asarray(prob_img.GetDirection(), float).reshape(3, 3)

    try:
        edt_mm = distance_transform_edt(mask_bin, sampling=tuple(spacing_zyx))
    except Exception:
        return empty5

    idx_zyx = np.argwhere(mask_bin)
    if idx_zyx.size == 0:
        return empty5

    zs, ys, xs = idx_zyx.T
    idx_xyz = np.vstack([xs, ys, zs]).T.astype(float)

    scaled   = idx_xyz * spacing_xyz[None, :]
    pts_phys = (direction @ scaled.T).T + origin[None, :]

    rad_vals = edt_mm[zs, ys, xs]


    curr = _to_np(curr_mm)
    prev = _to_np(prev_mm)
    old_dir = _unit(curr - prev)

    v    = pts_phys - curr[None, :]
    dist = np.linalg.norm(v, axis=1)
    diru = np.zeros_like(v, dtype=float)
    valid_dir = dist > 1e-12
    diru[valid_dir] = v[valid_dir] / dist[valid_dir, None]
    dotv = np.clip(np.sum(diru * old_dir[None, :], axis=1), -1.0, 1.0)
    valid = (
        (dist >= MIN_STRIDE_MM) &
        (dist <= MAX_CENTERLINE_LEN_MM) &
        (dotv >= cos_thr)
    )
    if not np.any(valid):
        return empty5

    pts_phys = pts_phys[valid]
    rad_vals = rad_vals[valid]
    dist     = dist[valid]

    Fcl = np.zeros_like(rad_vals)
    if vt is not None and getattr(vt, "_cov_pts", None) is not None and vt._cov_pts.shape[0] > 0:
        mask_un = ~vt._cov_mask
        if np.any(mask_un):
            pts_un = vt._cov_pts[mask_un]
            kdt_un = KDTree(pts_un)
            d_un, _ = kdt_un.query(pts_phys, k=1)
            sigma_un = SIGMA_FRAC * (rad_vals + 1e-8)
            Fp_un = np.exp(- (d_un ** 2) / (2.0 * (sigma_un ** 2)))
            Fcl = _norm_vec(Fp_un)

    s_rad  = _norm_vec(rad_vals)
    s_dist = _norm_vec(dist)
    s_cl   = _norm_vec(Fcl)
    score_endpoint = W_R * s_rad + W_CL * s_cl + W_LEN * s_dist

    idx_sorted = np.argsort(-score_endpoint)
    if ENDPOINT_SCORE_MIN > -1e9:
        idx_sorted = [i for i in idx_sorted if score_endpoint[i] >= ENDPOINT_SCORE_MIN]
    if not idx_sorted:
        return empty5

    max_try = min(k_best, len(idx_sorted))

    chosen_poly = None
    chosen_score = None

    for t in range(max_try):
        j = int(idx_sorted[t])
        end_phys = pts_phys[j]

        poly = backtrack(
            prob_img=prob_img,
            start_phys=curr_mm,
            end_phys=end_phys,
            bin_thr=BIN_THR,
            leak_frac=LEAK_FRAC,
            step_mm=float(cfg.get('BACK_STEP_MM', 0.25)),
            max_iters=int(cfg.get('BACK_MAX_ITERS', 2048)),
        )
        if poly is None or poly.shape[0] < 2:
            continue

        if np.linalg.norm(poly[0] - curr) > np.linalg.norm(poly[-1] - curr):
            poly = poly[::-1]

        poly = _smooth_polyline_spline_keep_ends(
            poly,
            resample_step_mm=float(cfg.get('CENTERLINE_RESAMPLE_MM', 0.5)),
            s=float(cfg.get('CENTERLINE_SMOOTH', 0.0)),
        )

        if poly.shape[0] >= 2:
            chosen_poly = poly
            chosen_score = float(score_endpoint[j])
            break

    if chosen_poly is None:
        return empty5

    poly = chosen_poly

    diffs = np.diff(poly, axis=0)
    seglen = np.linalg.norm(diffs, axis=1)
    s_al = np.concatenate(([0.0], np.cumsum(seglen)))

    idx_curr = int(np.argmin(np.sum((poly - curr[None, :]) ** 2, axis=1)))
    s_from_curr = s_al - s_al[idx_curr]

    mask_ahead = (s_from_curr >= MIN_STRIDE_MM) & (s_from_curr <= MAX_STRIDE_MM)
    cand_idx = np.where(mask_ahead)[0]
    if cand_idx.size == 0:
        nxt_idx = poly.shape[0] - 1
    else:
        best_local = cand_idx[np.argmax(s_from_curr[cand_idx])]
        nxt_idx = int(best_local)

    nxt_p = poly[nxt_idx]
    try:
        idx_xyz = prob_img.TransformPhysicalPointToContinuousIndex(tuple(map(float, nxt_p)))
        z = int(round(idx_xyz[2]))
        y = int(round(idx_xyz[1]))
        x = int(round(idx_xyz[0]))
        z = np.clip(z, 0, edt_mm.shape[0] - 1)
        y = np.clip(y, 0, edt_mm.shape[1] - 1)
        x = np.clip(x, 0, edt_mm.shape[2] - 1)
        nxt_r = float(edt_mm[z, y, x])
    except Exception:
        nxt_r = float(radius_mm)
    v_step = nxt_p - curr
    L_step = float(np.linalg.norm(v_step))
    if L_step > 1e-12:
        dir_step = v_step / L_step
        dot_step = np.clip(np.dot(dir_step, old_dir), -1.0, 1.0)
        ang_deg = float(np.degrees(np.arccos(dot_step)))
    else:
        ang_deg = 0.0

    # pack single candidate
    Pk = nxt_p.reshape(1, 3)
    Rk = np.asarray([nxt_r], float)
    Ak = np.asarray([ang_deg], float)
    Sk = np.asarray([chosen_score if chosen_score is not None else 0.0], float)

    best_poly = poly
    return Pk, Rk, Ak, Sk, best_poly


def _pick_reseed_node(
    vt: VesselTree,
    Gcent: nx.Graph,
    current_point: np.ndarray,
    *,
    alpha_prob: float = 0.3,
    beta_radius: float = 0.7,
    gamma_dist: float = 0.1,
) -> Optional[np.ndarray]:
    if getattr(vt, "_cov_kdt", None) is None or not vt.node_not_traversed:
        return None

    q = _to_np(current_point)
    best_score = -1e9
    best_p: Optional[np.ndarray] = None

    for nid in vt.node_not_traversed:
        dnode = Gcent.nodes[nid]
        p = _to_np(dnode.get("pos_phys", dnode.get("point")))
        if p is None or not np.all(np.isfinite(p)):
            continue

        r = float(dnode.get("radius_mm", dnode.get("MaximumInscribedSphereRadius", 0.1)))
        dist = float(np.linalg.norm(p - q))

        sc = beta_radius * r - gamma_dist * dist
        if sc > best_score:
            best_score = sc
            best_p = p

    return best_p


def _choose_graph_step_node(
    vt: VesselTree,
    Gcent: nx.Graph,
    cent_index: Dict[str, object],
    curr_pt: np.ndarray,
    prev_pt: np.ndarray,
    radius_mm: float,
    cfg: Dict,
) -> Tuple[Optional[int], Optional[int], Optional[np.ndarray]]:
    kd, ids = cent_index.get("kd"), cent_index.get("ids")
    if kd is None or not ids:
        return None, None, None

    curr = _to_np(curr_pt)
    prev = _to_np(prev_pt)

    disp = curr - prev
    nrm = float(np.linalg.norm(disp))
    if nrm > 1e-8:
        old_dir = disp / nrm
        use_angle = True
    else:
        old_dir = np.array([1.0, 0.0, 0.0], float)
        use_angle = False

    # nearest node to define u
    _, idx = kd.query(curr, k=1)
    u = int(ids[int(idx)])

    MIN_STRIDE_MM = float(cfg.get("MIN_STRIDE_MM", max(1.0, 0.5 * radius_mm)))
    MAX_STRIDE_MM = float(cfg.get("MAX_STRIDE_MM", max(8.0, 4.0 * radius_mm)))
    ANGLE_ALLOW_DEG = float(cfg.get("ANGLE_ALLOW_DEG", 150.0))+15
    cos_thr = float(np.cos(np.radians(ANGLE_ALLOW_DEG)))
    if getattr(vt, "_cov_kdt", None) is None or vt._cov_pts.shape[0] == 0:
        return u, None, None

    mask_un = ~vt._cov_mask
    if not np.any(mask_un):
        return u, None, None

    pts_un = vt._cov_pts[mask_un]
    ids_un = np.asarray(vt._cov_ids)[mask_un]

    kdt_un = KDTree(pts_un)
    R_SEARCH = float(cfg.get("GRAPH_SEARCH_RADIUS_MM", 1.5 * MAX_STRIDE_MM))
    idx_lists = kdt_un.query_ball_point(curr, r=R_SEARCH)
    if isinstance(idx_lists, int):
        cand_idx_local = [idx_lists]
    else:
        cand_idx_local = idx_lists

    if not cand_idx_local:
        return u, None, None

    cand_nodes = ids_un[np.array(cand_idx_local, dtype=int)]

    best_v: Optional[int] = None
    best_score = -1e9
    best_path_nodes: List[int] = []

    for v in cand_nodes:
        v = int(v)
        pv = _to_np(Gcent.nodes[v].get("pos_phys", Gcent.nodes[v].get("point")))
        if pv is None or not np.all(np.isfinite(pv)):
            continue

        step_vec = pv - curr
        step_len = float(np.linalg.norm(step_vec))
        if step_len < 1e-6:
            continue
        if step_len < MIN_STRIDE_MM or step_len > MAX_STRIDE_MM:
            continue

        diru = step_vec / step_len
        if use_angle:
            dotv = float(np.clip(np.dot(diru, old_dir), -1.0, 1.0))
            if dotv < cos_thr:
                continue
        else:
            dotv = 1.0
        score = 0.7 * dotv + 0.3 * (step_len / MAX_STRIDE_MM)

        if score > best_score:
            best_score = score
            best_v = v

    if best_v is None:
        return u, None, None
    try:
        path_nodes = nx.shortest_path(Gcent, source=u, target=best_v, weight="length")
    except Exception:
        path_nodes = [u, best_v]

    pts = [curr.copy()]
    for nid in path_nodes:
        p_n = _to_np(Gcent.nodes[nid].get("pos_phys", Gcent.nodes[nid].get("point")))
        if p_n is None:
            continue
        if np.linalg.norm(p_n - pts[-1]) > 1e-6:
            pts.append(p_n)

    if len(pts) < 2:
        return u, None, None

    poly_uv = np.vstack(pts)
    poly_uv = _smooth_polyline_spline_keep_ends(
        poly_uv,
        resample_step_mm=float(cfg.get("CENTERLINE_RESAMPLE_MM", 0.5)),
        s=float(cfg.get("CENTERLINE_SMOOTH", 0.0)),
    )
    return u, best_v, poly_uv


def _extract_inside_points_mm(
    prob_img: sitk.Image,
    thresh: float = 0.5,
    stride: int = 4,
) -> np.ndarray:
    arr = sitk.GetArrayFromImage(prob_img)
    mask = arr >= float(thresh)
    if stride > 1:
        mask[::stride, :, :] = mask[::stride, :, :]
        mask[:, ::stride, :] = mask[:, ::stride, :]
        mask[:, :, ::stride] = mask[:, :, ::stride]
    idx_zyx = np.argwhere(mask)
    if idx_zyx.size == 0:
        return np.empty((0, 3), dtype=float)

    idx_xyz = idx_zyx[:, [2, 1, 0]].astype(float)

    origin = np.asarray(prob_img.GetOrigin(), float)
    spacing = np.asarray(prob_img.GetSpacing(), float)
    direction = np.asarray(prob_img.GetDirection(), float).reshape(3, 3)

    scaled = idx_xyz * spacing
    phys = (direction @ scaled.T).T + origin
    return phys


def _surface_from_prob(
    prob_img: sitk.Image,
    iso: float = 0.5,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    arr = sitk.GetArrayFromImage(prob_img).astype(np.float32)
    sp = np.array(prob_img.GetSpacing(), float)
    sp_zyx = sp[::-1]

    try:
        verts_zyx, faces, _, _ = measure.marching_cubes(
            arr, level=float(iso), spacing=tuple(sp_zyx)
        )
    except Exception:
        return None

    verts_xyz = verts_zyx[:, [2, 1, 0]]
    origin = np.asarray(prob_img.GetOrigin(), float)
    direction = np.asarray(prob_img.GetDirection(), float).reshape(3, 3)
    verts_mm = (direction @ verts_xyz.T).T + origin

    faces = faces.astype(np.int32)
    return verts_mm.astype(np.float32), faces


#main tracing loop

def trace_centerline(
    output_folder: str,
    image_file: str,
    case: str,
    model_folder: str,
    fold: int,
    *,
    seed_node: int,
    target_nodes: List[int],
    centerline_graph: nx.Graph,  # Gcent
    max_steps_per_component: int = 500,
    global_config: Optional[dict] = None,
    unit: str = "cm",
    scale: float = 1.0,
    seg_file: Optional[sitk.Image] = None,
    start_seg: Optional[sitk.Image] = None,
    write_samples: bool = True,
):

    assert global_config is not None, "global_config (YAML dict) is required"
    cfg = dict(global_config)

    scale_unit = 0.1 if unit == "cm" else 1.0
    SEGMENTATION = bool(cfg.get("SEGMENTATION", False))
    if SEGMENTATION and isinstance(seg_file, str):
        reader_im, origin_im, size_im, spacing_im = import_image(seg_file)
        image_file_effective = seg_file
    else:
        reader_im, origin_im, size_im, spacing_im = import_image(image_file)
        image_file_effective = image_file

    try:
        geom_img = sitk.ReadImage(image_file_effective)
    except Exception:
        geom_img = sitk.Image(
            int(size_im[0]), int(size_im[1]), int(size_im[2]), sitk.sitkUInt8
        )
        geom_img.SetOrigin(tuple(map(float, origin_im)))
        geom_img.SetSpacing(tuple(map(float, spacing_im)))
        geom_img.SetDirection((1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0))
    predictor = initialize_predictor(model_folder, fold)
    print("[INFO] predictor initialized")

    Gcent = centerline_graph
    cent_index = _make_index(Gcent)

    vt = VesselTree(
        case=case,
        image_file=image_file_effective,
        seed_id=int(seed_node),
        target_ids=list(map(int, target_nodes)),
        graph=Gcent,
        centerline_pd=None,
        centerline_graph=Gcent,
    )
    vt.geom_img = geom_img

    vt_init_coverage(vt, Gcent)
    assembly_segs = Segmentation(
        case,
        image_file_effective,
        weighted=bool(cfg.get("WEIGHTED_ASSEMBLY", False)),
        weight_type=cfg.get("WEIGHT_TYPE", "radius"),
        start_seg=start_seg,
    )

    list_surfaces: List = []
    list_points: List[np.ndarray] = []
    list_inside_points: List[np.ndarray] = []
    list_centerlines: List[np.ndarray] = []

    seed_pt = _to_np(
        Gcent.nodes[int(seed_node)].get(
            "pos_phys", Gcent.nodes[int(seed_node)].get("point")
        )
    )
    seed_r = float(
        Gcent.nodes[int(seed_node)].get(
            "radius_mm",
            Gcent.nodes[int(seed_node)].get("MaximumInscribedSphereRadius", 0.1),
        )
    )
    vt.steps = [
        {
            "old point": seed_pt,
            "point": seed_pt,
            "old radius": seed_r,
            "radius": seed_r,
            "tangent": np.array([1.0, 0.0, 0.0], float),
            "angle change": 0.0,
        }
    ]
    list_points.append(seed_pt)

    VOLUME_SIZE_RATIO = float(cfg.get("VOLUME_SIZE_RATIO", 2.0))
    MAGN_RADIUS = float(cfg.get("MAGN_RADIUS", 0.5))
    ADD_RADIUS = float(cfg.get("ADD_RADIUS", 1)) * scale_unit
    MIN_RES = int(cfg.get("MIN_RES", 8))

    STALL_MAX_STEPS = int(cfg.get("STALL_MAX_STEPS", 7))
    RESEED_ALPHA_PROB = float(cfg.get("RESEED_ALPHA_PROB", 0.3))
    RESEED_BETA_RADIUS = float(cfg.get("RESEED_BETA_RADIUS", 0.7))
    RESEED_GAMMA_DIST = float(cfg.get("RESEED_GAMMA_DIST", 0.1))
    COVERAGE_STOP = float(cfg.get("COVERAGE_STOP", 0.99))
    MAX_RESEED_NO_GAIN = float(cfg.get('MAX_RESEED_NO_GAIN',100))
    STOP_WHEN_ALL_TARGETS = bool(cfg.get("STOP_WHEN_ALL_TARGETS", False))

    def _targets_remaining(vt_: VesselTree) -> List[int]:
        ids = set(getattr(vt_, "target_ids", []) or [])
        return [t for t in ids if t not in vt_.node_traversed]

    no_cover_streak = 0
    reseed_count = 0
    i = 1

    while i <= max_steps_per_component:
        step = vt.steps[-1]
        curr_pt = _to_np(step["point"])
        old_pt = _to_np(step["old point"])
        curr_rad = float(step["radius"])

        prob_prediction = None

        try:
            idx_clamped, size_clamped, border_flag = map_to_image(
                center_phys=curr_pt,
                box_radius_mm=(curr_rad + ADD_RADIUS) * MAGN_RADIUS,
                volume_size_ratio=VOLUME_SIZE_RATIO,
                image=geom_img,
                min_res=MIN_RES,
            )
            cropped_vol = extract_volume(reader_im, idx_clamped, size_clamped)

            if predictor is not None:
                spacing_pred_vec = (
                    np.asarray(geom_img.GetSpacing(), float) * float(scale)
                ).tolist()[::-1]
                img_np = sitk.GetArrayFromImage(cropped_vol)[None].astype("float32")
                t0 = time.time()
                pred = predictor.predict_single_npy_array(
                    img_np, {"spacing": spacing_pred_vec}, None, None, True
                )
                prob_arr = np.clip(pred[1][1], 0, 1).astype(np.float32)
                prob_prediction = sitk.GetImageFromArray(prob_arr)
                prob_prediction = sitk.GetImageFromArray((pred[0] > 0).astype(np.uint8))
                prob_prediction = copy_settings(prob_prediction, cropped_vol)
                prob_prediction = copy_settings(prob_prediction, cropped_vol)
                print(f"[FWD] forward_time={time.time() - t0:.3f}s")
                print('crop size:',prob_prediction.GetSize())
            else:
                print("[ERR] predictor not initialized properly")
                break

            seed_vox = (np.rint(np.array(size_clamped) / 2).astype(int)).tolist()
            #prob_prediction  = largest_cc(sitk.Cast(prob_prediction, sitk.sitkUInt8), background_value=0)
            prob_prediction = remove_other_vessels(prob_prediction, seed_vox)

            assembly_segs.add_segmentation(
                prob_prediction, idx_clamped, size_clamped, curr_rad
            )

            if write_samples and prob_prediction is not None:
                inside_pts_mm = _extract_inside_points_mm(
                    prob_prediction,
                    thresh=float(cfg.get("INSIDE_THRESH", 0.5)),
                    stride=int(cfg.get("INSIDE_SAMPLE_STRIDE", 1)),
                )
                if inside_pts_mm.size > 0:
                    list_inside_points.append(inside_pts_mm)

                surf = _surface_from_prob(prob_prediction, iso=0.5)
                if surf is not None:
                    list_surfaces.append(surf)

            if check_seg_border(
                size_clamped, idx_clamped, prob_prediction, np.array(geom_img.GetSize(), int)
            ):
                print("[STOP] segmentation crop touched global border; stopping component")
                #break

        except Exception as e:
            print(f"[ERR crop/predict] {e}")
        picked = False
        centerline_poly = None
        nxt_p = None
        nxt_r = curr_rad

        try:
            cl_pts, cl_rad, cl_ang, cl_sc, cl_poly = get_next_points_from_crop_centerline(
                prob_img=prob_prediction,
                curr_mm=curr_pt,
                prev_mm=old_pt,
                radius_mm=curr_rad,
                cent_index=cent_index,
                cfg=cfg,
                k_best=1,
                vol_index=None,
                vt=vt,
            )
            if cl_pts is not None and cl_pts.size > 0 and len(list_points) >= 3:
                last_pts = np.vstack(list_points[-3:])
                keep_mask = np.ones(cl_pts.shape[0], dtype=bool)
                rmin = float(cfg.get("RETRACE_MIN_DIST_MM", 3))
                for j in range(cl_pts.shape[0]):
                    dists = np.linalg.norm(last_pts - cl_pts[j], axis=1)
                    if np.min(dists) < rmin:
                        keep_mask[j] = False
                cl_pts = cl_pts[keep_mask]
                cl_rad = cl_rad[keep_mask]
                cl_ang = cl_ang[keep_mask]
                cl_sc = cl_sc[keep_mask]

            if (
                cl_pts is not None
                and cl_pts.size > 0
                and cl_poly is not None
                and cl_poly.size > 0
            ):
                centerline_poly = cl_poly
                nxt_p = cl_poly[-1]
                nxt_r = float(cl_rad[0]) if cl_rad.size > 0 else curr_rad
                picked = True
                print(
                    f"[STEP] local crop centerline, score={float(cl_sc[0]) if cl_sc.size>0 else 0:.3f}"
                )

            if (not picked) or nxt_p is None or centerline_poly is None:
                u, v, poly_uv = _choose_graph_step_node(
                    vt=vt,
                    Gcent=Gcent,
                    cent_index=cent_index,
                    curr_pt=curr_pt,
                    prev_pt=old_pt,
                    radius_mm=curr_rad,
                    cfg=cfg,
                )
                if v is not None and poly_uv is not None and poly_uv.size > 0:
                    centerline_poly = poly_uv
                    nxt_p = poly_uv[-1]
                    nxt_r = curr_rad
                    picked = True
                    print("[STEP] graph step used")

            if not picked or nxt_p is None or centerline_poly is None:
                print("[DBG] crop + graph step failed → GLOBAL reseed on Gcent")
                jump_pt = _pick_reseed_node(
                    vt,
                    Gcent,
                    curr_pt,
                    alpha_prob=RESEED_ALPHA_PROB,
                    beta_radius=RESEED_BETA_RADIUS,
                    gamma_dist=RESEED_GAMMA_DIST,
                )
                if jump_pt is None:
                    print("[DBG re-seed] no valid re-seed found; stopping.")
                    break
                reseed_count+=1


                vt.steps.append(
                    {
                        "old point": jump_pt,
                        "point": jump_pt,
                        "old radius": curr_rad,
                        "radius": curr_rad,
                        "tangent": step["tangent"],
                        "angle change": 0.0,
                    }
                )
                list_points.append(jump_pt)
                no_cover_streak = 0
                i += 1
                continue
            list_centerlines.append(centerline_poly)

            vt.steps.append(
                {
                    "old point": curr_pt,
                    "point": nxt_p,
                    "old radius": curr_rad,
                    "radius": nxt_r,
                    "tangent": _unit(nxt_p - curr_pt),
                    "angle change": 0.0,
                }
            )
            list_points.append(nxt_p)

            newly = vt_mark_covered_from_poly(
                vt,
                centerline_poly,
                nxt_r,
                radius_scale=float(cfg.get("COVER_RADIUS_SCALE", 1)),
            )
            cov = vt_coverage_ratio(vt)
            if newly > 0:
                no_cover_streak = 0
            else:
                no_cover_streak += 1
            # stall-based reseed
            if no_cover_streak >= STALL_MAX_STEPS:
                print("[STALL] triggering auto GLOBAL re-seed due to coverage stall.")
                jump_pt = _pick_reseed_node(
                    vt,
                    Gcent,
                    curr_pt,
                    alpha_prob=RESEED_ALPHA_PROB,
                    beta_radius=RESEED_BETA_RADIUS,
                    gamma_dist=RESEED_GAMMA_DIST,
                )
                if jump_pt is None:
                    print("[DBG re-seed] no valid re-seed found; stopping.")
                    break

                vt.steps.append(
                    {
                        "old point": jump_pt,
                        "point": jump_pt,
                        "old radius": curr_rad,
                        "radius": curr_rad,
                        "tangent": step["tangent"],
                        "angle change": 0.0,
                    }
                )
                list_points.append(jump_pt)
                no_cover_streak = 0
                reseed_count += 1
                i += 1
                continue

            if STOP_WHEN_ALL_TARGETS and len(_targets_remaining(vt)) == 0:
                print("[STOP] all targets reached; stopping component.")
                break
            if cov >= COVERAGE_STOP or vt._cov_mask.sum() >= vt._cov_mask.size:
                print(
                    f"[STOP] coverage {100*cov:.1f}% ≥ {100*COVERAGE_STOP:.0f}% "
                    f"or all nodes covered; stopping."
                )
                break
            if reseed_count > MAX_RESEED_NO_GAIN:
                print('[STOP] Reached maximum reseeds')
                break

            print(f"[STEP] local/graph -> coverage {100*cov:.1f}%")
            i += 1
            continue

        except Exception as e:
            print(f"[ERR decision step {i}] {e}")
            break
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
