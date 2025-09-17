# ---------------------------------------------------------------------
# Graph-guided SeqSeg tracer (free explore; unified candidates; verbose)
#   • Candidates per step = CL neighbors + KNN VOL nodes + IMG ray peaks
#   • Image evidence uses a fixed saturating squash (no per-step norm)
#   • Frontier bonus for coverage hunger
#   • ε-greedy Top-K selection + momentum extension
#   • Probabilities:
#       – Centerline (CL): constant probability for all CL edges/nodes
#       – Volumetric (VOL): use existing edge_prob on Gvol edges as-is
#       – Image (IMG): unchanged (ray tube integrals)
#   • Auto-reseed when:
#       – zero-coverage streak is hit
#       – fused best score stays below threshold for a streak
#       – no candidates are found at a step
#   • Early stop only when:
#       – coverage goal is met
#       – max steps per branch are reached
#   • Lean code: caches KD-trees, minimizes SITK⇄NumPy conversions
#   • VERBOSE logging of key decisions: ROI, candidates, pick, momentum, coverage
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import time
from typing import List, Tuple, Optional

import numpy as np
import SimpleITK as sitk
import networkx as nx
from scipy.spatial import cKDTree as KDTree

# --- SeqSeg modules (I/O + nnU-Net + assembly) ---
from SeqSeg.seqseg.modules.sitk_functions import (
    import_image, extract_volume, copy_settings, remove_other_vessels, check_seg_border
)
from SeqSeg.seqseg.modules.nnunet import initialize_predictor
from SeqSeg.seqseg.modules.assembly import Segmentation

from seqseg_built_from_scratch.assembly import VesselTree


# ----------------- small helpers -----------------

def _to_np(p) -> np.ndarray:
    return np.asarray(p, dtype=float)

def _phys_tuple(p):
    a = np.asarray(p, np.float64).reshape(3)
    return (float(a[0]), float(a[1]), float(a[2]))

def _unit(v) -> np.ndarray:
    v = _to_np(v)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else np.array([1.0, 0.0, 0.0], float)

def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = float(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    return 1.0 / (1.0 + np.exp(-z))

def _squash01_exp(x: np.ndarray, k: float) -> np.ndarray:
    """Absolute, fixed saturating map to [0,1] (no per-step normalization)."""
    x = np.asarray(x, float)
    x = np.maximum(x, 0.0)
    return 1.0 - np.exp(-k * x)

def _normalize_node_attrs_to_phys(G: nx.Graph) -> None:
    """Ensure 'point' (mm) and 'radius' exist if alternatives are present."""
    for _, d in G.nodes(data=True):
        if ('point' not in d) or (d['point'] is None):
            p = d.get('pos_phys', d.get('pos_mm', None))
            if p is not None:
                d['point'] = _to_np(p)
        if ('radius' not in d) or (d['radius'] is None):
            r = d.get('radius_mm', d.get('MaximumInscribedSphereRadius', None))
            if r is not None:
                try:
                    d['radius'] = float(r)
                except Exception:
                    pass

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

def _nearest_node(kdt, ids, point):
    if kdt is None or not ids:
        return None
    _, idx = kdt.query(_to_np(point), k=1)
    return ids[int(idx)]

def _mk_step_dict(old_point, old_radius, new_point, new_radius, tangent, angle_change: Optional[float] = None) -> dict:
    d = {
        'old point': _to_np(old_point),
        'point': _to_np(new_point),
        'old radius': float(old_radius),
        'radius': float(new_radius),
        'tangent': _unit(tangent),
        'prob_predicted_vessel': None,
    }
    if angle_change is not None:
        d['angle change'] = float(angle_change)
    return d


# -------- phys/idx transforms ----------

def _phys_to_idx(img: sitk.Image, p_phys: np.ndarray) -> np.ndarray:
    ci = np.array(img.TransformPhysicalPointToContinuousIndex(_phys_tuple(p_phys)), float)
    sz = np.array(img.GetSize(), float)
    return np.clip(ci, 0, sz - 1)

def _snap_phys_inside(image: sitk.Image, p_xyz: np.ndarray) -> np.ndarray:
    ci = np.array(image.TransformPhysicalPointToContinuousIndex(_phys_tuple(p_xyz)))
    sz = np.array(image.GetSize(), int)
    ci = np.clip(ci, 0, sz - 1)
    return np.array(image.TransformContinuousIndexToPhysicalPoint(tuple(ci.tolist())))


# -------- image scoring (SeqSeg-like) ----------

def _line_integral_prob(prob_img: sitk.Image,
                        start_phys: np.ndarray,
                        dir_unit_phys: np.ndarray,
                        spacing_xyz: np.ndarray,
                        ray_len_mm: float,
                        step_mm: float,
                        tube_sigma_mm: float) -> float:
    steps = max(1, int(np.ceil(ray_len_mm / max(step_mm, 1e-6))))
    sig_vox = np.maximum(tube_sigma_mm / np.maximum(spacing_xyz, 1e-6), 0.5)
    rad_vox = np.ceil(2.5 * sig_vox).astype(int)
    arr = sitk.GetArrayFromImage(prob_img)  # (z,y,x)
    sz_xyz = np.array(prob_img.GetSize(), int)
    total = 0.0
    p = np.asarray(start_phys, float)
    for _ in range(1, steps + 1):
        p = p + step_mm * dir_unit_phys
        ci_xyz = _phys_to_idx(prob_img, p)
        cx, cy, cz = [int(round(v)) for v in ci_xyz]
        x0, x1 = max(0, cx - rad_vox[0]), min(sz_xyz[0] - 1, cx + rad_vox[0])
        y0, y1 = max(0, cy - rad_vox[1]), min(sz_xyz[1] - 1, cy + rad_vox[1])
        z0, z1 = max(0, cz - rad_vox[2]), min(sz_xyz[2] - 1, cz + rad_vox[2])
        if x0 > x1 or y0 > y1 or z0 > z1:
            continue
        xs = np.arange(x0, x1 + 1) - ci_xyz[0]
        ys = np.arange(y0, y1 + 1) - ci_xyz[1]
        zs = np.arange(z0, z1 + 1) - ci_xyz[2]
        gx = np.exp(-0.5 * (xs / max(sig_vox[0], 1e-6)) ** 2)
        gy = np.exp(-0.5 * (ys / max(sig_vox[1], 1e-6)) ** 2)
        gz = np.exp(-0.5 * (zs / max(sig_vox[2], 1e-6)) ** 2)
        g = gz[:, None, None] * gy[None, :, None] * gx[None, None, :]
        patch = arr[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
        total += float((patch * g).sum())
    return total / float(steps)

def _ray_profile(prob_img: sitk.Image,
                 start_phys: np.ndarray,
                 dir_unit_phys: np.ndarray,
                 spacing_xyz: np.ndarray,
                 ray_len_mm: float,
                 step_mm: float,
                 tube_sigma_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return distances (mm) and per-step tube integrals along a ray."""
    steps = max(1, int(np.ceil(ray_len_mm / max(step_mm, 1e-6))))
    t = np.arange(1, steps + 1, dtype=float) * step_mm
    vals = []
    p = np.asarray(start_phys, float)
    for _ in t:
        v = _line_integral_prob(prob_img, p, dir_unit_phys, spacing_xyz, step_mm, step_mm, tube_sigma_mm)
        vals.append(v)
        p = p + step_mm * dir_unit_phys
    return t, np.asarray(vals, float)

def _nms1d(y: np.ndarray, radius: int = 2, min_prom: float = 0.0, max_k: int = 3):
    idx = []
    y = np.asarray(y, float)
    used = np.zeros_like(y, dtype=bool)
    for _ in range(max_k):
        j = int(np.nanargmax(y))
        if not np.isfinite(y[j]) or y[j] <= min_prom:
            break
        idx.append(j)
        lo, hi = max(0, j - radius), min(len(y), j + radius + 1)
        used[lo:hi] = True
        y[used] = -np.inf
    return idx


# -------- ROI helper ----------

def map_to_image_diraware(center_phys, box_radius_mm: float, volume_size_ratio: float, *, image: sitk.Image,
                          min_res: int = 8, require_odd: bool = True) -> Tuple[list[int], list[int], bool]:
    if image is None:
        raise ValueError("map_to_image_diraware requires a SimpleITK image.")
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


# ----------------- probabilities & coverage -----------------

def ensure_node_probability_from_edges(
    G: nx.Graph,
    node_prob_key: str = "node_prob",
    edge_prob_key: str = "edge_prob",
    default: float = 0.5
) -> None:
    """Derive a node probability as the mean of incident 'edge_prob' values; used as fallback."""
    for n in G.nodes():
        neigh = list(G.neighbors(n))
        if not neigh:
            G.nodes[n][node_prob_key] = float(G.nodes[n].get(node_prob_key, default))
            continue
        vals = []
        for u in neigh:
            e = G.edges[n, u]
            vals.append(float(e.get(edge_prob_key, e.get("prob", default))))
        G.nodes[n][node_prob_key] = float(np.mean(vals)) if vals else float(default)

def set_centerline_constant_probability(Gcl: nx.Graph, const: float = 1.0) -> None:
    """Assign a constant probability to all CL nodes and edges (stored as 'node_prob' and 'cl_prob')."""
    const = float(const)
    for n in Gcl.nodes():
        Gcl.nodes[n]['node_prob'] = const
        Gcl.nodes[n]['node_prob_smooth'] = const
    for u, v in Gcl.edges():
        Gcl.edges[u, v]['cl_prob'] = const

def _frontier_boost(vt: VesselTree, p_phys, k: int = 16, r_mm: float = 6.0) -> float:
    if vt is None or getattr(vt, '_cov_kdt', None) is None or not len(getattr(vt, '_cov_pts', [])):
        return 0.0
    d, j = vt._cov_kdt.query(_to_np(p_phys), k=min(k, len(vt._cov_pts)))
    idxs = np.atleast_1d(j); dists = np.atleast_1d(d)
    cnt = 0; tot = 0
    for di, ji in zip(dists, idxs):
        if di > r_mm:
            continue
        nid = vt._cov_ids[int(ji)]
        tot += 1
        if nid in vt.node_not_traversed:
            cnt += 1
    return float(cnt) / float(max(1, tot))

def vt_init_coverage(vt: VesselTree, G_for_cov: nx.Graph):
    if not hasattr(vt, "node_traversed") or vt.node_traversed is None:
        vt.node_traversed = set()
    if not hasattr(vt, "node_not_traversed") or vt.node_not_traversed is None:
        vt.node_not_traversed = set(G_for_cov.nodes())
    pts, ids = [], []
    for n, d in G_for_cov.nodes(data=True):
        p = d.get("point", None)
        if p is not None:
            pts.append(_to_np(p)); ids.append(n)
    vt._cov_ids = ids
    vt._cov_pts = np.vstack(pts) if pts else np.zeros((0, 3), float)
    vt._cov_kdt = KDTree(vt._cov_pts) if len(vt._cov_pts) else None
    deg = dict(G_for_cov.degree())
    mdeg = float(max(1, max(deg.values()) if len(deg) else 1))
    vt._centrality = {n: (deg.get(n, 0) / mdeg) for n in G_for_cov.nodes()}
def vt_mark_covered_by_segment(vt: VesselTree, G_for_cov: nx.Graph, p0: np.ndarray, p1: np.ndarray,
                               *, step_mm: float = 1.2, radius_scale: float = 1.5) -> int:
    if vt._cov_kdt is None or len(vt._cov_pts) == 0:
        return 0
    p0 = _to_np(p0); p1 = _to_np(p1)
    seg = p1 - p0
    L = float(np.linalg.norm(seg))
    if L < 1e-6:
        # point-coverage case
        idxs = vt._cov_kdt.query_ball_point(p0, r=radius_scale * 1.0)  # fallback r=1mm
        newly = {vt._cov_ids[i] for i in idxs if vt._cov_ids[i] not in vt.node_traversed}
        vt.node_traversed |= newly; vt.node_not_traversed -= newly
        return len(newly)

    n = max(1, int(np.ceil(L / max(step_mm, 1e-6))))
    newly = set()
    # radius per sample uses nearest node’s radius (cheap and works well)
    for t in np.linspace(0.0, 1.0, n+1):
        s = p0 + t * seg
        d, j = vt._cov_kdt.query(s, k=1)
        center_id = vt._cov_ids[int(j)]
        r = float(G_for_cov.nodes[center_id].get("radius",
                    G_for_cov.nodes[center_id].get("MaximumInscribedSphereRadius", 1.0)))
        idxs = vt._cov_kdt.query_ball_point(s, r=radius_scale * max(r, 1e-6))
        for i in idxs:
            nid = vt._cov_ids[int(i)]
            if nid not in vt.node_traversed:
                newly.add(nid)
    vt.node_traversed |= newly
    vt.node_not_traversed -= newly
    return len(newly)


def vt_coverage_ratio(vt: VesselTree, total_nodes: int) -> float:
    return len(vt.node_traversed) / max(1, int(total_nodes))

def _append_seed_step(vt: VesselTree, seed_pt: np.ndarray, seed_r: float, tangent_hint: Optional[np.ndarray] = None):
    if tangent_hint is None:
        tangent_hint = _unit(vt.steps[-1]['tangent']) if getattr(vt, 'steps', []) else np.array([1.0,0.0,0.0], float)
    vt.steps.append(_mk_step_dict(seed_pt, seed_r, seed_pt, seed_r, _unit(tangent_hint), angle_change=0.0))

def _targets_remaining(vt: VesselTree) -> List[int]:
    ids = set(getattr(vt, "target_ids", []) or [])
    return [t for t in ids if t not in vt.node_traversed]

def _pick_reseed_node(vt: VesselTree, Gcov: nx.Graph, current_point: np.ndarray, *,
                      prefer: str = "farthest", min_sep_mm: float = 8.0,
                      prob_key: str = "node_prob", alpha_prob: float = 0.6,
                      beta_radius: float = 0.25, gamma_dist: float = 0.08,
                      min_degree: int = 1, targets_first: bool = True) -> Optional[np.ndarray]:
    if vt._cov_kdt is None or not vt.node_not_traversed:
        return None
    targets_rem = set(_targets_remaining(vt))
    pool = list(vt.node_not_traversed)
    if targets_first and len(targets_rem) > 0:
        pool = [nid for nid in pool if nid in targets_rem]
    if len(pool) == 0:
        pool = list(vt.node_not_traversed)
    deg = dict(Gcov.degree())
    pool = [nid for nid in pool if deg.get(nid, 0) >= int(min_degree)] or pool
    # separation from already covered nodes
    if len(vt.node_traversed) > 0:
        pts_tr = [_to_np(Gcov.nodes[n]['point']) for n in vt.node_traversed if 'point' in Gcov.nodes[n]]
        if len(pts_tr) > 0:
            kdt_tr = KDTree(np.vstack(pts_tr))
            pool = [nid for nid in pool if kdt_tr.query(_to_np(Gcov.nodes[nid]['point']), k=1)[0] >= float(min_sep_mm)]
    q = _to_np(current_point)
    scores, cand_pts = [], []
    for nid in pool:
        dnode = Gcov.nodes[nid]
        if 'point' not in dnode:
            continue
        p = _to_np(dnode['point'])
        pr = float(dnode.get(prob_key, 1.0))  # CL uses constant node prob
        r = float(dnode.get('radius', dnode.get('MaximumInscribedSphereRadius', 1.0)))
        cent = float(getattr(vt, "_centrality", {}).get(nid, 0.0))
        dist = float(np.linalg.norm(p - q))
        scores.append(2.0 * cent + alpha_prob * pr + beta_radius * r - gamma_dist * dist)
        cand_pts.append(p)
    if not scores:
        return None
    order = np.argsort(-np.asarray(scores))
    topk = order[:min(64, len(order))]
    topk = sorted(topk, key=lambda i: np.linalg.norm(cand_pts[i] - q), reverse=(prefer == "farthest"))
    for i in topk:
        return cand_pts[int(i)]
    return None


# ----------------- candidate generation (unified pools) -----------------

def _gather_candidates_unified(
    vt: VesselTree,
    Gcent: nx.Graph,
    Gvol: nx.Graph,
    kdt_c, ids_c, kdt_v, ids_v,
    curr: np.ndarray,
    prev: np.ndarray,
    old_radius: float,
    prob_img: Optional[sitk.Image],
    *,
    # pool sizes
    K_VOL: int = 24,
    VOL_RADIUS_MM: float = 30.0,
    K_IMG: int = 6,
    IMG_MIN_PROM: float = 0.05,
    IMG_NMS_RADIUS: int = 2,
    # scoring params
    near_graph_tau_mm: float = 3.0,
    ray_len_mm: float = 20.0,
    ray_step_mm: float = 0.2,
    tube_sigma_mm: float = 0.3,
    # gentle priors
    prior_CL: float = 1.0,
    prior_VOL: float = 0.85,
    prior_IMG: float = 0.70,
    # misc
    min_radius: float = 0.0,
    add_radius: float = 0.0,
    dedup_eps_mm: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Return (points, radii, angles, features_dict) where features include:
      source_prior, near_score, eprob, img_raw, frontier, cl_prob, step_penalty, src
    """
    old_vec = _unit(curr - prev)
    spacing_xyz = None if prob_img is None else np.array(prob_img.GetSpacing(), float)

    pts, rads, angs = [], [], []
    source_prior, near_scores, eprobs, img_raw = [], [], [], []
    frontier, clprob, step_penalty, src_ids = [], [], [], []  # 0=CL, 1=VOL, 2=IMG

    def _append_candidate(p: np.ndarray, r_guess: float, src_prior: float, cl_prob_val: float = 0.0,
                          img_hint: Optional[float] = None, src_id: Optional[int] = None):
        # dedup
        if pts and np.min(np.linalg.norm(np.vstack(pts) - p, axis=1)) < float(dedup_eps_mm):
            return
        vec = _unit(p - curr)
        ang = _angle_deg(vec, old_vec)
        # micro-step guard
        r_eff = max((2.0/3.0)*r_guess + (1.0/3.0)*old_radius, float(min_radius))
        if np.linalg.norm(p - curr) <= 0.25 * max(r_eff, 1e-6):
            p = curr + 0.5 * r_eff * vec
        # near-graph proximity & VOL edge probability
        if kdt_v is not None and ids_v:
            d_vol, _ = kdt_v.query(p, k=1)
            near_val = float(np.exp(-max(0.0, d_vol) / max(near_graph_tau_mm, 1e-6)))
            nV_curr = _nearest_node(kdt_v, ids_v, curr)
            nV_to = _nearest_node(kdt_v, ids_v, p)
            if (nV_curr is not None) and (nV_to is not None) and Gvol.has_edge(nV_curr, nV_to):
                eprob = float(Gvol.edges[nV_curr, nV_to].get('edge_prob', Gvol.edges[nV_curr, nV_to].get('prob', 0.5)))
            else:
                # fallback if no edge exists
                eprob = float(Gvol.nodes[nV_to].get('node_prob', 0.5)) if (nV_to is not None) else 0.5
        else:
            near_val, eprob = 0.0, 0.5
        # image score (raw)
        if prob_img is not None:
            if img_hint is None:
                img_val = _line_integral_prob(prob_img, curr, vec, spacing_xyz, ray_len_mm, ray_step_mm, tube_sigma_mm)
            else:
                img_val = float(img_hint)
        else:
            img_val = 0.0
        # frontier & step penalty
        f = _frontier_boost(vt, p, k=16, r_mm=6.0)
        step_len = float(np.linalg.norm(p - curr))
        s_pref = 0.7 * max(old_radius, 1e-6)
        step_pen = max(0.0, step_len / max(s_pref, 1e-6) - 1.0)
        # append
        pts.append(p); rads.append(r_eff + add_radius); angs.append(ang)
        source_prior.append(src_prior); near_scores.append(near_val); eprobs.append(eprob)
        img_raw.append(img_val); frontier.append(f); clprob.append(float(cl_prob_val)); step_penalty.append(step_pen)
        if src_id is None:
            if abs(src_prior - prior_CL) < 1e-6: sid = 0
            elif abs(src_prior - prior_VOL) < 1e-6: sid = 1
            else: sid = 2
        else:
            sid = int(src_id)
        src_ids.append(sid)

    # --- Centerline neighbors (constant CL prob) ---
    nC = _nearest_node(kdt_c, ids_c, curr)
    if nC is not None:
        for nb in Gcent.neighbors(nC):
            if 'point' not in Gcent.nodes[nb]:
                continue
            p_c = _to_np(Gcent.nodes[nb]['point'])
            r_c = float(Gcent.nodes[nb].get('radius', old_radius))
            cprob = float(Gcent.edges[nC, nb].get('cl_prob', 1.0)) if Gcent.has_edge(nC, nb) else 1.0
            _append_candidate(p_c, r_c, prior_CL, cl_prob_val=cprob, src_id=0)

    # --- Volumetric KNN around current point ---
    if kdt_v is not None and ids_v:
        try:
            dK, jK = kdt_v.query(curr, k=min(K_VOL, len(ids_v)))
            if np.isscalar(jK):
                jK = [jK]
            for j in np.atleast_1d(jK):
                vn = ids_v[int(j)]
                p_v = _to_np(Gvol.nodes[vn].get('point', curr))
                if np.linalg.norm(p_v - curr) > VOL_RADIUS_MM:
                    continue
                r_v = float(Gvol.nodes[vn].get('radius', old_radius))
                _append_candidate(p_v, r_v, prior_VOL, cl_prob_val=0.0, src_id=1)
        except Exception:
            pass

    # --- Image ray peaks (forward + ±30°) ---
    if prob_img is not None:
        fwd = _unit(curr - prev)
        # build a stable basis
        a = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, fwd)) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        n1 = _unit(np.cross(fwd, a))
        def _rot(d, axis, deg):
            th = np.deg2rad(deg); axis = _unit(axis)
            return _unit(d*np.cos(th) + np.cross(axis, d)*np.sin(th) + axis*np.dot(axis, d)*(1-np.cos(th)))
        dirs = [fwd, _rot(fwd, n1, 30.0), _rot(fwd, n1, -30.0)]
        spacing_xyz = np.array(prob_img.GetSpacing(), float)
        added = 0
        for d in dirs:
            t, prof = _ray_profile(prob_img, np.asarray(curr, float), d, spacing_xyz, ray_len_mm, ray_step_mm, tube_sigma_mm)
            for idx in _nms1d(prof.copy(), radius=IMG_NMS_RADIUS, min_prom=IMG_MIN_PROM, max_k=2):
                p_img = curr + t[idx] * d
                # snap to nearest VOL node if very close (stay on rails when possible)
                if kdt_v is not None and ids_v:
                    dv, jn = kdt_v.query(p_img, k=1)
                    if dv < 3.0:
                        vn = ids_v[int(jn)]
                        p_img = _to_np(Gvol.nodes[vn].get('point', p_img))
                        r_guess = float(Gvol.nodes[vn].get('radius', old_radius))
                    else:
                        r_guess = old_radius
                else:
                    r_guess = old_radius
                _append_candidate(p_img, r_guess, prior_IMG, cl_prob_val=0.0, img_hint=float(prof[idx]), src_id=2)
                added += 1
                if added >= K_IMG: break
            if added >= K_IMG: break

    if not pts:
        return np.empty((0,3)), np.empty((0,)), np.empty((0,)), {}

    P = np.vstack(pts)
    R = np.asarray(rads, float)
    A = np.asarray(angs, float)
    feats = {
        'src_prior': np.asarray(source_prior, float),
        'near': np.asarray(near_scores, float),
        'eprob': np.asarray(eprobs, float),
        'img_raw': np.asarray(img_raw, float),
        'frontier': np.asarray(frontier, float),
        'cl_prob': np.asarray(clprob, float),
        'step_pen': np.asarray(step_penalty, float),
        'src': np.asarray(src_ids, int),
    }
    return P, R, A, feats


# ----------------- main tracer -----------------

def trace_centerline(
    output_folder: str,
    image_file: str,
    case: str,
    model_folder: str,
    fold: int,
    *,
    graph: nx.Graph,                 # volumetric graph (with edge_prob on edges)
    seed_node: int,
    target_nodes: List[int],
    centerline_graph: Optional[nx.Graph] = None,
    max_n_steps_per_branch: int = 5000,
    global_config: Optional[dict] = None,
    unit: str = 'cm',
    scale: float = 1.0,
    seg_file: Optional[str] = None,
    start_seg: Optional[sitk.Image] = None,
):
    assert global_config is not None, "global_config (YAML dict) is required"

    # units
    scale_unit = 0.1 if unit == 'cm' else 1.0

    # config
    cfg = dict(global_config)

    # Verbosity
    VERBOSE = int(cfg.get('VERBOSE', 1))  # 0=off, 1=brief, 2=full
    TOPK_LOG = int(cfg.get('TOPK_LOG', 8))

    def vlog(level: int, msg: str):
        if VERBOSE >= level:
            print(msg)

    SEGMENTATION = bool(cfg.get('SEGMENTATION', False))
    VOLUME_SIZE_RATIO = float(cfg.get('VOLUME_SIZE_RATIO', 2.0))
    MAGN_RADIUS = float(cfg.get('MAGN_RADIUS', 3))
    MIN_RADIUS = float(cfg.get('MIN_RADIUS', 4)) * scale_unit
    ADD_RADIUS = float(cfg.get('ADD_RADIUS', 1)) * scale_unit
    MIN_RES = int(cfg.get('MIN_RES', 8))

    # Exploration & reseed
    EXPLORATION_TOPK = int(cfg.get('EXPLORATION_TOPK', 5))
    EXPLORATION_EPS  = float(cfg.get('EXPLORATION_EPS', 0.15))

    # Threshold for RAW fused score (no normalization)
    SCORE_PASS_THR   = float(cfg.get('SCORE_PASS_THR', 0.35))
    BAD_STREAK_MAX   = int(cfg.get('BAD_STREAK_MAX', 2))

    MOMENTUM_ENABLE   = bool(cfg.get('MOMENTUM_ENABLE', True))
    MOMENTUM_MIN_COS  = float(cfg.get('MOMENTUM_MIN_COS', 0.90))
    MOMENTUM_MAX_MULT = float(cfg.get('MOMENTUM_MAX_MULT', 2.3))
    MOMENTUM_CONF_K   = float(cfg.get('MOMENTUM_CONF_K', 5.0))  # confidence steepness

    # Image squash
    IMG_SQUASH_K      = float(cfg.get('IMG_SQUASH_K', 0.01))

    # Early stop coverage
    COVERAGE_STOP = float(cfg.get('COVERAGE_STOP', 0.5))

    # Re-seed policy
    RESEED_MIN_SEP_MM   = float(cfg.get('RESEED_MIN_SEP_MM', 0.01))
    RESEED_POLICY       = cfg.get('RESEED_POLICY', 'farthest')
    RESEED_ALPHA_PROB   = float(cfg.get('RESEED_ALPHA_PROB', 0.6))
    RESEED_BETA_RADIUS  = float(cfg.get('RESEED_BETA_RADIUS', 0.25))
    RESEED_GAMMA_DIST   = float(cfg.get('RESEED_GAMMA_DIST', 0.08))
    RESEED_MIN_DEGREE   = int(cfg.get('RESEED_MIN_DEGREE', 1))
    RESEED_TARGETS_FIRST= bool(cfg.get('RESEED_TARGETS_FIRST', True))

    # Zero-coverage streak reseed knobs
    ZERO_COV_MAX = int(cfg.get('ZERO_COV_MAX', 12))
    RESEED_COOLDOWN_STEPS = int(cfg.get('RESEED_COOLDOWN_STEPS', 0))

    # Unified score weights
    W_SRC_PRIOR   = float(cfg.get('W_SRC_PRIOR',   0.10))
    W_NEAR_GRAPH  = float(cfg.get('W_NEAR_GRAPH',  0.9))
    W_EPROB       = float(cfg.get('W_EPROB',       0))
    W_IMG         = float(cfg.get('W_IMG',         0))
    W_FRONTIER    = float(cfg.get('W_FRONTIER',    0))
    W_CLPROB      = float(cfg.get('W_CLPROB',      0))
    W_ANG_PEN     = float(cfg.get('W_ANG_PEN',     0))
    W_STEP_PEN    = float(cfg.get('W_STEP_PEN',    0.1))

    # Image candidate params
    K_VOL          = int(cfg.get('K_VOL', 24))
    VOL_RADIUS_MM  = float(cfg.get('VOL_RADIUS_MM', 30.0))
    K_IMG          = int(cfg.get('K_IMG', 6))
    IMG_MIN_PROM   = float(cfg.get('IMG_MIN_PROM', 0.05))
    IMG_NMS_RADIUS = int(cfg.get('IMG_NMS_RADIUS', 12))
    DEDUP_EPS_MM   = float(cfg.get('DEDUP_EPS_MM', 0.2))
    NEAR_GRAPH_TAU_MM = float(cfg.get('NEAR_GRAPH_TAU_MM', 3.0))

    # Image score params
    UNIFY_RAY_LEN_MM     = float(cfg.get('UNIFY_RAY_LEN_MM', 20.0))
    UNIFY_RAY_STEP_MM    = float(cfg.get('UNIFY_RAY_STEP_MM', 4))
    UNIFY_TUBE_SIGMA_MM  = float(cfg.get('UNIFY_TUBE_SIGMA_MM', 0.3))

    # I/O (image or provided segmentation)
    if SEGMENTATION and seg_file:
        reader_im, origin_im, size_im, spacing_im = import_image(seg_file)
        image_file_effective = seg_file
        vlog(1, f"[IO] Using provided segmentation file: {seg_file}")
    else:
        reader_im, origin_im, size_im, spacing_im = import_image(image_file)
        image_file_effective = image_file
        vlog(1, f"[IO] Reading image: {image_file} (scale={scale})")

    try:
        geom_img = sitk.ReadImage(image_file_effective)
    except Exception:
        geom_img = sitk.Image(int(size_im[0]), int(size_im[1]), int(size_im[2]), sitk.sitkUInt8)
        geom_img.SetOrigin(tuple(map(float, origin_im)))
        geom_img.SetSpacing(tuple(map(float, spacing_im)))
        geom_img.SetDirection(tuple([1.0,0,0, 0,1.0,0, 0,0,1.0]))

    dir_mat = np.array(geom_img.GetDirection(), float).reshape(3, 3)
    vlog(2, f"[IMG] size={list(map(int, size_im))}, spacing={list(map(float, spacing_im))}, det(dir)={float(np.linalg.det(dir_mat)):.6f}")

    # Prepare graphs & probabilities
    _normalize_node_attrs_to_phys(graph)
    _normalize_node_attrs_to_phys(centerline_graph)
    ensure_node_probability_from_edges(graph, node_prob_key="node_prob", edge_prob_key="edge_prob", default=0.5)
    set_centerline_constant_probability(centerline_graph, const=float(cfg.get('CL_CONST_PROB', 1.0)))

    vlog(1, f"[GRAPH] CL nodes={centerline_graph.number_of_nodes()}, edges={centerline_graph.number_of_edges()} | "
             f"VOL nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}")

    # Predictor (for crops & scoring)
    predictor = None
    if not (SEGMENTATION and seg_file):
        try:
            t0 = time.time()
            predictor = initialize_predictor(model_folder, fold)
            vlog(1, f"[NN] nnU-Net predictor loaded (fold={fold}) in {time.time()-t0:.2f}s")
        except Exception as e:
            print("[WARN] predictor init failed:", e)

    # VesselTree & coverage
    vt = VesselTree(case=case, image_file=image_file_effective, seed_id=int(seed_node),
                    target_ids=list(map(int, target_nodes)), graph=graph,
                    centerline_pd=None, centerline_graph=centerline_graph)
    Gcov = centerline_graph
    vt_init_coverage(vt, Gcov)
    total_nodes_for_cov = Gcov.number_of_nodes()

    # KD-trees cache
    vt._kdt_c, vt._ids_c = _build_kdtree(centerline_graph)
    vt._kdt_v, vt._ids_v = _build_kdtree(graph)

    # Global assembly
    assembly_segs = Segmentation(case, image_file_effective,
                                 weighted=bool(cfg.get('WEIGHTED_ASSEMBLY', False)),
                                 weight_type=cfg.get('WEIGHT_TYPE', 'radius'),
                                 start_seg=start_seg)

    bad_streak = 0
    zero_cov_streak = 0
    zero_cov_streak_max = 0
    reseed_cooldown = 0
    reseed_count_zero_cov = 0

    # Seed step
    if not hasattr(vt, "steps") or len(vt.steps) == 0:
        seed_pt = _to_np(Gcov.nodes[int(seed_node)]['point'])
        seed_r = float(Gcov.nodes[int(seed_node)].get('radius', Gcov.nodes[int(seed_node)].get('MaximumInscribedSphereRadius', 1.0)))
        _append_seed_step(vt, seed_pt, seed_r, tangent_hint=np.array([1.0,0.0,0.0]))
        vlog(1, f"[START] case={case} seed_id={seed_node} seed_r={seed_r:.2f}mm targets={len(target_nodes)}")
        vlog(2, f"[CFG] eps_topk={EXPLORATION_TOPK}, eps_prob={EXPLORATION_EPS}, raw_thr={SCORE_PASS_THR}, bad_streak_max={BAD_STREAK_MAX}")

    i = 1

    # Centralized reseed helper enforcing the policy
    def _attempt_reseed(reason: str) -> bool:
        nonlocal i, bad_streak, zero_cov_streak, reseed_cooldown
        vlog(1, f"[RESEED] {reason} → picking new seed")
        curr_idx = min(i-1, len(vt.steps)-1)
        curr_pt_local = vt.steps[curr_idx]['point']
        curr_rad_local = vt.steps[curr_idx]['radius']
        step_local = vt.steps[curr_idx]
        jump_pt = _pick_reseed_node(
            vt, Gcov, curr_pt_local, prefer=RESEED_POLICY,
            min_sep_mm=RESEED_MIN_SEP_MM,
            prob_key="node_prob",
            alpha_prob=RESEED_ALPHA_PROB,
            beta_radius=RESEED_BETA_RADIUS,
            gamma_dist=RESEED_GAMMA_DIST,
            min_degree=RESEED_MIN_DEGREE,
            targets_first=RESEED_TARGETS_FIRST
        )
        # Fallback: permissive reseed before giving up
        if jump_pt is None:
            jump_pt = _pick_reseed_node(
                vt, Gcov, curr_pt_local, prefer="farthest",
                min_sep_mm=0.0, prob_key="node_prob",
                alpha_prob=RESEED_ALPHA_PROB,
                beta_radius=RESEED_BETA_RADIUS,
                gamma_dist=0.0, min_degree=0,
                targets_first=False
            )
        if jump_pt is None:
            vlog(1, "[STOP] max steps per branch reached (no reseed candidate)")
            i = max_n_steps_per_branch + 1
            return False
        jump_pt = _snap_phys_inside(geom_img, jump_pt)
        _append_seed_step(vt, jump_pt, curr_rad_local, tangent_hint=step_local['tangent'])
        bad_streak = 0
        zero_cov_streak = 0
        reseed_cooldown = RESEED_COOLDOWN_STEPS
        i += 1
        return True

    while i <= max_n_steps_per_branch:
        if reseed_cooldown > 0:
            reseed_cooldown -= 1

        step = vt.steps[i - 1] if i - 1 < len(vt.steps) else vt.steps[-1]
        curr_pt = step['point']
        old_pt = step['old point']
        curr_rad = step['radius']
        cov_now = vt_coverage_ratio(vt, total_nodes_for_cov)
        vlog(1, f"[STEP {i}] r={curr_rad:.2f}mm cov={100*cov_now:.1f}% pt={np.round(curr_pt,2)}")

        # Crop & predict (for scoring + assembly)
        prob_prediction = None
        try:
            idx_clamped, size_clamped, border_flag = map_to_image_diraware(
                center_phys=curr_pt,
                box_radius_mm=(curr_rad + ADD_RADIUS) * MAGN_RADIUS,
                volume_size_ratio=VOLUME_SIZE_RATIO,
                image=geom_img,
                min_res=MIN_RES,
            )
            vlog(2, f"[ROI] idx={tuple(idx_clamped)}, size={tuple(size_clamped)}, border={border_flag}")
            cropped_vol = extract_volume(reader_im, idx_clamped, size_clamped)
            if predictor is not None:
                t0 = time.time()
                spacing_pred = (np.asarray(geom_img.GetSpacing(), float) * float(scale)).tolist()[::-1]
                img_np = sitk.GetArrayFromImage(cropped_vol)[None].astype('float32')
                pred = predictor.predict_single_npy_array(img_np, {'spacing': spacing_pred}, None, None, True)
                prob_prediction = sitk.GetImageFromArray(np.clip(pred[1][1], 0, 1).astype(np.float32))
                pred_img = sitk.GetImageFromArray((pred[0] > 0).astype(np.uint8))
                pred_img = copy_settings(pred_img, cropped_vol)
                prob_prediction = copy_settings(prob_prediction, cropped_vol)
                vlog(2, f"[NN] forward {time.time()-t0:.3f}s | crop_size={list(map(int, prob_prediction.GetSize()))}")
            else:
                pred_img = cropped_vol
                prob_prediction = cropped_vol
            # keep island near center
            seed_vox = (np.rint(np.array(size_clamped) / 2).astype(int)).tolist()
            pred_img = remove_other_vessels(pred_img, seed_vox)
            # global assembly
            assembly_segs.add_segmentation(prob_prediction, idx_clamped, size_clamped,
                                           (1.0 / max(curr_rad, 1e-3)) ** 2 if cfg.get('WEIGHTED_ASSEMBLY', False) else curr_rad)
            if check_seg_border(size_clamped, idx_clamped, pred_img, size_im):
                vlog(1, "[BORDER] crop touched global border")
                if not _attempt_reseed("crop touched border"):
                    continue
                continue
        except Exception as e:
            print(f"[WARN] crop/predict error at step {i}: {e}")
            prob_prediction = None

        # Gather unified candidates
        P, R, A, F = _gather_candidates_unified(
            vt,
            centerline_graph,
            graph,
            vt._kdt_c, vt._ids_c, vt._kdt_v, vt._ids_v,
            _to_np(curr_pt), _to_np(old_pt), float(curr_rad),
            prob_prediction,
            K_VOL=K_VOL, VOL_RADIUS_MM=VOL_RADIUS_MM,
            K_IMG=K_IMG, IMG_MIN_PROM=IMG_MIN_PROM, IMG_NMS_RADIUS=IMG_NMS_RADIUS,
            near_graph_tau_mm=NEAR_GRAPH_TAU_MM,
            ray_len_mm=UNIFY_RAY_LEN_MM, ray_step_mm=UNIFY_RAY_STEP_MM, tube_sigma_mm=UNIFY_TUBE_SIGMA_MM,
            min_radius=MIN_RADIUS, add_radius=ADD_RADIUS, dedup_eps_mm=DEDUP_EPS_MM,
        )

        if P.size == 0:
            if not _attempt_reseed("no local candidates"):
                continue
            continue

        # Feature shaping (NO per-step normalization)
        img = _squash01_exp(F['img_raw'], k=IMG_SQUASH_K)    # absolute squash 0..1
        near = np.clip(F['near'], 0, 1)
        eprob = np.clip(F['eprob'], 0, 1)
        frontier = np.clip(F['frontier'], 0, 1)
        clprob = np.clip(F['cl_prob'], 0, 1)
        src_prior = np.clip(F['src_prior'], 0, 1)
        ang_pen = np.asarray(A, float) / 180.0
        step_pen = np.asarray(F['step_pen'], float)
        src = np.asarray(F.get('src', np.zeros(len(P), int)), int)

        fused = (W_SRC_PRIOR*src_prior + W_NEAR_GRAPH*near + W_EPROB*eprob +
                 W_IMG*img + W_FRONTIER*frontier + W_CLPROB*clprob -
                 W_ANG_PEN*ang_pen - W_STEP_PEN*step_pen)

        order = np.argsort(-fused)
        P, R, A, fused = P[order], R[order], A[order], fused[order]
        near, eprob, frontier, clprob, src_prior, ang_pen, step_pen, img, src = (
            near[order], eprob[order], frontier[order], clprob[order],
            src_prior[order], ang_pen[order], step_pen[order], img[order], src[order]
        )

        # Verbose candidate summary
        if VERBOSE:
            cnt_cl  = int(np.sum(src==0)); cnt_vol = int(np.sum(src==1)); cnt_img = int(np.sum(src==2))
            print(f"[CANDS] total={len(P)} (CL={cnt_cl}, VOL={cnt_vol}, IMG={cnt_img}); best={fused[0]:.3f}")
            if VERBOSE >= 2:
                src_names = {0:"CL",1:"VOL",2:"IMG"}
                topn = int(min(TOPK_LOG, len(P)))
                print("[SCORES] top candidates:")
                for jj in range(topn):
                    dmm = np.linalg.norm(P[jj]-curr_pt)
                    print(
                        f"  #{jj:02d} src={src_names.get(int(src[jj]),'?')} dist={dmm:.2f}mm ang={A[jj]:.1f}"
                        f" near={near[jj]:.2f} eprob={eprob[jj]:.2f} img={img[jj]:.2f} front={frontier[jj]:.2f}"
                        f" clp={clprob[jj]:.2f} step_pen={step_pen[jj]:.2f} fused={fused[jj]:.3f}"
                    )

        # RAW fused score threshold (no normalization)
        if float(fused[0]) < SCORE_PASS_THR:
            bad_streak += 1
            print(f"[THR] fused_best={float(fused[0]):.3f} < {SCORE_PASS_THR:.3f} (streak {bad_streak}/{BAD_STREAK_MAX})")
            if bad_streak >= BAD_STREAK_MAX:
                if not _attempt_reseed("low-score streak"):
                    continue
                continue
        else:
            bad_streak = 0

        # Pick next (Top-K ε-greedy)
        K = int(min(EXPLORATION_TOPK, len(P)))
        pick = 0
        explore_flag = False
        if K > 1 and np.random.rand() < EXPLORATION_EPS:
            pick = int(np.random.choice(np.arange(K)))
            explore_flag = True

        src_names = {0:"CL",1:"VOL",2:"IMG"}
        if VERBOSE:
            print(
                f"[PICK] j={pick} src={src_names.get(int(src[pick]),'?')} fused={fused[pick]:.3f}"
                f" eps_explore={explore_flag} pt={np.round(P[pick],2)} r={R[pick]:.2f} ang={A[pick]:.1f}"
            )
            comp = {
                'src_prior': (W_SRC_PRIOR, src_prior[pick]),
                'near': (W_NEAR_GRAPH, near[pick]),
                'eprob': (W_EPROB, eprob[pick]),
                'img': (W_IMG, img[pick]),
                'frontier': (W_FRONTIER, frontier[pick]),
                'clprob': (W_CLPROB, clprob[pick]),
                'ang_pen': (-W_ANG_PEN, ang_pen[pick]),
                'step_pen': (-W_STEP_PEN, step_pen[pick]),
            }
            parts = [f"{k}={w:+.2f}×{v:.2f}={w*v:+.3f}" for k,(w,v) in comp.items()]
            print("[FUSE] " + ", ".join(parts) + f" → fused={fused[pick]:.3f}")

        nxt_p = _snap_phys_inside(geom_img, P[pick])
        nxt_r = float(R[pick])
        step_vec = (nxt_p - curr_pt)
        if (not np.any(np.isfinite(step_vec))) or (np.linalg.norm(step_vec) < 1e-9):
            vlog(1, "[STALL] degenerate step_vec → try next candidate")
            if len(P) > 1:
                P, R, A, fused = P[1:], R[1:], A[1:], fused[1:]
                near, eprob, frontier, clprob, src_prior, ang_pen, step_pen, img, src = (
                    near[1:], eprob[1:], frontier[1:], clprob[1:], src_prior[1:], ang_pen[1:], step_pen[1:], img[1:], src[1:]
                )
                continue
            else:
                if not _attempt_reseed("no viable movement from candidates"):
                    continue
                continue

        tangent = _unit(step_vec)
        # Momentum extension when aligned & confident
        if MOMENTUM_ENABLE:
            prev_tan = _unit(step.get('tangent', np.array([1.0,0.0,0.0])))
            cosang = float(np.clip(np.dot(prev_tan, tangent), -1.0, 1.0))
            if cosang >= MOMENTUM_MIN_COS:
                conf = float(_sigmoid(MOMENTUM_CONF_K * (float(fused[pick]) - SCORE_PASS_THR)))
                mul = min(MOMENTUM_MAX_MULT, 1.0 + 1.2 * conf)
                cand = _to_np(curr_pt) + mul * (nxt_p - _to_np(curr_pt))
                nxt_p = _snap_phys_inside(geom_img, cand)
                tangent = _unit(nxt_p - curr_pt)
                vlog(2, f"[MOMENTUM] cos={cosang:.3f} conf={conf:.2f} mul={mul:.2f}")

        vt.steps.append(_mk_step_dict(curr_pt, curr_rad, nxt_p, nxt_r, tangent, angle_change=float(A[pick])))
        newly = vt_mark_covered_by_segment(vt, Gcov, curr_pt, nxt_p, step_mm=1, radius_scale=12)
        cov = vt_coverage_ratio(vt, total_nodes_for_cov)
        vlog(1, f"[COVER] +{newly} → {len(vt.node_traversed)}/{total_nodes_for_cov} ({100*cov:.1f}%) targets_left={len(_targets_remaining(vt))}")

        # zero-coverage streak auto-reseed (with cooldown)
        if newly <= 0:
            zero_cov_streak += 1
            zero_cov_streak_max = max(zero_cov_streak_max, zero_cov_streak)
            if zero_cov_streak == 1 or zero_cov_streak % 10 == 0:
                vlog(1, f"[STALL] zero_cov={zero_cov_streak}/{ZERO_COV_MAX} cooldown={reseed_cooldown}")
        else:
            zero_cov_streak = 0

        if (zero_cov_streak >= ZERO_COV_MAX) and (reseed_cooldown == 0):
            if _attempt_reseed("zero-coverage streak"):
                reseed_count_zero_cov += 1
                continue
            continue

        # Early stop: coverage goal only
        if cov >= COVERAGE_STOP:
            print(f"[STOP] coverage {100*cov:.1f}%")
            break

        i += 1

    # free VRAM if torch is around
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    vlog(1, f"[SUMMARY] zero_cov_max_streak={zero_cov_streak_max} reseeds_from_zero_cov={reseed_count_zero_cov}")

    return ([],                # list_centerlines (unused)
            [],                # list_surfaces (unused)
            [],                # list_points (unused)
            [],                # list_inside_pts (unused)
            assembly_segs,     # Segmentation object (global assembly)
            vt,                # VesselTree with steps/history/coverage
            i)