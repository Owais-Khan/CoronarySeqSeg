# graph_only_centerline_medial_oriented.py
# GRAPH-ONLY centerline extraction with medial- & orientation-aware cost
# and optional graph-only "medial snap" back to high-radius nodes.

from typing import Optional, Tuple, List
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from SeqSeg.seqseg.modules.centerline import post_process_centerline
from SeqSeg.seqseg.modules.vtk_functions import write_vtk_polydata,smooth_polydata,smooth_surface



import SimpleITK as sitk
from gnn_model.gnn_modules import gm_load, gm_predict_graph, save_predicted_graph_to_vtp
from seqseg_modules_modified.trace_centerline import trace_centerline
from scipy.spatial import cKDTree as KDTree
import faulthandler, time, os, yaml, vtk

# --- Constants (STEP_MM controls output resampling density) ---
STEP_MM = 0.5
JUMP_MM = 1.0
LEAK_FRAC = 0.05
GRAD_FLOOR = 1e-6
MAX_ITERS = 4000
TT_MARGIN = 1e-6


# ============================== I/O utils ==============================

def create_directories(output_folder: str, write_samples: bool) -> None:
    base = Path(output_folder)
    for sub in ("", "errors", "assembly"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    if write_samples:
        for sub in ("volumes", "predictions", "centerlines", "surfaces",
                    "points", "animation", "images", "labels", "trace_artifacts"):
            (base / sub).mkdir(parents=True, exist_ok=True)


# ============================== Graph helpers ==============================

def _inside_frac_order(G: nx.Graph, img: sitk.Image, pos_key='pos', order=(2, 1, 0), sample=512) -> float:
    size = np.array(img.GetSize(), dtype=float)
    nodes = list(G.nodes())
    if not nodes:
        return 0.0
    if len(nodes) > sample:
        nodes = list(np.random.choice(nodes, size=sample, replace=False))
    ok = 0
    for n in nodes:
        p = np.asarray(G.nodes[n].get(pos_key, [np.inf, np.inf, np.inf]), float)
        if p.shape != (3,):
            continue
        idx_xyz = p[list(order)]
        if np.all(idx_xyz >= 0) and np.all(idx_xyz < size):
            ok += 1
    return ok / max(len(nodes), 1)


def attach_pos_idx_xyz(G: nx.Graph, img: sitk.Image, pos_key='pos',
                       try_orders=((2, 1, 0), (0, 1, 2))) -> Tuple[int, int, int]:
    best = max(try_orders, key=lambda ord_: _inside_frac_order(G, img, pos_key, ord_))
    for n in G.nodes():
        p = np.asarray(G.nodes[n].get(pos_key, [0, 0, 0]), float)
        if p.shape == (3,):
            G.nodes[n]['pos_idx_xyz'] = p[list(best)]
    return best


def _pad_vox(pad_mm, spacing_xyz):
    return np.ceil(np.asarray(pad_mm, float) / np.maximum(np.asarray(spacing_xyz, float), 1e-6)).astype(int)


def bbox_from_graph_component(Gc: nx.Graph, img: sitk.Image,
                              pos_idx_key='pos_idx_xyz',
                              pad_mm=(4, 4, 4),
                              min_size_mm=(12, 12, 12)) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    pts = []
    for n in Gc.nodes():
        if pos_idx_key in Gc.nodes[n]:
            pts.append(np.asarray(Gc.nodes[n][pos_idx_key], float))
    if not pts:
        return None
    P = np.vstack(pts)
    pmin = np.floor(P.min(axis=0)).astype(int)
    pmax = np.ceil(P.max(axis=0)).astype(int)

    pad_vx = _pad_vox(pad_mm, img.GetSpacing())
    min_vx = _pad_vox(min_size_mm, img.GetSpacing())

    pmin -= pad_vx
    pmax += pad_vx
    size = np.maximum(pmax - pmin + 1, min_vx)

    img_size = np.array(img.GetSize(), dtype=int)
    pmin = np.maximum(pmin, 0)
    pmax = np.minimum(pmin + size - 1, img_size - 1)
    size = (pmax - pmin + 1).astype(int)
    size = np.maximum(size, 1)

    start_xyz = tuple(int(v) for v in pmin.tolist())
    size_xyz = tuple(int(v) for v in size.tolist())
    return start_xyz, size_xyz


def largest_components(G: nx.Graph, k: int = 2, *, by: str = "nodes") -> List[nx.Graph]:
    if G.number_of_nodes() == 0:
        return []
    comps = [set(c) for c in nx.connected_components(G.to_undirected())]
    if not comps:
        return []
    if by == "edge_weight_sum":
        def score(nodeset):
            H = G.subgraph(nodeset)
            return sum(float(d.get('edge_prob', 1.0)) for _, _, d in H.edges(data=True))
    else:
        def score(nodeset):
            return len(nodeset)
    comps.sort(key=score, reverse=True)
    return [G.subgraph(c).copy() for c in comps[:max(1, int(k))]]


# ---------- metric attachments (selection + graph-only cost) ----------

def attach_edge_metrics_mm_from_phys(G: nx.Graph,
                                     *,
                                     prob_key: str = 'edge_prob',
                                     pos_phys_key: str = 'pos_phys',
                                     cost_key: str = 'length_cost',
                                     length_key: str = 'length_mm',
                                     prob_exp: float = 1.5) -> None:
    """
    Adds length_mm and a Dijkstra-friendly 'length_cost' used for seed/target selection.
    """
    for u, v in G.edges():
        pu = np.asarray(G.nodes[u][pos_phys_key], float)
        pv = np.asarray(G.nodes[v][pos_phys_key], float)
        L = float(np.linalg.norm(pu - pv))
        p = float(G.edges[u, v].get(prob_key, 0.1))
        G.edges[u, v][length_key] = L
        G.edges[u, v][cost_key] = L / max(p, 1e-3) ** prob_exp


def prune_short_or_lowprob_spurs(G: nx.Graph,
                                 *,
                                 length_key: str = 'length_mm',
                                 prob_key: str = 'edge_prob',
                                 Lspur_min_mm: float = 2.0,
                                 prob_min: float = 0.15) -> None:
    changed = True
    while changed:
        changed = False
        leaves = [n for n, d in G.degree() if d == 1]
        for n in leaves:
            nbrs = list(G.neighbors(n))
            if not nbrs:
                continue
            u = nbrs[0]
            e = G.edges[n, u]
            L = float(e.get(length_key, np.inf))
            p = float(e.get(prob_key, 1.0))
            if (L < Lspur_min_mm) or (p < prob_min):
                G.remove_node(n)
                changed = True


def prune_spurs_graph_only(G: nx.Graph,
                           length_key: str = 'length_mm',
                           prob_key: str = 'edge_prob',
                           Lspur_min_mm: float = 2.0,
                           prob_min: float = 0.15) -> None:
    prune_short_or_lowprob_spurs(G, length_key=length_key, prob_key=prob_key,
                                 Lspur_min_mm=Lspur_min_mm, prob_min=prob_min)


# ---------- medial & orientation aware routing cost ----------

def _safe_unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / max(n, 1e-12)

def attach_graph_only_cost_medial_oriented(
    G: nx.Graph,
    *,
    base_len_key: str = 'length_mm',       # present from attach_edge_metrics_mm_from_phys or gm_predict_graph
    prob_key: str = 'edge_prob',           # present from gm_predict_graph
    edge_r_key: str = 'radius_min_mm',     # prefer per-edge min radius (from gm_predict_graph)
    node_r_key: str = 'radius_mm',         # per-node radius (from gm_predict_graph)
    tangent_key: str = 'tangent',          # per-node tangent (from gm_predict_graph)
    out_cost_key: str = 'graph_cost',

    prob_exp: float = 1.5,                 # reliability preference
    beta_med: float = 1.5,                 # medial preference (↑ makes more central)
    lambda_drop: float = 0.75,             # penalize necking
    gamma_align: float = 1.0,              # reward alignment to tangents
    align_eps: float = 1e-3
) -> None:
    """
    cost(u,v) =  L / (p^prob_exp * r_edge^beta_med) * (1 + lambda_drop * drop) * (1 / align_mean^gamma_align)

      L: Euclidean length (mm)
      p: edge probability
      r_edge: per-edge radius (radius_min_mm preferred; fallback to radius_mean_mm or min(node radii))
      drop: ReLU((r_u - r_v) / r_u)
      align_mean: 0.5*( |dot(t_u, dir_uv)| + |dot(t_v, dir_vu)| ) \in (0,1]
    """
    EPS = 1e-6
    for u, v, e in G.edges(data=True):
        # length & prob
        L = float(e.get(base_len_key, 1.0))
        p = float(e.get(prob_key, 1.0))

        # choose a robust edge radius
        r_edge = e.get(edge_r_key, None)
        if r_edge is None:
            r_edge = e.get('radius_mean_mm', None)
        if r_edge is None:
            ru_tmp = float(G.nodes[u].get(node_r_key, 1.0))
            rv_tmp = float(G.nodes[v].get(node_r_key, ru_tmp))
            r_edge = min(ru_tmp, rv_tmp)

        # drop penalty (necking u -> v)
        ru = float(G.nodes[u].get(node_r_key, r_edge))
        rv = float(G.nodes[v].get(node_r_key, r_edge))
        drop = max(0.0, (ru - rv) / max(ru, EPS))

        # orientation reward (favor alignment with tangents)
        pu = np.asarray(G.nodes[u].get('pos_phys', None), float)
        pv = np.asarray(G.nodes[v].get('pos_phys', None), float)
        if pu is None or pv is None:
            align_mean = 1.0
        else:
            dir_uv = _safe_unit(pv - pu)
            dir_vu = -dir_uv
            tu = _safe_unit(G.nodes[u].get(tangent_key, dir_uv))
            tv = _safe_unit(G.nodes[v].get(tangent_key, dir_vu))
            a_u = abs(float(np.dot(tu, dir_uv)))
            a_v = abs(float(np.dot(tv, dir_vu)))
            align_mean = max(0.5 * (a_u + a_v), align_eps)

        base = L / (max(p, EPS)**prob_exp * max(r_edge, EPS)**beta_med)
        orient = 1.0 / (align_mean**gamma_align)
        e[out_cost_key] = float(base * (1.0 + lambda_drop * drop) * orient)


# ---------- seed/targets from graph features ----------

def _seed_from_core_radius(G: nx.Graph, radius_key='radius_mm') -> Optional[int]:
    if G.number_of_nodes() == 0:
        return None
    core = nx.core_number(G) if G.number_of_edges() else {n: 0 for n in G.nodes()}
    kmax = max(core.values()) if core else 0
    core_nodes = [n for n, k in core.items() if k == kmax] or list(G.nodes())
    if any(radius_key in G.nodes[n] for n in core_nodes):
        return max(core_nodes, key=lambda n: float(G.nodes[n].get(radius_key, 0.0)))
    return max(core_nodes, key=lambda n: G.degree(n))


def _endpoints(G: nx.Graph) -> List[int]:
    return [n for n, d in G.degree() if d == 1]


def _pos_phys(G: nx.Graph, n: int, pos_phys_key='pos_phys') -> np.ndarray:
    return np.asarray(G.nodes[n][pos_phys_key], float)


def _dedupe_by_mm(G: nx.Graph, cand: List[int], *, min_sep_mm: float = 5.0, pos_phys_key='pos_phys') -> List[int]:
    kept, kept_pos = [], []
    for n in cand:
        p = _pos_phys(G, n, pos_phys_key=pos_phys_key)
        if not kept:
            kept.append(n); kept_pos.append(p); continue
        dmin = np.min([np.linalg.norm(p - q) for q in kept_pos])
        if dmin >= float(min_sep_mm):
            kept.append(n); kept_pos.append(p)
    return kept


def select_seed_and_targets_from_features(Gc: nx.Graph,
                                          *,
                                          max_targets: int = 25,
                                          prob_exp: float = 1.5,
                                          Lspur_min_mm: float = 2.0,
                                          prob_min: float = 0.15,
                                          min_sep_mm: float = 5.0,
                                          length_key: str = 'length_mm',
                                          cost_key: str = 'length_cost',
                                          pos_phys_key: str = 'pos_phys') -> Tuple[Optional[int], List[int]]:
    if Gc.number_of_nodes() == 0:
        return None, []

    H = Gc.copy()
    attach_edge_metrics_mm_from_phys(H, prob_key='edge_prob', pos_phys_key=pos_phys_key,
                                     cost_key=cost_key, length_key=length_key, prob_exp=prob_exp)
    prune_short_or_lowprob_spurs(H, length_key=length_key, prob_key='edge_prob',
                                 Lspur_min_mm=Lspur_min_mm, prob_min=prob_min)

    if H.number_of_nodes() == 0:
        return None, []

    seed = _seed_from_core_radius(H, radius_key='radius_mm')
    if seed is None:
        return None, []

    eps = _endpoints(H)
    dist = nx.single_source_dijkstra_path_length(H, seed, weight=cost_key)
    ranked = sorted((eps if eps else [n for n in H.nodes() if n != seed]),
                    key=lambda n: dist.get(n, -np.inf), reverse=True)
    ranked = _dedupe_by_mm(H, ranked, min_sep_mm=min_sep_mm, pos_phys_key=pos_phys_key)
    if max_targets and max_targets > 0:
        ranked = ranked[:max_targets]
    return seed, ranked


# ---------- graph-only SPT union + resample + VTK export ----------

def _resample_polyline_mm(P_xyz: np.ndarray, step_mm: float = 0.5) -> np.ndarray:
    P = np.asarray(P_xyz, float)
    if len(P) < 2:
        return P.copy()
    seg = np.linalg.norm(P[1:] - P[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(s[-1])
    if L < 1e-6:
        return P[:1].copy()
    n = max(2, int(np.round(L / step_mm)) + 1)
    s_new = np.linspace(0.0, L, n)
    out = np.stack([np.interp(s_new, s, P[:, d]) for d in range(3)], axis=1)
    return out


def spt_union_from_seed_targets(G: nx.Graph,
                                seed: int,
                                targets: List[int],
                                weight_key: str = 'graph_cost') -> nx.Graph:
    if seed is None or not targets:
        return nx.Graph()
    _, paths = nx.single_source_dijkstra(G, seed, weight=weight_key)
    U = nx.Graph()
    for t in targets:
        path = paths.get(t, None)
        if not path or len(path) < 2:
            continue
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            U.add_edge(u, v, **G.edges[u, v])
    for n in U.nodes():
        U.nodes[n].update(G.nodes[n])
    return U


# ---- optional graph-only medial snapping helpers ----

def build_graph_kd(G: nx.Graph, pos_key='pos_phys'):
    ids, P = [], []
    for n in G.nodes():
        if pos_key in G.nodes[n]:
            ids.append(n)
            P.append(np.asarray(G.nodes[n][pos_key], float))
    P = np.asarray(P, float)
    kd = KDTree(P) if len(P) else None
    return kd, np.asarray(ids, int), P

def snap_polyline_to_graph_medial(P_xyz: np.ndarray,
                                  G: nx.Graph,
                                  kd: KDTree, kd_ids: np.ndarray, kd_pts: np.ndarray,
                                  k: int = 6, iters: int = 1,
                                  beta_r: float = 2.0, gamma_d: float = 1.0,
                                  eta: float = 0.5, radius_key: str = 'radius_mm') -> np.ndarray:
    """
    Move each polyline point toward a radius-weighted barycenter of k nearest graph nodes.
    weights w_i = (radius_i^beta_r) / (dist_i^gamma_d)
    position <- position + eta * (weighted_mean - position)
    """
    P = np.asarray(P_xyz, float).copy()
    if kd is None or len(P) == 0:
        return P
    for _ in range(iters):
        for i in range(len(P)):
            d, idx = kd.query(P[i], k=min(k, len(kd_pts)))
            d   = np.atleast_1d(d)
            idx = np.atleast_1d(idx)
            neigh_ids = kd_ids[idx]
            R  = np.array([float(G.nodes[n].get(radius_key, 1.0)) for n in neigh_ids])
            X  = kd_pts[idx]
            D  = np.maximum(d, 1e-6)
            w  = (R**beta_r) / (D**gamma_d)
            xc = np.sum(X * w[:, None], axis=0) / np.sum(w)
            P[i] = P[i] + eta * (xc - P[i])
    return P


import numpy as np
import networkx as nx
import vtk
from typing import List

def build_seqseg_centerline_polydata_from_graph_only(
        G_union: nx.Graph,
        seed_node: int,
        target_nodes: List[int],
        *,
        resample_step_mm: float = 0.5,
        weight_key: str = 'graph_cost',
        radius_key: str = 'radius_mm',
        use_medial_snap: bool = True,
        snap_k: int = 6,
        snap_iters: int = 1,
        snap_beta_r: float = 2.0,
        snap_gamma_d: float = 1.0,
        snap_eta: float = 0.5,
        # --- new smoothing options (VTK) ---
        apply_vtk_smoothing: bool = True,
        smooth_iters: int = 50,
        relax_factor: float = 0.01,
        preserve_endpoints: bool = True
) -> vtk.vtkPolyData:
    """
    Build centerline polydata from graph paths and optionally smooth coordinates using VTK.
    The function keeps the original structure: it computes Dijkstra paths, resamples and snaps,
    then builds an intermediate unsmoothed vtkPolyData, applies vtkSmoothPolyDataFilter (optional),
    and finally writes out a polydata with the smoothed points and the original per-point arrays
    (MaximumInscribedSphereRadius, GlobalNodeID, CenterlineId) transferred by nearest-point mapping.
    """

    _, paths = nx.single_source_dijkstra(G_union, seed_node, weight=weight_key)

    # Build KD once (used only by medial snap codepath)
    kd, kd_ids, kd_pts = build_graph_kd(G_union, pos_key='pos_phys') if use_medial_snap else (None, None, None)

    # Intermediate containers (unsmoothed points/lines and per-point arrays)
    unsm_pts = vtk.vtkPoints()
    unsm_lines = vtk.vtkCellArray()

    unsm_radii = vtk.vtkDoubleArray(); unsm_radii.SetName("MaximumInscribedSphereRadius")
    unsm_gnode = vtk.vtkIntArray();    unsm_gnode.SetName("GlobalNodeID")
    unsm_clid  = vtk.vtkIntArray();    unsm_clid.SetName("CenterlineId")

    branch_id = 0
    # Keep track of original endpoint point ids for optional preservation
    branch_endpoint_point_ids = []

    for t in target_nodes:
        path = paths.get(t, None)
        if not path or len(path) < 2:
            continue

        # Collect raw positions along graph path
        P = np.vstack([np.asarray(G_union.nodes[n]['pos_phys'], float) for n in path])
        P = _resample_polyline_mm(P, resample_step_mm)

        if use_medial_snap:
            P = snap_polyline_to_graph_medial(
                P, G_union, kd, kd_ids, kd_pts,
                k=snap_k, iters=snap_iters, beta_r=snap_beta_r,
                gamma_d=snap_gamma_d, eta=snap_eta, radius_key=radius_key
            )

        # Build an intermediate vtkPolyLine for this branch
        poly = vtk.vtkPolyLine()
        poly.GetPointIds().SetNumberOfIds(len(P))
        first_pid = None
        last_pid = None
        for i, p in enumerate(P):
            pid = unsm_pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
            if i == 0:
                first_pid = pid
            if i == len(P)-1:
                last_pid = pid
            poly.GetPointIds().SetId(i, pid)
            # choose radius from nearest original path node (as before)
            mid = path[min(i, len(path)-1)]
            r = float(G_union.nodes[mid].get(radius_key, 1.0))
            unsm_radii.InsertNextValue(r)
            unsm_gnode.InsertNextValue(int(mid))
            unsm_clid.InsertNextValue(int(branch_id))
        unsm_lines.InsertNextCell(poly)
        branch_endpoint_point_ids.append((first_pid, last_pid))
        branch_id += 1

    # Build unsmoothed polydata
    unsm_pd = vtk.vtkPolyData()
    unsm_pd.SetPoints(unsm_pts)
    unsm_pd.SetLines(unsm_lines)
    unsm_pd.GetPointData().AddArray(unsm_radii)
    unsm_pd.GetPointData().AddArray(unsm_gnode)
    unsm_pd.GetPointData().AddArray(unsm_clid)

    if not apply_vtk_smoothing:
        # Return unsmoothed result (same arrays already attached)
        return unsm_pd

    # --- Apply VTK smoothing filter to the intermediate polydata ---
    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputData(unsm_pd)
    smoother.SetNumberOfIterations(int(max(0, smooth_iters)))
    smoother.SetRelaxationFactor(float(relax_factor))
    # Turn off feature/edge smoothing to avoid collapsing topology; keep boundaries less moved
    smoother.FeatureEdgeSmoothingOff()
    smoother.BoundarySmoothingOff()  # boundary smoothing off helps preserve endpoints; can set On for gentle movement
    smoother.SetConvergence(0.0)  # iterate fixed number of iterations
    smoother.Update()

    smooth_pd = smoother.GetOutput()

    # --- Build final output polydata with smoothed points but original arrays transferred ---
    sm_pts = smooth_pd.GetPoints()
    n_sm = sm_pts.GetNumberOfPoints()

    # Build a locator on the unsmoothed points so we can transfer per-point arrays by nearest original point
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(unsm_pd)
    locator.BuildLocator()

    # Prepare final arrays
    final_pts = vtk.vtkPoints()
    final_lines = vtk.vtkCellArray()

    final_radii = vtk.vtkDoubleArray(); final_radii.SetName("MaximumInscribedSphereRadius")
    final_gnode = vtk.vtkIntArray();    final_gnode.SetName("GlobalNodeID")
    final_clid  = vtk.vtkIntArray();    final_clid.SetName("CenterlineId")

    # We need to preserve the same polyline topology (same number of lines and point counts per line).
    # The smoothing filter preserves topology (lines) so we can read the lines from smooth_pd directly.
    smooth_lines = smooth_pd.GetLines()
    smooth_lines.InitTraversal()
    id_list = vtk.vtkIdList()
    while smooth_lines.GetNextCell(id_list):
        n_ids = id_list.GetNumberOfIds()
        poly = vtk.vtkPolyLine()
        poly.GetPointIds().SetNumberOfIds(n_ids)
        for j in range(n_ids):
            sid = id_list.GetId(j)
            p = sm_pts.GetPoint(sid)
            new_pid = final_pts.InsertNextPoint(p)
            poly.GetPointIds().SetId(j, new_pid)
            # find nearest original (unsmoothed) point and copy arrays
            orig_id = locator.FindClosestPoint(p)
            r = unsm_radii.GetValue(orig_id)
            gn = unsm_gnode.GetValue(orig_id)
            cl = unsm_clid.GetValue(orig_id)
            final_radii.InsertNextValue(r)
            final_gnode.InsertNextValue(gn)
            final_clid.InsertNextValue(cl)
        final_lines.InsertNextCell(poly)

    # Optional: preserve exact original endpoints (seed/target) if requested
    if preserve_endpoints and len(branch_endpoint_point_ids) > 0:
        # Build mapping from old unsmoothed point id -> final point id by nearest match (should exist)
        final_locator = vtk.vtkPointLocator()
        tmp_pd = vtk.vtkPolyData()
        tmp_pd.SetPoints(final_pts)
        final_locator.SetDataSet(tmp_pd)
        final_locator.BuildLocator()

        for (unsm_first, unsm_last) in branch_endpoint_point_ids:
            # original coords
            orig_first = unsm_pts.GetPoint(unsm_first)
            orig_last = unsm_pts.GetPoint(unsm_last)
            # find nearest in final and replace coordinates exactly
            f_first = final_locator.FindClosestPoint(orig_first)
            f_last = final_locator.FindClosestPoint(orig_last)
            final_pts.SetPoint(f_first, orig_first)
            final_pts.SetPoint(f_last, orig_last)
        final_pts.Modified()

    # Compose final polydata
    out_pd = vtk.vtkPolyData()
    out_pd.SetPoints(final_pts)
    out_pd.SetLines(final_lines)
    out_pd.GetPointData().AddArray(final_radii)
    out_pd.GetPointData().AddArray(final_gnode)
    out_pd.GetPointData().AddArray(final_clid)

    return out_pd



def export_seqseg_centerline_from_graph_only(
        Gc: nx.Graph,
        out_vtp_path: str,
        seed_node: int,
        target_nodes: List[int],
        *,
        resample_step_mm: float = 0.5,
        clean_and_smooth: bool = True,
        use_medial_snap: bool = True
) -> Tuple[vtk.vtkPolyData, nx.Graph]:
    G_union = spt_union_from_seed_targets(Gc, seed_node, target_nodes, weight_key='graph_cost')
    pd = build_seqseg_centerline_polydata_from_graph_only(
        G_union, seed_node, target_nodes,
        resample_step_mm=resample_step_mm,
        use_medial_snap=use_medial_snap
    )
    if clean_and_smooth and pd.GetNumberOfPoints() > 0:
        try:
            pd = post_process_centerline(pd, verbose=False)
        except Exception:
            pass
    os.makedirs(os.path.dirname(out_vtp_path), exist_ok=True)
    write_vtk_polydata(pd, out_vtp_path)
    return pd, G_union


# ============================== Misc helpers ==============================

def rebind_points_from_indices(G: nx.Graph, img: sitk.Image,
                               idx_key='pos_idx_xyz', out_key='point'):
    for n, d in G.nodes(data=True):
        if idx_key in d:
            ix, iy, iz = map(int, d[idx_key])
            p = img.TransformIndexToPhysicalPoint((ix, iy, iz))
            d[out_key] = np.asarray(p, float)
            d['pos_phys'] = d[out_key]  # ensure alias exists


def attach_radii_from_global_edt(G: nx.Graph, edt_mm_zyx: np.ndarray, idx_key='pos_idx_xyz'):
    for n, d in G.nodes(data=True):
        if idx_key in d:
            x, y, z = map(int, d[idx_key])  # xyz indices
            r_mm = float(edt_mm_zyx[z, y, x])  # edt is z,y,x
            d['radius_mm'] = r_mm
            d['MaximumInscribedSphereRadius'] = r_mm


def inside_frac_phys(G: nx.Graph, img: sitk.Image, key='pos_phys', sample=1024) -> float:
    nodes = list(G.nodes())
    if not nodes:
        return 0.0
    if len(nodes) > sample:
        nodes = list(np.random.choice(nodes, size=sample, replace=False))
    ok = 0
    size = np.array(img.GetSize(), int)
    for n in nodes:
        p = G.nodes[n].get(key, None)
        if p is None:
            continue
        ci = np.array(img.TransformPhysicalPointToContinuousIndex(tuple(p)), float)
        if np.all(ci >= 0) and np.all(ci <= (size - 1)):
            ok += 1
    return ok / max(1, len(nodes))


def largest_cc_simple(img: sitk.Image, background_value=0) -> sitk.Image:
    relabeled = sitk.RelabelComponent(
        sitk.ConnectedComponent(img != background_value),
        sortByObjectSize=True
    )
    return sitk.Cast(relabeled == 1, img.GetPixelID())


def blank_image(ref: sitk.Image, pixel_id=sitk.sitkFloat32) -> sitk.Image:
    out = sitk.Image(ref.GetSize(), pixel_id)
    out.CopyInformation(ref)
    return out


# --- numpy -> VTK helpers (used by trace artifacts saving) ---

def np_polyline_to_vtk(points_xyz: np.ndarray) -> vtk.vtkPolyData:
    points_xyz = np.asarray(points_xyz, float)
    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    for p in points_xyz:
        pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    line = vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(points_xyz.shape[0])
    for i in range(points_xyz.shape[0]):
        line.GetPointIds().SetId(i, i)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(line)
    pd.SetPoints(pts)
    pd.SetLines(cells)
    return pd


def np_points_to_vtk(points_xyz: np.ndarray) -> vtk.vtkPolyData:
    points_xyz = np.asarray(points_xyz, float)
    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    verts = vtk.vtkCellArray()
    for p in points_xyz:
        pid = pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
        verts.InsertNextCell(1); verts.InsertCellPoint(pid)
    pd.SetPoints(pts)
    pd.SetVerts(verts)
    return pd


def np_mesh_to_vtk(verts_xyz: np.ndarray, faces_tri: np.ndarray) -> vtk.vtkPolyData:
    verts_xyz = np.asarray(verts_xyz, float)
    faces_tri = np.asarray(faces_tri, dtype=np.int64)
    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    for v in verts_xyz:
        pts.InsertNextPoint(float(v[0]), float(v[1]), float(v[2]))
    polys = vtk.vtkCellArray()
    for tri in faces_tri:
        cell = vtk.vtkTriangle()
        cell.GetPointIds().SetId(0, int(tri[0]))
        cell.GetPointIds().SetId(1, int(tri[1]))
        cell.GetPointIds().SetId(2, int(tri[2]))
        polys.InsertNextCell(cell)
    pd.SetPoints(pts)
    pd.SetPolys(polys)
    return pd

# ----------- Save assembled image while preserving metadata (skimage + SimpleITK) -------------
import numpy as np
import SimpleITK as sitk
from skimage import morphology, filters, measure

def save_assembled_with_skimage(sitk_img: sitk.Image, out_path: str,
                                bin_thr: float = 0.5,
                                do_closing: bool = True,
                                closing_radius_vox: int = 2,
                                remove_small_objects_vox: int = 64,
                                fill_holes: bool = True):
    """
    Take a sitk.Image (probability or mask), run a few skimage ops and save a sitk image
    with exactly the same spacing/origin/direction as the input.
    - sitk_img may be float probs or a binary mask (0/1).
    - out_path: full filename, e.g. 'case_assembled_mask.nii.gz'
    """
    # 1) pull numpy array in z,y,x order
    arr = sitk.GetArrayFromImage(sitk_img)  # shape (Z,Y,X)
    arr = arr.astype(np.float32)

    # 2) binarize if needed
    mask = arr >= float(bin_thr)

    # 3) optional: morphological closing to remove tiny holes
    if do_closing and closing_radius_vox > 0:
        selem = morphology.ball(closing_radius_vox)
        mask = morphology.closing(mask, selem)

    # 4) remove small objects (in voxels)
    if remove_small_objects_vox and remove_small_objects_vox > 0:
        mask = morphology.remove_small_objects(mask.astype(bool), min_size=int(remove_small_objects_vox))

    # 5) fill holes within connected components (optional)
    if fill_holes:
        mask = morphology.remove_small_holes(mask.astype(bool), area_threshold=remove_small_objects_vox)

    # 6) convert back to sitk.Image and copy geometry from original
    out_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))  # still ZYX -> sitk expects ZYX
    out_sitk.CopyInformation(sitk_img)  # copies origin, spacing, direction
    sitk.WriteImage(out_sitk, out_path)
    print(f"[WRITE] assembled mask (skimage processed): {out_path}")
    return out_sitk

# ---------- Merge + smooth polylines -> single VTP -------------
import vtk
import numpy as np
from scipy.spatial import cKDTree as KDTree
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

def vtkpoly_to_numpy_polylines(vtk_pd):
    """Return list of Nx3 numpy arrays for each polyline in vtkPolyData."""
    out = []
    if vtk_pd is None or vtk_pd.GetNumberOfPoints() == 0:
        return out
    pts = vtk_pd.GetPoints()
    pts_np = np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())], dtype=float)
    lines = vtk_pd.GetLines()
    lines.InitTraversal()
    id_list = vtk.vtkIdList()
    while lines.GetNextCell(id_list):
        n = id_list.GetNumberOfIds()
        if n < 2: continue
        idxs = [id_list.GetId(i) for i in range(n)]
        out.append(pts_np[idxs, :])
    return out

def numpy_polyline_to_vtk(poly):
    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    for p in poly: pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    line = vtk.vtkPolyLine(); line.GetPointIds().SetNumberOfIds(poly.shape[0])
    for i in range(poly.shape[0]): line.GetPointIds().SetId(i, i)
    cells = vtk.vtkCellArray(); cells.InsertNextCell(line)
    pd.SetPoints(pts); pd.SetLines(cells)
    return pd

def resample_polyline_by_arclength(pts, step_mm=0.5):
    pts = np.asarray(pts, float)
    if pts.shape[0] < 2: return pts.copy()
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.hstack(([0.0], np.cumsum(seg_len)))
    L = cum[-1]
    if L <= 0: return pts.copy()
    n = max(2, int(np.ceil(L / float(step_mm))) + 1)
    s = np.linspace(0.0, L, n)
    out = np.empty((len(s), 3), dtype=float)
    for i, si in enumerate(s):
        j = np.searchsorted(cum, si, side='right') - 1
        j = min(max(j, 0), len(seg_len)-1)
        t = (si - cum[j]) / (seg_len[j] + 1e-12)
        out[i] = (1 - t) * pts[j] + t * pts[j+1]
    return out

def moving_average_smooth(pts, window=5):
    pts = np.asarray(pts, float)
    if pts.shape[0] <= 2 or window <= 1: return pts.copy()
    w = int(window);
    if w % 2 == 0: w += 1
    pad = w // 2
    out = np.empty_like(pts)
    for dim in range(3):
        a = pts[:, dim]; A = np.pad(a, pad, mode='reflect')
        csum = np.cumsum(A, dtype=float)
        out[:, dim] = (csum[w:] - csum[:-w]) / float(w)
    out[0] = pts[0]; out[-1] = pts[-1]
    return out

def snap_endpoints_and_align(polys, snap_tol_mm=0.8):
    polys = [np.asarray(p, float) for p in polys]
    changed = True
    while changed:
        changed = False
        for i in range(len(polys)):
            for j in range(i+1, len(polys)):
                a0, a1 = polys[i][0], polys[i][-1]
                b0, b1 = polys[j][0], polys[j][-1]
                pairs = [ (0,0,a0,b0), (0,1,a0,b1), (1,0,a1,b0), (1,1,a1,b1) ]
                for si, sj, pi, pj in pairs:
                    d = np.linalg.norm(pi - pj)
                    if d <= snap_tol_mm:
                        meanp = 0.5*(pi + pj)
                        if si == 0: polys[i][0] = meanp
                        else:      polys[i][-1] = meanp
                        if sj == 0: polys[j][0] = meanp
                        else:       polys[j][-1] = meanp
                        if si == sj:
                            polys[j] = polys[j][::-1]
                        changed = True
    return polys

import numpy as np
from scipy.spatial import cKDTree as KDTree
import vtk

# ---------------- safer moving-average smoother ----------------
def moving_average_smooth(pts, window=5):
    """
    Robust moving-average smoothing for Nx3 points.
    - If poly shorter than window, returns original pts (no oversmooth).
    - Uses reflect padding + np.convolve (stable lengths).
    """
    pts = np.asarray(pts, dtype=float)
    n = pts.shape[0]
    if n < 3 or window <= 1:
        return pts.copy()
    w = int(window)
    if w % 2 == 0: w += 1
    if n <= w:
        # too few points to apply the filter safely — return original
        return pts.copy()

    pad = w // 2
    out = np.empty_like(pts)
    for dim in range(3):
        a = pts[:, dim]
        A = np.pad(a, pad, mode='reflect')
        kernel = np.ones(w, dtype=float) / float(w)
        sm = np.convolve(A, kernel, mode='valid')  # length == n
        out[:, dim] = sm
    # preserve endpoints exactly to avoid tiny endpoint drift
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out

# --------------- merged & smoothed VTK exporter (safe) ---------------
def merge_and_smooth_vtk(merged_centerlines_list,
                         out_vtp_path,
                         resample_step_mm=0.5,
                         smooth_window=5,
                         snap_tol_mm=0.8,
                         connect_max_mm=3.5,
                         vtk_smooth_iter=20,
                         vtk_passband=0.1,
                         spacing_mm=None):
    """
    Merge list of vtkPolyData (or Nx3 numpy arrays) into one smoothed .vtp.
    spacing_mm: tuple/list (sx, sy, sz) used to pick a clean tolerance; if None, fallback used.
    Returns vtkPolyData written to out_vtp_path.
    """
    # ---- collect polylines as numpy arrays ----
    def vtkpoly_to_numpy_polylines(vtk_pd):
        out = []
        if vtk_pd is None or vtk_pd.GetNumberOfPoints() == 0:
            return out
        pts = vtk_pd.GetPoints()
        pts_np = np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())], dtype=float)
        lines = vtk_pd.GetLines()
        lines.InitTraversal()
        id_list = vtk.vtkIdList()
        while lines.GetNextCell(id_list):
            n = id_list.GetNumberOfIds()
            if n < 2: continue
            idxs = [id_list.GetId(i) for i in range(n)]
            out.append(pts_np[idxs, :])
        return out

    def numpy_polyline_to_vtk(poly):
        pd = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        for p in poly: pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
        line = vtk.vtkPolyLine(); line.GetPointIds().SetNumberOfIds(poly.shape[0])
        for i in range(poly.shape[0]): line.GetPointIds().SetId(i, i)
        cells = vtk.vtkCellArray(); cells.InsertNextCell(line)
        pd.SetPoints(pts); pd.SetLines(cells)
        return pd

    all_polys = []
    for pd in merged_centerlines_list:
        try:
            if isinstance(pd, vtk.vtkPolyData):
                polys = vtkpoly_to_numpy_polylines(pd)
                all_polys.extend(polys)
            else:
                arr = np.asarray(pd)
                if arr.ndim == 2 and arr.shape[1] == 3:
                    all_polys.append(arr)
        except Exception:
            continue

    if not all_polys:
        print("[MERGE] no polylines to merge")
        return None

    # simple helper: resample by arc-length
    def resample_polyline_by_arclength(pts, step_mm=0.5):
        pts = np.asarray(pts, float)
        if pts.shape[0] < 2: return pts.copy()
        seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.hstack(([0.0], np.cumsum(seg_len)))
        L = cum[-1]
        if L <= 0: return pts.copy()
        n = max(2, int(np.ceil(L / float(step_mm))) + 1)
        s = np.linspace(0.0, L, n)
        out = np.empty((len(s), 3), dtype=float)
        for i, si in enumerate(s):
            j = np.searchsorted(cum, si, side='right') - 1
            j = min(max(j, 0), len(seg_len)-1)
            t = (si - cum[j]) / (seg_len[j] + 1e-12)
            out[i] = (1 - t) * pts[j] + t * pts[j+1]
        return out

    # 1) resample + per-poly smoothing
    proc = []
    for p in all_polys:
        r = resample_polyline_by_arclength(p, step_mm=resample_step_mm)
        r = moving_average_smooth(r, window=smooth_window)   # now robust
        proc.append(r)

    # 2) snap endpoints + align (reuse your function but ensure it's present)
    def snap_endpoints_and_align_local(polys, snap_tol_mm=0.8):
        polys = [np.asarray(p, float) for p in polys]
        changed = True
        while changed:
            changed = False
            for i in range(len(polys)):
                for j in range(i+1, len(polys)):
                    a0, a1 = polys[i][0], polys[i][-1]
                    b0, b1 = polys[j][0], polys[j][-1]
                    pairs = [ (0,0,a0,b0), (0,1,a0,b1), (1,0,a1,b0), (1,1,a1,b1) ]
                    for si, sj, pi, pj in pairs:
                        d = np.linalg.norm(pi - pj)
                        if d <= snap_tol_mm:
                            meanp = 0.5*(pi + pj)
                            if si == 0: polys[i][0] = meanp
                            else:      polys[i][-1] = meanp
                            if sj == 0: polys[j][0] = meanp
                            else:       polys[j][-1] = meanp
                            if si == sj:
                                polys[j] = polys[j][::-1]
                            changed = True
        return polys

    proc = snap_endpoints_and_align_local(proc, snap_tol_mm=snap_tol_mm)

    # 3) small straight connectors via KD (keep simple)
    endpoints = []
    idx_map = []
    for i,p in enumerate(proc):
        endpoints.append(p[0]); idx_map.append((i,0))
        endpoints.append(p[-1]); idx_map.append((i,1))
    pts_ep = np.vstack(endpoints)
    kd = KDTree(pts_ep)
    connectors = []
    # find neighbors within connect_max_mm
    for i in range(pts_ep.shape[0]):
        dists, ids = kd.query(pts_ep[i], k=min(8, pts_ep.shape[0]-1))
        for dj, idj in zip(np.atleast_1d(dists)[1:], np.atleast_1d(ids)[1:]):
            if dj <= connect_max_mm:
                connectors.append((i, int(idj), float(dj)))
    used = set();
    for a,b,d in sorted(connectors, key=lambda x: x[2]):
        key = tuple(sorted((a,b)))
        if key in used: continue
        used.add(key)
        ia, sa = idx_map[a]; ib, sb = idx_map[b]
        pa = proc[ia][0] if sa==0 else proc[ia][-1]
        pb = proc[ib][0] if sb==0 else proc[ib][-1]
        proc.append(np.vstack([pa, pb]))

    # 4) greedy concatenation walk (seed longest)
    lengths = [np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in proc]
    cur_idx = int(np.argmax(lengths))
    used_poly = set()
    chain = []
    cur_side = 0
    while True:
        p = proc[cur_idx]
        if cur_side == 0:
            chain.extend(list(p))
            cur_pt = p[-1]
        else:
            chain.extend(list(p[::-1]))
            cur_pt = p[0]
        used_poly.add(cur_idx)
        # find nearest unused poly
        cand = []
        for j in range(len(proc)):
            if j in used_poly: continue
            pj = proc[j]
            d0 = np.linalg.norm(cur_pt - pj[0])
            d1 = np.linalg.norm(cur_pt - pj[-1])
            cand.append((min(d0,d1), j, 0 if d0<=d1 else 1))
        if not cand: break
        cand.sort(key=lambda x: x[0])
        cur_idx = cand[0][1]; cur_side = cand[0][2]

    merged_chain = np.vstack(chain)

    # 5) final resample + smoothing
    merged_chain = resample_polyline_by_arclength(merged_chain, step_mm=resample_step_mm)
    merged_chain = moving_average_smooth(merged_chain, window=max(3, smooth_window))

    # 6) vtk WindowedSinc smoothing
    pd_merge = numpy_polyline_to_vtk(merged_chain)
    try:
        filt = vtk.vtkWindowedSincPolyDataFilter()
        filt.SetInputData(pd_merge)
        filt.SetNumberOfIterations(int(vtk_smooth_iter))
        filt.SetPassBand(float(vtk_passband))
        filt.NonManifoldSmoothingOn()
        filt.NormalizeCoordinatesOn()
        filt.Update()
        pd_smooth = filt.GetOutput()
    except Exception:
        pd_smooth = pd_merge

    # 7) final clean: use spacing_mm if provided to compute tolerance
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(pd_smooth)
    clean.ToleranceIsAbsoluteOn()
    try:
        tol = 0.25 * float(min(spacing_mm)) if spacing_mm is not None else 1e-3
    except Exception:
        tol = 1e-3
    clean.SetAbsoluteTolerance(tol)
    clean.PointMergingOn()
    clean.Update()

    # write out
    write_vtk_polydata(clean.GetOutput(), out_vtp_path)
    print(f"[WRITE] merged/smoothed centerline: {out_vtp_path}")
    return clean.GetOutput()

# ============================== Main pipeline ==============================

import argparse
from nnUNet.nnunetv2.paths import nnUNet_results


def parse_args():
    p = argparse.ArgumentParser("GNN model (graph-only centerline | medial-oriented)")
    p.add_argument("--output_dir", required=True, type=str,
                   help="Folder to save traced segmentations")
    p.add_argument("--pred_dir", required=True, type=str,
                   help="Folder to retrieve predictions")
    p.add_argument("--data_dir", required=True, type=str,
                   help="Folder to retrieve raw data")
    p.add_argument("--gnn_folder", required=True, type=str,
                   help="GNN folder path")
    p.add_argument("--config_file", default=None, type=str,
                   help="path to SeqSeg config file")
    p.add_argument("--fold", default=5, type=int,
                   help="nnU-Net fold to use")
    p.add_argument("--img_ext", default='.nii.gz', type=str,
                   help="Image extension")
    p.add_argument("--dataset_id", type=str, help="Dataset id to initialize predictor")
    return p.parse_args()


def main():
    faulthandler.enable()
    args = parse_args()
    t0 = time.time()

    OUTPUT_DIR = args.output_dir
    seqseg_cfg = args.config_file
    gnn_cfg = os.path.join(args.gnn_folder, 'gnn_cfg.yaml')

    with open(seqseg_cfg, 'r') as f:
        params_seqseg = yaml.safe_load(f) or {}
    with open(gnn_cfg, 'r') as f:
        params_gnn = yaml.safe_load(f) or {}

    img_format = args.img_ext
    seg_dir = args.pred_dir
    images_dir = args.data_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_cases = [f.replace(img_format, "") for f in os.listdir(seg_dir) if f.endswith(img_format)]

    model, cfg_gnn, device, _ = gm_load(
        ckpt_path=os.path.join(args.gnn_folder, "best_gnn_checkpoint.pt"),
        cfg=params_gnn
    )

    merged_centerlines = []
    merged_points = []
    merged_inside_points = []
    merged_surfaces = []

    model_tuple = (model, cfg_gnn, device, "best_gnn_checkpoint.pt")

    node_min_spacing_mm = float(params_gnn["node_min_spacing_mm"]) if "node_min_spacing_mm" in params_gnn else None
    knn_k = int(params_gnn.get("knn_k", 16))
    knn_radius_mm = float(params_gnn.get("knn_radius_mm", 8.0))
    edge_prob_thresh = float(params_gnn.get("edge_prob_thresh", 0.5))
    top_k_components = int(params_seqseg.get("top_k_components", 10))

    seqseg_model_folder = os.path.join(
        nnUNet_results, f"{args.dataset_id}/nnUNetTrainer__nnUNetPlans__3d_fullres"
    )

    max_steps_per_component = int(params_seqseg.get("MAX_STEPS_PER_COMPONENT", 300))

    ASSEMBLY_THRESH = float(params_seqseg.get("ASSEMBLY_THRESHOLD", 0.5))
    WRITE_CENTERLINE_MERGE = bool(params_seqseg.get("WRITE_CENTERLINE_MERGE", True))

    def read_vtp_polydata(path):
        r = vtk.vtkXMLPolyDataReader()
        r.SetFileName(path)
        r.Update()
        return r.GetOutput()

    for case_id in all_cases:
        case_t0 = time.time()
        print(f"\n{'=' * 60}\nProcessing case: {case_id}\n{'=' * 60}")
        dir_output_case = os.path.join(OUTPUT_DIR, case_id)
        create_directories(dir_output_case, write_samples=True)

        dir_image = os.path.join(images_dir, f"{case_id}{img_format}")
        dir_seg = os.path.join(seg_dir, f"{case_id}{img_format}")
        if not (os.path.exists(dir_image) and os.path.exists(dir_seg)):
            print("  - Missing image/seg; skipping.")
            continue

        image_ref = sitk.ReadImage(dir_image)
        segmentation_image = sitk.ReadImage(dir_seg)
        segmentation_image.CopyInformation(image_ref)

        # per-case accumulators
        start_prob_global = None
        coverage_union = None
        merged_centerlines = []

        # --- ECC graph ---
        print("--- Building graph with ECC model ---")
        G, dbg = gm_predict_graph(
            seg_img=segmentation_image, prob_img=None,
            model_tuple=model_tuple,
            node_min_spacing_mm=node_min_spacing_mm,
            knn_k=knn_k,
            knn_radius_mm=knn_radius_mm,
            edge_prob_thresh=edge_prob_thresh,
            use_adaptive_spacing=False,
            return_debug=True,
        )

        print("  Graph:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")
        try:
            save_predicted_graph_to_vtp(G, out_path=os.path.join(dir_output_case, f"graph_full_{case_id}.vtp"))
        except Exception:
            pass
        if G.number_of_nodes() == 0:
            print("  - Empty graph; skipping.")
            continue

        # Normalize pos indices
        order = attach_pos_idx_xyz(G, image_ref, pos_key='pos', try_orders=((2, 1, 0), (0, 1, 2)))
        inside_frac = _inside_frac_order(G, image_ref, pos_key='pos', order=order)
        print(f"  - graph→index order: {order}, inside_frac: {inside_frac:.3f}")

        comps = largest_components(G, k=top_k_components, by='nodes')
        print(f"  - Cropping from top {len(comps)} graph component(s)")

        # Global EDT in mm (ZYX) for radii
        seg_np_zyx = (sitk.GetArrayFromImage(segmentation_image) > 0).astype(np.uint8)
        edt_mm_zyx_global = distance_transform_edt(seg_np_zyx, sampling=segmentation_image.GetSpacing()[::-1])

        pad_mm = tuple(params_seqseg.get("pad_mm", (4, 4, 4)))
        min_size_mm = tuple(params_seqseg.get("min_size_mm", (12, 12, 12)))
        prob_exp = float(params_seqseg.get("prob_exp", 1.5))
        max_targets_per_comp = int(params_seqseg.get("max_targets_per_comp", 25))
        spur_len_min_mm = float(params_seqseg.get("spur_len_min_mm", 2.0))
        prob_min = float(params_seqseg.get("prob_min", 0.15))
        min_target_sep_mm = float(params_seqseg.get("min_target_sep_mm", 5.0))

        for gi, Gc in enumerate(comps):
            roi = bbox_from_graph_component(
                Gc, image_ref, pos_idx_key='pos_idx_xyz',
                pad_mm=pad_mm, min_size_mm=min_size_mm
            )
            if roi is None:
                print(f"  - comp {gi}: no points; skip.")
                continue

            start_xyz, size_xyz = roi
            # crop (for saving artifacts & later trace_centerline assembly)
            cropped_img = sitk.RegionOfInterest(image_ref, size_xyz, start_xyz)
            cropped_seg = sitk.RegionOfInterest(segmentation_image, size_xyz, start_xyz)
            cropped_seg.CopyInformation(cropped_img)

            comp_tag = f"{case_id}_comp{gi:02d}"
            img_out = os.path.join(dir_output_case, "images", f"{comp_tag}{img_format}")
            lab_out = os.path.join(dir_output_case, "labels", f"{comp_tag}{img_format}")
            os.makedirs(os.path.dirname(img_out), exist_ok=True)
            os.makedirs(os.path.dirname(lab_out), exist_ok=True)
            sitk.WriteImage(cropped_img, img_out)
            sitk.WriteImage(cropped_seg, lab_out)
            print(f"    saved ROI {comp_tag}: start={start_xyz}, size={size_xyz}")

            # ---------- GRAPH NORMALIZATION (volumetric component) ----------
            rebind_points_from_indices(Gc, image_ref, idx_key='pos_idx_xyz', out_key='pos_phys')
            attach_radii_from_global_edt(Gc, edt_mm_zyx_global, idx_key='pos_idx_xyz')
            print(f"[DBG inside] comp{gi} volumetric_graph inside_frac (cropped img): "
                  f"{inside_frac_phys(Gc, cropped_img, key='pos_phys'):.3f}")

            # ---- Seed/targets from graph features
            seed_node, target_nodes = select_seed_and_targets_from_features(
                Gc,
                max_targets=max_targets_per_comp,
                prob_exp=prob_exp,
                Lspur_min_mm=spur_len_min_mm,
                prob_min=prob_min,
                min_sep_mm=min_target_sep_mm,
            )
            if seed_node is None or not target_nodes:
                print("  - no valid seed/targets for this component; skipping")
                continue

            # ================== GRAPH-ONLY CENTERLINE (MEDIAL & ORIENTED) ==================
            # 1) ensure lengths & prune spurs
            attach_edge_metrics_mm_from_phys(Gc, prob_key='edge_prob', pos_phys_key='pos_phys',
                                             cost_key='length_cost', length_key='length_mm', prob_exp=prob_exp)
            prune_spurs_graph_only(Gc, length_key='length_mm', prob_key='edge_prob',
                                   Lspur_min_mm=spur_len_min_mm, prob_min=prob_min)

            # 2) attach medial & orientation-aware routing cost and export
            attach_graph_only_cost_medial_oriented(
                Gc,
                base_len_key='length_mm',
                prob_key='edge_prob',
                edge_r_key='radius_min_mm',
                node_r_key='radius_mm',
                tangent_key='tangent',
                out_cost_key='graph_cost',
                prob_exp=prob_exp,
                beta_med=1.5,          # tune 1.5–2.0 if needed to hug the center
                lambda_drop=0.75,
                gamma_align=1.0
            )

            cent_vtp = os.path.join(dir_output_case, "centerlines", f"{comp_tag}_centerline.vtp")
            os.makedirs(os.path.dirname(cent_vtp), exist_ok=True)
            guide_poly, final_graph = export_seqseg_centerline_from_graph_only(
                Gc, cent_vtp, seed_node=seed_node, target_nodes=target_nodes,
                resample_step_mm=STEP_MM, clean_and_smooth=True,
                use_medial_snap=True   # enable snap
            )
            print(f"  - SeqSeg-compatible centerline (graph-only) saved: {os.path.basename(cent_vtp)}")

            guide_centerline = guide_poly  # already vtkPolyData

            # ================== CONTINUE WITH SeqSeg TRACER (UNCHANGED) ==================
            prev_prob_for_tracer = start_prob_global if start_prob_global is not None else None

            seed_id_tracer = int(seed_node)        # from volumetric graph Gc
            target_ids_tracer = list(map(int, target_nodes))

            _lc, _ls, _lp, _li, assembly_segs, vt, i = trace_centerline(
                output_folder=dir_output_case,
                image_file=dir_image,
                case=case_id,
                model_folder=seqseg_model_folder,
                fold=args.fold,
                graph=Gc,
                centerline_graph=final_graph,  # graph-only union
                seed_node=seed_id_tracer,
                target_nodes=target_ids_tracer,
                max_steps_per_component=max_steps_per_component,
                global_config=params_seqseg,
                unit='cm',
                scale=1,
                seg_file=None,
                start_seg=prev_prob_for_tracer
            )

            prev_prob = start_prob_global if start_prob_global is not None else blank_image(image_ref, sitk.sitkFloat32)
            curr_prob = sitk.Cast(assembly_segs.assembly, sitk.sitkFloat32)
            curr_prob.CopyInformation(prev_prob)

            new_bin = sitk.Greater(curr_prob, ASSEMBLY_THRESH)
            old_bin = sitk.Greater(prev_prob, ASSEMBLY_THRESH)
            delta_bin = sitk.And(new_bin, sitk.Not(old_bin))
            largest_delta = largest_cc_simple(sitk.Cast(delta_bin, sitk.sitkUInt8), background_value=0)

            stats = sitk.StatisticsImageFilter()
            stats.Execute(largest_delta)
            if stats.GetSum() == 0:
                print("    [assembly] no novel region above threshold; skipping")
                continue

            inc_prob = sitk.Mask(curr_prob, largest_delta)
            start_prob_global = sitk.Maximum(prev_prob, inc_prob)

            if coverage_union is None:
                coverage_union = sitk.Cast(largest_delta, sitk.sitkUInt8)
                coverage_union.CopyInformation(prev_prob)
            else:
                coverage_union = sitk.Or(coverage_union, sitk.Cast(largest_delta, sitk.sitkUInt8))

            # ---- Save trace artifacts
            trace_dir = os.path.join(dir_output_case, "trace_artifacts")
            os.makedirs(trace_dir, exist_ok=True)
            save_base = f"{case_id}_seed{seed_id_tracer}"

            # 1) centerlines from tracer
            if _lc:
                for k, poly in enumerate(_lc):
                    if poly is None or len(poly) == 0:
                        continue
                    pd = np_polyline_to_vtk(np.asarray(poly))
                    pd = smooth_polydata(pd)
                    out_path = os.path.join(trace_dir, f"{save_base}_centerline_{k:02d}.vtp")
                    write_vtk_polydata(pd, out_path)
                    print(f"[WRITE] centerline: {out_path}")
                    merged_centerlines.append(pd)

            # 2) visited points
            if _lp and len(_lp) > 0:
                try:
                    P = np.vstack([np.asarray(p, float).reshape(3) for p in _lp])
                    pd_pts = np_points_to_vtk(P)
                    out_path = os.path.join(trace_dir, f"{save_base}_points.vtp")
                    write_vtk_polydata(pd_pts, out_path)
                    print(f"[WRITE] points: {out_path}")
                    merged_points.append(pd_pts)
                except Exception as e:
                    print(f"[WARN] saving points failed: {e}")

            # 3) inside points
            if _li and len(_li) > 0:
                try:
                    Pin = np.vstack([np.asarray(p, float).reshape(3) for p in _li])
                    pd_in = np_points_to_vtk(Pin)
                    out_path = os.path.join(trace_dir, f"{save_base}_inside_points.vtp")
                    write_vtk_polydata(pd_in, out_path)
                    print(f"[WRITE] inside-points: {out_path}")
                    merged_inside_points.append(pd_in)
                except Exception as e:
                    print(f"[WARN] saving inside-points failed: {e}")

            # 4) QA surfaces
            if _ls and len(_ls) > 0:
                for sidx, surf in enumerate(_ls):
                    try:
                        if surf is None:
                            continue
                        verts, faces = surf
                        if verts is None or faces is None or len(verts) == 0 or len(faces) == 0:
                            continue
                        pd_mesh = np_mesh_to_vtk(np.asarray(verts), np.asarray(faces))
                        pd_mesh = smooth_polydata(pd_mesh,5)
                        out_path = os.path.join(trace_dir, f"{save_base}_surface_{sidx:02d}.vtp")
                        write_vtk_polydata(pd_mesh, out_path)
                        print(f"[WRITE] surface: {out_path}")
                        merged_surfaces.append(pd_mesh)
                    except Exception as e:
                        print(f"[WARN] saving surface {sidx} failed: {e}")

        # ---- case-level assembly outputs
        assembly_dir = os.path.join(dir_output_case, "assembly")
        os.makedirs(assembly_dir, exist_ok=True)
        if WRITE_CENTERLINE_MERGE:
            merged_centerlines.append(guide_centerline)

        if start_prob_global is not None:
            assembled_prob = start_prob_global
            assembled_prob_path = os.path.join(assembly_dir, f"{case_id}_assembled_prob.nii.gz")
            sitk.WriteImage(assembled_prob, assembled_prob_path)
            print(f"[WRITE] assembled probability: {assembled_prob_path}")

            thr = ASSEMBLY_THRESH
            assembled_mask = sitk.BinaryThreshold(
                assembled_prob, lowerThreshold=thr, upperThreshold=1e9,
                insideValue=1, outsideValue=0
            )
            assembled_mask.CopyInformation(assembled_prob)
            assembled_mask_path = os.path.join(assembly_dir, f"{case_id}_assembled_mask_thr{thr:.2f}.nii.gz")
            sitk.WriteImage(assembled_mask, assembled_mask_path)
            print(f"[WRITE] assembled mask: {assembled_mask_path}")
        else:
            print("[WARN] No assembled volume produced for this case.")

        # merged artifacts
        sp = tuple(map(float, image_ref.GetSpacing()))
        # assembled_mask is a sitk.Image from your pipeline (binary thresholded)
        assembled_mask_path = os.path.join(assembly_dir, f"{case_id}_assembled_mask_postproc.nii.gz")
        # fine-tune parameters as you like
        save_assembled_with_skimage(assembled_mask, assembled_mask_path,
                                    bin_thr=0.5, do_closing=True,
                                    closing_radius_vox=2, remove_small_objects_vox=100, fill_holes=True)

        if len(merged_centerlines) > 0:
            app = vtk.vtkAppendPolyData()
            for pd in merged_centerlines:
                app.AddInputData(pd)
            app.Update()
            clean = vtk.vtkCleanPolyData()
            clean.SetInputConnection(app.GetOutputPort())
            clean.ToleranceIsAbsoluteOn()
            clean.SetAbsoluteTolerance(0.25 * float(min(sp)))
            clean.PointMergingOn()
            clean.Update()
            merged_out = os.path.join(assembly_dir, f"{case_id}_centerlines_merged_seqseg.vtp")
            write_vtk_polydata(clean.GetOutput(), merged_out)
            print(f"[WRITE] merged centerlines: {merged_out}")
        else:
            print("[INFO] no centerlines to merge for this case.")
        merged_centerline_vtp = os.path.join(assembly_dir, f"{case_id}__post_centerlines_merged_seqseg.vtp")
        merge_and_smooth_vtk(merged_centerlines, merged_centerline_vtp,
                             resample_step_mm=0.5,
                             smooth_window=1,
                             snap_tol_mm=0.5,
                             connect_max_mm=2,
                             vtk_smooth_iter=2,
                             vtk_passband=0.1)

        if len(merged_points) > 0:
            app = vtk.vtkAppendPolyData()
            for pd in merged_points:
                app.AddInputData(pd)
            app.Update()
            clean = vtk.vtkCleanPolyData()
            clean.SetInputConnection(app.GetOutputPort())
            clean.ToleranceIsAbsoluteOn()
            clean.SetAbsoluteTolerance(0.25 * float(min(sp)))
            clean.PointMergingOn()
            clean.Update()
            merged_out = os.path.join(assembly_dir, f"{case_id}_points_merged.vtp")
            write_vtk_polydata(clean.GetOutput(), merged_out)
            print(f"[WRITE] merged points: {merged_out}")
        else:
            print("[INFO] no visited points to merge for this case.")

        if len(merged_inside_points) > 0:
            app = vtk.vtkAppendPolyData()
            for pd in merged_inside_points:
                app.AddInputData(pd)
            app.Update()
            clean = vtk.vtkCleanPolyData()
            clean.SetInputConnection(app.GetOutputPort())
            clean.ToleranceIsAbsoluteOn()
            clean.SetAbsoluteTolerance(0.25 * float(min(sp)))
            clean.PointMergingOn()
            clean.Update()
            merged_out = os.path.join(assembly_dir, f"{case_id}_inside_points_merged_seqseg.vtp")
            write_vtk_polydata(clean.GetOutput(), merged_out)
            print(f"[WRITE] merged inside points: {merged_out}")
        else:
            print("[INFO] no inside points to merge for this case.")

        if len(merged_surfaces) > 0:
            app = vtk.vtkAppendPolyData()
            for pd in merged_surfaces:
                app.AddInputData(pd)
            app.Update()
            tri = vtk.vtkTriangleFilter()
            tri.SetInputConnection(app.GetOutputPort())
            tri.PassLinesOff()
            tri.PassVertsOff()
            tri.Update()

            clean = vtk.vtkCleanPolyData()
            clean.SetInputConnection(tri.GetOutputPort())
            clean.ToleranceIsAbsoluteOn()
            clean.SetAbsoluteTolerance(0.25 * float(min(sp)))
            clean.PointMergingOn()
            clean.Update()
            cleaned = clean.GetOutput()

            passBand = 0.01  # 0.001
            # featureAngle = 120.0
            smoother = vtk.vtkWindowedSincPolyDataFilter()
            smoother.SetInputData(cleaned)
            smoother.SetNumberOfIterations(5)
            smoother.BoundarySmoothingOff()
            smoother.FeatureEdgeSmoothingOff()
            # smoother.SetFeatureAngle(featureAngle)
            smoother.SetPassBand(passBand)
            smoother.NonManifoldSmoothingOn()
            smoother.NormalizeCoordinatesOn()
            smoother.Update()

            merged_out = os.path.join(assembly_dir, f"{case_id}_surfaces_merged_seqseg.vtp")
            write_vtk_polydata(smoother.GetOutput(), merged_out)
            print(f"[WRITE] merged surfaces: {merged_out}")
        else:
            print("[INFO] no surfaces to merge for this case.")

        print(f"\nCase time: {((time.time() - case_t0) / 60):.2f} min\n")
        break
    print(f"Total execution time: {((time.time() - t0) / 60):.2f} min")


if __name__ == '__main__':
    main()
