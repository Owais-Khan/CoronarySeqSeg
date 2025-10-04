from typing import Optional, Tuple, List

import numpy as np
import networkx as nx
from pathlib import Path
from scipy.ndimage import distance_transform_edt, map_coordinates
from SeqSeg.seqseg.modules.centerline import post_process_centerline
from SeqSeg.seqseg.modules.vtk_functions import write_vtk_polydata

import SimpleITK as sitk
from gnn_model.gnn_modules import gm_load, gm_predict_graph,save_predicted_graph_to_vtp
import skfmm
from seqseg_modules_modified.trace_centerline import trace_centerline
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree as KDTree
import faulthandler, time, os, yaml, vtk



class CFG:
    # I/O

    # ECC graph settidongs
    knn_k: int = 12
    knn_radius_mm: float = 6.0
    edge_prob_thresh: float = 0.5

    # Component cropping
    top_k_components: int = 10
    pad_mm = (0,0,0)
    min_size_mm = (12.0, 12.0, 12.0)

    # FMM/target selection
    max_targets_per_comp: int = 25
    min_target_sep_mm: float = 9.0
    spur_len_min_mm: float = 9.0
    prob_min: float = 0.6
    prob_exp: float = 9

    # Tracer limits (defaults; most are read from YAML)
    MAX_STEP_SIZE: int = 10
    MAX_STEPS_PER_COMPONENT: int = 220



cfg = CFG()

STEP_MM   = 0.5
JUMP_MM   = 1.0
LEAK_FRAC = 0.05
GRAD_FLOOR = 1e-6
MAX_ITERS  = 4000
TT_MARGIN  = 1e-6

def create_directories(output_folder: str, write_samples: bool) -> None:
    base = Path(output_folder)
    for sub in ("", "errors", "assembly"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    if write_samples:
        for sub in ("volumes", "predictions", "centerlines", "surfaces",
                    "points", "animation", "images", "labels"):
            (base / sub).mkdir(parents=True, exist_ok=True)


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


def largest_components(G: nx.Graph, k: int = 1, *, by: str = "nodes") -> List[nx.Graph]:
    if G.number_of_nodes() == 0:
        return []
    comps = [set(c) for c in nx.connected_components(G.to_undirected())]
    if not comps:
        return []
    if by == "edge_weight_sum":
        def score(nodeset):
            H = G.subgraph(nodeset)
            return sum(float(d.get('weight', 1.0)) for _, _, d in H.edges(data=True))
    else:
        def score(nodeset):
            return len(nodeset)
    comps.sort(key=score, reverse=True)
    return [G.subgraph(c).copy() for c in comps[:max(1, int(k))]]


def _edge_length_mm_from_phys(G: nx.Graph, u, v, pos_phys_key='pos_phys') -> float:
    pu = np.asarray(G.nodes[u][pos_phys_key], float)
    pv = np.asarray(G.nodes[v][pos_phys_key], float)
    return float(np.linalg.norm(pu - pv))


def attach_edge_metrics_mm_from_phys(G: nx.Graph,
                                     *,
                                     prob_key: str = 'edge_prob',
                                     pos_phys_key: str = 'pos_phys',
                                     cost_key: str = 'length_cost',
                                     length_key: str = 'length_mm',
                                     prob_exp: float = 1.5) -> None:
    for u, v in G.edges():
        L = _edge_length_mm_from_phys(G, u, v, pos_phys_key=pos_phys_key)
        p = float(G.edges[u, v].get(prob_key, 1.0))
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

def graph_corridor_mask_zyx(Gc: nx.Graph, cropped_img: sitk.Image,
                            start_xyz: Tuple[int,int,int], size_xyz: Tuple[int,int,int],
                            pos_idx_key: str = 'pos_idx_xyz', radius_mm: float = 2.0) -> np.ndarray:

    sz_zyx = np.array(cropped_img.GetSize(), int)[::-1]
    seeds = np.zeros(sz_zyx, np.uint8)  # Z,Y,X
    for n, d in Gc.nodes(data=True):
        if pos_idx_key not in d:
            continue
        gx, gy, gz = np.asarray(d[pos_idx_key], int)  # global X,Y,Z
        lx, ly, lz = gx - start_xyz[0], gy - start_xyz[1], gz - start_xyz[2]
        if 0 <= lz < sz_zyx[0] and 0 <= ly < sz_zyx[1] and 0 <= lx < sz_zyx[2]:
            seeds[lz, ly, lx] = 1
    if seeds.max() == 0:
        return np.zeros_like(seeds, bool)

    bg = np.ones_like(seeds, np.uint8)
    bg[seeds == 1] = 0
    dist_mm = distance_transform_edt(bg, sampling=cropped_img.GetSpacing()[::-1])
    return (dist_mm <= float(radius_mm))



def _idx_from_phys(img: sitk.Image, p_phys_xyz: np.ndarray) -> np.ndarray:
    return np.array(img.TransformPhysicalPointToContinuousIndex(p_phys_xyz))[::-1]

def _phys_from_idx(img: sitk.Image, cont_zyx: np.ndarray) -> np.ndarray:
    return np.array(img.TransformContinuousIndexToPhysicalPoint(cont_zyx[::-1].tolist()))

def backtrack_gradient(start_point_phys: np.ndarray, end_point_phys: np.ndarray,
                       gradient_field: list[np.ndarray], reference_image: sitk.Image,
                       travel_time: np.ndarray | None = None,
                       *,
                       graph_kd_phys: KDTree | None = None,  # <— NEW (optional)
                       graph_jump_mm: float = 1.5) -> list[np.ndarray]:
    path = [np.array(start_point_phys, float)]
    cur  = np.array(start_point_phys, float)
    spacing = np.array(reference_image.GetSpacing(), float)
    mean_sp = float(np.mean(spacing))
    step_idx = STEP_MM / mean_sp
    jump_idx = JUMP_MM / mean_sp
    graph_jump_idx = graph_jump_mm / mean_sp

    for _ in range(MAX_ITERS):
        if np.linalg.norm(cur - end_point_phys) <= mean_sp:
            break

        cur_zyx = _idx_from_phys(reference_image, cur)
        gz = map_coordinates(gradient_field[0], cur_zyx.reshape(3,1), order=1, mode='nearest')[0]
        gy = map_coordinates(gradient_field[1], cur_zyx.reshape(3,1), order=1, mode='nearest')[0]
        gx = map_coordinates(gradient_field[2], cur_zyx.reshape(3,1), order=1, mode='nearest')[0]
        g = np.array([gz, gy, gx], float)
        gnorm = float(np.linalg.norm(g))

        moved = False
        if gnorm > GRAD_FLOOR:
            new_zyx = cur_zyx + (-g / gnorm) * step_idx
            if travel_time is None:
                cur = _phys_from_idx(reference_image, new_zyx); path.append(cur.copy()); moved = True
            else:
                T_cur = float(map_coordinates(travel_time, cur_zyx.reshape(3,1),  order=1, mode='nearest')[0])
                T_new = float(map_coordinates(travel_time, new_zyx.reshape(3,1), order=1, mode='nearest')[0])
                if T_new <= T_cur - TT_MARGIN:
                    cur = _phys_from_idx(reference_image, new_zyx); path.append(cur.copy()); moved = True

        if not moved:
            # 1) try a tiny jump toward the goal
            end_zyx = _idx_from_phys(reference_image, end_point_phys)
            d = end_zyx - cur_zyx; dn = np.linalg.norm(d)
            if dn > 1e-9:
                new_zyx = cur_zyx + (d / dn) * jump_idx
                cur = _phys_from_idx(reference_image, new_zyx); path.append(cur.copy())
                moved = True

        if not moved and graph_kd_phys is not None:
            # 2) if still stuck, snap toward nearest graph node (≤ graph_jump_mm)
            dist, idx = graph_kd_phys.query(cur, k=1)
            if np.isfinite(dist) and dist <= graph_jump_mm:
                # move part-way toward that node (stays smooth)
                target_phys = graph_kd_phys.data[idx]
                v = target_phys - cur
                vn = np.linalg.norm(v)
                if vn > 1e-9:
                    step = min(graph_jump_mm, vn)
                    cur = cur + (v / vn) * step
                    path.append(cur.copy())
                    moved = True

        if not moved:
            break

    path.append(np.array(end_point_phys, float))
    return path


from scipy.ndimage import gaussian_filter

def trace_centerline_fmm(vessel_mask_np: np.ndarray,
                         seed_coords_vox_zyx: np.ndarray,
                         target_coords_list_vox_zyx: list[np.ndarray],
                         reference_image: sitk.Image,
                         *,
                         corridor_mask_zyx: np.ndarray | None = None,   # <— NEW
                         graph_kd_phys: KDTree | None = None            # <— NEW
                         ) -> nx.Graph:
    mask = vessel_mask_np.astype(bool)
    edt  = distance_transform_edt(mask)
    speed = np.full_like(edt, 1e-8, dtype=float)
    if mask.any():
        med_in = float(np.median((edt + 1e-6)[mask]))
    else:
        med_in = 1.0
    speed[mask] = edt[mask] + 1e-6
    if corridor_mask_zyx is not None:
        speed[~mask & corridor_mask_zyx] = max(1e-6, med_in * LEAK_FRAC)

    speed = gaussian_filter(speed, sigma=0.6)

    phi = np.ones_like(speed, float)
    sz, sy, sx = [int(np.clip(v, 0, s-1)) for v, s in zip(seed_coords_vox_zyx, speed.shape)]
    phi[sz, sy, sx] = -1.0

    travel_time = skfmm.travel_time(phi, speed=speed, dx=reference_image.GetSpacing()[::-1])
    gradient = np.gradient(travel_time)  # z,y,x

    seed_phys = np.array(reference_image.TransformIndexToPhysicalPoint(seed_coords_vox_zyx[::-1].tolist()), float)
    targets_phys = [np.array(reference_image.TransformIndexToPhysicalPoint(np.asarray(t)[::-1].tolist()), float)
                    for t in target_coords_list_vox_zyx]

    G, node_counter, node_map = nx.Graph(), 0, {}
    def add_node(p_phys_xyz: np.ndarray) -> int:
        nonlocal node_counter
        key = tuple(np.asarray(p_phys_xyz, float))
        if key in node_map: return node_map[key]
        vox_xyz = reference_image.TransformPhysicalPointToIndex(p_phys_xyz.tolist())
        pos_vox_zyx = np.array(vox_xyz)[::-1]
        pos_vox_zyx = np.clip(pos_vox_zyx, 0, np.array(speed.shape) - 1).astype(int)
        G.add_node(node_counter, pos_phys=np.asarray(p_phys_xyz, float), pos=pos_vox_zyx)
        node_map[key] = node_counter; node_counter += 1
        return node_map[key]

    for t_phys in targets_phys:
        path = backtrack_gradient(
            start_point_phys=t_phys, end_point_phys=seed_phys,
            gradient_field=gradient, reference_image=reference_image,
            travel_time=travel_time,
            graph_kd_phys=graph_kd_phys  # <— NEW
        )
        if len(path) < 2: continue
        ids = [add_node(p) for p in path]
        for i in range(len(ids) - 1):
            u, v = ids[i], ids[i + 1]
            if u != v and not G.has_edge(u, v):
                d = float(np.linalg.norm(G.nodes[u]['pos_phys'] - G.nodes[v]['pos_phys']))
                G.add_edge(u, v, length_mm=d)
    return G


def points_poly(poly: vtk.vtkPolyData, spacing_xyz) -> vtk.vtkPolyData:
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(poly)
    clean.ToleranceIsAbsoluteOn()
    clean.SetAbsoluteTolerance(0.25 * float(min(spacing_xyz)))
    clean.PointMergingOn()
    clean.Update()
    return clean.GetOutput()


def _edt_mm_from_seg(seg_img: sitk.Image) -> np.ndarray:
    seg_np_zyx = (sitk.GetArrayFromImage(seg_img) > 0).astype(np.uint8)
    edt_mm_zyx = distance_transform_edt(seg_np_zyx, sampling=seg_img.GetSpacing()[::-1])
    return edt_mm_zyx


def build_seqseg_centerline_polydata_from_graph(
    G: nx.Graph,
    seg_img: sitk.Image,
    seed_node: Optional[int] = None,
    target_nodes: Optional[List[int]] = None,
    *,
    weight_keys: Tuple[str, ...] = ("length_cost", "length_mm", "weight"),
    dedupe_targets_mm: float = 5.0,
    max_targets: Optional[int] = None,
) -> vtk.vtkPolyData:
    poly = vtk.vtkPolyData()
    if G.number_of_nodes() == 0:
        return poly

    # Resolve seed/targets if not given
    if seed_node is None or not target_nodes:
        s, ts = select_seed_and_targets_from_features(
            G,
            max_targets=(max_targets if max_targets is not None else 25),
            prob_exp=1.5,
            Lspur_min_mm=2.0,
            prob_min=0.15,
            min_sep_mm=dedupe_targets_mm,
        )
        seed_node = seed_node if seed_node is not None else s
        target_nodes = target_nodes if target_nodes else ts
    if seed_node is None or not target_nodes:
        return poly

    edt_mm_zyx = _edt_mm_from_seg(seg_img)
    size_xyz = np.array(seg_img.GetSize(), dtype=int)

    def _edge_w(u, v) -> float:
        d = G.edges[u, v]
        for k in weight_keys:
            if k in d:
                try:
                    return float(d[k])
                except Exception:
                    pass
        p1 = np.asarray(G.nodes[u]["pos_phys"], float)
        p2 = np.asarray(G.nodes[v]["pos_phys"], float)
        return float(np.linalg.norm(p1 - p2))

    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    radii = vtk.vtkDoubleArray(); radii.SetName("MaximumInscribedSphereRadius")
    radii_f = vtk.vtkDoubleArray(); radii_f.SetName("f")
    gnode = vtk.vtkIntArray(); gnode.SetName("GlobalNodeID")
    clid  = vtk.vtkIntArray(); clid.SetName("CenterlineId")

    branch_id = 0
    for t in target_nodes:
        try:
            path = nx.shortest_path(G, seed_node, t, weight=_edge_w)
        except nx.NetworkXNoPath:
            continue
        except Exception:
            try:
                path = nx.shortest_path(G, seed_node, t)
            except Exception:
                continue

        if len(path) < 2:
            continue

        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(path))

        for i, nid in enumerate(path):
            p_xyz = np.asarray(G.nodes[nid]["pos_phys"], float)
            pid = pts.InsertNextPoint(p_xyz.tolist())
            polyline.GetPointIds().SetId(i, pid)

            ix, iy, iz = seg_img.TransformPhysicalPointToIndex(tuple(p_xyz))
            ix = int(np.clip(ix, 0, size_xyz[0] - 1))
            iy = int(np.clip(iy, 0, size_xyz[1] - 1))
            iz = int(np.clip(iz, 0, size_xyz[2] - 1))
            r_mm = float(edt_mm_zyx[iz, iy, ix])

            radii.InsertNextValue(r_mm)
            radii_f.InsertNextValue(r_mm)
            gnode.InsertNextValue(int(nid))
            clid.InsertNextValue(branch_id)

        lines.InsertNextCell(polyline)
        branch_id += 1

    poly.SetPoints(pts)
    poly.SetLines(lines)
    poly.GetPointData().AddArray(radii)
    poly.GetPointData().AddArray(radii_f)
    poly.GetPointData().AddArray(gnode)
    poly.GetPointData().AddArray(clid)
    return poly


def export_seqseg_centerline_from_graph(
    G: nx.Graph,
    seg_img: sitk.Image,
    out_vtp_path: str,
    seed_node: Optional[int] = None,
    target_nodes: Optional[List[int]] = None,
    *,
    clean_and_smooth: bool = True
) -> vtk.vtkPolyData:
    poly = build_seqseg_centerline_polydata_from_graph(
        G, seg_img, seed_node=seed_node, target_nodes=target_nodes
    )
    if clean_and_smooth and poly.GetNumberOfPoints() > 0:
        try:
            poly = post_process_centerline(poly, verbose=False)
        except Exception:
            pass
    os.makedirs(os.path.dirname(out_vtp_path), exist_ok=True)
    write_vtk_polydata(poly, out_vtp_path)
    return poly

def rebind_points_from_indices(G: nx.Graph, img: sitk.Image,
                               idx_key='pos_idx_xyz', out_key='point'):
    for n, d in G.nodes(data=True):
        if idx_key in d:
            ix, iy, iz = map(int, d[idx_key])
            p = img.TransformIndexToPhysicalPoint((ix, iy, iz))
            d[out_key] = np.asarray(p, float)


def attach_radii_from_global_edt(G: nx.Graph, edt_mm_zyx: np.ndarray, idx_key='pos_idx_xyz'):
    for n, d in G.nodes(data=True):
        if idx_key in d:
            x, y, z = map(int, d[idx_key])  # xyz indices
            r_mm = float(edt_mm_zyx[z, y, x])  # edt is z,y,x
            d['radius'] = r_mm
            d['MaximumInscribedSphereRadius'] = r_mm


def attach_edge_metrics(G: nx.Graph, prob_key='edge_prob', prob_exp: float = 1.5):
    EPS = 1e-6
    for u, v, e in G.edges(data=True):
        p1 = np.asarray(G.nodes[u]['point'], float)
        p2 = np.asarray(G.nodes[v]['point'], float)
        L = float(np.linalg.norm(p1 - p2))
        prob = float(e.get('prob', e.get(prob_key, 1.0)))
        e['length'] = L
        e['prob'] = prob
        e['cost'] = L / max(prob, EPS)**prob_exp


def inside_frac_phys(G: nx.Graph, img: sitk.Image, key='point', sample=1024) -> float:
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


import os, glob, time, argparse
import numpy as np
import torch
from nnUNet.nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
from nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def parse_args():
    p = argparse.ArgumentParser("GNN model")
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
        params_seqseg = yaml.safe_load(f)
    with open(gnn_cfg, 'r') as f:
        params_gnn = yaml.safe_load(f)

    img_format = args.img_ext
    seg_dir = args.pred_dir
    images_dir = args.data_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_cases = [f.replace(img_format, "") for f in os.listdir(seg_dir) if f.endswith(img_format)]

    model, cfg_gnn, device, _ = gm_load(
        ckpt_path=os.path.join(
            args.gnn_folder,
            "gnn_checkpoint.pt"
        ),
        runtime_cfg=params_gnn,
        prefer_cfg='runtime'
    )
    model_tuple = (model, cfg_gnn, device, "gnn_checkpoint.pt")

    # SeqSeg model folder
    seqseg_model_folder = os.path.join(
        nnUNet_results, f"{args.dataset_id}/nnUNetTrainer__nnUNetPlans__3d_fullres"
    )

    max_steps_per_component = int(params_seqseg.get("MAX_STEPS_PER_COMPONENT", 300))

    # Assembly / writing params
    ASSEMBLY_THRESH = float(params_seqseg.get("ASSEMBLY_THRESHOLD", 0.5))
    WRITE_CENTERLINE_MERGE = bool(params_seqseg.get("WRITE_CENTERLINE_MERGE", True))

    def read_vtp_polydata(path):
        r = vtk.vtkXMLPolyDataReader()
        r.SetFileName(path)
        r.Update()
        return r.GetOutput()

    for case_id in all_cases:
        case_t0 = time.time()
        print(f"\n{'='*60}\nProcessing case: {case_id}\n{'='*60}")
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
            node_min_spacing_mm=float(getattr(params_gnn, "node_min_spacing_mm", 0.2)) if hasattr(params_gnn, "node_min_spacing_mm") else None,
            knn_k=int(params_gnn.knn_k),
            knn_radius_mm=float(params_gnn.knn_radius_mm),
            edge_prob_thresh=float(params_gnn.edge_prob_thresh),
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

        comps = largest_components(G, k=params_gnn.top_k_components, by='nodes')
        print(f"  - Cropping from top {len(comps)} graph component(s)")

        # Global EDT in mm (ZYX) for radii
        seg_np_zyx = (sitk.GetArrayFromImage(segmentation_image) > 0).astype(np.uint8)
        edt_mm_zyx_global = distance_transform_edt(seg_np_zyx, sampling=segmentation_image.GetSpacing()[::-1])

        for gi, Gc in enumerate(comps):
            roi = bbox_from_graph_component(
                Gc, image_ref, pos_idx_key='pos_idx_xyz',
                pad_mm=params_seqseg.pad_mm, min_size_mm=params_seqseg.min_size_mm
            )
            if roi is None:
                print(f"  - comp {gi}: no points; skip.")
                continue

            start_xyz, size_xyz = roi
            # Crop image & seg for FMM centerline bootstrap (tracer will run on FULL image)
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
            rebind_points_from_indices(Gc, image_ref, idx_key='pos_idx_xyz', out_key='point')
            attach_radii_from_global_edt(Gc, edt_mm_zyx_global, idx_key='pos_idx_xyz')
            attach_edge_metrics(Gc, prob_key='edge_prob', prob_exp=params_seqseg.prob_exp)
            print(f"[DBG inside] comp{gi} volumetric_graph inside_frac (cropped img): "
                  f"{inside_frac_phys(Gc, cropped_img, key='point'):.3f}")

            # ---- Seed/targets from graph features
            seed_node, target_nodes = select_seed_and_targets_from_features(
                Gc,
                max_targets=params_seqseg.max_targets_per_comp,
                prob_exp=params_seqseg.prob_exp,
                Lspur_min_mm=params_seqseg.spur_len_min_mm,
                prob_min=params_seqseg.prob_min,
                min_sep_mm=params_seqseg.min_target_sep_mm,
            )
            if seed_node is None or not target_nodes:
                print("  - no valid seed/targets for this component; skipping")
                continue

            def to_local_zyx(nid):
                g_xyz = np.asarray(Gc.nodes[nid]['pos_idx_xyz'], int)
                l_xyz = g_xyz - np.asarray(start_xyz, int)
                l_zyx = np.array([l_xyz[2], l_xyz[1], l_xyz[0]], dtype=int)
                sz = np.array(cropped_seg.GetSize(), int)[::-1]
                return np.minimum(np.maximum(l_zyx, 0), sz - 1)

            seed_zyx = to_local_zyx(seed_node)
            targets_zyx = [to_local_zyx(t) for t in target_nodes]
            vessel_mask_np = (sitk.GetArrayFromImage(cropped_seg) > 0).astype(np.uint8)

            print(f"    comp {gi}: tracing FMM centerline with {len(targets_zyx)} targets...")
            corridor = graph_corridor_mask_zyx(Gc, cropped_seg, start_xyz, size_xyz, radius_mm=2.0)
            graph_pts_phys = np.array([np.asarray(Gc.nodes[n]['point'], float)
                                       for n in Gc.nodes() if 'point' in Gc.nodes[n]])
            graph_kd = KDTree(graph_pts_phys) if len(graph_pts_phys) else None

            final_graph = trace_centerline_fmm(
                vessel_mask_np, seed_zyx, targets_zyx, cropped_seg,
                corridor_mask_zyx=corridor,
                graph_kd_phys=graph_kd
            )
            if final_graph.number_of_nodes() == 0:
                print(f"    comp {gi}: FMM produced empty graph; skipping.")
                continue

            # Normalize centerline_graph: ensure 'point' and 'radius' (mm)
            edt_mm_zyx_local = distance_transform_edt(
                (sitk.GetArrayFromImage(cropped_seg) > 0).astype(np.uint8),
                sampling=cropped_seg.GetSpacing()[::-1]
            )
            for n, d in final_graph.nodes(data=True):
                if 'pos_phys' in d and 'point' not in d:
                    d['point'] = np.asarray(d['pos_phys'], float)
                if 'radius' not in d:
                    if 'pos' in d and len(d['pos']) == 3:
                        z, y, x = map(int, d['pos'])
                    else:
                        ci = np.array(cropped_seg.TransformPhysicalPointToContinuousIndex(tuple(d['point'])), float)
                        z, y, x = map(int, np.clip(ci[::-1], [0, 0, 0], np.array(edt_mm_zyx_local.shape) - 1))
                    r = float(edt_mm_zyx_local[z, y, x])
                    d['radius'] = r
                    d['MaximumInscribedSphereRadius'] = r

            # Export SeqSeg-compatible polyline & choose seed/targets in that graph
            pts = np.array([final_graph.nodes[n]['pos_phys'] for n in final_graph.nodes()])
            seed_phys = np.array(cropped_seg.TransformIndexToPhysicalPoint(tuple(seed_zyx[::-1].tolist())))
            d2 = np.sum((pts - seed_phys[None, :]) ** 2, axis=1)
            seed_final = list(final_graph.nodes())[int(np.argmin(d2))]
            target_final_nodes = [n for n, deg in final_graph.degree() if deg == 1 and n != seed_final]

            cent_vtp = os.path.join(dir_output_case, "centerlines", f"{comp_tag}_centerline.vtp")
            os.makedirs(os.path.dirname(cent_vtp), exist_ok=True)
            guide_poly = export_seqseg_centerline_from_graph(
                final_graph, cropped_seg, cent_vtp,
                seed_node=seed_final,
                target_nodes=target_final_nodes,
                clean_and_smooth=True
            )
            print(f"  - SeqSeg-compatible centerline saved: {os.path.basename(cent_vtp)}")
            guide_centerline = read_vtp_polydata(cent_vtp)
            guide_centerline = points_poly(guide_centerline, cropped_seg.GetSpacing())
            write_vtk_polydata(guide_centerline, cent_vtp)
            if WRITE_CENTERLINE_MERGE:
                merged_centerlines.append(guide_centerline)
            prev_prob_for_tracer = start_prob_global if start_prob_global is not None else None

            # Use IDs from the volumetric graph Gc (these were computed earlier)
            seed_id_tracer = int(seed_node)  # from Gc
            target_ids_tracer = list(map(int, target_nodes))  # from Gc

            _lc, _ls, _lp, _li, assembly_segs, vt, i = trace_centerline(
                output_folder=dir_output_case,
                image_file=dir_image,
                case=case_id,
                model_folder=seqseg_model_folder,
                fold=args.fold,
                graph=Gc,
                centerline_graph=final_graph,
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
            delta_bin = sitk.And(new_bin, sitk.Not(old_bin))  # only voxels newly above threshold

            largest_delta = largest_cc_simple(sitk.Cast(delta_bin, sitk.sitkUInt8), background_value=0)

            stats = sitk.StatisticsImageFilter()
            stats.Execute(largest_delta)
            if stats.GetSum() == 0:
                print("    [assembly] no novel region above threshold; skipping")
                continue

            inc_prob = sitk.Mask(curr_prob, largest_delta)  # keep values only in largest new CC
            start_prob_global = sitk.Maximum(prev_prob, inc_prob)

            if coverage_union is None:
                coverage_union = sitk.Cast(largest_delta, sitk.sitkUInt8)
                coverage_union.CopyInformation(prev_prob)
            else:
                coverage_union = sitk.Or(coverage_union, sitk.Cast(largest_delta, sitk.sitkUInt8))

        # ➌ After all components: write assembled volumes (case-level)
        assembly_dir = os.path.join(dir_output_case, "assembly")
        os.makedirs(assembly_dir, exist_ok=True)

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

            if coverage_union is not None:
                coverage_mask_path = os.path.join(assembly_dir, f"{case_id}_assembly_coverage_mask.nii.gz")
                sitk.WriteImage(coverage_union, coverage_mask_path)
                print(f"[WRITE] coverage mask: {coverage_mask_path}")
        else:
            print("[WARN] No assembled volume produced for this case.")

        if WRITE_CENTERLINE_MERGE and len(merged_centerlines) > 0:
            app = vtk.vtkAppendPolyData()
            for pd in merged_centerlines:
                app.AddInputData(pd)
            app.Update()
            clean = vtk.vtkCleanPolyData()
            clean.SetInputConnection(app.GetOutputPort())
            clean.ToleranceIsAbsoluteOn()
            clean.SetAbsoluteTolerance(0.25 * min(image_ref.GetSpacing()))
            clean.Update()
            merged_out = os.path.join(assembly_dir, f"{case_id}_centerlines_merged.vtp")
            write_vtk_polydata(clean.GetOutput(), merged_out)
            print(f"[WRITE] merged centerlines: {merged_out}")

        print(f"\nCase time: {((time.time() - case_t0) / 60):.2f} min\n")

    print(f"Total execution time: {((time.time() - t0) / 60):.2f} min")


if __name__ == '__main__':

    main()


