from SeqSeg.seqseg.modules.sitk_functions import (read_image, create_new, sitk_to_numpy,
                             numpy_to_sitk, keep_component_seeds,
                             is_point_in_image)
from SeqSeg.seqseg.modules.centerline import calc_centerline_fmm
from SeqSeg.seqseg.modules.vtk_functions import (write_vtk_polydata,
                            points2polydata, appendPolyData)
import numpy as np
import SimpleITK as sitk
import operator
from datetime import datetime
import sys
sys.stdout.flush()


class Segmentation:
    """
    Class to keep track of a global segmentation,
    and update it with new segmentations
    """

    def __init__(self,
                 case=None,
                 image_file=None,
                 weighted=False,
                 weight_type=None,
                 image=None,
                 start_seg=None):
        """
        Args:
            case: name of the case
            image_file: image file to create the global segmentation
            in the same space
            weighted: whether to use a weighted average for the segmentation
            weight_type: type of weight to use for the weighted average
            image: image object to create the global segmentation
            start_seg: initial segmentation to start with
        """
        if case:
            self.name = case
        if image_file:
            self.image_reader = read_image(image_file)

            if start_seg is not None:
                self.assembly = start_seg
            else:
                new_img = create_new(self.image_reader)
                self.assembly = new_img

        elif image:

            self.image_reader = image
            self.assembly = image

        else:
            print("Please provide either an image file or an image object")

        self.number_updates = np.zeros(sitk_to_numpy(self.assembly).shape)

        self.weighted = weighted

        if weighted:
            # also keep track of how many updates to pixels
            if start_seg is None:
                self.n_updates = np.zeros(sitk_to_numpy(self.assembly).shape)
            else:
                self.n_updates = sitk_to_numpy(self.assembly)
            # print("Creating weighted segmentation")
            assert weight_type, "Please provide a weight type"
            assert weight_type in ['radius', 'gaussian'], """Weight type
            not recognized"""
            self.weight_type = weight_type

    def add_segmentation(self,
                         volume_seg,
                         index_extract,
                         size_extract,
                         weight=None):
        """
        Function to add a new segmentation to the global assembly
        Args:
            volume_seg: the new segmentation to add
            index_extract: index for sitk volume extraction
            size_extract: number of voxels to extract in each dim
            weight: weight for the weighted average
        """
        # Load the volumes
        np_arr = sitk_to_numpy(self.assembly).astype(float)
        np_arr_add = sitk_to_numpy(volume_seg).astype(float)

        # Calculate boundaries
        cut = 0
        edges = np.array(index_extract) + np.array(size_extract) - cut
        index_extract = np.array(index_extract) + cut

        # Keep track of number of updates
        curr_n = self.number_updates[index_extract[2]:edges[2],
                                     index_extract[1]:edges[1],
                                     index_extract[0]:edges[0]]

        # Isolate current subvolume of interest
        curr_sub_section = np_arr[index_extract[2]:edges[2],
                                  index_extract[1]:edges[1],
                                  index_extract[0]:edges[0]]
        np_arr_add = np_arr_add[cut:size_extract[2]-cut,
                                cut:size_extract[1]-cut,
                                cut:size_extract[0]-cut]
        # Find indexes where we need to average predictions
        ind = curr_n > 0
        # Where this is the first update, copy directly
        curr_sub_section[curr_n == 0] = np_arr_add[curr_n == 0]

        if not self.weighted:  # Then we do plain average
            # Update those values, calculating an average
            curr_sub_section[ind] = 1/(curr_n[ind]+1) * (
                np_arr_add[ind] + (curr_n[ind])*curr_sub_section[ind])
            # Add to update counter for these voxels
            self.number_updates[index_extract[2]:edges[2],
                                index_extract[1]:edges[1],
                                index_extract[0]:edges[0]] += 1

        else:
            if self.weight_type == 'radius':
                curr_sub_section[ind] = 1/(curr_n[ind]+weight)*(
                    weight*np_arr_add[ind] + (
                        curr_n[ind])*curr_sub_section[ind])
                # Add to update weight sum for these voxels
                self.number_updates[index_extract[2]:edges[2],
                                    index_extract[1]:edges[1],
                                    index_extract[0]:edges[0]] += weight
                self.n_updates[index_extract[2]:edges[2],
                               index_extract[1]:edges[1],
                               index_extract[0]:edges[0]] += 1

            elif self.weight_type == 'gaussian':
                # now the weight varies with the distance to the center of
                # the volume, and the distance to the border
                weight_array = self.calc_weight_array_gaussian(size_extract)
                # print(f"weight array size: {weight_array.shape},
                # ind size: {ind.shape}")
                # Update those values, calculating an average
                curr_sub_section[ind] = 1/(
                    curr_n[ind]+weight_array[ind])*(
                        weight_array[ind]*np_arr_add[ind]
                        + (curr_n[ind])*curr_sub_section[ind])
                # Add to update weight sum for these voxels
                self.number_updates[index_extract[2]:edges[2],
                                    index_extract[1]:edges[1],
                                    index_extract[0]:edges[0]] += weight_array
                self.n_updates[index_extract[2]:edges[2],
                               index_extract[1]:edges[1],
                               index_extract[0]:edges[0]] += 1

        # Update the global volume
        np_arr[index_extract[2]:edges[2],
               index_extract[1]:edges[1],
               index_extract[0]:edges[0]] = curr_sub_section

        self.assembly = numpy_to_sitk(np_arr, self.image_reader)

    def calc_weight_array_gaussian(self, size_extract):
        """Function to calculate the weight array for
        a gaussian weighted segmentation"""
        # define std as 10% of the size of the volume
        std = 0.5*np.ones_like(size_extract)  # 0.1
        # create a grid of distances to the center of the volume
        x = np.linspace(-size_extract[0]/2, size_extract[0]/2, size_extract[0])
        y = np.linspace(-size_extract[1]/2, size_extract[1]/2, size_extract[1])
        z = np.linspace(-size_extract[2]/2, size_extract[2]/2, size_extract[2])
        # normalize
        x = x/(size_extract[0]/2)
        y = y/(size_extract[1]/2)
        z = z/(size_extract[2]/2)
        # create a meshgrid
        x, y, z = np.meshgrid(z, y, x)
        # now transpose
        x = x.transpose(1, 0, 2)
        y = y.transpose(1, 0, 2)
        z = z.transpose(1, 0, 2)
        # calculate the weight array
        weight_array = np.exp(-0.5*(x**2/std[0]**2
                                    + y**2/std[1]**2
                                    + z**2/std[2]**2))
        # print(f"Max weight: {np.max(weight_array)}")
        # print(f"Min weight: {np.min(weight_array)}")
        return weight_array

    def create_mask(self):
        "Function to create a global image mask of areas that were segmented"
        mask = (self.number_updates > 0).astype(int)
        mask = numpy_to_sitk(mask, self.image_reader)
        self.mask = mask

        return mask

    def upsample(self, template_size=[1000, 1000, 1000]):
        from SeqSeg.seqseg.modules.prediction import centering
        or_im = sitk.GetImageFromArray(self.assembly)
        or_im.SetSpacing(self.image_resampled.GetSpacing())
        or_im.SetOrigin(self.image_resampled.GetOrigin())
        or_im.SetDirection(self.image_resampled.GetDirection())
        target_im = create_new(or_im)
        target_im.SetSize(template_size)
        new_spacing = (
            np.array(or_im.GetSize())*np.array(
                or_im.GetSpacing()))/np.array(template_size)
        target_im.SetSpacing(new_spacing)

        resampled = centering(or_im, target_im, order=0)
        return resampled

    def upsample_sitk(self, template_size=[1000, 1000, 1000]):

        from SeqSeg.seqseg.modules.prediction import resample_spacing
        resampled, _ = resample_spacing(self.assembly,
                                        template_size=template_size)
        return resampled


import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Iterable, Set, Tuple
from dataclasses import dataclass

# Reuse SeqSeg helpers you already have
from SeqSeg.seqseg.modules.sitk_functions import is_point_in_image
from SeqSeg.seqseg.modules.vtk_functions import points2polydata, write_vtk_polydata

# --------- small vector helpers ----------
def _to_np(p) -> np.ndarray:
    return np.asarray(p, dtype=float)

def _vec(a, b) -> np.ndarray:
    return _to_np(b) - _to_np(a)

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return v / n

# --------- create SeqSeg-style step dict ----------
def create_step_dict_from_nodes(
    G: nx.Graph,
    prev_id: int,
    curr_id: int,
    angle_change: Optional[float] = None
) -> Dict[str, Any]:
    p0 = _to_np(G.nodes[prev_id]["point"])
    p1 = _to_np(G.nodes[curr_id]["point"])
    r0 = float(G.nodes[prev_id].get("radius", G.nodes[curr_id].get("radius", 0.0)))
    r1 = float(G.nodes[curr_id].get("radius", r0))
    tangent = _unit(_vec(p0, p1))

    step_dict: Dict[str, Any] = {
        "old point": p0,
        "point": p1,
        "old radius": r0,
        "radius": r1,
        "tangent": tangent,
        "chances": 0,
        "seg_file": None,
        "img_file": None,
        "surf_file": None,
        "cent_file": None,
        "prob_predicted_vessel": None,
        "point_pd": None,
        "surface": None,
        "centerline": None,
        "is_inside": False,
        "time": None,
        "dice": None,
    }
    if angle_change is not None:
        step_dict["angle change"] = angle_change
    return step_dict

@dataclass
class BuildCfg:
    weight_attr: str = "cost"       # edge weight to use for shortest paths
    use_prob_cost: bool = True      # derive cost = length / prob if possible
    eps_prob: float = 1e-6

class VesselTree:
    """
    Graph-first VesselTree. No potential_branches/branches.
    Maintains the 'vessel' as an ordered path (node ids) plus an edge set
    and SeqSeg-style step dicts compatible with your existing pipeline.
    """

    def __init__(
        self,
        case: str,
        image_file: Optional[str],
        seed_id: int,
        target_ids: Iterable[int],
        graph: nx.Graph,
        centerline_pd=None,            # optional vtkPolyData centerline
        centerline_graph: Optional[nx.Graph] = None,
        cfg: Optional[BuildCfg] = None,
    ):
        self.name = case
        self.image = image_file
        self.G: nx.Graph = graph
        self.seed_id: int = int(seed_id)
        self.target_ids: List[int] = [int(t) for t in target_ids]
        self.centerline_pd = centerline_pd
        self.centerline_graph = centerline_graph
        self.cfg = cfg or BuildCfg()

        # Vessel state (graph-backed)
        self.path: List[int] = [self.seed_id]      # ordered nodes visited
        self.visited_nodes: Set[int] = {self.seed_id}
        self.vessel_edges: Set[Tuple[int, int]] = set()  # undirected edges (u<v)
        self.steps: List[Dict[str, Any]] = []      # SeqSeg-style steps
        self.caps: List[np.ndarray] = []           # computed endpoints later

        # bookkeeping sets
        self.id_traversed: List[int] = [self.seed_id]
        self.id_not_traversed: Set[int] = set(self.G.nodes) - self.visited_nodes

        # Initialize a seed "step 0" so downstream code doesn’t choke.
        seed_pt = _to_np(self.G.nodes[self.seed_id]["point"])
        seed_r = float(self.G.nodes[self.seed_id].get("radius", 0.0))
        seed_step = {
            "old point": seed_pt,
            "point": seed_pt,
            "old radius": seed_r,
            "radius": seed_r,
            "tangent": np.array([1.0, 0.0, 0.0], dtype=float),  # dummy
            "chances": 0,
            "seg_file": None,
            "img_file": None,
            "surf_file": None,
            "cent_file": None,
            "prob_predicted_vessel": None,
            "point_pd": None,
            "surface": None,
            "centerline": None,
            "is_inside": False,
            "time": None,
            "dice": None,
        }
        self.steps.append(seed_step)

        # Make sure edges have a usable weight for path finding.
        self._ensure_edge_costs()

    # ---------- internal helpers ----------
    def _edge_tuple(self, u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u < v else (v, u)

    def _ensure_edge_costs(self) -> None:
        for u, v, data in self.G.edges(data=True):
            # If no geometric length, compute from node coordinates.
            if "length" not in data:
                p0 = _to_np(self.G.nodes[u]["point"])
                p1 = _to_np(self.G.nodes[v]["point"])
                data["length"] = float(np.linalg.norm(p1 - p0))
            if self.cfg.use_prob_cost:
                prob = float(data.get("prob", 1.0))
                data["cost"] = data["length"] / max(prob, self.cfg.eps_prob)
            else:
                data.setdefault("cost", data["length"])

    # ---------- building the vessel ----------
    def add_step(self, curr_id: int, prev_id: Optional[int] = None) -> None:
        """Append one node to the ordered vessel path and record a step."""
        if prev_id is None:
            prev_id = self.path[-1]
        if not self.G.has_edge(prev_id, curr_id):
            raise ValueError(f"No edge in G between {prev_id} and {curr_id}")

        # record path/node/edge
        self.path.append(curr_id)
        self.visited_nodes.add(curr_id)
        self.id_traversed.append(curr_id)
        self.id_not_traversed.discard(curr_id)
        self.vessel_edges.add(self._edge_tuple(prev_id, curr_id))

        # make a SeqSeg-compatible step dict
        self.steps.append(create_step_dict_from_nodes(self.G, prev_id, curr_id))

    def add_path(self, node_seq: List[int]) -> None:
        """Append a whole path (seed .. target), building steps for each hop."""
        if len(node_seq) < 2:
            return
        # glue to end if needed
        if self.path[-1] != node_seq[0]:
            # If they connect, add a single hop to the front node
            if self.G.has_edge(self.path[-1], node_seq[0]):
                self.add_step(node_seq[0], prev_id=self.path[-1])
            else:
                # Otherwise we assume node_seq starts from seed; re-initialize stitching.
                pass
        for u, v in zip(node_seq[:-1], node_seq[1:]):
            self.add_step(v, prev_id=u)

    def build_union_paths(self) -> None:
        """Build the vessel as the union of shortest paths from seed to each target."""
        for t in self.target_ids:
            if t not in self.G:
                continue
            try:
                sp = nx.shortest_path(self.G, self.seed_id, t, weight=self.cfg.weight_attr)
            except nx.NetworkXNoPath:
                continue
            self.add_path(sp)

    # ---------- edits & queries ----------
    def remove_previous_n(self, n: int) -> None:
        """Undo the last n *steps* (not counting the initial seed step)."""
        n = max(0, int(n))
        if n == 0:
            return
        if n >= len(self.steps) - 1:
            # keep seed only
            self.path = [self.seed_id]
            self.visited_nodes = {self.seed_id}
            self.vessel_edges.clear()
            self.steps = [self.steps[0]]
            self.id_traversed = [self.seed_id]
            self.id_not_traversed = set(self.G.nodes) - self.visited_nodes
            return

        # pop steps and update structures
        for _ in range(n):
            # last step corresponds to path[-1] and edge (path[-2], path[-1])
            last = self.path.pop()
            prev = self.path[-1]
            self.steps.pop()
            self.vessel_edges.discard(self._edge_tuple(prev, last))
            # Node may still be present earlier in path; only drop from visited if not used
            if last not in self.path:
                self.visited_nodes.discard(last)
                self.id_not_traversed.add(last)
        # id_traversed is a history list; we keep it as a log

    def get_previous_n(self, n: int) -> List[int]:
        """Return the last n node IDs from the ordered vessel path (excluding seed if needed)."""
        tail = self.path[-n:] if n > 0 else []
        if len(tail) > 1 and tail[0] == self.seed_id:
            tail = tail[1:]
        return tail

    def get_previous_step(self, step_index: int) -> Dict[str, Any]:
        """Return the step dict at a given index (0 is the initial seed step)."""
        return self.steps[step_index]

    # ---------- structural analysis (degree-based) ----------
    def vessel_subgraph(self) -> nx.Graph:
        """Return the visited subgraph."""
        return self.G.subgraph(self.visited_nodes).copy()

    def bifurcation_nodes(self) -> List[int]:
        """Nodes with degree >= 3 in the visited subgraph."""
        H = self.vessel_subgraph()
        return [n for n, d in H.degree() if d >= 3]

    def endpoint_nodes(self) -> List[int]:
        """Nodes with degree == 1 in the visited subgraph (true endpoints)."""
        H = self.vessel_subgraph()
        return [n for n, d in H.degree() if d == 1]

    def get_end_points(self) -> List[np.ndarray]:
        """
        Return 3D points slightly beyond each endpoint, pointing outward along the local tangent.
        """
        ends = []
        H = self.vessel_subgraph()
        for n in self.endpoint_nodes():
            neigh = next(iter(H.neighbors(n)), None)
            if neigh is None:
                continue
            p = _to_np(self.G.nodes[n]["point"])
            q = _to_np(self.G.nodes[neigh]["point"])
            r = float(self.G.nodes[n].get("radius", 0.0))
            t = _unit(p - q)
            ends.append(p + t * r)
        return ends

    # ---------- metrics ----------
    def calc_ave_dice(self) -> Optional[float]:
        total, cnt = 0.0, 0
        for st in self.steps[1:]:
            d = st.get("dice", None)
            if d and d != 0:
                total += float(d)
                cnt += 1
        if cnt > 1:
            ave = total / cnt
            print(f"Average dice per step: {ave}")
            return ave
        return None

    def calc_ave_time(self) -> Optional[float]:
        total, cnt = 0.0, 0
        for st in self.steps[1:]:
            t = st.get("time", None)
            if t:
                total += float(t) if np.isscalar(t) else sum(t)
                cnt += 1
        if cnt > 1:
            ave = total / cnt
            print(f"Average time: {ave}")
            return ave
        return None

    # ---------- visualization / export ----------
    def create_tree_polydata(self, out_path: str) -> None:
        """
        Export the visited subgraph as a VTK PolyData (lines).
        Writes point arrays: Radius, ID, Deg, IsBifurcation, IsEndpoint
        """
        import vtk

        H = self.vessel_subgraph()

        # map graph node id -> vtk point id
        vtk_points = vtk.vtkPoints()
        id_map: Dict[int, int] = {}
        for i, n in enumerate(H.nodes):
            vtk_points.InsertNextPoint(_to_np(self.G.nodes[n]["point"]))
            id_map[n] = i

        # build lines from visited edges only (ensure undirected)
        vtk_lines = vtk.vtkCellArray()
        for u, v in H.edges():
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, id_map[u])
            line.GetPointIds().SetId(1, id_map[v])
            vtk_lines.InsertNextCell(line)

        poly = vtk.vtkPolyData()
        poly.SetPoints(vtk_points)
        poly.SetLines(vtk_lines)

        # point arrays
        npts = len(H.nodes)
        arr_radius = vtk.vtkDoubleArray(); arr_radius.SetName("Radius")
        arr_id     = vtk.vtkIntArray();    arr_id.SetName("ID")
        arr_deg    = vtk.vtkIntArray();    arr_deg.SetName("Degree")
        arr_bif    = vtk.vtkIntArray();    arr_bif.SetName("IsBifurcation")
        arr_end    = vtk.vtkIntArray();    arr_end.SetName("IsEndpoint")

        deg_map = dict(H.degree())
        bif_set = set(self.bifurcation_nodes())
        end_set = set(self.endpoint_nodes())

        for n in H.nodes:
            arr_radius.InsertNextValue(float(self.G.nodes[n].get("radius", 0.0)))
            arr_id.InsertNextValue(int(n))
            arr_deg.InsertNextValue(int(deg_map[n]))
            arr_bif.InsertNextValue(1 if n in bif_set else 0)
            arr_end.InsertNextValue(1 if n in end_set else 0)

        poly.GetPointData().AddArray(arr_radius)
        poly.GetPointData().AddArray(arr_id)
        poly.GetPointData().AddArray(arr_deg)
        poly.GetPointData().AddArray(arr_bif)
        poly.GetPointData().AddArray(arr_end)

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(out_path)
        writer.SetInputData(poly)
        writer.Write()

    def plot_radius_distribution(self, dir_output: str) -> None:
        import matplotlib.pyplot as plt
        radii = [float(self.G.nodes[n].get("radius", 0.0)) for n in self.path]
        n_step = len(radii)
        plt.hist(radii, bins=20)
        plt.savefig(f"{dir_output}/radius_distribution.png")
        plt.close()

        plt.figure(figsize=(25, 5))
        plt.plot(range(n_step), radii)
        plt.xlabel("Step"); plt.ylabel("Radius"); plt.title("Radius Change")
        plt.savefig(f"{dir_output}/radius_evolution.png")
        plt.close()

    # ---------- caps / outlets ----------
    def calc_caps(self, global_assembly) -> List[np.ndarray]:
        """
        Re-uses the old idea: caps = endpoints that are *outside* the global assembly.
        With a graph, endpoints = degree==1 nodes in visited subgraph.
        """
        caps = []
        for p in self.get_end_points():
            if not is_point_in_image(global_assembly, p):
                caps.append(p)
        print(f"Number of outlets: {len(caps)}")
        self.caps = caps
        return caps




def print_error(output_folder,
                i,
                step_seg,
                image=None,
                predicted_vessel=None,
                old_point_ref=None,
                centerline_poly=None):

    now = datetime.now()
    dt_string = now.strftime("_%d_%m_%Y_%H_%M_%S")
    directory = output_folder + 'errors/'+str(i) + '_error_'+dt_string

    polydata_point = points2polydata([step_seg['point'].tolist()])
    write_vtk_polydata(polydata_point, directory + 'point.vtp')

    try:
        if step_seg['img_file'] and not step_seg['is_inside']:
            sitk.WriteImage(image, directory + 'img.vtk')

            if step_seg['seg_file']:
                sitk.WriteImage(predicted_vessel, directory + 'seg.vtk')

                if step_seg['surf_file']:
                    write_vtk_polydata(step_seg['surface'], directory + 'surf.vtp')

                    if step_seg['centerline']:
                        polydata_point = points2polydata(
                            [step_seg['old_point_ref'].tolist()])
                        write_vtk_polydata(polydata_point,
                                        directory + 'old_point_ref.vtp')
                        write_vtk_polydata(centerline_poly,
                                        directory + 'cent.vtp')
    except Exception as e:
        print('Didnt work to save error')
        print(e)

