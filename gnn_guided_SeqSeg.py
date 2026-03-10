from typing import Optional, Tuple, List
import numpy as np
import networkx as nx
from pathlib import Path
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from SeqSeg.seqseg.modules.centerline import post_process_centerline
from SeqSeg.seqseg.modules.vtk_functions import write_vtk_polydata,smooth_polydata,smooth_surface
from scipy.spatial import cKDTree as KDTree
import vtk
from SeqSeg.seqseg.modules.sitk_functions import (
    import_image,
    extract_volume,
    copy_settings,
    remove_other_vessels,
    check_seg_border,
)
import SimpleITK as sitk
from gnn_model.gnn_modules import gm_load, gm_predict_graph, save_predicted_graph_to_vtp,process_case_infer
from seqseg_modules_modified.trace_centerline import trace_centerline
from scipy.spatial import cKDTree as KDTree
import faulthandler, time, os, yaml, vtk
import numpy as np
import SimpleITK as sitk
import vtk
from skimage.morphology import skeletonize
from SeqSeg.seqseg.modules.vtk_functions import write_vtk_polydata
import numpy as np
import SimpleITK as sitk
from skimage import morphology, filters, measure
import vtk
import numpy as np
from scipy.spatial import cKDTree as KDTree
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import argparse
from nnUNet.nnunetv2.paths import nnUNet_results
import numpy as np
import SimpleITK as sitk
import vtk
import networkx as nx
from skimage.morphology import skeletonize
from SeqSeg.seqseg.modules.vtk_functions import write_vtk_polydata



#Helpers

def create_directories(output_folder: str, write_samples: bool) -> None:
    base = Path(output_folder)
    for sub in ("", "errors", "assembly"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    if write_samples:
        for sub in ("volumes", "predictions", "centerlines", "surfaces",
                    "points", "animation", "images", "labels", "trace_artifacts"):
            (base / sub).mkdir(parents=True, exist_ok=True)

def verify_coord_order(G: nx.Graph, img: sitk.Image, pos_key='pos', order=(2, 1, 0), sample=512) -> float:
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

def save_skeletonized(
    mask_img: sitk.Image,
    out_vtp_path: str,
    bin_thr: float = 0.5,
):
    arr = sitk.GetArrayFromImage(mask_img).astype(np.float32)
    mask = arr >= float(bin_thr)
    if not mask.any():
        pd_empty = vtk.vtkPolyData()
        write_vtk_polydata(pd_empty, out_vtp_path)
        return pd_empty
    skel = skeletonize(mask)
    coords_zyx = np.argwhere(skel)
    if coords_zyx.shape[0] == 0:
        pd_empty = vtk.vtkPolyData()
        write_vtk_polydata(pd_empty, out_vtp_path)
        return pd_empty
    spacing = np.asarray(mask_img.GetSpacing(), float)
    origin  = np.asarray(mask_img.GetOrigin(), float)
    direction = np.asarray(mask_img.GetDirection(), float).reshape(3, 3)
    def vox_to_phys(z, y, x):
        idx_xyz = np.array([x, y, z], float) * spacing
        return (direction @ idx_xyz) + origin
    pts = vtk.vtkPoints()
    verts = vtk.vtkCellArray()

    for z, y, x in coords_zyx:
        p = vox_to_phys(z, y, x)
        pid = pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
        verts.InsertNextCell(1)
        verts.InsertCellPoint(pid)

    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetVerts(verts)

    write_vtk_polydata(pd, out_vtp_path)
    return pd

def largest_connected_component(img: sitk.Image, background_value=0) -> sitk.Image:
    relabeled = sitk.RelabelComponent(
        sitk.ConnectedComponent(img != background_value),
        sortByObjectSize=True
    )
    return sitk.Cast(relabeled == 1, img.GetPixelID())


def blank_image(ref: sitk.Image, pixel_id=sitk.sitkFloat32) -> sitk.Image:
    out = sitk.Image(ref.GetSize(), pixel_id)
    out.CopyInformation(ref)
    return out

#assembly functions -------------------------------------------------------------
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


def merge_centerlines_vtk(merged_centerlines,
                               out_vtp_path: str,
                               spacing_mm,
                               spline_len_mm: float = 0.5,
                               smooth_iter: int = 15,
                               pass_band: float = 0.1):

    if not merged_centerlines:
        print("no polylines to merge.")
        return None

    app = vtk.vtkAppendPolyData()
    for pd in merged_centerlines:
        if isinstance(pd, vtk.vtkPolyData):
            app.AddInputData(pd)
    app.Update()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(app.GetOutputPort())
    clean.ToleranceIsAbsoluteOn()
    clean.SetAbsoluteTolerance(0.25 * float(min(spacing_mm)))
    clean.PointMergingOn()
    clean.Update()

    spline = vtk.vtkSplineFilter()
    spline.SetInputConnection(clean.GetOutputPort())
    spline.SetSubdivideToLength()
    spline.SetLength(float(spline_len_mm))
    spline.SetGenerateTCoordsToOff()
    spline.Update()

    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(spline.GetOutputPort())
    smooth.SetNumberOfIterations(int(smooth_iter))
    smooth.SetPassBand(float(pass_band))
    smooth.NonManifoldSmoothingOn()
    smooth.NormalizeCoordinatesOn()
    smooth.BoundarySmoothingOff()
    smooth.Update()

    out_pd = smooth.GetOutput()
    write_vtk_polydata(out_pd, out_vtp_path)
    print(f"merged centerline: {out_vtp_path}")
    return out_pd

def save_similar_to_seg(sitk_img: sitk.Image, out_path: str,                #save seg in similar format
                                bin_thr: float = 0.5):
    arr = sitk.GetArrayFromImage(sitk_img)
    arr = arr.astype(np.float32)
    mask = arr >= float(bin_thr)
    out_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
    out_sitk.CopyInformation(sitk_img)
    sitk.WriteImage(out_sitk, out_path)
    print(f"[WRITE] assembled mask (skimage processed): {out_path}")
    return out_sitk

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

    model = gm_load(
        ckpt_path=os.path.join(args.gnn_folder, "gnn_best.pt"),
        cfg=params_gnn
    )

    merged_points = []
    merged_inside_points = []
    merged_surfaces = []

    seqseg_model_folder = os.path.join(
        nnUNet_results, f"{args.dataset_id}/nnUNetTrainer__nnUNetPlans__3d_fullres"
    )

    max_steps_per_component = int(params_seqseg.get("MAX_STEPS_PER_COMPONENT", 1000))
    ASSEMBLY_THRESH = float(params_seqseg.get("ASSEMBLY_THRESHOLD", 0.5))

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

        start_prob_global = None
        coverage_union = None
        merged_centerlines = []
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Centerline extraction")
        G0,G1,seed_list,targets_list = process_case_infer(
            img_p= dir_image,pred_p= dir_seg,model = model, cfg = params_gnn, device = device, out_dir = None)

        print("  Graph0:", G0.number_of_nodes(), "nodes,", G0.number_of_edges(), "edges")
        print("  Graph1:", G1.number_of_nodes(), "nodes,", G1.number_of_edges(), "edges")
        try:
            save_predicted_graph_to_vtp(G0, out_path=os.path.join(dir_output_case, f"centerlines/centerline_{case_id}_comp00.vtp"))
            save_predicted_graph_to_vtp(G1, out_path=os.path.join(dir_output_case, f"centerlines/centerline_{case_id}_comp01.vtp"))
        except Exception:
            pass
        if G0.number_of_nodes() == 0:
            print("  - Empty graph; skipping.")
            continue
        if G1.number_of_nodes() == 0:
            print("  - Empty graph; skipping.")
            continue

        comps = []
        comps.append(G0)
        comps.append(G1)


        for gi, Gc in enumerate(comps):

            comp_tag = f"{case_id}_comp{gi:02d}"
            img_out = os.path.join(dir_output_case, "images", f"{comp_tag}{img_format}")
            lab_out = os.path.join(dir_output_case, "labels", f"{comp_tag}{img_format}")
            os.makedirs(os.path.dirname(img_out), exist_ok=True)
            os.makedirs(os.path.dirname(lab_out), exist_ok=True)
            seed_node, target_nodes = seed_list[gi], targets_list[gi]
            print('seed location', Gc.nodes[int(seed_node)].get('pos_phys'))
            if seed_node is None or not target_nodes:
                print("  - no valid seed/targets for this component; skipping")
                continue
            Gcent = Gc
            prev_prob_for_tracer = start_prob_global if start_prob_global is not None else None
            seed_id = int(seed_node)
            target_ids = list(map(int, target_nodes))

            _lc, _ls, _lp, _li, assembly_segs, vt, i = trace_centerline(
                output_folder=dir_output_case,
                image_file=dir_image,
                case=case_id,
                model_folder=seqseg_model_folder,
                fold=args.fold,
                centerline_graph=Gcent,
                seed_node=seed_id,
                target_nodes=target_ids,
                max_steps_per_component=max_steps_per_component,
                global_config=params_seqseg,
                unit='cm',
                scale=1,
                seg_file=None,
                start_seg=prev_prob_for_tracer,
                write_samples = True,
            )

            prev_prob = start_prob_global if start_prob_global is not None else blank_image(image_ref, sitk.sitkFloat32)
            curr_prob = sitk.Cast(assembly_segs.assembly, sitk.sitkFloat32)
            curr_prob.CopyInformation(prev_prob)

            new_bin = sitk.Greater(curr_prob, ASSEMBLY_THRESH)
            old_bin = sitk.Greater(prev_prob, ASSEMBLY_THRESH)
            delta_bin = sitk.And(new_bin, sitk.Not(old_bin))

            #largest_delta = remove_other_vessels(delta_bin, see)
            largest_delta = largest_connected_component(sitk.Cast(delta_bin, sitk.sitkUInt8), background_value=0)
            stats = sitk.StatisticsImageFilter()
            stats.Execute(largest_delta)
            if stats.GetSum() == 0:
                print("[assembly] no region above threshold, skipping")
                continue
            inc_prob = sitk.Mask(curr_prob, largest_delta)
            start_prob_global = sitk.Maximum(prev_prob, inc_prob)
            if coverage_union is None:
                coverage_union = sitk.Cast(largest_delta, sitk.sitkUInt8)
                coverage_union.CopyInformation(prev_prob)
            else:
                coverage_union = sitk.Or(coverage_union, sitk.Cast(largest_delta, sitk.sitkUInt8))
            trace_dir = os.path.join(dir_output_case, "trace_artifacts")
            os.makedirs(trace_dir, exist_ok=True)
            save_base = f"{case_id}_seed{seed_id}"

            if _lc:
                for k, poly in enumerate(_lc):
                    if poly is None or len(poly) == 0:
                        continue
                    pd = np_polyline_to_vtk(np.asarray(poly))
                    pd = smooth_polydata(pd)
                    out_path = os.path.join(os.path.join(dir_output_case, "centerlines"), f"{save_base}_centerline_{k:02d}.vtp")
                    write_vtk_polydata(pd, out_path)
                    print(f"[WRITE] centerline: {out_path}")
                    merged_centerlines.append(pd)

            if _lp and len(_lp) > 0:
                try:
                    P = np.vstack([np.asarray(p, float).reshape(3) for p in _lp])
                    pd_pts = np_points_to_vtk(P)
                    out_path = os.path.join(os.path.join(dir_output_case, "points"), f"{save_base}_points.vtp")
                    write_vtk_polydata(pd_pts, out_path)
                    print(f"[WRITE] points: {out_path}")
                    merged_points.append(pd_pts)
                except Exception as e:
                    print(f"[WARN] saving points failed: {e}")

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
                        out_path = os.path.join(os.path.join(dir_output_case, "surfaces"), f"{save_base}_surface_{sidx:02d}.vtp")
                        write_vtk_polydata(pd_mesh, out_path)
                        print(f"[WRITE] surface: {out_path}")
                        merged_surfaces.append(pd_mesh)
                    except Exception as e:
                        print(f"[WARN] saving surface {sidx} failed: {e}")

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
        else:
            print("[WARN] No assembled volume produced for this case.")

        sp = tuple(map(float, image_ref.GetSpacing()))
        assembled_mask_post_path = f"seqseg_results/test/{case_id}.nii.gz"
        assembled_mask_post = save_similar_to_seg(
            assembled_mask,
            assembled_mask_post_path,
            bin_thr=0.5,
        )
        if len(merged_centerlines) > 0:
            merged_centerline_vtp = os.path.join(
                assembly_dir, f"{case_id}_centerlines_merged.vtp"
            )
            merge_centerlines_vtk(
                merged_centerlines,
                merged_centerline_vtp,
                spacing_mm=sp,
                spline_len_mm=0.5,
                smooth_iter=10,
                pass_band=0.1,
            )
        else:
            print("[INFO] no centerlines to merge for this case.")

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
            merged_out = os.path.join(assembly_dir, f"{case_id}_inside_points_merged.vtp")
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

            passBand = 0.01
            smoother = vtk.vtkWindowedSincPolyDataFilter()
            smoother.SetInputData(cleaned)
            smoother.SetNumberOfIterations(5)
            smoother.BoundarySmoothingOff()
            smoother.FeatureEdgeSmoothingOff()
            smoother.SetPassBand(passBand)
            smoother.NonManifoldSmoothingOn()
            smoother.NormalizeCoordinatesOn()
            smoother.Update()

            merged_out = os.path.join(assembly_dir, f"{case_id}_surfaces_merged.vtp")
            write_vtk_polydata(smoother.GetOutput(), merged_out)
            print(f"[WRITE] merged surfaces: {merged_out}")
        else:
            print("[INFO] no surfaces to merge for this case.")


        print(f"\nCase time: {((time.time() - case_t0) / 60):.2f} min\n")
    print(f"Total execution time: {((time.time() - t0) / 60):.2f} min")


if __name__ == '__main__':
    main()
