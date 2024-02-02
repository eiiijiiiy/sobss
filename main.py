import numpy as np
import open3d as o3d
import open3d.visualization as vis
import evaluate
import os
import shutil
import ctypes
import argparse
import json
from sobss import coarse_segmentation as cs

vote_lib = ctypes.cdll.LoadLibrary("build/libvote.dylib")
merge_lib = ctypes.cdll.LoadLibrary("build/libmerge.dylib")


def parse_bss_segm(path):
    bss_segm_geom = [], bss_segm_volume = []
    return bss_segm_geom, bss_segm_volume

def initialize_working_folder(working_folder):
    if os.path.exists(working_folder):
        shutil.rmtree(working_folder)
    os.makedir(working_folder)

    # TODO: set the default parameters
    params = {}

    conf_path = os.path.join(working_folder, "config.json")
    with open(conf_path, "w") as f:
        json.dump(params, f)

def actions(pcd_path, working_folder):
    input_pcd = o3d.io.read_point_cloud(pcd_path)
    INPUT_NAME = "innput pcd"
    ALIGN_NAME = "aligned pcd (relevant points highlighted)"
    BSS_ATOMS_NAME = "BSS atoms"
    BSS_COARSE_SEGM_NAME = "BSS coarse segm"
    BSS_COARSE_VOLUME_NAME = "BSS coarse segm"
    BSS_MERGED_SEGM_NAME = "BSS merged segm"
    BSS_MERGED_VOLUME_NAME = "BSS merged volume"
    BSS_MERGED_TRI_MESH_NAME = "BSS merged mesh (tri)"
    DISTANCE_TO_ALIGN_NAME = "Distance to input (0-1 m)"

    RESULT_NAME = "Result (Poisson reconstruction)"
    TRUTH_NAME = "Ground truth"

    bunny = o3d.data.BunnyMesh()
    bunny_mesh = o3d.io.read_triangle_mesh(bunny.path)
    bunny_mesh.compute_vertex_normals()

    bunny_mesh.paint_uniform_color((1, 0.75, 0))
    bunny_mesh.compute_vertex_normals()
    cloud = o3d.geometry.PointCloud()
    cloud.points = bunny_mesh.vertices
    cloud.normals = bunny_mesh.vertex_normals

    def align(o3dvis):
        pcd_path_c = ctypes.create_string_buffer(
                    pcd_path.encode('utf-8'))
        working_folder_c = ctypes.create_string_buffer(
                    working_folder.encode('utf-8'))
        # read conf from conf file in the working folder???
        some_lib.align(pcd_path_c, working_folder_c)

        align_pcd_path = os.path.join(working_folder, "aligned.ply")
        align_pcd = o3d.io.read_point_cloud(align_pcd_path)

        o3dvis.add_geometry({"name": ALIGN_NAME, "geometry": align_pcd})
        
    def skeletonize(o3dvis):
        pcd_path_c = ctypes.create_string_buffer(
                    pcd_path.encode('utf-8'))
        working_folder_c = ctypes.create_string_buffer(
                    working_folder.encode('utf-8'))
        
        some_lib.skeletonize(pcd_path_c, working_folder_c)

        bss_atoms_path = os.path.join(working_folder, "bss_atoms.txt")
        bss_atoms_pts = np.loadtxt(bss_atoms_path)[:,:3]
        bss_atoms = o3d.geometry.PointCloud()
        bss_atoms.points = o3d.utility.Vector3dVector(bss_atoms_pts)

        o3dvis.add_geometry({"name": BSS_ATOMS_NAME, "geometry": bss_atoms})
    
    def coarse_segment(o3dvis):
        bss_atoms_path = os.path.join(working_folder, "bss_atoms.txt")
        
        cs.coarse_segment(bss_atoms_path, working_folder)

        bss_coarse_segm_path = os.path.join(working_folder, "bss_coarse_segm.txt")
        bss_coarse_segm, bss_coarse_volume = parse_bss_segm(bss_coarse_segm_path)

        o3dvis.add_geometry({"name": BSS_COARSE_SEGM_NAME, "geometry": bss_coarse_segm})
        o3dvis.add_geometry({"name": BSS_COARSE_VOLUME_NAME, "geometry": bss_coarse_volume})
    
    def merge_segments(o3dvis):
        pcd_path_c = ctypes.create_string_buffer(
                    pcd_path.encode('utf-8'))
        working_folder_c = ctypes.create_string_buffer(
                    working_folder.encode('utf-8'))
        
        some_lib.merge_segments(pcd_path_c, working_folder_c)

        bss_merged_segm_path = os.path.join(working_folder, "bss_merged_segm.txt")
        bss_merged_segm, bss_merged_volume = parse_bss_segm(bss_merged_segm_path)

        o3dvis.add_geometry({"name": BSS_MERGED_SEGM_NAME, "geometry": bss_merged_segm})
        o3dvis.add_geometry({"name": BSS_MERGED_VOLUME_NAME, "geometry": bss_merged_volume})

        bss_merged_trimesh_path = os.path.join(working_folder, "bss_merged_segm.obj")
        bss_merged_trimesh = o3d.io.read_triangle_mesh(bss_merged_trimesh_path)

        o3dvis.add_geometry({"name": BSS_MERGED_TRI_MESH_NAME, "geometry": bss_merged_trimesh})
    
    def evaluate_merged_segments(o3dvis):
        bss_merged_trimesh_path = os.path.join(working_folder, "bss_merged_segm.obj")
        align_pcd_path = os.path.join(working_folder, "aligned.ply")

        distance_to_align = evaluate.distance_from_pcd_to_trimesh(align_pcd_path, bss_merged_trimesh_path, truncation=1.0)

        o3dvis.add_geometry({"name": DISTANCE_TO_ALIGN_NAME, "geometry": distance_to_align})


    vis.draw([{
        "name": INPUT_NAME,
        "geometry": input_pcd}
        ],
        actions=[("Align", align),
                ("Skeletonize (collect bss atoms)", skeletonize),
                ("Coarse segment", coarse_segment),
                ("Merge segments", merge_segments),
                ("Evaluate merged segments", evaluate_merged_segments)
                ])

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, required=True, help="input point cloud with normals (.ply)")
parser.add_argument('-w', type=str, required=True, help="working folder")
args = parser.parse_args()

initialize_working_folder(args.w)

actions(args.i, args.w)




