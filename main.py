import numpy as np
import open3d as o3d
import open3d.visualization as vis
import evaluate as evaluate
import os
import shutil
import ctypes
import argparse
import json
import coarse_segmentation as cs
import platform

if platform.system() == "Darwin":
    sobss_lib = ctypes.cdll.LoadLibrary("build/libsobss.dylib")
elif platform.system() == "Linux":
    sobss_lib = ctypes.cdll.LoadLibrary("build/libsobss.so")
elif platform.system() == "WINdows":
    sobss_lib = ctypes.cdll.LoadLibrary("build/sobss.dll")
else:
    raise Exception("Unsupported platform: {}".format(platform.system()))

def parse_bss_segm(path):
    # 0 x, 1 y, 2 z, 3 width, 4 height (of front/back face), 
    # 5 radius, 6 length of the horizontal normals (default normals point to the (0,1,0))
    segms = np.loadtxt(path)
    num = segms.shape[0]
    NV = np.sqrt(1 - segms[:, -1] ** 2)
    Z_TOP = segms[:, 2] + segms[:, 5] * \
	    NV + segms[:, 4] * segms[:, -1] / 2
    Z_BTM = segms[:, 2] + segms[:, 5] * \
	    NV - segms[:, 4] * segms[:, -1] / 2

    Y_FRONT_TOP = segms[:, 1] + segms[:, 5] * \
	    segms[:, -1] - segms[:, 4] / 2 * NV
    Y_FRONT_BTM = segms[:, 1] + segms[:, 5] * \
	    segms[:, -1] + segms[:, 4] / 2 * NV
    Y_BACK_TOP = segms[:, 1] - segms[:, 5] * \
	    segms[:, -1] + segms[:, 4] / 2 * NV
    Y_BACK_BTM = segms[:, 1] - segms[:, 5] * \
	    segms[:, -1] - segms[:, 4] / 2 * NV

    X_RIGHT = segms[:, 0] + segms[:, 3] / 2
    X_LEFT = segms[:, 0] - segms[:, 3] / 2

    # segm
    segm_coords = np.zeros((num, 4, 3))
    segm_coords[:, 0, :] = np.vstack(
	    [X_LEFT, segms[:, 1], Z_TOP]).transpose()  # LT 0
    segm_coords[:, 1, :] = np.vstack(
	    [X_RIGHT, segms[:, 1], Z_TOP]).transpose()  # RT 1
    segm_coords[:, 2, :] = np.vstack(
	    [X_LEFT, segms[:, 1], Z_BTM]).transpose()  # LB 2
    segm_coords[:, 3, :] = np.vstack(
	    [X_RIGHT, segms[:, 1], Z_BTM]).transpose()  # RB 3
    
    segm_edge_conn = np.array((
        [0, 1], # LT-RT
        [1, 3], # RT-RB
        [3, 2], # RB-LB
        [2, 0]  # LB-LT
        )).reshape((1, 4, 2))

    segm_edges = np.repeat(segm_edge_conn, num, axis=0) # shape: num, 4, 2
    shift = (np.arange(num) * 4).reshape((num, 1))
    shift = np.repeat(shift, 4, axis=1).reshape((num, 4, 1))
    shift = np.repeat(shift, 2, axis=2)
    segm_edges += shift

    bss_segm_wf = o3d.geometry.LineSet()
    bss_segm_wf.points = o3d.utility.Vector3dVector(segm_coords.reshape(-1, 3))
    bss_segm_wf.lines = o3d.utility.Vector2iVector(segm_edges.reshape(-1, 2))
    bss_segm_wf.colors = o3d.utility.Vector3dVector(np.array([[1, 0.253, 1]] * num * 4))
    
    #  vol
    vol_coords = np.zeros((num, 8, 3))

    vol_coords[:, 0, :] = np.vstack(
	    [X_LEFT, Y_FRONT_TOP, Z_TOP]).transpose()  # FLT 0
    vol_coords[:, 1, :] = np.vstack(
	    [X_RIGHT, Y_FRONT_TOP, Z_TOP]).transpose()  # FRT 1
    vol_coords[:, 2, :] = np.vstack(
	    [X_LEFT, Y_FRONT_BTM, Z_BTM]).transpose()  # FLB 2
    vol_coords[:, 3, :] = np.vstack(
	    [X_RIGHT, Y_FRONT_BTM, Z_BTM]).transpose()  # FRB 3

    vol_coords[:, 4, :] = np.vstack(
	    [X_LEFT, Y_BACK_TOP, Z_TOP]).transpose()  # BLT 4
    vol_coords[:, 5, :] = np.vstack(
	    [X_RIGHT, Y_BACK_TOP, Z_TOP]).transpose()  # BRT 5
    vol_coords[:, 6, :] = np.vstack(
	    [X_LEFT, Y_BACK_BTM, Z_BTM]).transpose()  # BLB 6
    vol_coords[:, 7, :] = np.vstack(
	    [X_RIGHT, Y_BACK_BTM, Z_BTM]).transpose()  # BRB 7
    
    vol_edge_conn = np.array((
        [0, 1], # FLT-FRT
        [1, 5], # FRT-BRT
        [5, 4], # BRT-BLT
        [4, 0], # BLT-FLT
        [2, 3], # FLB-FRB 
        [3, 7], # FRB-BRB
        [7, 6], # BRB-BLB
        [6, 2], # BLB-FLB
        [0, 2], # FLT-FLB
        [1, 3], # FRT-FRB
        [4, 6], # BLT-BLB
        [5, 7]  # BRT-BRB
        )).reshape((1, 12, 2))

    vol_edges = np.repeat(vol_edge_conn, num, axis=0) # shape: num, 12, 2
    shift = (np.arange(num) * 8).reshape((num, 1))
    shift = np.repeat(shift, 12, axis=1).reshape((num, 12, 1))
    shift = np.repeat(shift, 2, axis=2)
    vol_edges += shift

    bss_segm_volume_wf = o3d.geometry.LineSet()
    bss_segm_volume_wf.points = o3d.utility.Vector3dVector(vol_coords.reshape(-1, 3))
    bss_segm_volume_wf.lines = o3d.utility.Vector2iVector(vol_edges.reshape(-1, 2))

    return bss_segm_wf, bss_segm_volume_wf


def initialize_working_folder(
        working_folder, vs = 0.25, p = 2.0, nmi = np.pi/9, sigma = 2.0, lmbd = 2.0):
    if os.path.exists(working_folder):
        shutil.rmtree(working_folder)
    os.makedirs(working_folder)

    params = {
        'voxel_size': vs, 
        'ps': p,
        'dxi': p / 2,
        'xi': p,
        'zci': p,
        'nmi': nmi,
        'sigma': sigma,
        'lambda': lmbd}

    conf_path = os.path.join(working_folder, "config.json")
    with open(conf_path, "w") as f:
        json.dump(params, f)


def actions(pcd_path, working_folder):
    input_pcd = o3d.io.read_point_cloud(pcd_path)
    INPUT_NAME = "input pcd"
    ALIGN_NAME = "aligned pcd"
    NH_NAME = "aligned non-horizontal pcd"
    BSS_ATOM_NAME = "BSS atoms"
    BSS_COARSE_SEGM_NAME = "BSS coarse segm"
    BSS_COARSE_VOLUME_NAME = "BSS coarse volume"
    BSS_MERGED_SEGM_NAME = "BSS merged segm"
    BSS_MERGED_VOLUME_NAME = "BSS merged volume"
    BSS_MERGED_TRI_MESH_NAME = "BSS merged mesh (tri)"
    DISTANCE_TO_ALIGN_NAME = "Distance to input (0-1 m)"

        
    def skeletonize(o3dvis):
        pcd_path_c = ctypes.create_string_buffer(
                    pcd_path.encode('utf-8'))
        working_folder_c = ctypes.create_string_buffer(
                    working_folder.encode('utf-8'))
        
        sobss_lib.skeletonize(pcd_path_c, working_folder_c)

        align_pcd_path = os.path.join(working_folder, "aligned.ply")
        align_pcd = o3d.io.read_point_cloud(align_pcd_path)
        o3dvis.add_geometry({"name": ALIGN_NAME, "geometry": align_pcd})

        aa_nh_pcd_path = os.path.join(working_folder, "non_horizontal.ply")
        aa_nh_pcd = o3d.io.read_point_cloud(aa_nh_pcd_path)
        o3dvis.add_geometry({"name": NH_NAME, "geometry": aa_nh_pcd})

        bss_atom_path = os.path.join(working_folder, "bss_atom.txt")
        bss_atom_pts = np.loadtxt(bss_atom_path)[:,:3]
        bss_atom = o3d.geometry.PointCloud()
        bss_atom.points = o3d.utility.Vector3dVector(bss_atom_pts)
        bss_atom.colors = o3d.utility.Vector3dVector(np.array([[1, 0.253, 1]] * bss_atom_pts.shape[0]))

        o3dvis.add_geometry({"name": BSS_ATOM_NAME, "geometry": bss_atom})
        o3dvis.show_geometry(INPUT_NAME, False)
        o3dvis.setup_camera(45, np.array([0,0,0]), np.array([0, 0, 30]), np.array([0,0,1]))
        o3dvis.post_redraw()
        # view_ctl = o3dvis.get_view_control()
        # view_ctl.set_lookat([0, 0, 0])
        # view_ctl.set_front([0, 0, 1])
    
    def coarse_segment(o3dvis):        
        cs.coarse_segment(working_folder)

        bss_coarse_segm_path = os.path.join(working_folder, "bss_coarse_segm.txt")
        bss_coarse_segm, bss_coarse_volume = parse_bss_segm(bss_coarse_segm_path)
        o3dvis.show_geometry(ALIGN_NAME, False)
        o3dvis.show_geometry(NH_NAME, False)
        o3dvis.add_geometry({"name": BSS_COARSE_SEGM_NAME, "geometry": bss_coarse_segm})
        o3dvis.add_geometry({"name": BSS_COARSE_VOLUME_NAME, "geometry": bss_coarse_volume})
    
    def merge_segments(o3dvis):
        working_folder_c = ctypes.create_string_buffer(
                    working_folder.encode('utf-8'))
        
        sobss_lib.merge(working_folder_c)

        bss_merged_segm_path = os.path.join(working_folder, "bss_merged_segm.txt")
        bss_merged_segm, bss_merged_volume = parse_bss_segm(bss_merged_segm_path)

        o3dvis.show_geometry(BSS_COARSE_SEGM_NAME, False)
        o3dvis.show_geometry(BSS_COARSE_VOLUME_NAME, False)
        o3dvis.add_geometry({"name": BSS_MERGED_SEGM_NAME, "geometry": bss_merged_segm})
        o3dvis.add_geometry({"name": BSS_MERGED_VOLUME_NAME, "geometry": bss_merged_volume})

        bss_merged_trimesh_path = os.path.join(working_folder, "bss_merged_tri.obj")
        bss_merged_trimesh = o3d.io.read_triangle_mesh(bss_merged_trimesh_path)

        o3dvis.add_geometry({"name": BSS_MERGED_TRI_MESH_NAME, "geometry": bss_merged_trimesh})
    
    def evaluate_merged_segments(o3dvis):        
        if evaluate.distance_from_pcd_to_trimesh(working_folder, truncation=1.0):
            evaluate_pcd_path = os.path.join(working_folder, "evaluated.ply")
            evaluated_pcd = o3d.io.read_point_cloud(evaluate_pcd_path)
            o3dvis.show_geometry(BSS_MERGED_TRI_MESH_NAME, False)
            o3dvis.add_geometry({"name": DISTANCE_TO_ALIGN_NAME, "geometry": evaluated_pcd})
        else:
            print("failed to evaluate")

    vis.draw([{
        "name": INPUT_NAME,
        "geometry": input_pcd}
        ],
        actions=[("skeletonize", skeletonize),
                ("coarse segment", coarse_segment),
                ("merge", merge_segments),
                ("evaluate", evaluate_merged_segments)
                ])

# main
parser = argparse.ArgumentParser()
# set input and working folder
parser.add_argument('-i', type=str, required=True, help="input point cloud with normals (.ply)")
parser.add_argument('-w', type=str, required=True, help="working folder")

# set parameters
parser.add_argument('-v', type=float, default=0.25, help="voxel size in skeletonization")
parser.add_argument('-p', type=float, default=2.0, help="distance interval to group BSS atoms in coarse segmentation")
parser.add_argument('-n', type=float, default=np.pi/9, help="angular interval to group BSS atoms in coarse segmentation")
parser.add_argument('-s', type=float, default=2.0, help="truncate distance in merging")
parser.add_argument('-l', type=float, default=2.0, help="weighting factor in merging")

args = parser.parse_args()

initialize_working_folder(
    args.w, vs = args.v, p = args.p, nmi = args.n, sigma = args.s, lmbd = args.l)

actions(args.i, args.w)




