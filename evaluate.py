import numpy as np
import open3d as o3d

def distance_from_pcd_to_trimesh(pcd, mesh, truncation = 1.0):
    pcd = o3d.io.read_point_cloud(pcd)
    dist_pcd = pcd

    mesh = o3d.io.read_triangle_mesh(mesh)
    # Sample the mesh to get the points
    # Compute distance
    # Colorize dist_pcd accordingly

    return dist_pcd
