import numpy as np
import open3d as o3d
import geopandas
from tqdm import tqdm
from shapely.geometry import Point, Polygon
import os

cmap = plt.cm.get_cmap('jet')
colors = cmap(np.arange(cmap.N)) # (256, 4) including the alpha channel

def remove_btm(mesh, sample_pts = 10000, thresh_pts = 10):
    
    max_bound = mesh.get_max_bound()
    min_bound = mesh.get_min_bound()
    H = max_bound[2] - min_bound[2]
    thresh_H = min_bound[2] + H * 0.1
    pcd = mesh.sample_points_uniformly(sample_pts)     
    pts = np.asarray(pcd.points)
    # find horizontal faces
    mesh.compute_triangle_normals()
    tn = np.asarray(mesh.triangle_normals)
    
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    tri_heights = vertices[triangles, 2] # (#tri,3)
    tri_minz = np.min(tri_heights, axis=1)
    hid = np.where((np.abs(tn[:,2])>0.5) * (tri_minz < thresh_H))[0]
    # create a bbox of the faces
    to_remove = []
    for h in tqdm(hid):
        tri = triangles[h]
        min_z = np.min(vertices[tri, 2])
        below_pts = pts[pts[:, 2] < min_z-0.1, :2]
        below_pt_set = [Point(p[0], p[1]) for p in below_pts]
        pt_series = geopandas.GeoSeries(below_pt_set)
        tri_plg = Polygon(vertices[tri, :2])
        within = pt_series.within(tri_plg)
        count = len(np.where(within)[0])
        if count < thresh_pts:
            to_remove.append(h)
        
    to_remove = np.asarray(to_remove)
    print('# to remove face: {}'.format(len(to_remove)))
    mesh.remove_triangles_by_index(to_remove)
    return mesh

def distance_from_pcd_to_trimesh(working_folder, truncation = 1.0):
    pcd_path = os.path.join(working_folder, "aligned.ply")
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd = pcd.voxel_down_sample(voxel_size=0.5)
    pt_num = np.asarray(pcd.points).shape[0]

    mesh_path = os.path.join(working_folder, "bss_merged_tri.obj")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    tri_num = np.asarray(mesh.triangles).shape[0]
    if tri_num == 0:
        return False
    mesh = remove_btm(mesh)
    tri_num = np.asarray(mesh.triangles).shape[0]
    if tri_num == 0:
        return False

    tri_pcd = mesh.sample_points_uniformly(pt_num)
    dist = pcd.compute_point_cloud_distance(tri_pcd)
    dist = np.asarray(dist)

    dist = dist / truncation
    dist = np.clip(dist, 0, 1)
    dist = np.floor(dist * 255).astype(np.uint8)
    dist_colors = colors[dist, :-1]
    pcd.colors = o3d.utility.Vector3dVector(dist_colors)
    evaluate_path = os.path.join(working_folder, "evaluated.ply")
    o3d.io.write_point_cloud(evaluate_path, pcd)
    return True