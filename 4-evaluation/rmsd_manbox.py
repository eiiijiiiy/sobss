import pymeshlab as pml
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import geopandas
from tqdm import tqdm
import openmesh as om



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


cmap = plt.cm.get_cmap('jet')
colors = cmap(np.arange(cmap.N)) # (256, 4) including the alpha channel

names = [str(i+1) for i in range(10)]

# for DEBUG
# param_names = ['lb']
# param_values = ['1']
# names = ['1']
# sampled_num = np.array([2]) * 1000000

truncated_dist = [0.5, 1., 2.]

this_type = '3-select_primary_normal_manhattan_revision'
pcd_dir = '/media/y/18T/parallelism_2022/revision_exper/{}/0-input'.format(this_type)
poly_dir = '/media/y/18T/parallelism_2022/revision_exper/{}/comparison/ManhattanWorld/poly'.format(this_type)
tri_dir = '/media/y/18T/parallelism_2022/revision_exper/{}/comparison/ManhattanWorld/tri'.format(this_type)
tri_rb_dir = '/media/y/18T/parallelism_2022/revision_exper/{}/comparison/ManhattanWorld/tri_rb'.format(this_type)
r2s_dir = '/media/y/18T/parallelism_2022/revision_exper/{}/comparison/ManhattanWorld/r2s'.format(this_type)
s2r_dir = '/media/y/18T/parallelism_2022/revision_exper/{}/comparison/ManhattanWorld/s2r'.format(this_type)
rmsd_dir = '/media/y/18T/parallelism_2022/revision_exper/{}/comparison/ManhattanWorld/rmsd'.format(this_type)

smoothness = [12.5, 100, 800]
sample_num = 10
names = [str(i+1) for i in range(sample_num)]
for s in smoothness:

    print('processing smoothness {}'.format(s))
    param_rmsd_path = '{}/{}.txt'.format(rmsd_dir, s)
    p_rmsd = np.zeros((sample_num, 5))
    for ni in range(sample_num):
        print("sample id {}".format(ni))
        n = names[ni]
        poly_path = '{}/{}(5000, {}).obj'.format(poly_dir, n, s)
        tri_path = '{}/{}(5000, {}).obj'.format(tri_dir, n, s)
        tri_rb_path = '{}/{}(5000, {}).obj'.format(tri_rb_dir, n, s)

        mesh = om.read_polymesh(poly_path)
        if mesh.n_faces() == 0: continue

        ms = pml.MeshSet()
        ms.load_new_mesh(poly_path)
        ms.meshing_poly_to_tri()
        ms.save_current_mesh(tri_path)

        mesh = o3d.io.read_triangle_mesh(tri_path)
        tri_num = np.asarray(mesh.triangles).shape[0]
        if tri_num == 0:
            p_rmsd[ni, :] = 100 
            continue
        
        mesh = remove_btm(mesh)
        o3d.io.write_triangle_mesh(tri_rb_path, mesh)
        tri_num = np.asarray(mesh.triangles).shape[0]
        if tri_num == 0:
            p_rmsd[ni, :] = 100 
            continue
        
        i_pcd = o3d.io.read_point_cloud('{}/{}.ply'.format(pcd_dir, n))
        i_pcd = i_pcd.voxel_down_sample(voxel_size=0.5)
        pt_num = np.asarray(i_pcd.points).shape[0]
        try:
            pcd = mesh.sample_points_uniformly(pt_num)
        except:
            from IPython import embed
            embed()
        
        r2s_dists = i_pcd.compute_point_cloud_distance(pcd)
        r2s_dists = np.asarray(r2s_dists)
        s2r_dists = pcd.compute_point_cloud_distance(i_pcd)
        s2r_dists = np.asarray(s2r_dists)
        rmsd_r2s = np.sqrt(np.sum(r2s_dists * r2s_dists)/r2s_dists.shape[0])
        rmsd_s2r = np.sqrt(np.sum(s2r_dists * s2r_dists)/s2r_dists.shape[0])
        p_rmsd[ni, 0] = rmsd_r2s 
        p_rmsd[ni, 1] = rmsd_s2r # s2r
        p_rmsd[ni, 2] = np.max(r2s_dists)
        p_rmsd[ni, 3] = np.max(s2r_dists)
        p_rmsd[ni, 4] = max(p_rmsd[ni, 2], p_rmsd[ni, 3])

        # colorize r2s pcd
        for td in truncated_dist:
            r2s_dists_ = r2s_dists / td
            r2s_dists_ = np.clip(r2s_dists_, 0, 1)
            r2s_dists_ = np.floor(r2s_dists_*255).astype(np.uint8) # 0,1,2,...,255
            r2s0_colors = colors[r2s_dists_, :-1]
            i_pcd.colors = o3d.utility.Vector3dVector(r2s0_colors)
            o3d.io.write_point_cloud('{}/{}_{}_td{}.ply'.format(r2s_dir, names[ni], s, td), i_pcd)

        # colorize s2r pcd
        for td in truncated_dist:
            s2r_dists_ = s2r_dists/ td
            s2r_dists_ = np.clip(s2r_dists_, 0, 1)
            s2r_dists_ = np.floor(s2r_dists_*255).astype(np.uint8) # 0,1,2,...,255
            s2r_colors = colors[s2r_dists_, :-1]
            pcd.colors = o3d.utility.Vector3dVector(s2r_colors)
            o3d.io.write_point_cloud('{}/{}_{}_td{}.ply'.format(s2r_dir, names[ni], s, td), pcd)

    np.savetxt(param_rmsd_path ,p_rmsd, fmt='%.4f') 