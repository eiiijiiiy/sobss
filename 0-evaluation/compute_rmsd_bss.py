from tokenize import Double
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import geopandas
from tqdm import tqdm
import os



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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c', type=float, required=True)
parser.add_argument('-t', type=str, required=True)
args = parser.parse_args()
assert args.t in 'mi'
if args.t == 'm':
    sample_num = 10
else:
    sample_num = 5

this_type = '3-select_primary_normal_manhattan_revision' if args.t == 'm' else '4-select_primary_normal_incline_roof_revision'
aa_dir = '/media/y/18T/parallelism_2022/exper_revision/{}/1-axisalign'.format(this_type)
root_union_tri_dir = '/media/y/18T/parallelism_2022/exper_revision/{}/sensitivity/merge_revision3_1/union_tri'.format(this_type)
root_union_tri_rb_dir = '/media/y/18T/parallelism_2022/exper_revision/{}/sensitivity/merge_revision3_1/union_tri_rb'.format(this_type)
root_r2s_dir = '/media/y/18T/parallelism_2022/exper_revision/{}/sensitivity/merge_revision3_1/r2s'.format(this_type)
root_s2r_dir = '/media/y/18T/parallelism_2022/exper_revision/{}/sensitivity/merge_revision3_1/s2r'.format(this_type)
root_rmsd_dir = '/media/y/18T/parallelism_2022/exper_revision/{}/sensitivity/merge_revision3_1/rmsd'.format(this_type)



this_union_tri_dir = '{}/{}'.format(root_union_tri_dir, args.c)
this_union_tri_rb_dir = '{}/{}'.format(root_union_tri_rb_dir, args.c)
this_r2s_dir = '{}/{}'.format(root_r2s_dir, args.c)
this_s2r_dir = '{}/{}'.format(root_s2r_dir, args.c)
this_rmsd_dir = '{}/{}'.format(root_rmsd_dir, args.c)

sigmas = [0.25, 0.5, 1, 2, 4]
lambdas = [0.125, 0.25, 0.5, 1, 2, 4, 8]
names = [str(i+1) for i in range(sample_num)]
for s in sigmas:
    for lbd in lambdas:
        print('processing sigma {} lamdba {}'.format(s, lbd))
        param_union_tri_dir = '{}/sigma_{:.6f}_lambda_{:.6f}'.format(
            this_union_tri_dir, s, lbd)
        param_union_tri_rb_dir = '{}/sigma_{:.6f}_lambda_{:.6f}'.format(
            this_union_tri_rb_dir, s, lbd)
        param_r2s_dir = '{}/sigma_{:.6f}_lambda_{:.6f}'.format(
            this_r2s_dir, s, lbd)
        param_s2r_dir = '{}/sigma_{:.6f}_lambda_{:.6f}'.format(
            this_s2r_dir, s, lbd)
        param_rmsd_path = '{}/sigma_{:.6f}_lambda_{:.6f}.txt'.format(
            this_rmsd_dir, s, lbd)
        
        if not os.path.exists(param_union_tri_rb_dir):
            os.mkdir(param_union_tri_rb_dir)
        if not os.path.exists(param_r2s_dir):
            os.mkdir(param_r2s_dir)
        if not os.path.exists(param_s2r_dir):
            os.mkdir(param_s2r_dir)
        
        p_rmsd = np.zeros((sample_num, 5))
        for ni in range(sample_num):
            print("sample id {}".format(ni))
            n = names[ni]
            union_tri_path = '{}/{}.obj'.format(param_union_tri_dir, n)
            mesh = o3d.io.read_triangle_mesh(union_tri_path)
            tri_num = np.asarray(mesh.triangles).shape[0]
            if tri_num == 0:
                p_rmsd[ni, :] = 100 
                continue
            
            mesh = remove_btm(mesh)
            rb_path = '{}/{}.obj'.format(param_union_tri_rb_dir, n)
            o3d.io.write_triangle_mesh(rb_path, mesh)
            tri_num = np.asarray(mesh.triangles).shape[0]
            if tri_num == 0:
                p_rmsd[ni, :] = 100 
                continue
            
            aa_pcd = o3d.io.read_point_cloud('{}/{}.ply'.format(aa_dir, n))
            aa_pcd = aa_pcd.voxel_down_sample(voxel_size=0.5)
            aa_pt_num = np.asarray(aa_pcd.points).shape[0]
            try:
                pcd = mesh.sample_points_uniformly(aa_pt_num)
            except:
                from IPython import embed
                embed()
            
            r2s_dists = aa_pcd.compute_point_cloud_distance(pcd)
            r2s_dists = np.asarray(r2s_dists)
            s2r_dists = pcd.compute_point_cloud_distance(aa_pcd)
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
                aa_pcd.colors = o3d.utility.Vector3dVector(r2s0_colors)
                o3d.io.write_point_cloud('{}/{}_td{}.ply'.format(param_r2s_dir, names[ni], td), aa_pcd)

            # colorize s2r pcd
            for td in truncated_dist:
                s2r_dists_ = s2r_dists/ td
                s2r_dists_ = np.clip(s2r_dists_, 0, 1)
                s2r_dists_ = np.floor(s2r_dists_*255).astype(np.uint8) # 0,1,2,...,255
                s2r_colors = colors[s2r_dists_, :-1]
                pcd.colors = o3d.utility.Vector3dVector(s2r_colors)
                o3d.io.write_point_cloud('{}/{}_td{}.ply'.format(param_s2r_dir, names[ni], td), pcd)

        np.savetxt(param_rmsd_path ,p_rmsd, fmt='%.4f') 