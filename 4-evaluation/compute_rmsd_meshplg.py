import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


cmap = plt.cm.get_cmap('jet')
colors = cmap(np.arange(cmap.N)) # (256, 4) including the alpha channel
names = [str(i+1) for i in range(6)] + [str(i+1) for i in range(7,10)]
sampled_num = np.array([2,2,3,5,5,8,4,4,3,2]) * 1000000

# for DEBUG
# param_names = ['lb']
# param_values = ['1']
# names = ['1']
# sampled_num = np.array([2]) * 1000000

truncated_dist = [0.5, 1, 2]

root_dir = '/home/y/Dropbox'
# root_dir = '/Users/yijie/Dropbox'
mesh_dir = '{}/2022.Building_Section_Skeleton/exper_bdlg_only/7-MeshPlg-1.5-tri_rb'.format(root_dir)
in_dir = '{}/2022.Building_Section_Skeleton/exper_bdlg_only/0-input'.format(root_dir)
r2s_dir =  '{}/2022.Building_Section_Skeleton/exper_bdlg_only/7-MeshPlg-1.5-r2s'.format(root_dir)
s2r_dir = '{}/2022.Building_Section_Skeleton/exper_bdlg_only/7-MeshPlg-1.5-s2r'.format(root_dir)
rmsd_log_path = '{}/2022.Building_Section_Skeleton/exper_bdlg_only/7-MeshPlg-1.5-rmsd.txt'.format(root_dir)

rmsd = np.zeros((10, 5))

for ni in tqdm(range(len(names))):
    in_pcd = o3d.io.read_point_cloud('{}/{}.ply'.format(in_dir, names[ni]))
    mesh = o3d.io.read_triangle_mesh('{}/{}.ply'.format(mesh_dir, names[ni]))
    pcd = mesh.sample_points_uniformly(sampled_num[ni])
    r2s_dists = in_pcd.compute_point_cloud_distance(pcd)
    r2s_dists = np.asarray(r2s_dists)
    s2r_dists = pcd.compute_point_cloud_distance(in_pcd)
    s2r_dists = np.asarray(s2r_dists)
    rmsd_r2s = np.sqrt(np.sum(r2s_dists * r2s_dists)/r2s_dists.shape[0])
    rmsd_s2r = np.sqrt(np.sum(s2r_dists * s2r_dists)/s2r_dists.shape[0])
    rmsd[ni, 0] = rmsd_r2s 
    rmsd[ni, 1] = rmsd_s2r 
    rmsd[ni, 2] = np.max(r2s_dists) 
    rmsd[ni, 3] = np.max(s2r_dists) 
    rmsd[ni, 4] = max(rmsd[ni, 2], rmsd[ni, 3])
    
    # colorize r2s pcd
    for td in truncated_dist:
        r2s_dists_ = r2s_dists / td
        r2s_dists_ = np.clip(r2s_dists_, 0, 1)
        r2s_dists_ = np.floor(r2s_dists_*255).astype(np.uint8) # 0,1,2,...,255
        r2s_colors = colors[r2s_dists_, :-1]
        in_pcd.colors = o3d.utility.Vector3dVector(r2s_colors)
        o3d.io.write_point_cloud('{}/{}_td{}.ply'.format(r2s_dir, names[ni], td), in_pcd)

    # colorize s2r pcd
    for td in truncated_dist:
        s2r_dists_ = s2r_dists / td
        s2r_dists_ = np.clip(s2r_dists_, 0, 1)
        s2r_dists_ = np.floor(s2r_dists_*255).astype(np.uint8) # 0,1,2,...,255
        s2r_colors = colors[s2r_dists_, :-1]
        pcd.colors = o3d.utility.Vector3dVector(s2r_colors)
        o3d.io.write_point_cloud('{}/{}_td{}.ply'.format(s2r_dir, names[ni], td), pcd)


np.savetxt(rmsd_log_path ,rmsd, fmt='%.4f')
