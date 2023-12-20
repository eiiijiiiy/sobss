import numpy as np
import open3d as o3d
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--c', type=float, required=True)
parser.add_argument('--s', type=float, required=True)
parser.add_argument('--l1', type=float, required=True)
parser.add_argument('--l2', type=float, required=True)
args = parser.parse_args()

vote_log_path = '/media/y/18T/parallelism_2022/revision_exper/3-select_primary_normal_manhattan_revision/vs_0.250000.txt'
cs_log_dir = '/media/y/18T/parallelism_2022/revision_exper/3-select_primary_normal_manhattan_revision/sensitivity/coarse_segmentation/log'
cs_log_path = '{}/{}.txt'.format(cs_log_dir, args.c)
root_merge_log_dir = '/media/y/18T/parallelism_2022/revision_exper/3-select_primary_normal_manhattan_revision/sensitivity/merge/log'
param_merge_log_path = '{}/{}/sigma_{:.6f}_lambda_1_{:.6f}_lambda_2_{:.6f}.txt'.format(
    root_merge_log_dir, args.c, args.s, args.l1, args.l2)
root_union_tri_dir = '/media/y/18T/parallelism_2022/revision_exper/3-select_primary_normal_manhattan_revision/sensitivity/merge/union_tri'
param_union_tri_dir = '{}/{}/sigma_{:.6f}_lambda_1_{:.6f}_lambda_2_{:.6f}'.format(
    root_union_tri_dir, args.c, args.s, args.l1, args.l2)
root_rmsd_dir = '/media/y/18T/parallelism_2022/revision_exper/3-select_primary_normal_manhattan_revision/sensitivity/merge/rmsd'
param_rmsd_path = '{}/{}/sigma_{:.6f}_lambda_1_{:.6f}_lambda_2_{:.6f}.txt'.format(
    root_rmsd_dir, args.c, args.s, args.l1, args.l2)
root_summary_dir = '/media/y/18T/parallelism_2022/revision_exper/3-select_primary_normal_manhattan_revision/sensitivity/merge/summary' 
param_summary_path = '{}/{}/sigma_{:.6f}_lambda_1_{:.6f}_lambda_2_{:.6f}.txt'.format(
    root_summary_dir, args.c, args.s, args.l1, args.l2)

sample_map = {
    0: 1, 1: 2, 2: 10, 3: 7, 4: 3,
    5: 9, 6: 8, 7: 4, 8: 5, 9: 6}
sample_num = len(sample_map.keys())
vote_log = np.loadtxt(vote_log_path)
cs_log = np.loadtxt(cs_log_path)
merge_log = np.loadtxt(param_merge_log_path)
rmsd = np.loadtxt(param_rmsd_path)

metric_table = np.zeros((sample_num+1, 7)) # tri num | rmsd | total time | vote time | cs time | merge time
for sk in sample_map.keys():
    n = sample_map[sk]
    # sum processing time
    vote_time = np.sum(vote_log[n-1, :])
    metric_table[sk, 4] = vote_time
    cs_time = cs_log[n-1, 0]
    metric_table[sk, 5] = vote_time
    merge_time = np.sum(merge_log[n-1, :-3])
    metric_table[sk, 6] = merge_time
    process_time = vote_time + cs_time + merge_time
    metric_table[sk, 3] = process_time
    # count triangles
    union_tri_path = '{}/{}.obj'.format(param_union_tri_dir, str(n))
    mesh = o3d.io.read_triangle_mesh(union_tri_path)
    tri_num = np.asarray(mesh.triangles).shape[0]
    metric_table[sk, 0] = tri_num
    # get rmsd of r2s
    metric_table[sk, 1] = rmsd[n-1, 0]
    # get rmsd of s2r
    metric_table[sk, 2] = rmsd[n-1, 1]

metric_table[-1, :] = np.average(metric_table[:-1, :], axis=0)
np.savetxt(param_summary_path, metric_table, fmt='%.4f')




    





