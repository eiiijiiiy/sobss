import matplotlib
matplotlib.rcParams.update({
    'font.family': 'times',
    'font.size': 10,
    'text.usetex': True,
})
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import open3d as o3d
import openmesh as om

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--t', type=str, required=True) # interval of coarse segmentation
parser.add_argument('--c', type=float, required=True) # interval of coarse segmentation
parser.add_argument('--s', type=float, required=True) # interval of coarse segmentation
parser.add_argument('--l', type=float, required=True) # interval of coarse segmentation

args = parser.parse_args()

assert args.c in [0.5, 1, 2, 4]

sample_map = {
    0: 1, 1: 2, 2: 10, 3: 7, 4: 3,
    5: 9, 6: 8, 7: 4, 8: 5, 9: 6}


# root_dir = '/media/y/18T/parallelism_2022/revision_exper'
root_dir = '/Volumes/SSD/yijie_revision_exper'
man = '3-select_primary_normal_manhattan_revision'
roof = '4-select_primary_normal_incline_roof_revision'
out_dir = '{}/param_fig'.format(root_dir)
assert args.t in 'mi'

if args.t == 'i':
    t = roof
else:
    t = man

vote_log = np.loadtxt('{}/{}/vs_0.250000.txt'.format(root_dir, t))
cs_log = np.loadtxt(
    '{}/{}/sensitivity/coarse_segmentation/log/{}.txt'.format(root_dir, t, args.c))
merge_log = np.loadtxt('{}/{}/sensitivity/merge_revision3_1/log/{}/sigma_{:.6f}_lambda_{:.6f}.txt'.format(root_dir, t, args.c, args.s, args.l))
merge_rmsd = np.loadtxt('{}/{}/sensitivity/merge_revision3_1/rmsd/{}/sigma_{:.6f}_lambda_{:.6f}.txt'.format(root_dir, t, args.c, args.s, args.l))
mesh_dir = '{}/{}/sensitivity/merge_revision3_1/union/{}/sigma_{:.6f}_lambda_{:.6f}'.format(root_dir, t, args.c, args.s, args.l)
output_path = '{}/{}/sensitivity/param_sta_cs_{:1f}_sigma_{:6f}_lambda_{:6f}.txt'.format(root_dir, t, args.c, args.s, args.l)

m_sample_num = 10
i_sample_num = 5
if args.t == 'i':
    sample_num = i_sample_num
else:
    sample_num = m_sample_num

output = np.zeros((sample_num+1, 3))
output[:-1, 1] = merge_rmsd[:, 0]
output[:-1, 2] =  np.sum(vote_log[:, :], axis=1) + cs_log[:, 0] + np.sum(merge_log[:, :-3], axis = 1)
names = [str(i+1) for i in range(sample_num)]
for i in range(sample_num):
    sample_mesh_path = '{}/{}.obj'.format(mesh_dir, names[i])
    mesh = om.read_polymesh(sample_mesh_path)
    output[i, 0] = mesh.n_faces()

output[-1, 0] = np.mean(output[:-1, 0])
output[-1, 1] = np.mean(output[:-1, 1])
output[-1, 2] = np.mean(output[:-1, 2])

np.savetxt(output_path, output, fmt="%3f")