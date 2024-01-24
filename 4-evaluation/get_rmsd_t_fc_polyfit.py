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
parser.add_argument('--c', type=int, required=True) # interval of coarse segmentation
parser.add_argument('--l', type=int, required=True) # interval of coarse segmentation

args = parser.parse_args()

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

assert args.l in [0,1,2,3]
if args.l == 0:
    param = [0.33, 0.33, 0.33]
elif args.l == 1:
    param = [0.2, 0.2 ,0.6]
elif args.l == 2:
    param = [0.2, 0.2, 0.6]
elif args.l == 3:
    param = [0.1, 0.1, 0.9]

t_log = np.loadtxt('{}/{}/comparison/Polyfit/{}.txt'.format(root_dir, t, args.c))
time = t_log[:, [0, 1, 2, 3+args.l]]
rmsd = np.loadtxt('{}/{}/comparison/Polyfit/rmsd/{}/({}, {}, {}).txt'.format(root_dir, t, args.c, param[0], param[1], param[2]))
mesh_dir = '{}/{}/comparison/Polyfit/poly/{}'.format(root_dir, t, args.c)
output_path = '{}/{}/comparison/Polyfit/param_sta_ransac_{}_({},{},{}).txt'.format(root_dir, t, args.c, param[0], param[1], param[2])

m_sample_num = 10
i_sample_num = 5
if args.t == 'i':
    sample_num = i_sample_num
else:
    sample_num = m_sample_num

output = np.zeros((sample_num+1, 3))
output[:-1, 1] = rmsd[:, 0]
output[:-1, 2] =  np.sum(time, axis=1)
names = [str(i+1) for i in range(sample_num)]
for i in range(sample_num):
    sample_mesh_path = '{}/{}({}, {}, {}).obj'.format(mesh_dir, names[i],param[0], param[1], param[2])
    mesh = om.read_polymesh(sample_mesh_path)
    output[i, 0] = mesh.n_faces()

output[-1, 0] = np.mean(output[:-1, 0])
output[-1, 1] = np.mean(output[:-1, 1])
output[-1, 2] = np.mean(output[:-1, 2])

np.savetxt(output_path, output, fmt="%3f")