import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import openmesh as om
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--l', type=float, required=True)
args = parser.parse_args()

assert args.l in [0.125, 0.25, 0.5, 1, 2, 4, 8]



root_dir = '/media/y/18T/parallelism_2022/revision_exper'
man = '3-select_primary_normal_manhattan_revision'
roof = '4-select_primary_normal_incline_roof_revision'
out_dir = '{}/param_fig'.format(root_dir)

m_vote_log = np.loadtxt('{}/{}/vs_0.250000.txt'.format(root_dir, man))
m_cs_log_dir = '{}/{}/sensitivity/coarse_segmentation/log'.format(root_dir, man)
m_cs_rmsd_dir = '{}/{}/sensitivity/coarse_segmentation/rmsd'.format(root_dir, man)
m_merge_log_dir = '{}/{}/sensitivity/merge/log'.format(root_dir, man)
m_merge_rmsd_dir = '{}/{}/sensitivity/merge/rmsd'.format(root_dir, man)
m_union_dir = '{}/{}/sensitivity/merge/union'.format(root_dir, man)

r_vote_log = np.loadtxt('{}/{}/vs_0.250000.txt'.format(root_dir, roof))
r_cs_log_dir = '{}/{}/sensitivity/coarse_segmentation/log'.format(root_dir, roof)
r_cs_rmsd_dir = '{}/{}/sensitivity/coarse_segmentation/rmsd'.format(root_dir, roof)
r_merge_log_dir = '{}/{}/sensitivity/merge/log'.format(root_dir, roof)
r_merge_rmsd_dir = '{}/{}/sensitivity/merge/rmsd'.format(root_dir, roof)
r_union_dir = '{}/{}/sensitivity/merge/union'.format(root_dir, roof)

sigmas = [0.25, 0.5, 1, 2, 4]
css = [0.5, 1, 2, 4]

num_sigmas = len(sigmas)
num_css = len(css)
CS = np.zeros(num_sigmas * num_css)
SIGMA = np.zeros(num_sigmas * num_css)
RMSD = np.zeros(num_sigmas * num_css)
FC = np.zeros(num_sigmas * num_css)
T = np.zeros(num_sigmas * num_css)

m_sample_num = 10
r_sample_num = 5

for ci in tqdm(range(num_css)):
    CS[ci * num_sigmas : (ci+1) * num_sigmas] = css[ci]
    m_cs_log = np.loadtxt('{}/{:.1f}.txt'.format(m_cs_log_dir, css[ci]))
    r_cs_log = np.loadtxt('{}/{:.1f}.txt'.format(r_cs_log_dir, css[ci]))

    for si in tqdm(range(num_sigmas)):
        SIGMA[ci*num_sigmas + si] = sigmas[si]

        m_merge_rmsd = np.loadtxt(
            '{}/{:.1f}/sigma_{:.6f}_lambda_{:.6f}.txt'.format(m_merge_rmsd_dir, css[ci], sigmas[si],args.l))
        m_merge_log = np.loadtxt(
            '{}/{:.1f}/sigma_{:.6f}_lambda_{:.6f}.txt'.format(m_merge_log_dir, css[ci], sigmas[si], args.l))
        rft = np.zeros((m_sample_num + r_sample_num, 3)) # rmsd, fc, t
        rft[:m_sample_num, 2] = np.sum(m_vote_log, axis=1) + m_cs_log[:, 0] + np.sum(m_merge_log[:, :-3], axis = 1)
        rft[:m_sample_num, 0] = m_merge_rmsd[:, 0]
        for mi in range(m_sample_num):
            union_path = '{}/{:.1f}/sigma_{:.6f}_lambda_{:.6f}/{}.obj'.format(
                m_union_dir, css[ci], sigmas[si], args.l, mi+1)
            union = om.read_polymesh(union_path)
            rft[mi, 1] = union.n_faces()
        
        r_merge_rmsd = np.loadtxt(
            '{}/{:.1f}/sigma_{:.6f}_lambda_{:.6f}.txt'.format(r_merge_rmsd_dir, css[ci], sigmas[si], args.l))
        r_merge_log = np.loadtxt(
            '{}/{:.1f}/sigma_{:.6f}_lambda_{:.6f}.txt'.format(r_merge_log_dir, css[ci], sigmas[si], args.l))
        rft[m_sample_num:, 2] = np.sum(r_vote_log[:, :], axis=1) + r_cs_log[:, 0] + np.sum(r_merge_log[:, :-3], axis = 1)
        rft[m_sample_num:, 0] = r_merge_rmsd[:, 0]
        for ri in range(r_sample_num):
            union_path = '{}/{:.1f}/sigma_{:.6f}_lambda_{:.6f}/{}.obj'.format(
                r_union_dir, css[ci], sigmas[si], args.l, ri+1)
            union = om.read_polymesh(union_path)
            rft[m_sample_num + ri, 1] = union.n_faces()
        idx = np.where(rft[:, 0] < 100)[0]

        RMSD[ci*num_sigmas + si] = np.mean(rft[idx, 0])
        FC[ci*num_sigmas + si] = np.mean(rft[:, 1])
        T[ci*num_sigmas + si] = np.mean(rft[:, 2])

CS = CS.reshape((num_css, num_sigmas))
SIGMA = SIGMA.reshape((num_css, num_sigmas))

RMSD = RMSD.reshape((num_css, num_sigmas))
FC = FC.reshape((num_css, num_sigmas))
T = T.reshape((num_css, num_sigmas))


# plot figure
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
RMSD_surf = ax.plot_surface(SIGMA, CS, RMSD, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
fig.colorbar(RMSD_surf, shrink=0.5, aspect=5)
plt.savefig('{}/cs_RMSD_l_{}.png'.format(out_dir, args.l))
plt.clf()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
FC_surf = ax.plot_surface(CS, SIGMA, FC, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
fig.colorbar(FC_surf, shrink=0.5, aspect=5)
plt.savefig('{}/cs_FC_l_{}.png'.format(out_dir, args.l))
plt.clf()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
T_surf = ax.plot_surface(CS, SIGMA, T, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
fig.colorbar(T_surf, shrink=0.5, aspect=5)
plt.savefig('{}/cs_T_l_{}.png'.format(out_dir, args.l))


                                                    