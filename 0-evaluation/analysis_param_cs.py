import matplotlib
matplotlib.rcParams.update({
    'font.family': 'times',
    'font.size': 30,
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
parser.add_argument('--l', type=float, required=True) # lambda
args = parser.parse_args()

assert args.l in [0.125, 0.25, 0.5, 1, 2, 4, 8]

sample_map = {
    0: 1, 1: 2, 2: 10, 3: 7, 4: 3,
    5: 9, 6: 8, 7: 4, 8: 5, 9: 6}


root_dir = '/Volumes/SSD/yijie_revision_exper'
man = '3-select_primary_normal_manhattan_revision'
roof = '4-select_primary_normal_incline_roof_revision'
out_dir = '{}/param_fig'.format(root_dir)

m_vote_log = np.loadtxt('{}/{}/vs_0.250000.txt'.format(root_dir, man))
r_vote_log = np.loadtxt('{}/{}/vs_0.250000.txt'.format(root_dir, roof))


css = [0.5, 1, 2, 4]
sigmas = [0.25, 0.5, 1, 2, 4]

num_sigmas = len(sigmas)
num_css = len(css)
SIGMA = np.zeros(num_css * num_sigmas)
CS = np.zeros(num_css * num_sigmas)
RMSD = np.zeros(num_css * num_sigmas)
FC = np.zeros(num_css * num_sigmas)
T = np.zeros(num_css * num_sigmas)

m_sample_num = 10
r_sample_num = 5

names = [str(i+1) for i in range(m_sample_num)]
for ci in range(num_css):
    CS[ci*num_sigmas : (ci+1) * num_sigmas] = css[ci]
    for si in range(num_sigmas):
        SIGMA[ci*num_sigmas + si] = sigmas[si]

        m_cs_log = np.loadtxt(
            '{}/{}/sensitivity/coarse_segmentation/log/{:.1f}.txt'.format(root_dir, man, css[ci]))
        m_cs_rmsd= np.loadtxt(
            '{}/{}/sensitivity/coarse_segmentation/rmsd/{:.1f}.txt'.format(root_dir, man, css[ci]))
        r_cs_log = np.loadtxt(
            '{}/{}/sensitivity/coarse_segmentation/log/{:.1f}.txt'.format(root_dir, roof, css[ci]))
        r_cs_rmsd= np.loadtxt(
            '{}/{}/sensitivity/coarse_segmentation/rmsd/{:.1f}.txt'.format(root_dir, roof, css[ci]))
        
        m_merge_rmsd_dir = '{}/{}/sensitivity/merge/rmsd/{:.1f}'.format(root_dir, man, css[ci])
        m_merge_rmsd = np.loadtxt(
            '{}/sigma_{:.6f}_lambda_{:.6f}.txt'.format(m_merge_rmsd_dir, sigmas[si], args.l))
        m_merge_log_dir = '{}/{}/sensitivity/merge/log/{:.1f}'.format(root_dir, man, css[ci])
        m_merge_log = np.loadtxt(
            '{}/sigma_{:.6f}_lambda_{:.6f}.txt'.format(m_merge_log_dir, sigmas[si], args.l))
        rft = np.zeros((m_sample_num + r_sample_num, 3)) # rmsd, fc, t
        rft[:m_sample_num, 2] = np.sum(m_vote_log[:, :], axis=1) + m_cs_log[:, 0] + np.sum(m_merge_log[:, :-3], axis = 1)
        rft[:m_sample_num, 0] = m_merge_rmsd[:, 0]
        m_union_dir = '{}/{}/sensitivity/merge/union/{:.1f}'.format(root_dir, man, css[ci])
        for mi in range(m_sample_num):
            union_path = '{}/sigma_{:.6f}_lambda_{:.6f}/{}.obj'.format(
                m_union_dir, sigmas[si], args.l, mi+1)
            union = om.read_polymesh(union_path)
            rft[mi, 1] = union.n_faces()

        r_merge_rmsd_dir = '{}/{}/sensitivity/merge/rmsd/{:.1f}'.format(root_dir, roof, css[ci])
        r_merge_rmsd = np.loadtxt(
            '{}/sigma_{:.6f}_lambda_{:.6f}.txt'.format(r_merge_rmsd_dir, sigmas[si], args.l))
        r_merge_log_dir = '{}/{}/sensitivity/merge/log/{:.1f}'.format(root_dir, roof, css[ci])
        r_merge_log = np.loadtxt(
            '{}/sigma_{:.6f}_lambda_{:.6f}.txt'.format(r_merge_log_dir, sigmas[si], args.l))
        rft[m_sample_num:, 2] = np.sum(r_vote_log[:, :], axis=1) + r_cs_log[:, 0] + np.sum(r_merge_log[:, :-3], axis = 1)
        rft[m_sample_num:, 0] = r_merge_rmsd[:, 0]
        r_union_dir = '{}/{}/sensitivity/merge/union/{:.1f}'.format(root_dir, roof, css[ci])
        for ri in range(r_sample_num):
            union_path = '{}/sigma_{:.6f}_lambda_{:.6f}/{}.obj'.format(
                r_union_dir, sigmas[si], args.l, ri+1)
            union = om.read_polymesh(union_path)
            rft[m_sample_num + ri, 1] = union.n_faces()
        idx = np.where(rft[:, 0] < 100)[0]
        RMSD[ci * num_sigmas + si] = np.mean(rft[idx, 0])
        FC[ci * num_sigmas + si] = np.mean(rft[:, 1])
        T[ci * num_sigmas + si] = np.mean(rft[:, 2])

np.savetxt('{}/cs_lambda{}.txt'.format(out_dir, args.l), 
           np.vstack((np.log2(CS), np.log2(SIGMA), RMSD, FC, T)).transpose())


# SIGMA = SIGMA.reshape((num_css, num_sigmas))
# LAMBDA = LAMBDA.reshape((num_css, num_sigmas))

# RMSD = RMSD.reshape((num_css, num_sigmas))
# FC = FC.reshape((num_css, num_sigmas))
# T = T.reshape((num_css, num_sigmas))


# # plot figure
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# RMSD_surf = ax.plot_surface(CS, SIGMA, RMSD, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=True)
# fig.colorbar(RMSD_surf, shrink=0.5, aspect=5)
# plt.savefig('{}/cs_RMSD_lambda_{}.png'.format(out_dir, args.l))
# plt.clf()

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# FC_surf = ax.plot_surface(LAMBDA, SIGMA, FC, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=True)
# fig.colorbar(FC_surf, shrink=0.5, aspect=5)
# plt.savefig('{}/cs_FC_lambda_{}.png'.format(out_dir, args.l))
# plt.clf()

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# T_surf = ax.plot_surface(LAMBDA, SIGMA, T, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=True)
# fig.colorbar(T_surf, shrink=0.5, aspect=5)
# plt.savefig('{}/cs_T_lambda_{}.png'.format(out_dir, args.l))


                                                            


