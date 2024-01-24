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
parser.add_argument('--c', type=float, required=True) # interval of coarse segmentation
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

m_vote_log = np.loadtxt('{}/{}/vs_0.250000.txt'.format(root_dir, man))
m_cs_log = np.loadtxt(
    '{}/{}/sensitivity/coarse_segmentation/log/{}.txt'.format(root_dir, man, args.c))
m_cs_rmsd= np.loadtxt(
    '{}/{}/sensitivity/coarse_segmentation/rmsd/{}.txt'.format(root_dir, man, args.c))
m_merge_log_dir = '{}/{}/sensitivity/merge/log/{}'.format(root_dir, man, args.c)
m_merge_rmsd_dir = '{}/{}/sensitivity/merge/rmsd/{}'.format(root_dir, man, args.c)
m_union_dir = '{}/{}/sensitivity/merge/union/{}'.format(root_dir, man, args.c)

r_vote_log = np.loadtxt('{}/{}/vs_0.250000.txt'.format(root_dir, roof))
r_cs_log = np.loadtxt(
    '{}/{}/sensitivity/coarse_segmentation/log/{}.txt'.format(root_dir, roof, args.c))
r_cs_rmsd= np.loadtxt(
    '{}/{}/sensitivity/coarse_segmentation/rmsd/{}.txt'.format(root_dir, roof, args.c))
r_merge_log_dir = '{}/{}/sensitivity/merge/log/{}'.format(root_dir, roof, args.c)
r_merge_rmsd_dir = '{}/{}/sensitivity/merge/rmsd/{}'.format(root_dir, roof, args.c)
r_union_dir = '{}/{}/sensitivity/merge/union/{}'.format(root_dir, roof, args.c)

sigmas = [0.25, 0.5, 1, 2, 4]
lambdas = [0.125, 0.25, 0.5, 1, 2, 4, 8]
num_sigmas = len(sigmas)
num_lambdas = len(lambdas)
SIGMA = np.zeros(num_sigmas * num_lambdas)
LAMBDA = np.zeros(num_sigmas * num_lambdas)
RMSD = np.zeros(num_sigmas * num_lambdas)
FC = np.zeros(num_sigmas * num_lambdas)
T = np.zeros(num_sigmas * num_lambdas)

m_sample_num = 10
r_sample_num = 5

names = [str(i+1) for i in range(m_sample_num)]
for si in range(num_sigmas):
    SIGMA[si*num_lambdas : (si+1) * num_lambdas] = sigmas[si]
    for li in range(num_lambdas):
        LAMBDA[si*num_lambdas + li] = lambdas[li]

        m_merge_rmsd = np.loadtxt(
            '{}/sigma_{:.6f}_lambda_{:.6f}.txt'.format(m_merge_rmsd_dir, sigmas[si], lambdas[li]))
        m_merge_log = np.loadtxt(
            '{}/sigma_{:.6f}_lambda_{:.6f}.txt'.format(m_merge_log_dir, sigmas[si], lambdas[li]))
        rft = np.zeros((m_sample_num + r_sample_num, 3)) # rmsd, fc, t
        rft[:m_sample_num, 2] = np.sum(m_vote_log[:, :], axis=1) + m_cs_log[:, 0] + np.sum(m_merge_log[:, :-3], axis = 1)
        rft[:m_sample_num, 0] = m_merge_rmsd[:, 0]
        for mi in range(m_sample_num):
            union_path = '{}/sigma_{:.6f}_lambda_{:.6f}/{}.obj'.format(
                m_union_dir, sigmas[si], lambdas[li], mi+1)
            union = om.read_polymesh(union_path)
            rft[mi, 1] = union.n_faces()
        
        r_merge_rmsd = np.loadtxt(
            '{}/sigma_{:.6f}_lambda_{:.6f}.txt'.format(r_merge_rmsd_dir, sigmas[si], lambdas[li]))
        r_merge_log = np.loadtxt(
            '{}/sigma_{:.6f}_lambda_{:.6f}.txt'.format(r_merge_log_dir, sigmas[si], lambdas[li]))
        rft[m_sample_num:, 2] = np.sum(r_vote_log[:, :], axis=1) + r_cs_log[:, 0] + np.sum(r_merge_log[:, :-3], axis = 1)
        rft[m_sample_num:, 0] = r_merge_rmsd[:, 0]
        for ri in range(r_sample_num):
            union_path = '{}/sigma_{:.6f}_lambda_{:.6f}/{}.obj'.format(
                r_union_dir, sigmas[si], lambdas[li], ri+1)
            union = om.read_polymesh(union_path)
            rft[m_sample_num + ri, 1] = union.n_faces()
        idx = np.where(rft[:, 0] < 100)[0]
        RMSD[si*num_lambdas + li] = np.mean(rft[idx, 0])
        FC[si*num_lambdas + li] = np.mean(rft[:, 1])
        T[si*num_lambdas + li] = np.mean(rft[:, 2])


np.savetxt('{}/merge_cs{}.txt'.format(out_dir, args.c), 
           np.vstack((np.log2(SIGMA), np.log2(LAMBDA), RMSD, FC, T)).transpose())
np.savetxt('{}/merge_cs{}_sigma{}.txt'.format(out_dir, args.c, 0.5), 
           np.vstack((np.log2(LAMBDA[7:14]), RMSD[7:14], FC[7:14], T[7:14])).transpose())

# SIGMA = SIGMA.reshape((num_sigmas, num_lambdas))
# LAMBDA = LAMBDA.reshape((num_sigmas, num_lambdas))

# RMSD = RMSD.reshape((num_sigmas, num_lambdas))
# FC = FC.reshape((num_sigmas, num_lambdas))
# T = T.reshape((num_sigmas, num_lambdas))


# # plot figure
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# RMSD_surf = ax.plot_surface(SIGMA, LAMBDA, RMSD, cmap=cm.jet,
#                        linewidth=0.2, antialiased=True)
# wire = ax.plot_wireframe(SIGMA, LAMBDA, RMSD,color='k',linewidth=0.3)
# fig.colorbar(RMSD_surf, shrink=0.5, aspect=15, location='bottom')
# plt.savefig('{}/merge_RMSD_cs_{}.png'.format(out_dir, args.c))
# plt.clf()

# # 交换xy的vis效果更好，比换成透明的好
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# FC_surf = ax.plot_surface(LAMBDA, SIGMA, FC, cmap=cm.jet,
#                        linewidth=0.2, antialiased=True)
# wire = ax.plot_wireframe(LAMBDA, SIGMA, FC, color='k',linewidth=0.3)
# fig.colorbar(FC_surf, shrink=0.5, aspect=15, location='bottom')
# plt.savefig('{}/merge_FC_cs_{}.png'.format(out_dir, args.c))
# plt.clf()

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# T_surf = ax.plot_surface(LAMBDA, SIGMA, T, cmap=cm.jet,
#                        linewidth=0.2, antialiased=True) 
# wire = ax.plot_wireframe(LAMBDA, SIGMA, T, color='k',linewidth=0.3)
# fig.colorbar(T_surf, shrink=0.5, aspect=15, location='bottom')
# plt.savefig('{}/merge_T_cs_{}.png'.format(out_dir, args.c))


                                                            


