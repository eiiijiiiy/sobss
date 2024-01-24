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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--c1', type=float, required=True)
parser.add_argument('--l1', type=float, required=True)
parser.add_argument('--s1', type=float, required=True)
parser.add_argument('--c2', type=float, required=True)
parser.add_argument('--l2', type=float, required=True)
parser.add_argument('--s2', type=float, required=True)
args = parser.parse_args()


root_dir = '/Volumes/SSD/yijie_revision_exper'
man = '3-select_primary_normal_manhattan_revision'
roof = '4-select_primary_normal_incline_roof_revision'
out_dir = '{}/param_fig'.format(root_dir)
m_sample_num = 10
r_sample_num = 5

m_pcd_dir = '{}/{}/0-input'.format(root_dir, man)
r_pcd_dir = '{}/{}/0-input'.format(root_dir, roof)
m_pcd_size = np.zeros(m_sample_num)
r_pcd_size = np.zeros(r_sample_num)

pcd_size = np.zeros((m_sample_num + r_sample_num))
for mi in range(m_sample_num):
    pcd_path = '{}/{}.ply'.format(m_pcd_dir, mi+1)
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd_size[mi] = np.asarray(pcd.points).shape[0]

for ri in range(r_sample_num):
    pcd_path = '{}/{}.ply'.format(r_pcd_dir, ri+1)
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd_size[m_sample_num+ri] = np.asarray(pcd.points).shape[0]

vote_time = np.zeros((m_sample_num + r_sample_num))
m_vote_log = np.loadtxt('{}/{}/vs_0.250000.txt'.format(root_dir, man))
vote_time[:m_sample_num] = np.sum(m_vote_log, axis=1)

r_vote_log = np.loadtxt('{}/{}/vs_0.250000.txt'.format(root_dir, roof))
vote_time[m_sample_num:] = np.sum(r_vote_log, axis=1)

cs_time = np.zeros((m_sample_num + r_sample_num))
m_cs_log = np.loadtxt(
    '{}/{}/sensitivity/coarse_segmentation/log/{:.1f}.txt'.format(root_dir, man, args.c1))
cs_time[:m_sample_num] = m_cs_log[:, 0]

r_cs_log = np.loadtxt(
    '{}/{}/sensitivity/coarse_segmentation/log/{}.txt'.format(root_dir, roof, args.c2))
cs_time[m_sample_num:] = r_cs_log[:, 0]

merge_time = np.zeros((m_sample_num + r_sample_num))
merge_pre_time = np.zeros((m_sample_num + r_sample_num))
merge_opt_time = np.zeros((m_sample_num + r_sample_num))
merge_other_time = np.zeros((m_sample_num + r_sample_num))
m_merge_log = np.loadtxt(
    '{}/{}/sensitivity/merge_revision3_1/log/{:.1f}/sigma_{:.6f}_lambda_{:.6f}.txt'.format(
        root_dir, man, args.c1, args.s1, args.l1))
merge_time[:m_sample_num] = np.sum(m_merge_log[:, :-3], axis=1)
merge_pre_time[:m_sample_num] = m_merge_log[:, 1]
merge_opt_time[:m_sample_num] = m_merge_log[:, 2]
merge_other_time[:m_sample_num] = np.sum(m_merge_log[:, [0,3,4]], axis=1)

r_merge_log = np.loadtxt(
    '{}/{}/sensitivity/merge_revision3_1/log/{:.1f}/sigma_{:.6f}_lambda_{:.6f}.txt'.format(
        root_dir, roof, args.c2, args.s2, args.l2))
merge_time[m_sample_num:] = np.sum(r_merge_log[:, :-3], axis=1)
merge_pre_time[m_sample_num:] = r_merge_log[:, 1]
merge_opt_time[m_sample_num:] = r_merge_log[:, 2]
merge_other_time[m_sample_num:] = np.sum(r_merge_log[:, [0,3,4]], axis=1)

np.savetxt('{}/time_stage_m{}_{}_{}_r{}_{}_{}_all.txt'.format(out_dir, args.c1, args.s1, args.l1, args.c2, args.s2, args.l2), 
        np.vstack((
            pcd_size, vote_time, cs_time, merge_time,
            merge_time + vote_time+cs_time)).transpose())
# # plot 
# pcd_idx = np.arange(m_sample_num + r_sample_num)
# plt.bar(pcd_idx, vote_time, width = 0.5, label='Pairing')
# plt.bar(pcd_idx, cs_time,  width = 0.5, bottom=vote_time, label='Coarse segmentation')
# # plt.bar(pcd_idx, merge_time, bottom=vote_time+cs_time, label='Merging')
# plt.bar(pcd_idx, merge_pre_time, width = 0.5, bottom=vote_time+cs_time+merge_other_time, label='Merging: Cost pre-computation', color='g', hatch='+++', linewidth=0.01)
# plt.bar(pcd_idx, merge_opt_time, width = 0.5, bottom=vote_time+cs_time+merge_other_time+merge_pre_time, label='Merging: Optimization', color='g', hatch='///', linewidth=0.01)
# plt.bar(pcd_idx, merge_other_time, width = 0.5, bottom=vote_time+cs_time, label='Merging: Other', color='g')
# plt.legend()
# plt.savefig('{}/time_stage_c{}_s{}_l{}.png'.format(out_dir, args.c, args.s, args.l))
# plt.clf()

# pcd_idx = np.arange(r_sample_num)
# plt.bar(pcd_idx, vote_time[:5], width = 0.5, label='Pairing')
# plt.bar(pcd_idx, cs_time[:5],  width = 0.5, bottom=vote_time[:5], label='Coarse segmentation')
# # plt.bar(pcd_idx, merge_time[:5], width = 0.5, bottom=vote_time[:5]+cs_time[:5], label='Merging')
# plt.bar(pcd_idx, merge_pre_time[:5], width = 0.5, bottom=vote_time[:5]+cs_time[:5]+merge_other_time[:5], label='Merging: Cost pre-computation', color='g', hatch='+++', linewidth=0.01)
# plt.bar(pcd_idx, merge_opt_time[:5], width = 0.5, bottom=vote_time[:5]+cs_time[:5]+merge_other_time[:5]+merge_pre_time[:5], label='Merging: Optimization', color='g', hatch='///', linewidth=0.01)
# plt.bar(pcd_idx, merge_other_time[:5], width = 0.5, bottom=vote_time[:5]+cs_time[:5], label='Merging: Other', color='g')
# # plt.legend()
# plt.savefig('{}/time_stage_c{}_s{}_l{}_inc.png'.format(out_dir, args.c, args.s, args.l))
# plt.clf()

# pcd_idx = np.arange(m_sample_num)
# plt.bar(pcd_idx, vote_time[5:], width = 0.5, label='Pairing')
# plt.bar(pcd_idx, cs_time[5:],  width = 0.5, bottom=vote_time[5:], label='Coarse segmentation')
# # plt.bar(pcd_idx, merge_time[5:], width = 0.5, bottom=vote_time[5:]+cs_time[5:], label='Merging')
# plt.bar(pcd_idx, merge_pre_time[5:], width = 0.5, bottom=vote_time[5:]+cs_time[5:]+merge_other_time[5:], label='Merging: Cost pre-computation', color='g', hatch='+++', linewidth=0.01)
# plt.bar(pcd_idx, merge_opt_time[5:], width = 0.5, bottom=vote_time[5:]+cs_time[5:]+merge_other_time[5:]+merge_pre_time[5:], label='Merging: Optimization', color='g', hatch='///', linewidth=0.01)
# plt.bar(pcd_idx, merge_other_time[5:], width = 0.5, bottom=vote_time[5:]+cs_time[5:], label='Merging: Other', color='g')
# plt.legend()
# plt.savefig('{}/time_stage_c{}_s{}_l{}_man.png'.format(out_dir, args.c, args.s, args.l))





