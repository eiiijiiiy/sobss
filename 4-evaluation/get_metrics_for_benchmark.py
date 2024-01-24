import numpy as np
import os
import matplotlib
matplotlib.rcParams.update({
    'font.family': 'times',
    'font.size': 10,
    'text.usetex': True,
})
import matplotlib.pyplot as plt
from matplotlib import cm
import openmesh as om

root_dir = '/Volumes/SSD/yijie_revision_exper'
out_dir = '{}/param_fig'.format(root_dir)
man = '3-select_primary_normal_manhattan_revision'

# get all bss results
m_vote_log = np.loadtxt('{}/{}/vs_0.250000.txt'.format(root_dir, man))
cs_log_dir = '{}/{}/sensitivity/coarse_segmentation/log'.format(root_dir, man)
merge_log_dir = '{}/{}/sensitivity/merge/log'.format(root_dir, man)
rmsd_dir = '{}/{}/sensitivity/merge/rmsd'.format(root_dir, man)
poly_dir = '{}/{}/sensitivity/merge/union'.format(root_dir, man)
time_dir = '{}/{}/sensitivity/merge/total_time'.format(root_dir, man)


cs = [0.5, 1, 2, 4]
sigmas = [0.25, 0.5, 1, 2, 4]
lambdas = [0.125, 0.25, 0.5, 1, 2, 4, 8]
num_cs = len(cs)
num_sigmas = len(sigmas)
num_lambdas = len(lambdas)
samples = [str(i+1) for i in range(10)]

bss_results = np.zeros((num_cs * num_sigmas * num_lambdas, 3)) # rmsd, num of faces, T
for ci in range(num_cs):
    c = cs[ci]
    cs_log_path = '{}/{:.1f}.txt'.format(cs_log_dir, c)
    cs_log = np.loadtxt(cs_log_path)
    for si in range(num_sigmas):
        s = sigmas[si]
        for li in range(num_lambdas):
            l = lambdas[li]
            merge_log_path = '{}/{:.1f}/sigma_{:.6f}_lambda_{:.6f}.txt'.format(
                merge_log_dir, c, s, l)
            merge_log = np.loadtxt(merge_log_path)
            rmsd_path = '{}/{:.1f}/sigma_{:.6f}_lambda_{:.6f}.txt'.format(
                rmsd_dir, c, s, l)
            rmsd = np.loadtxt(rmsd_path)
            p_poly_dir = '{}/{:.1f}/sigma_{:.6f}_lambda_{:.6f}'.format(
                poly_dir, c, s, l)
            bss_results[ci * (num_sigmas * num_lambdas) + si * num_lambdas + li, 0] \
                    = np.mean(rmsd[(rmsd[:,0]>0)*(rmsd[:,0]<100), 0])
            T = np.sum(m_vote_log, axis=1) + cs_log[:, 0] + np.sum(merge_log[:, :-3], axis=1)
            np.savetxt('{}/{:.1f}/sigma_{:.6f}_lambda_{:.6f}.txt'.format(time_dir, c, s, l), T.transpose())
            bss_results[ci * (num_sigmas * num_lambdas) + si * num_lambdas + li, 2] \
                    = np.mean(T[(rmsd[:,0]>0)*(rmsd[:,0]<100)])
        
            total_face_num = 0
            valid_count = 0
            for i in samples:
                poly_path = '{}/{}.obj'.format(
                        p_poly_dir, i)
                poly = om.read_polymesh(poly_path)
                if poly.n_faces() == 0: continue
                total_face_num += poly.n_faces()
                valid_count += 1
            bss_results[ci * (num_sigmas * num_lambdas) + si * num_lambdas + li, 1] \
                    = total_face_num / valid_count

# get all Polyfit results
sps = [5000, 10000] # minimum supported points
params=['(0.33, 0.33, 0.33)', '(0.2, 0.2, 0.6)', '(0.1, 0.1, 0.8)', '(0.05, 0.05, 0.9)'] # too large rmsd (36+) for (0.005, 0.05, 0.9)
params=['(0.33, 0.33, 0.33)', '(0.2, 0.2, 0.6)', '(0.1, 0.1, 0.8)']

num_sps = len(sps)
num_params = len(params)

poly_dir = '{}/{}/comparison/Polyfit/poly'.format(root_dir, man)
rmsd_dir = '{}/{}/comparison/Polyfit/rmsd'.format(root_dir, man)
log_dir = '{}/{}/comparison/Polyfit'.format(root_dir, man)
samples = [str(i+1) for i in range(10)]

polyfit_results = np.zeros((num_sps * num_params, 3)) # rmsd, num of faces, T
for si in range(num_sps):
    s = sps[si]
    log_path = '{}/{}.txt'.format(log_dir, s)
    log = np.loadtxt(log_path)
    p_poly_dir = '{}/{}'.format(poly_dir, s)
    for pi in range(num_params):
        p = params[pi]
        print('polyfit : support points: {} params: {}'.format(s, p))
        rmsd_path = '{}/{}/{}.txt'.format(rmsd_dir, s, p)
        rmsd = np.loadtxt(rmsd_path)
        total_face_num = 0
        valid_count = 0
        polyfit_results[si * num_params + pi, 0] = np.mean(
            rmsd[(rmsd[:,0]>0)*(rmsd[:,0]<100), 0])
        T = np.sum(log[:, :3], axis=1) + log[:, 3+pi]
        polyfit_results[si * num_params + pi, 2] = np.mean(
            T[(rmsd[:,0]>0)*(rmsd[:,0]<100)])
        for i in samples:
            poly_path = '{}/{}{}.obj'.format(p_poly_dir, i, p)
            if not os.path.exists(poly_path): continue
            poly = om.read_polymesh(poly_path)
            # from IPython import embed
            # embed()
            if poly.n_faces() == 0: continue
            total_face_num += poly.n_faces()
            valid_count += 1
        polyfit_results[si * num_params + pi, 1] = total_face_num / valid_count

# get all ManhattanBox results
params = [12.5, 100, 800]
num_params = len(params)
num_params = len(params)
poly_dir = '{}/{}/comparison/Manbox/poly'.format(root_dir, man)
rmsd_dir = '{}/{}/comparison/Manbox/rmsd'.format(root_dir, man)
log_dir = '{}/{}/comparison/Manbox/log'.format(root_dir, man)
samples = [str(i+1) for i in range(10)]

manbox_results = np.zeros((num_params, 3)) # rmsd, num of faces, T
for pi in range(num_params):
    p = params[pi]
    p_log_path = '{}/5000_{}.txt'.format(log_dir, p)
    log = np.loadtxt(p_log_path)
    T = np.sum(log, axis=1)
    p_rmsd_path = '{}/{}.txt'.format(rmsd_dir, p)
    rmsd = np.loadtxt(p_rmsd_path)
    total_face_num = 0
    valid_count = 0
    manbox_results[pi, 0] = np.mean(rmsd[(rmsd[:,0]>0)*(rmsd[:,0]<100), 0])
    manbox_results[pi, 2] = np.mean(T[(rmsd[:,0]>0)*(rmsd[:,0]<100)])
    for i in samples:
        poly_path = '{}/{}(5000, {}).obj'.format(poly_dir, i, p)
        poly = om.read_polymesh(poly_path)
        if poly.n_faces() == 0: continue
        total_face_num += poly.n_faces()
        valid_count += 1
    manbox_results[pi, 1] = total_face_num / valid_count

# get all 2.5D dualing results
poly_dir = '{}/{}/comparison/25D_Dual/obj_1m_1'.format(root_dir, man)
rmsd_path = '{}/{}/comparison/25D_Dual/rmsd.txt'.format(root_dir, man)
log_path = '{}/{}/comparison/25D_Dual/log_1m_1.txt'.format(root_dir, man)
samples = [str(i+1) for i in range(10)]
rmsd = np.loadtxt(rmsd_path)
log = np.loadtxt(log_path).reshape((len(samples)))

dual_results = np.zeros((1, 3))
dual_results[0, 0] = np.mean(rmsd[(rmsd[:,0]>0)*(rmsd[:,0]<100), 0])
dual_results[0, 2] = np.mean(log[(rmsd[:,0]>0)*(rmsd[:,0]<100)])

total_face_num = 0
valid_count = 0
for i in samples:
    poly_path = '{}/{}.obj'.format(poly_dir, i)
    poly = om.read_polymesh(poly_path)
    if poly.n_faces() == 0: continue
    total_face_num += poly.n_faces()
    valid_count += 1
dual_results[0, 1] = total_face_num / valid_count

# get Meshplg results
poly_dir = '{}/{}/comparison/Meshplg/poly'.format(root_dir, man)
rmsd_path = '{}/{}/comparison/Meshplg/rmsd.txt'.format(root_dir, man)
rmsd = np.loadtxt(rmsd_path)
log_path = '{}/{}/comparison/Meshplg/log.txt'.format(root_dir, man)
log = np.loadtxt(log_path)
T = np.sum(log, axis=1)
samples = [str(i+1) for i in range(10)]

meshplg_results = np.zeros((1, 3))
total_face_num = 0
valid_count = 0
meshplg_results[0, 0] = np.mean(rmsd[(rmsd[:,0]>0)*(rmsd[:,0]<100), 0])
meshplg_results[0, 2] = np.mean(T[(rmsd[:,0]>0)*(rmsd[:,0]<100)])
for i in samples:
    poly_path = "{}/{}.off-result.ply".format(poly_dir, i)
    if not os.path.exists(poly_path): continue
    poly = om.read_polymesh(poly_path)
    if poly.n_faces() == 0: continue
    total_face_num += poly.n_faces()
    valid_count += 1
meshplg_results[0, 1] = total_face_num / valid_count

# save data as txt for Veusz 3D Plot
np.savetxt('{}/bss_results.txt'.format(out_dir), bss_results)
np.savetxt('{}/polyfit_results.txt'.format(out_dir), polyfit_results)
np.savetxt('{}/manbox_results.txt'.format(out_dir), manbox_results)
np.savetxt('{}/dual_results.txt'.format(out_dir), dual_results)
np.savetxt('{}/meshplg_results.txt'.format(out_dir), meshplg_results)

# Plot All
# fig = plt.figure()
# ax = plt.axes(projection = '3d')
# stack = np.vstack((bss_results, polyfit_results, manbox_results, dual_results, meshplg_results))
# ax.scatter3D(bss_results[:, 1], bss_results[:, 2],  bss_results[:, 0], color = 'red', label='BSS')
# ax.scatter3D(polyfit_results[:, 1], polyfit_results[:, 2], polyfit_results[:, 0], color = 'blue', label='PolyFit')
# ax.scatter3D(manbox_results[:, 1], manbox_results[:, 2], manbox_results[:, 0], color = 'green', label='ManBox')
# ax.scatter3D(dual_results[:, 1], dual_results[:, 2], dual_results[:, 0], color = 'black', label='DualCont')
# ax.scatter3D(meshplg_results[:, 1], meshplg_results[:, 2], meshplg_results[:, 0], color = 'cyan', label='MeshPlg')
# # ax.set_xlim(0, np.max(stack[:, 2]))
# # ax.set_ylim(0, np.max(stack[:, 1]))
# ax.zaxis._axinfo['juggled'] = (1, 2, 0)
# ax.legend()
# plt.tight_layout()
# plt.savefig('{}/all_benchmarks.png'.format(out_dir), dpi=600, bbox_inches='tight')

# plt.clf()
# ax = plt.axes(projection = '3d')
# stack = np.vstack((bss_results, polyfit_results))
# ax.scatter3D(bss_results[:, 1], bss_results[:, 2], bss_results[:, 0], color = 'red', label='BSS')
# ax.scatter3D(polyfit_results[:, 1], polyfit_results[:, 2],  polyfit_results[:, 0], color = 'blue', label='PolyFit')
# # ax.scatter3D(manbox_results[:, 2], manbox_results[:, 1], manbox_results[:, 0], color = 'green')
# # ax.scatter3D(dual_results[:, 2], dual_results[:, 1], dual_results[:, 0], color = 'black')
# # ax.scatter3D(meshplg_results[:, 2], meshplg_results[:, 1], meshplg_results[:, 0], color = 'cyan')
# # ax.set_xlim(0, np.max(stack[:, 2]))
# # ax.set_ylim(0, np.max(stack[:, 1]))
# # ax.set_zlim(0, np.max(stack[:, 0])+1)
# ax.zaxis._axinfo['juggled'] = (1, 2, 0)
# plt.tight_layout()
# plt.savefig('{}/all_benchmarks_spot_bss_polyfit.png'.format(out_dir), dpi=600, bbox_inches='tight')

