import readline
import cv2
import numpy as np
from tqdm import tqdm
import open3d as o3d
import time
import argparse
import os
import json

fill_ker_size_dict = {0.5: 7, 1: 5, 2: 3, 4: 1}
rm_out_ker_size_dict = {0.5: 5, 1: 3, 2: 1, 4: 1}
pixel_zero_filter = {0.5: 2, 1:8, 2: 32, 4: 128}


def create_vol_mesh_from_params(volumes, output_mesh_path, names=None):
	num_vol = volumes.shape[0]
	# parameters of a volume
	# 0 x, 1 y, 2 z, 3 width, 4 height (of front/back face), 5 radius, 6 length of the horizontal normals (default normals point to the (0,1,0))
	# 0 FLT, 1 BLT, 2 FRT, 3 BRT, 4 FRB, 5 BRB, 6 FLB, 7 BLB
	vol_coords = np.zeros((num_vol, 8, 3))
	NV = np.sqrt(1 - volumes[:, -1] ** 2)

	Z_TOP = volumes[:, 2] + volumes[:, 5] * \
	    NV + volumes[:, 4] * volumes[:, -1] / 2
	Z_BTM = volumes[:, 2] + volumes[:, 5] * \
	    NV - volumes[:, 4] * volumes[:, -1] / 2

	Y_FRONT_TOP = volumes[:, 1] + volumes[:, 5] * \
	    volumes[:, -1] - volumes[:, 4] / 2 * NV
	Y_FRONT_BTM = volumes[:, 1] + volumes[:, 5] * \
	    volumes[:, -1] + volumes[:, 4] / 2 * NV
	Y_BACK_TOP = volumes[:, 1] - volumes[:, 5] * \
	    volumes[:, -1] + volumes[:, 4] / 2 * NV
	Y_BACK_BTM = volumes[:, 1] - volumes[:, 5] * \
	    volumes[:, -1] - volumes[:, 4] / 2 * NV

	X_RIGHT = volumes[:, 0] + volumes[:, 3] / 2
	X_LEFT = volumes[:, 0] - volumes[:, 3] / 2

	vol_coords[:, 0, :] = np.vstack(
	    [X_LEFT, Y_FRONT_TOP, Z_TOP]).transpose()  # FLT
	vol_coords[:, 1, :] = np.vstack(
	    [X_RIGHT, Y_FRONT_TOP, Z_TOP]).transpose()  # FRT
	vol_coords[:, 2, :] = np.vstack(
	    [X_LEFT, Y_FRONT_BTM, Z_BTM]).transpose()  # FLB
	vol_coords[:, 3, :] = np.vstack(
	    [X_RIGHT, Y_FRONT_BTM, Z_BTM]).transpose()  # FRB

	vol_coords[:, 4, :] = np.vstack(
	    [X_LEFT, Y_BACK_TOP, Z_TOP]).transpose()  # BLT
	vol_coords[:, 5, :] = np.vstack(
	    [X_RIGHT, Y_BACK_TOP, Z_TOP]).transpose()  # BRT
	vol_coords[:, 6, :] = np.vstack(
	    [X_LEFT, Y_BACK_BTM, Z_BTM]).transpose()  # BLB
	vol_coords[:, 7, :] = np.vstack(
	    [X_RIGHT, Y_BACK_BTM, Z_BTM]).transpose()  # BRB

	objf = open(output_mesh_path, 'w')
	objf.write('Ka 1.000000 1.000000 1.000000\n')
	objf.write('Kd 1.000000 1.000000 1.000000\n')
	objf.write('Ks 0.000000 0.000000 0.000000\n')
	objf.write('Tr 1.000000\n')
	objf.write('illum 1\n')
	objf.write('Ns 0.000000 1\n')

	# write vertices
	for i in range(num_vol):
		for j in range(8):
			objf.write("v {:.3f} {:.3f} {:.3f} \n".format(
				vol_coords[i, j, 0],
				vol_coords[i, j, 1],
				vol_coords[i, j, 2]))
	
	# write faces
	if names is None:
		names = [str(i+1) for i in range(num_vol)]

	for i in range(num_vol):
		bi = i * 8 + 1  # !! vertex idx should begin from 1 in an obj
		objf.write("o solid {} \n".format(names[i]))
        # 0 FLT, 1 FRT, 2 FLB, 3 FRB, 4 BLT, 5 BRT, 6 BLB, 7 BRB

        # top: FLT -> BLT -> BRT -> FRT 
		objf.write("f {} {} {} {} \n".format(bi, bi+4, bi+5, bi+1))
        # bottom: FLB ->FRB -> BRB -> BLB
		objf.write("f {} {} {} {} \n".format(bi+2, bi+3, bi+7, bi+6))
        # front: FLT -> FRT -> FRB -> FLB
		objf.write("f {} {} {} {} \n".format(bi, bi+1, bi+3, bi+2))
        # back: BLT -> BLB -> BRB -> BRT
		objf.write("f {} {} {} {} \n".format(bi+4, bi+6, bi+7, bi+5))
        # left: FLT -> FLB -> BLB -> BLT
		objf.write("f {} {} {} {} \n".format(bi, bi+2, bi+6, bi+4))
        # right: FRT -> BRT -> BRB -> FRB
		objf.write("f {} {} {} {} \n".format(bi+1, bi+5, bi+7, bi+3))

	objf.close()
     

def rmse_value(input, interval):
    print(np.unique(input, return_counts=True))
    num_input = input.shape[0]
    input = np.arccos(input)/np.pi * 180
    min = np.min(input)
    max = np.max(input)
    if max - min == 0: return np.cos(min / 180 * np.pi)
    num_bin = np.ceil((max - min)/interval).astype(int)
    bin_value = min + interval * np.arange(num_bin)
    E = np.repeat(input.reshape(num_input, 1), num_bin, axis=1) \
        - np.repeat(bin_value.reshape(1, num_bin), num_input, axis=0)
    SE = E ** 2
    RMSE = np.sqrt(np.mean(SE, axis=0))
    # from IPython import embed
    # embed()
    min_idx_RMSE = np.argmin(RMSE)
    return np.cos(bin_value[min_idx_RMSE] / 180 * np.pi)


def B_divide(im, grad_h, grad_w, c_t, h_b, w_b):
    h = im.shape[0]
    w = im.shape[1]
    # print("h {} w {}".format(h, w))
    count = np.nonzero(im)[0].shape[0]
    zero_count =  (h * w) - count
    c =  count / (h * w)
    if (zero_count <= 5) & (c >= c_t):
        # print(c)
        return [[h_b, h_b + h, w_b, w_b + w]]
    if c > 0:

        if (h > 1) & (w > 1):
            h_grad = np.sum(grad_h, axis=1)
            w_grad = np.sum(grad_w, axis=0)
            max_h = np.max(h_grad)
            max_w = np.max(w_grad)
            max_hid = np.argmax(h_grad)
            max_wid = np.argmax(w_grad)
            # print("max h {} max w {} max hid {} max wid {}".format(max_h, max_w, max_hid, max_wid))
            if max_h > max_w:
                if max_hid == 0: max_hid=1
                rects = B_divide(im[0:max_hid, :], grad_h[0:max_hid, :], grad_w[0:max_hid, :], c_t, h_b, w_b) \
                        + B_divide(im[max_hid:, :], grad_h[max_hid:, :], grad_w[max_hid:, :], c_t, h_b+max_hid, w_b)
                return rects

            elif max_w > max_h:
                if max_wid == 0: max_wid = 1
                rects = B_divide(im[:, 0:max_wid], grad_h[:, 0:max_wid], grad_w[:, 0:max_wid], c_t, h_b, w_b)\
                    + B_divide(im[:, max_wid:], grad_h[:, max_wid:], grad_w[:, max_wid:], c_t, h_b, w_b+max_wid)
                return rects

            else:
                if w>=h:
                    if max_wid == 0: max_wid=1
                    rects = B_divide(im[:, 0:max_wid], grad_h[:, 0:max_wid], grad_w[:, 0:max_wid], c_t, h_b, w_b)\
                        + B_divide(im[:, max_wid:], grad_h[:, max_wid:], grad_w[:, max_wid:], c_t, h_b, w_b+max_wid)
                    return rects
                if h>w:
                    if max_hid == 0: max_hid=1
                    rects = B_divide(im[0:max_hid, :], grad_h[0:max_hid, :], grad_w[0:max_hid, :], c_t, h_b, w_b) \
                        + B_divide(im[max_hid:, :], grad_h[max_hid:, :], grad_w[max_hid:, :], c_t, h_b+max_hid, w_b)
                    return rects

        elif w == 1:
            h_grad = np.sum(grad_h, axis=1)
            max_hid = np.argmax(h_grad)
            # print("max h {}  max hid {} ".format(h_grad[max_hid], max_hid))
            if max_hid == 0: max_hid=1
            rects = B_divide(im[0:max_hid, :], grad_h[0:max_hid, :], grad_w[0:max_hid, :], c_t, h_b, w_b) \
                + B_divide(im[max_hid:, :], grad_h[max_hid:, :], grad_w[max_hid:, :], c_t, h_b+max_hid, w_b)
            return rects
        else:
            w_grad = np.sum(grad_w, axis=0)
            max_wid = np.argmax(w_grad)
            # print("max w {}  max wid {} ".format(w_grad[max_wid], max_wid))
            if max_wid == 0: max_wid=1
            rects = B_divide(im[:, 0:max_wid], grad_h[:, 0:max_wid], grad_w[:, 0:max_wid], c_t, h_b, w_b)\
                + B_divide(im[:, max_wid:], grad_h[:, max_wid:], grad_w[:, max_wid:], c_t, h_b, w_b+max_wid)
            return rects
    else:
        return []


def coarse_segment(working_folder):
    pcd_path = os.path.join(working_folder, "bss_atom.txt")
    conf_path = os.path.join(working_folder, "config.json")
    options = json.load(open(conf_path))
    
    ps = options['ps'] if 'ps' in options else 2
    xi = options['xi'] if 'xi' in options else 2
    dxi = options['dxi']/2 if 'dxi' in options else 1
    zci = options['zci'] if 'zci' in options else 2
    nmi = options['nmi'] if 'nmi' in options else np.pi/18

    rect_count = 0
    vote_count = 0
    boxes = []
    other_solids = []
    volumes = []
    pcd = np.loadtxt(pcd_path)
    pixel_size = ps
    MINY = np.min(pcd[:,1])
    MINDIM = np.min(pcd[:,3])

    tempzc = pcd[:, 5].copy()
    MINZC = np.min(tempzc)
    check_idx = np.where((tempzc > 1000) * (tempzc < 9999.0))[0]
    assert check_idx.shape[0] == 0
    tempzc[tempzc > 1000] = 1003
    normals = np.arccos(pcd[:, 4]) 
    normals[normals < np.pi / 6] = 0
    MINNM = np.min(normals)

    STEPY = ((pcd[:, 1] - MINY) / xi).astype(int)
    STEPDIM = ((pcd[:, 3] - MINDIM) / dxi).astype(int)
    STEPZC = ((tempzc - MINZC) / zci).astype(int)
    STEPN = ((normals - MINNM) / nmi).astype(int)

    t_start = time.time()
    filtered_box_num = 0
    # group_pt_sta = []
    for nid in tqdm(np.unique(STEPN)):
        if nid == 0:
            STEP_PARAM_UNIQUE = np.unique(STEPDIM)
        else:
            STEP_PARAM_UNIQUE = np.unique(STEPZC)
        for yid in tqdm(np.unique(STEPY)):
            for pid in tqdm(STEP_PARAM_UNIQUE):
                if nid == 0:
                    bin_pcd_idx = np.where((STEPN == nid) * (STEPY == yid) * (STEPDIM == pid))
                else:
                    bin_pcd_idx = np.where((STEPN == nid) * (STEPY == yid) * (STEPZC == pid))
                if bin_pcd_idx[0].shape[0] == 0:
                    continue
                bin_pcd = pcd[bin_pcd_idx[0]]
                max_x, min_x = np.max(bin_pcd[:, 0]), np.min(bin_pcd[:, 0])
                max_z, min_z = np.max(bin_pcd[:, 2]), np.min(bin_pcd[:, 2])

                Y = np.mean(bin_pcd[:, 1])
                pixel_size_w = pixel_size
                if nid > 0:
                    if nid > 1:
                        prev_nid = nid - 1
                        next_nid = nid + 1
                        prev_nid_bin_pcd_idx = np.where((STEPN == prev_nid) * (STEPY == yid) * (STEPZC == pid))
                        next_nid_bin_pcd_idx = np.where((STEPN == next_nid) * (STEPY == yid) * (STEPZC == pid))
                        prev_bin_pcd = pcd[prev_nid_bin_pcd_idx[0]]
                        next_bin_pcd = pcd[next_nid_bin_pcd_idx[0]]
                        # COS = np.mean(np.hstack((prev_bin_pcd[:, 4], bin_pcd[:, 4], next_bin_pcd[:, 4])))
                        COS = rmse_value(np.hstack((prev_bin_pcd[:, 4], bin_pcd[:, 4], next_bin_pcd[:, 4])), 1)
                        print (len(prev_bin_pcd[:, 4]), len(bin_pcd[:, 4]), len(next_bin_pcd[:, 4]))
                    else:
                        # COS = np.mean(bin_pcd[:, 4])
                        COS = rmse_value(bin_pcd[:, 4], 1)
                    pixel_size_h = pixel_size / COS
                else:
                    pixel_size_h = pixel_size
                
                im_w = (np.ceil((max_x - min_x) / pixel_size_w) + 1).astype(int)
                im_h = (np.ceil((max_z - min_z) / pixel_size_h) + 1).astype(int)
                im = np.zeros((im_h, im_w))
                bin_pcd_h = np.floor((max_z - bin_pcd[:,2])/pixel_size_h).astype(int)
                bin_pcd_w = np.floor((bin_pcd[:,0] - min_x)/pixel_size_w).astype(int)
                for idx in range(bin_pcd_h.shape[0]):
                    im[bin_pcd_h[idx], bin_pcd_w[idx]] += 1.0
                im_ = im.copy()
                vote_count += np.nonzero(im)[0].shape[0]
                im_count = im.copy()
                im[im<pixel_zero_filter[pixel_size]] = 0
                im[im>0] = 1
                # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,1))
                # im = cv2.dilate(im,kernel)
                # im = cv2.erode(im,kernel)
                # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,7))
                # im = cv2.dilate(im, kernel)
                # im = cv2.erode(im, kernel)
                remove_outliners_ks = rm_out_ker_size_dict[pixel_size]
                if remove_outliners_ks > 0:
                    kernel = np.ones((remove_outliners_ks, remove_outliners_ks), np.uint8)
                    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel) # remove the outlines
                fill_ker_size = fill_ker_size_dict[pixel_size]
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(fill_ker_size,1))
                im = cv2.dilate(im,kernel)
                im = cv2.erode(im,kernel)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,fill_ker_size))
                im = cv2.dilate(im,kernel)
                im = cv2.erode(im,kernel)
                # im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel) # close the holes
                # im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel) # remove the outlines

                grad_h = np.zeros((im_h, im_w))
                grad_w = np.zeros((im_h, im_w))
                grad_h[1:, :] = np.abs(im[1:, :] - im[:-1, :])
                grad_w[:, 1:] = np.abs(im[:, 1:] - im[:, :-1])
                rects = B_divide(im, grad_h, grad_w, 0.5, 0, 0)
                rects = np.asarray(rects)
                # if rects.shape[0] > 0:
                #     group_pt_sta.append(bin_pcd.shape[0])
                for rect in rects:
                    pt_count = np.sum(im_count[rect[0]:rect[1], rect[2]:rect[3]])

                    this_rect_dim_id = np.where(
                        (bin_pcd_h >= rect[0]) *
                        (bin_pcd_h < rect[1]) *
                        (bin_pcd_w >= rect[2]) *
                        (bin_pcd_w < rect[3]))[0]
                    c = np.sum(im[rect[0]:rect[1], rect[2]:rect[3]]) / ((rect[1] - rect[0]) * (rect[3] - rect[2]))

                    if len(this_rect_dim_id) == 0:
                        continue
                    
                    b = np.min(bin_pcd[this_rect_dim_id, 2]) - 0.25 / 2
                    t = np.max(bin_pcd[this_rect_dim_id, 2]) + 0.25 / 2
                    l = np.min(bin_pcd[this_rect_dim_id, 0]) - 0.25 / 2
                    r = np.max(bin_pcd[this_rect_dim_id, 0]) + 0.25 / 2
                    
                    if r - l == 0  or t - b == 0:
                        continue
                    
                    x = (l + r) / 2
                    z = (t + b) / 2
                    if nid == 0:
                        dim = np.mean(bin_pcd[:, 3])
                        if dim/((r-l)*(t-b)) >= 10 or pt_count/((r-l)*(t-b)) < 10 or dim/(r-l) > 20:
                            filtered_box_num += 1
                            continue
                    else:
                        # COS = np.mean(bin_pcd[:, 4])
                        SIN = np.sqrt(1 - COS ** 2)
                        TAN = SIN / COS
                        zc = bin_pcd[:, 5]
                        ZC = np.mean(zc)
                        if ZC < t:
                            ZC = t
                        if ZC < z: # too vertical
                            continue
                        radius = (ZC - z) * SIN
                        if radius/((r-l)*(t-b)) >= 10 or pt_count/((r-l)*(t-b)*COS) < 10 or radius/(r-l) > 20:
                            filtered_box_num += 1
                            continue
                    if nid == 0:
                        this_param = [x, Y, z, r-l, t-b, dim, 1, pt_count]
                        boxes.append(this_param)
                    else:
                        this_param = [x, Y, z, r-l, (t-b) * COS, radius, COS, pt_count]
                        other_solids.append(this_param)
                
           
        t_end = time.time()
        # print("rect_0 n {} time {}".format(n, t_end-t_start))
        # if group_dir:
        #     group_pt_sta = np.asarray(group_pt_sta)
        #     np.savetxt("{}/{}.txt".format(group_dir, n), group_pt_sta, fmt="%i")
        print("rect count {} im pt count {} pcd point num {} ".format(rect_count, vote_count, pcd.shape[0]))

    # fill holes
    # speed test, comment the output
    boxes = np.asarray(boxes).reshape((-1, 8))
    other_solids = np.asarray(other_solids).reshape((-1, 8))
    volumes = np.vstack((boxes, other_solids))

    volume_mesh_path = os.path.join(working_folder, "bss_coarse_segm.obj")
    create_vol_mesh_from_params(
        volumes[:, :-1], volume_mesh_path)
    
    volume_param_path = os.path.join(working_folder, "bss_coarse_segm.txt")
    np.savetxt(volume_param_path, volumes[:, :-1], fmt = "%.3f")
    