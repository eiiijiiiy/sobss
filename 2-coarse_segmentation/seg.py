import readline
import cv2
import numpy as np
from tqdm import tqdm
import open3d as o3d
import time
import argparse
import os
import util


# fill_ker_size_dict = {0.5: 5, 1: 3, 2: 1}
# rm_out_ker_size_dict = {0.5: 3, 1: 1, 2: 0}

fill_ker_size_dict = {0.5: 7, 1: 5, 2: 3, 4: 1}
rm_out_ker_size_dict = {0.5: 5, 1: 3, 2: 1, 4: 1}
pixel_zero_filter = {0.5: 2, 1:8, 2: 32, 4: 128}


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


def coarse_segment(options):
    pcd_dir = options['pcd_dir']
    out_dir = options['out_dir']
    param_out_dir = options['param_out_dir']
    group_dir = options['group_dir'] if 'group_dir' in options else None
    log_path = options['log_path']
    ps = options['ps'] if 'ps' in options else 2
    xi = options['xi'] if 'xi' in options else 2
    dxi = options['dxi']/2 if 'dxi' in options else 1
    zci = options['zci'] if 'zci' in options else 2
    nmi = options['nmi'] if 'nmi' in options else np.pi/18
    box_pts_dir = options['box_pts_dir'] if 'box_pts_dir' in options else ""
    roof_pts_dir = options['roof_pts_dir'] if 'roof_pts_dir' in options else ""
    names = options['names'] if 'names' in options else os.listdir(pcd_dir)
    names = [int(n.split('.txt')[0]) for n in names if '.txt' in n]
    names.sort()
    
    print(pcd_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(param_out_dir):
        os.mkdir(param_out_dir)
    # names = ['{}.txt'.format(i+1) for i in range(10)]
    # out_names = [str(i+1) for i in range(10)]

    # names = [n[:-4] for n in names]

    sta = np.zeros((len(names), 2))
    for n in names:
        this_sample_box_pt_dir = '{}/{}'.format(box_pts_dir, n)
        this_sample_roof_pt_dir = '{}/{}'.format(roof_pts_dir, n)
        rect_count = 0
        other_solid_count = 0
        vote_count = 0
        boxes = []
        other_solids = []
        volumes = []
        pcd_path = "{}/{}.txt".format(pcd_dir, n)
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

        E_Y = np.max(STEPY) + 1
        E_DIM = np.max(STEPDIM) + 1
        E_ZC = np.max(STEPZC) + 1
        E_N = np.max(STEPN) + 1
        t_start = time.time()
        filtered_box_num = 0
        group_pt_sta = []
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
                    if rects.shape[0] > 0:
                        group_pt_sta.append(bin_pcd.shape[0])
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
        if group_dir:
            group_pt_sta = np.asarray(group_pt_sta)
            np.savetxt("{}/{}.txt".format(group_dir, n), group_pt_sta, fmt="%i")
        sta[n-1, 0] = t_end - t_start
        sta[n-1, 1] = filtered_box_num
        # fill holes
        # speed test, comment the output
        boxes = np.asarray(boxes)
        print("rect count {} im pt count {} pcd point num {} ".format(rect_count, vote_count, pcd.shape[0]))
        np.savetxt("{}/{}_b.txt".format(param_out_dir, n), boxes, fmt = "%.3f")
        
        util.create_vol_mesh_from_params(boxes[:, :-1], 
            "{}/{}_b.obj".format(out_dir, n))
        other_solids = np.asarray(other_solids).reshape((-1, 8))
        np.savetxt("{}/{}_os.txt".format(param_out_dir, n), other_solids, fmt = "%.3f")
        util.create_vol_mesh_from_params(
            other_solids[:, :-1], "{}/{}_os.obj".format(out_dir, n))
        
        volumes = np.vstack((boxes, other_solids))
        np.savetxt("{}/{}.txt".format(param_out_dir, n), volumes, fmt = "%.3f")
    
    np.savetxt(log_path, sta, fmt="%.2f")



