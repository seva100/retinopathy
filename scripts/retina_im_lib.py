# -*- coding: utf-8 -*-

"""

@author: Artem S.
"""

import os
import glob
from math import sqrt
import numpy as np
import scipy as sp
import scipy.misc
import mahotas as mh
from numba import jit
from collections import deque
# imhandle - self-written module for general images processing.
from imhandle import rgb_to_grayscale, load_image

# --------------------------------------------------
# CROPPING
def _non_black_part(distr, ztol=3):
    '''receives vector distr - distribution of pixels over horizontal or vertical axis.
    Returns (a, b): 2 ints that mean that [a, b] part of distr corresponds to image
    and everything else corresponds to black region.
    '''
    a, b = 0, distr.shape[0] - 1
    while distr[a] < ztol and a < distr.shape[0]:
        a += 1
    while distr[b] < ztol and b >= 0:
        b -= 1
    return (a, b)


def crop_black_border(img):
    gs = rgb_to_grayscale(img)
    distr_hor = gs.mean(axis=0)
    distr_ver = gs.mean(axis=1)
    ver_crop = _non_black_part(distr_hor)
    hor_crop = _non_black_part(distr_ver)
    return img[hor_crop[0]:hor_crop[1], ver_crop[0]:ver_crop[1], :]


def crop_by_dist_from_center(img, frac_radius=1.0):
    radius = img.shape[0] / 2.0
    height = min(radius, radius * frac_radius / sqrt(2))    # half of side of square inside eye circle
    gap = int(radius - height)
    crop = img[gap:-gap - 1, gap:-gap - 1]
    return crop


def prepare_img(img):
    #img = sp.misc.imresize(img, 0.2)
    img = sp.misc.imresize(img, (518, 777))
    img = crop_black_border(img)
    gb = rgb_to_grayscale(img)
    gb = gb.astype(np.uint8)
    return gb


# --------------------------------------------------
# BFS clustering

#@jit
def expand(img, ans, row, col, cluster, radius=3, tol=2):
    #ans[row, col] = np.random.randint(0, 255)
    q = deque([(row, col)])
    while q:
        last = q.popleft()
        neighb_row = np.arange(max(last[0] - radius, 0),
                               min(last[0] + radius + 1, img.shape[0]))
        neighb_col = np.arange(max(last[1] - radius, 0),
                               min(last[1] + radius + 1, img.shape[1]))
        for nrow in neighb_row:
            for ncol in neighb_col:
                #print(nrow, ncol)
                if ans[nrow, ncol] == -1 and \
                    abs(img[nrow, ncol] - img[row, col]) < tol:
                    ans[nrow, ncol] = cluster
                    q.append((nrow, ncol))


def bfs_clustered(img, radius=3, tol=2):
    # TODO add mask parameter
    # ! Clustering may be faster if using DSU not BFS.
    label = np.full_like(img, -1, dtype=int)
    cluster = 0
    q = deque()
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            if label[i, j] == -1:
                label[i, j] = cluster
                #expand(img, label, i, j, cluster, radius, tol)
                q.clear()
                q.append((i, j))
                while q:
                    last = q.popleft()
                    neighb_row = np.arange(max(last[0] - radius, 0),
                                           min(last[0] + radius + 1, img.shape[0]))
                    neighb_col = np.arange(max(last[1] - radius, 0),
                                           min(last[1] + radius + 1, img.shape[1]))
                    for nrow in neighb_row:
                        for ncol in neighb_col:
                            #print(nrow, ncol)
                            if label[nrow, ncol] == -1 and \
                                abs(img[nrow, ncol] - img[i, j]) < tol:
                                label[nrow, ncol] = cluster
                                q.append((nrow, ncol))
                cluster += 1
    return label


def region_growing_1(gb, radius, tol1, tol2, tol3, edges=None, seeds=None):
    grad = mh.laplacian_2D(gb)
    label = np.full_like(gb, -1, dtype=int)
    cluster = 0
    q = deque()
    for i in xrange(gb.shape[0]):
        for j in xrange(gb.shape[1]):
            if label[i, j] == -1 and \
                (edges is None or not edges[i, j]) and \
                (seeds is None or seeds[i, j]):
                label[i, j] = cluster
                #expand(img, label, i, j, cluster, radius, tol)
                q.clear()
                start_pnt = (i, j)
                q.append(start_pnt)
                sum_color = gb[start_pnt]
                compn_size = 1
                while q:
                    last = q.popleft()
                    neighb_row = np.arange(max(last[0] - radius, 0),
                                           min(last[0] + radius + 1, gb.shape[0]))
                    neighb_col = np.arange(max(last[1] - radius, 0),
                                           min(last[1] + radius + 1, gb.shape[1]))
                    for nrow in neighb_row:
                        for ncol in neighb_col:
                            #print(nrow, ncol)
                            if label[nrow, ncol] == -1 and \
                                (edges is None or not edges[nrow, ncol]) and \
                                grad[nrow, ncol] < tol1 and \
                                abs(gb[nrow, ncol] * float(compn_size) - sum_color) < tol2 * float(compn_size) and \
                                abs(gb[nrow, ncol] - gb[start_pnt]) < tol3:
                                label[nrow, ncol] = cluster
                                q.append((nrow, ncol))
                                sum_color += gb[nrow, ncol]
                                compn_size += 1
                cluster += 1
    return label


def region_growing_2(gb, radius, edges, tol1, tol2, tol3):
    pass


@jit
def erase_clusters(segm, small_tol=4, large_tol=200, small_replace=-1):
    cl_num = np.zeros(segm.max() + 1)
    ans = segm.copy()
    for i in xrange(segm.shape[0]):
        for j in xrange(segm.shape[1]):
            cl_num[segm[i, j]] += 1
    for i in xrange(segm.shape[0]):
        for j in xrange(segm.shape[1]):
            if cl_num[segm[i, j]] < small_tol or cl_num[segm[i, j]] > large_tol:
                ans[i, j] = small_replace
    return ans


@jit
def cluster_sizes(segm):
    cl_num = np.zeros(segm.max() + 1)
    for i in xrange(segm.shape[0]):
        for j in xrange(segm.shape[1]):
            cl_num[segm[i, j]] += 1
    return cl_num


#@jit(nopython=True)
def get_clusters(segm, box_shape=10):
    # Another method:
    # 1. compute cluster centroids
    # 2. put them in rectangle
    # (it can be put in rectangle larger than itself by zero enlargement)

    #if segm.dtype == np.uint8:
    #    raise Exception('uint8 not supported')
    N = box_shape
    cl_num = segm.max() + 1
    cl_size = cluster_sizes(segm) 
    cl_box = np.empty((cl_num, N ** 2), dtype=segm.dtype)
    cl_bnd = np.full((cl_num, 4), -1, dtype=np.int64)    
    cl_no = []
    # top, bottom, left, right bounds
    for i in xrange(segm.shape[0]):
        for j in xrange(segm.shape[1]):
            if cl_size[segm[i, j]] != 0:
                if cl_bnd[segm[i, j], 0] == -1:
                    cl_bnd[segm[i, j], 0] = i
                if cl_bnd[segm[i, j], 2] == -1 or cl_bnd[segm[i, j], 2] > j:
                    cl_bnd[segm[i, j], 2] = j
                cl_bnd[segm[i, j], 1] = i
                if cl_bnd[segm[i, j], 3] == -1 or cl_bnd[segm[i, j], 3] < j:
                    cl_bnd[segm[i, j], 3] = j
    
    for cl in xrange(cl_num):
        if cl_size[cl] != 0:
            top, bottom, left, right = list(cl_bnd[cl])
            top -= 2; bottom += 2; left -= 2; right += 2
            box = segm[top:(bottom + 1), left:(right + 1)] == cl
            if box.shape[0] == box.shape[1]:
                box_enl = box
            elif box.shape[0] < box.shape[1]:
                box_enl = np.zeros((box.shape[1], box.shape[1]))
                box_enl[(box.shape[1] - box.shape[0]) / 2 : \
                    (box.shape[1] + box.shape[0]) / 2, :] = box
            else:
                box_enl = np.zeros((box.shape[0], box.shape[0]))
                box_enl[:, (box.shape[0] - box.shape[1]) / 2 :\
                    (box.shape[0] + box.shape[1]) / 2] = box
            if box_enl.shape[0] != box_shape:
                box_enl = scipy.misc.imresize(box_enl, size=(box_shape, box_shape),
                                              interp='nearest')
            box_enl_ravel = box_enl.reshape((box_enl.size,))
            cl_box[cl] = box_enl_ravel
            cl_no.append(cl)
    
    cl_box = cl_box[cl_size != 0, :]
    return (cl_box, cl_no)


def get_clusters_from_img_set(folder, box_shape=30):
    boxes = []
    for img_fn in sorted(glob.glob(os.path.join(folder, '*.jpeg'))):
        img = load_image(img_fn)
        gb = prepare_img(img)

        t = np.percentile(gb.ravel(), 99)
        bright = gb > t
        bright = (bright * 255).astype(int)

        cl1 = bfs_clustered(bright, radius=2, tol=1)
        cl2 = erase_clusters(cl1, small_tol=0.0004 * cl1.shape[0] * cl1.shape[1],
                             large_tol=2 * cl1.size, small_replace=0)
        
        cl_boxes, _ = get_clusters(cl2, box_shape=box_shape)
        cl_boxes = cl_boxes[1:]    # ignoring background
        cl_boxes = cl_boxes[cl_boxes.mean(axis=1) != 0.0]
        
        boxes.append(cl_boxes)
    
    boxes = np.array(boxes)
    boxes = np.vstack(boxes)
    return boxes


#@jit(nopython=True)
def leave_segments(segm, cl_no, replace_value=0):
    ans = segm.copy()
    for i in xrange(segm.shape[0]):
        for j in xrange(segm.shape[1]):
            # the following can be replaced with binsearch
            if segm[i, j] not in cl_no:    
                ans[i, j] = replace_value
    return ans

def leave_segments_by_mask(segm, cl_to_leave_mask, replace_value=0):
    ans = segm.copy()
    for i in xrange(segm.shape[0]):
        for j in xrange(segm.shape[1]):
            if segm[i, j] >= 0 and not cl_to_leave_mask[segm[i, j]]:    
                ans[i, j] = replace_value
    return ans