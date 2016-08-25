'''
Clusters features.

Following [] Mingqiang Yang - "A Survey of Shape Feature Extraction Techniques".

@author: Artem Sevastopolsky
'''


import os
import glob
import cv2
import bisect
from operator import attrgetter
from collections import deque, defaultdict
from math import sqrt
import numpy as np
import scipy as sp
from scipy import ndimage as ndi
import pandas as pd
import sklearn
import sklearn.cluster, sklearn.preprocessing, sklearn.decomposition
import skimage
import skimage.morphology, skimage.filters, skimage.feature, \
    skimage.segmentation, skimage.color
import mahotas as mh
import matplotlib
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from numba import jit
# my libraries that deal with images and this data set specifically:
from imhandle import load_image, save_image, show_image, rmse, \
    normalize_image, load_set, rgb_to_grayscale, pxl_distr, plot_subfigures
from retina_im_lib import crop_black_border, prepare_img, expand, \
    bfs_clustered, erase_clusters, cluster_sizes, get_clusters


def cl_areas(labels):
    labels_new = labels + 1    # +1 added because cluster #0 is interpreted as background 
                               # by some skimage functions which is not useful here.
    areas = map(attrgetter('area'), skimage.measure.regionprops(labels_new))
    return np.array(areas)


def borders(labels):
    #lapl = mh.laplacian_2D(labels)
    #ans = np.where(lapl > 0, labels, -1)
    bnd = np.where(skimage.segmentation.find_boundaries(labels),
                   labels, 0)
    return bnd

@jit
def cl_perimeter(labels):
    cl_no = labels.max() + 1
    stats = np.zeros(cl_no)
    brd = borders(labels)
    for i in xrange(labels.shape[0]):
        for j in xrange(labels.shape[1]):
            cl = brd[i, j]
            if cl != -1:
                stats[cl] += 1
    return stats
    

def cl_circularity(labels):
    area = cl_areas(labels)
    perimeter = cl_perimeter(labels)
    circ = np.where(perimeter != 0, area / (perimeter ** 2), 0)
    return circ


@jit
def cl_centroids(labels):
    cl_no = labels.max() + 1
    pos_i = np.zeros(cl_no)
    pos_j = np.zeros(cl_no)
    for i in xrange(labels.shape[0]):
        for j in xrange(labels.shape[1]):
            cl = labels[i, j]
            if cl != -1:
                pos_i[cl] += i
                pos_j[cl] += j
    area = cl_areas(labels)
    centr_i = np.where(area != 0, pos_i / area, 0)
    centr_j = np.where(area != 0, pos_j / area, 0)
    return (centr_i, centr_j)


@jit
def cl_principal_axes(labels):
    cl_no = labels.max() + 1
    centr_i, centr_j = cl_centroids(labels)
    brd = borders(labels)
    per = cl_perimeter(labels)
    
    cxx = np.zeros(cl_no)
    cyy = np.zeros(cl_no)
    cxy = np.zeros(cl_no)
    
    for i in xrange(labels.shape[0]):
        for j in xrange(labels.shape[1]):
            cl = brd[i, j]
            if cl != -1:
                cxx[cl] += (i - centr_i[cl]) ** 2
                cxy[cl] += (i - centr_i[cl]) * (j - centr_j[cl])
                cyy[cl] += (j - centr_j[cl]) ** 2
    
    cxx /= np.where(per != 0, per, 1)
    cyy /= np.where(per != 0, per, 1)
    cxy /= np.where(per != 0, per, 1)
    
    l1 = 0.5 * (cxx + cyy + np.sqrt((cxx + cyy) ** 2 - \
                                    4 * (cxx * cyy - cxy ** 2)))
    l2 = 0.5 * (cxx + cyy - np.sqrt((cxx + cyy) ** 2 - \
                                    4 * (cxx * cyy - cxy ** 2)))
    return l1, l2


def cl_eccentricity(labels):
    eps = 1e-8
    
    pr_axes_1, pr_axes_2 = cl_principal_axes(labels)
    ecc = pr_axes_2 / (pr_axes_1 + eps)
    return ecc


@jit
def cl_mean_intensity(labels, gb):
    cl_no = labels.max() + 1
    stats = np.zeros(cl_no)
    area = cl_areas(labels)
    for i in xrange(labels.shape[0]):
        for j in xrange(labels.shape[1]):
            cl = labels[i, j]
            if cl != -1:
                stats[cl] += gb[i, j]
    stats /= np.where(area != 0, area, 1)
    return stats


def actual_clusters_no(segm):
    areas = cl_areas(segm)
    actual_clusters = np.arange(segm.max() + 1)[areas != 0]
    return actual_clusters