'''
Pixel-level features.

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
import imhandle as imh
import objects_detection as od
import cluster_features as cf


def pxls_mean_intensity(gb, stride_width=5, neighbd_rad=7, mask=None):
    mean_intensity = mh.mean_filter(gb, Bc=np.ones((neighbd_rad, neighbd_rad)))
    if mask is not None:
        return mean_intensity[mask]
    return mean_intensity[::stride_width, ::stride_width]


@jit
def _jit_pxls_std_intensity(gb, stride_width=5, neighbd_rad=3, mask=None):
    neighbd_size = 2 * neighbd_rad + 1
    if mask is None:
        ans = np.empty(gb[::stride_width, ::stride_width].size)
    else:
        ans = np.empty(mask.sum())
    k = 0
    for i in xrange(0, gb.shape[0], stride_width):
        for j in xrange(0, gb.shape[1], stride_width):
            if mask is None or mask[i, j]:
                neighbd = gb[(i - neighbd_rad + 1):(i + neighbd_rad), 
                             (j - neighbd_rad + 1):(j + neighbd_rad)]
                # padding neighborhood with zeros
                if neighbd.shape[0] < neighbd_size:
                    neighbd = np.vstack((neighbd, 
                                         np.zeros((neighbd_size - neighbd.shape[0], 
                                                   neighbd.shape[1]))))
                if neighbd.shape[1] < neighbd_size:
                    neighbd = np.hstack((neighbd,
                                         np.zeros((neighbd.shape[0], 
                                                   neighbd_size - neighbd.shape[1]))))
                ans[k] = neighbd.std()
                k += 1
    return ans


def pxls_std_intensity(gb, stride_width=5, neighbd_rad=3, mask=None):
    return _jit_pxls_std_intensity(gb, stride_width, neighbd_rad, mask)


@jit
def _jit_pxls_std_intensity(gb, stride_width=5, neighbd_rad=3, mask=None):
    neighbd_size = 2 * neighbd_rad + 1
    if mask is None:
        ans = np.empty(gb[::stride_width, ::stride_width].size)
    else:
        ans = np.empty(mask.sum())
    k = 0
    for i in xrange(0, gb.shape[0], stride_width):
        for j in xrange(0, gb.shape[1], stride_width):
            if mask is None or mask[i, j]:
                neighbd = gb[(i - neighbd_rad + 1):(i + neighbd_rad), 
                             (j - neighbd_rad + 1):(j + neighbd_rad)]
                # padding neighborhood with zeros
                if neighbd.shape[0] < neighbd_size:
                    neighbd = np.vstack((neighbd, 
                                         np.zeros((neighbd_size - neighbd.shape[0], 
                                                   neighbd.shape[1]))))
                if neighbd.shape[1] < neighbd_size:
                    neighbd = np.hstack((neighbd,
                                         np.zeros((neighbd.shape[0], 
                                                   neighbd_size - neighbd.shape[1]))))
                ans[k] = neighbd.std()
                k += 1
    return ans


@jit
def _jit_pxls_mean_intensity_of_masked(gb, mask, stride_width=5, neighbd_rad=3):
    ans = np.empty(gb.shape[0] * gb.shape[1], dtype=np.float64)
    k = 0
    for i in xrange(0, gb.shape[0], stride_width):
        for j in xrange(0, gb.shape[1], stride_width):
            neighbd = gb[(i - neighbd_rad + 1):(i + neighbd_rad), 
                         (j - neighbd_rad + 1):(j + neighbd_rad)]
            neighbd_mask = mask[(i - neighbd_rad + 1):(i + neighbd_rad), 
                                (j - neighbd_rad + 1):(j + neighbd_rad)]
            if neighbd_mask.sum() == 0:
                ans[k] = 1.0
            else:
                ans[k] = neighbd[neighbd_mask].mean()
            k += 1
    return ans


def pxls_mean_intensity_of_masked(gb, mask, stride_width=5, neighbd_rad=3):
    '''Returns mean intensity for neighborhood of each pixel accounting only
    those pixels where mask == 1. If there are no such pixels in 
    a neighborhood, 1.0 intensity is returned for that pixel.'''
    return _jit_pxls_mean_intensity_of_masked(gb, mask, stride_width=5, neighbd_rad=3)


def pxls_dist_to_pnt(gb_shape, pnt, stride_width=5, mask=None):
    i_dist_sq = (np.arange(0, gb_shape[0], stride_width) - pnt[0]) ** 2
    j_dist_sq = (np.arange(0, gb_shape[1], stride_width) - pnt[1]) ** 2
    grid = np.meshgrid(j_dist_sq, i_dist_sq)
    dist = np.sqrt(grid[0] + grid[1])
    if mask is not None:
        dist = dist[mask[::stride_width, ::stride_width]]
    return dist


@jit
def local_minima(gb, stride_width=5, neighbd_rad=4, mask=None):
    #neighbd_size = 2 * neighbd_rad + 1
    ans = np.zeros_like(gb, dtype=np.bool)
    for i in xrange(0, gb.shape[0], stride_width):
        for j in xrange(0, gb.shape[1], stride_width):
            if mask is None or mask[i, j]:
                bounds = (max(i - neighbd_rad + 1, 0),
                          min(i + neighbd_rad, gb.shape[0]),
                          max(j - neighbd_rad + 1, 0),
                          min(j + neighbd_rad, gb.shape[1]))
                neighbd = gb[bounds[0]:bounds[1], bounds[2]:bounds[3]]
                min_idx = np.unravel_index(np.argmin(neighbd), neighbd.shape)
                ans[bounds[0] + min_idx[0], bounds[2] + min_idx[1]] = 1
    return ans