# -*- coding: utf-8 -*-

import os
import glob
import cv2
import bisect
from math import sqrt
import numpy as np
import pandas as pd
import sklearn
import sklearn.cluster
import sklearn.preprocessing
import skimage
import skimage.segmentation
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
from joblib import Parallel, delayed
from numba import jit
# my library that deals with images and this data set specifically:
from imhandle import load_image, save_image, show_image, rmse, load_set


@jit
def near_background(mask):
    segments = []
    neighbors = 8
    background = 0
    di = np.array([-1, 0, 1, 1, 1, 0, -1, -1])
    dj = np.array([1, 1, 1, 0, -1, -1, -1, 0])
    width, height = mask.shape

    for i in xrange(width):
        for j in xrange(height):
            if mask[i, j] == background:
                continue
            for k in xrange(neighbors):
                new_i = i + di[k]
                new_j = j + dj[k]
                if 0 <= new_i < width and 0 <= new_j < height and mask[new_i, new_j] == background:
                    segments.append(mask[i, j])
                    break
    return np.unique(np.array(segments))


def blank_by_edges_and_fzw(img, img_orig):
    edges = skimage.segmentation.boundaries.find_boundaries(img).astype(float)
    #show_image(edges, fig_size=(20, 20))
    edges_fzw = skimage.segmentation.felzenszwalb(edges, scale=1000, sigma=0.15, min_size=10)
    edges_fzw_max = edges_fzw.max()
    edges_fzw_img = edges_fzw / float(edges_fzw_max)
    #show_image(edges_fzw_img)

    '''
    # leaving only segments that lie on boundary
    segments_to_leave = edges_fzw[edges.astype(bool)]
    '''
    # leaving only segments that lie near background
    segments_to_leave_1 = near_background(edges_fzw)
    
    # Leaving only segments that have mean color very different from letters mean color
    selected = np.full_like(edges_fzw, False, dtype=bool)
    segm_mean = []
    for segm in segments_to_leave_1:
        selected |= (edges_fzw == segm)
        segm_mean.append(img[(edges_fzw == segm) & (img < 0.95)].mean())
    plt.figure(figsize=(15, 10))
    plt.title("Mean color of segments")
    plt.plot(segm_mean)
    all_mean = img[selected & (img < 0.95)].mean()
    all_var = img[selected & (img < 0.95)].var()
    #print all_mean, all_var
    #all_mean = np.mean(segm_mean)
    #all_var = np.var(segm_mean)
    #print all_mean, all_var
    
    segments_to_leave_2 = []
    for i, el in enumerate(segm_mean):
        #if abs(el - all_mean) < mean_color_threshold:
        if abs(el - all_mean) < 15 * all_var:
            segments_to_leave_2.append(segments_to_leave_1[i])
    
    blanked = leave_segments(img, edges_fzw, segments_to_leave_2, img_orig)
    #print "error for train image: ", rmse(y_train[0], X_train[0])
    #print "error for train image: ", rmse(y_train[0], blanked)
    #show_image(blanked)
    return blanked

img_idx = 0
blanked = blank_by_edges_and_fzw(X_train[img_idx], X_train_orig[img_idx])
print "error for train image: ", rmse(y_train[img_idx], X_train[img_idx])
print "error for train image: ", rmse(y_train[img_idx], blanked)
show_image(blanked)
