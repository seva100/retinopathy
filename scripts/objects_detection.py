# -*- coding: utf-8 -*-

import os
from operator import attrgetter
import numpy as np
import scipy as sp
from scipy import ndimage as ndi
import pandas as pd
import sklearn
import sklearn.decomposition, sklearn.preprocessing
import skimage
import skimage.morphology, skimage.filters, skimage.feature, \
    skimage.segmentation, skimage.color, skimage.exposure
import matplotlib.pyplot as plt
import mahotas as mh
from numba import jit
import cv2
# my libraries that deal with images and this data set specifically:
from imhandle import load_image, save_image, show_image, rmse, \
    load_set, rgb_to_grayscale, pxl_distr, normalize_image, rgb_to_hsi, \
    add_salt_and_pepper, image_names_in_folder, moat_operator
from retina_im_lib import crop_black_border, expand, bfs_clustered, \
    erase_clusters, leave_segments, get_clusters, get_clusters_from_img_set, \
    region_growing_1, leave_segments_by_mask
from cluster_features import cl_areas, cl_circularity, cl_eccentricity, \
    cl_mean_intensity
import pixels_features as pf


def exclude_backgr(gb):
    # CLAHE
    cl1 = skimage.exposure.equalize_adapthist(gb)
    cl1 = (cl1 * 256).astype(np.uint8)

    # Average filter
    aver = cl1 - mh.mean_filter(cl1, Bc=np.ones((12, 12)))
    #aver[aver < 0] = 0
    aver = np.abs(aver)
    aver = aver.astype(np.uint8)

    # CLAHE
    cl2 = skimage.exposure.equalize_adapthist(aver)

    return cl2


def detect_vessels_1(gb):
    # Excluding background
    gb2 = exclude_backgr(gb)
    # There is a lot of small noise and details; denoising
    gb3 = mh.gaussian_filter(gb2, sigma=1.0)

    # Computing response of Gabor filter (real part) at different angles
    resp = []
    for theta in np.linspace(0.0, 17.0 / 18.0 * np.pi, num=10):
        kernel = skimage.filters.gabor_kernel(frequency=0.4, theta=theta,
                                              sigma_x=1.0 * 1.5, sigma_y=(1.0 / 1.75) * 1.5,
                                              n_stds=15)
        #resp.append(power(img, kernel))    # equivalent to Gaussian (cos^2 + sin^2 = 1)
        resp.append(ndi.convolve(gb3, np.real(kernel), mode='wrap'))

    response = np.max(resp, axis=0)
    response[response < 0] = 0
    #show_image(response)

    # freq=0.3, bandwidth=1.0, n_stds=15
    # freq=0.05, sigma_x=2.0, sigma_y=6.0, n_stds=15
    # 0.6, 2.0, 3.5, 15
    # 0.25, 1.0*2.0, (1.0 / 1.75) * 2.0, 15
    # 0.4, 1.0*1.5, (1.0 / 1.75) * 1.5, 15  - ok! except small line segments
    #                                        (but there are some lines there in the image)

    # Histogram equalization
    response *= 1.0 / response.max()
    resp2 = skimage.exposure.equalize_adapthist(response)
    #show_image(resp2)
    resp2 = (resp2 * 256).astype('uint8')

    # Removing noise
    noise_level = np.percentile(resp2[resp2 != 0], 80)
    resp3 = np.where(resp2 > noise_level, resp2, 0)
    #show_image(resp3, fig_size=(12, 12))

    # Removing small objects
    mask_large = skimage.morphology.remove_small_objects(resp3 != 0, min_size=30, connectivity=2)
    #pxl_distr(mask_large)
    resp4 = np.where(mask_large, resp3, 0)

    return resp4

@jit
def modified_gabor_kernel(radius, theta, _lambda, s, phi, alpha, beta, kappa, n_directions):
    # Following Qui Li "Automated Retinal Vessel Segmentation Using Gabor Filters and Scale Multiplication"
    
    kernel = np.empty((2 * radius + 1, 2 * radius + 1), dtype=np.complex128)
    freq = n_directions * _lambda / float(alpha * np.pi * s)
    # kernel[radius, radius] is the center point
    # x is headed rightwards from center, y -- upwards
    for x in xrange(-radius, radius + 1):
        for y in xrange(-radius, radius + 1):
            i = radius - y    # = 2 * radius - (y + radius)
            j = x + radius
            x_theta = x * np.cos(theta) + y * np.sin(theta)
            y_theta = -x * np.sin(theta) + y * np.cos(theta)
            resp = np.exp(-np.pi * ((x_theta / s) ** 2 + (y_theta / float(kappa * s)) ** 2)) * \
                np.cos(2 * np.pi * freq * x_theta)
            kernel[i, j] = resp
    return kernel


def modified_gabor_convolve(gb, sigma_0=10.0):
    gb2 = mh.median_filter(gb)
    gb2 = skimage.exposure.equalize_adapthist(gb2)

    # Computing response of Gabor filter (real part) at different angles
    resp = []
    n_directions = 7
    for theta in np.linspace(0.0, np.pi, num=n_directions):
        #for factor in [0.25, 0.5, 0.75, 1.0]:
        #sigma = sigma_0 * factor
        s = sigma_0
        kernel = modified_gabor_kernel(radius=20, _lambda=np.sqrt(2 * np.log(2) / np.pi), 
                                       theta=theta, phi=np.pi, s=s,
                                       alpha=1.25, beta=0.75, kappa=0.85, n_directions=n_directions)
        resp.append(ndi.convolve(gb2, kernel, mode='wrap'))

    resp = np.array(resp)
    resp_abs_argmax = np.argmax(np.abs(resp), axis=0)
    response = np.empty_like(gb2)
    for k in xrange(resp.shape[0]):
        response = np.where(resp_abs_argmax == k, resp[k], response)
    
    return response


def detect_vessels_3(img, sigma=16, gabor_perc=95.0, one_channel_img=False):
    # Preprocessing
    if one_channel_img:
        intensity = img.copy()
    else:
        intensity = img.mean(axis=2) / 256.0
    show_image(intensity); plt.axis('off'); plt.savefig('../presentation/pics/vessels/1.png')
    intensity = skimage.exposure.equalize_adapthist(intensity)
    intensity -= mh.median_filter(intensity, Bc=np.ones((25, 25)))
    show_image(intensity); plt.axis('off'); plt.savefig('../presentation/pics/vessels/2.png')
    intensity = normalize_image(intensity)
    intensity_inv = 1.0 - intensity
    show_image(intensity_inv); plt.axis('off'); plt.savefig('../presentation/pics/vessels/3.png')
    
    vessels = modified_gabor_convolve(intensity_inv, sigma)
    show_image(vessels); plt.axis('off'); plt.savefig('../presentation/pics/vessels/4.png')
    
    # Thresholding result
    gabor_thresholded = vessels > np.percentile(vessels.ravel(), gabor_perc)
    gabor_thresholded = skimage.morphology.remove_small_objects(gabor_thresholded, min_size=200)
    show_image(gabor_thresholded); plt.axis('off'); plt.savefig('../presentation/pics/vessels/5.png')
    return gabor_thresholded


def objects_map(gb):
    bfs = bfs_clustered(gb, radius=7, tol=5)
    bfs2 = erase_clusters(bfs, small_tol=20, large_tol=500, small_replace=0)
    return bfs2


def exudate_detect_1(gb):
    mask = objects_map(gb)
    result = np.where(mask, gb, 0.0)
    ans2 = skimage.morphology.remove_small_objects(result != 0, min_size=20, connectivity=2)
    return ans2


def exudate_detect_2(img):
    # Partially following H. Li et al.
    img_luv = skimage.color.rgb2luv(img)
    img_luv = normalize_image(img_luv)
    img_luv2 = skimage.exposure.equalize_adapthist(img_luv)
    img_luv2 = normalize_image(img_luv2)
    img_luv3 = sp.signal.wiener(img_luv2, noise=10.0)
    img_luv3 = normalize_image(img_luv3)
    edges = skimage.feature.canny(img_luv3[:, :, 1], sigma=2.0)
    tol1, tol2, tol3 = 100, 40, 100
    img_obj_map = region_growing_1(img_luv3[:, :, 1] * 255, radius=3, edges=edges,
                                   tol1=tol1, tol2=tol2, tol3=tol3)
    segm_to_leave = list(np.where((cl_circularity(img_obj_map) > 0.05) & \
                              (cl_areas(img_obj_map) > 10) & \
                              (cl_areas(img_obj_map) < 1000) & \
                              (cl_eccentricity(img_obj_map) > 0.1))[0])
                              #(cl_mean_intensity(img_obj_map, img_luv3[:, :, 1]) < 20))[0])
    exud1 = leave_segments(img_obj_map, segm_to_leave, \
                           replace_value=-1)
    return exud1


def exudates_get_features_3(folder, elements_folder, fundus_mask, num_imgs=10, 
                            stride_width=6):

    thr_high = 189.0 / 255.0
    
    y = []
    pxl_df = pd.DataFrame(columns=['hue', 'intensity', 'mean intensity', 'std intensity', 'dist to optic disk'])
    
    for img_fn in image_names_in_folder(folder)[:num_imgs]:
        print 'Image filename:', img_fn
        img = load_image(img_fn)
        short_fn = os.path.split(img_fn)[-1]
        hard_exud = load_image(os.path.join(elements_folder, short_fn))
        
        hsi = rgb_to_hsi(img)
        intensity = hsi[:, :, 2].copy()
        #i_sp = add_salt_and_pepper(intensity, 0.005)
        i_med = mh.median_filter(intensity)    # use Wiener filter instead?
        i_clahe = skimage.exposure.equalize_adapthist(i_med)
        
        optic_disk_map = optic_disk_detect_3(img)
        mask, optic_disk_center = optic_disk_mask(optic_disk_map, return_center=True)
        print 'Disk center: ', optic_disk_center
        fundus_wo_disk_mask = fundus_mask & (~mask)
        
        pxl_mask = np.zeros_like(i_clahe, dtype=np.bool)
        pxl_mask[::stride_width, ::stride_width] = True
        pxl_mask[~fundus_wo_disk_mask] = False
        
        bbox = bounding_box(hard_exud >= thr_high, bbox_radius_ratio=2)
        pxl_mask &= bbox
    
        cur_pxl_df = pd.DataFrame(columns=['hue', 'intensity', 'mean intensity', 'std intensity', 'dist to optic disk'])
        cur_pxl_df['hue'] = hsi[:, :, 0][pxl_mask].ravel()
        cur_pxl_df['intensity'] = i_clahe[pxl_mask].ravel()
        cur_pxl_df['mean intensity'] = pf.pxls_mean_intensity(i_clahe, stride_width, neighbd_rad=7, mask=pxl_mask).ravel()
        cur_pxl_df['std intensity'] = pf.pxls_std_intensity(i_clahe, stride_width, neighbd_rad=3, mask=pxl_mask).ravel()
        cur_pxl_df['dist to optic disk'] = pf.pxls_dist_to_pnt(i_clahe.shape, optic_disk_center, 
                                                               stride_width, mask=pxl_mask).ravel()
        print 'Accounted pixels: ', len(cur_pxl_df)
    
        y.append(hard_exud[pxl_mask])        # we can use fraction of white points in a neighborhood instead, 
                                             # but it becomes regression task then
        pxl_df = pd.concat((pxl_df, cur_pxl_df), axis=0, ignore_index=True)
    
    y = np.hstack(y)
    y = y >= thr_high
    
    return pxl_df, y


def hemorrhages_get_features_3(folder, elements_folder, fundus_mask, num_imgs=10, 
                               stride_width=6):

    thr_high = 189.0 / 255.0
    
    y = []
    pxl_df = pd.DataFrame(columns=['hue', 'saturation', 'intensity', 'mean intensity', 'std intensity'])
    
    for img_fn in image_names_in_folder(folder)[:num_imgs]:
        print 'Image filename:', img_fn
        img = load_image(img_fn)
        short_fn = os.path.split(img_fn)[-1]
        hemorrhages = load_image(os.path.join(elements_folder, short_fn))
        
        hsi = rgb_to_hsi(img)
        intensity = hsi[:, :, 2].copy()
        #i_sp = add_salt_and_pepper(intensity, 0.005)
        i_med = mh.median_filter(intensity)    # use Wiener filter instead?
        i_clahe = skimage.exposure.equalize_adapthist(i_med)
        
        optic_disk_map = optic_disk_detect_3(img)
        mask, optic_disk_center = optic_disk_mask(optic_disk_map, return_center=True)
        print 'Disk center: ', optic_disk_center
        fundus_wo_disk_mask = fundus_mask & (~mask)
        
        pxl_mask = np.zeros_like(i_clahe, dtype=np.bool)
        pxl_mask[::stride_width, ::stride_width] = True
        pxl_mask[~fundus_wo_disk_mask] = False
        
        bbox = bounding_box(hemorrhages >= thr_high, bbox_radius_ratio=2)
        pxl_mask &= bbox
    
        cur_pxl_df = pd.DataFrame(columns=['hue', 'saturation', 'intensity', 'mean intensity', 'std intensity'])
        cur_pxl_df['hue'] = hsi[:, :, 0][pxl_mask].ravel()
        cur_pxl_df['saturation'] = hsi[:, :, 1][pxl_mask].ravel()
        cur_pxl_df['intensity'] = i_clahe[pxl_mask].ravel()
        cur_pxl_df['mean intensity'] = pf.pxls_mean_intensity(i_clahe, stride_width, neighbd_rad=3, mask=pxl_mask).ravel()
        cur_pxl_df['std intensity'] = pf.pxls_std_intensity(i_clahe, stride_width, neighbd_rad=7, mask=pxl_mask).ravel()
        print 'Accounted pixels: ', len(cur_pxl_df)
    
        y.append(hemorrhages[pxl_mask])        # we can use fraction of white points in a neighborhood instead, 
                                               # but it becomes regression task then
        pxl_df = pd.concat((pxl_df, cur_pxl_df), axis=0, ignore_index=True)
    
    y = np.hstack(y)
    y = y >= thr_high
    
    return pxl_df, y


def exudates_detect_3(img, fundus_mask, clf, stride_width=6):
    #thr_high = 189.0 / 255.0

    #img = imh.load_image(img_fn)
    #short_fn = os.path.split(img_fn)[-1]
    #hard_exud = imh.load_image(os.path.join(hard_exud_folder, short_fn))
    
    hsi = rgb_to_hsi(img)
    intensity = hsi[:, :, 2].copy()
    show_image(intensity); plt.axis('off'); plt.savefig('../presentation/pics/exudates/1.png', 
                                                        bbox_inches='tight', pad_inches=0)
    #i_sp = add_salt_and_pepper(intensity, 0.005)
    i_med = mh.median_filter(intensity)    # use Wiener filter instead?
    i_clahe = skimage.exposure.equalize_adapthist(i_med)
    show_image(i_clahe); plt.axis('off'); plt.savefig('../presentation/pics/exudates/2.png', 
                                                      bbox_inches='tight', pad_inches=0)
    
    optic_disk_map = optic_disk_detect_3(img)
    mask, optic_disk_center = optic_disk_mask(optic_disk_map, return_center=True)
    print 'Disk center: ', optic_disk_center
    fundus_wo_disk_mask = fundus_mask | mask
    show_image(mask)
    show_image(fundus_wo_disk_mask); plt.axis('off'); plt.savefig('../presentation/pics/exudates/3.png', 
                                                                  bbox_inches='tight', pad_inches=0)
    
    pxl_mask = np.zeros_like(i_clahe, dtype=np.bool)
    pxl_mask[::stride_width, ::stride_width] = True
    pxl_mask[~fundus_wo_disk_mask] = False
    
    '''
    cur_pxl_df = pd.DataFrame(columns=['hue', 'intensity', 'mean intensity', 'std intensity', 'dist to optic disk'])
    cur_pxl_df['hue'] = hsi[:, :, 0][pxl_mask].ravel()
    cur_pxl_df['intensity'] = i_clahe[pxl_mask].ravel()
    cur_pxl_df['mean intensity'] = pf.pxls_mean_intensity(i_clahe, stride_width, neighbd_rad=7, mask=pxl_mask).ravel()
    cur_pxl_df['std intensity'] = pf.pxls_std_intensity(i_clahe, stride_width, neighbd_rad=3, mask=pxl_mask).ravel()
    cur_pxl_df['dist to optic disk'] = pf.pxls_dist_to_pnt(i_clahe.shape, optic_disk_center, 
                                                           stride_width, mask=pxl_mask).ravel()
    print 'Accounted pixels: ', len(cur_pxl_df)
    '''
    cur_pxl_df = pd.DataFrame(columns=['hue', 'intensity', 'mean intensity', 'std intensity', 'dist to optic disk'])
    cur_pxl_df['hue'] = hsi[:, :, 0][::stride_width, ::stride_width].ravel()
    cur_pxl_df['intensity'] = i_clahe[::stride_width, ::stride_width].ravel()
    cur_pxl_df['mean intensity'] = pf.pxls_mean_intensity(i_clahe, stride_width, neighbd_rad=7).ravel()
    cur_pxl_df['std intensity'] = pf.pxls_std_intensity(i_clahe, stride_width, neighbd_rad=3).ravel()
    cur_pxl_df['dist to optic disk'] = pf.pxls_dist_to_pnt(i_clahe.shape, optic_disk_center, 
                                                           stride_width).ravel()
    
    scaler = sklearn.preprocessing.StandardScaler()
    cur_pxl_df_scaled = scaler.fit_transform(cur_pxl_df)
    
    pred = clf.predict(cur_pxl_df_scaled)
    '''
    pred_map = np.zeros_like(pxl_mask, dtype=np.uint8)
    k = 0
    for i in xrange(pred_map.shape[0]):
        for j in xrange(pred_map.shape[1]):
            if pxl_mask[i, j]:
                pred_map[i, j] = 255 * pred[k]
                if pred[k]:
                    cv2.circle(pred_map, (i, j), 8, (255, 255, 255))
                k += 1
    '''
    
    
    pred_map = pred.reshape(pxl_mask[::stride_width, ::stride_width].shape)
    pred_map &= pxl_mask[::stride_width, ::stride_width]
    pred_map_full = np.zeros_like(pxl_mask, dtype=np.uint8)
    pred_map_full[::stride_width, ::stride_width] = pred_map
    pred_map_full_enl = np.zeros_like(pred_map_full, dtype=np.uint8)
    for i in xrange(pred_map_full.shape[0]):
        for j in xrange(pred_map_full.shape[1]):
            if pred_map_full[i, j]:
                #cv2.circle(pred_map_full_enl, (i, j), 3, (255, 255, 255))
                sz = stride_width        
                pred_map_full_enl[i - sz + 1:i + sz, j - sz + 1:j + sz] = 1
    return pred_map_full_enl


def hemorrhages_detect_3(img, fundus_mask, clf, stride_width=6):
    #thr_high = 189.0 / 255.0

    #img = imh.load_image(img_fn)
    #short_fn = os.path.split(img_fn)[-1]
    #hard_exud = imh.load_image(os.path.join(hard_exud_folder, short_fn))
    
    hsi = rgb_to_hsi(img)
    intensity = hsi[:, :, 2].copy()
    #i_sp = add_salt_and_pepper(intensity, 0.005)
    i_med = mh.median_filter(intensity)    # use Wiener filter instead?
    i_clahe = skimage.exposure.equalize_adapthist(i_med)
    
    optic_disk_map = optic_disk_detect_3(img)
    mask, optic_disk_center = optic_disk_mask(optic_disk_map, return_center=True)
    #print 'Disk center: ', optic_disk_center
    fundus_wo_disk_mask = fundus_mask & (~mask)
    
    pxl_mask = np.zeros_like(i_clahe, dtype=np.bool)
    pxl_mask[::stride_width, ::stride_width] = True
    pxl_mask[~fundus_wo_disk_mask] = False
    
    cur_pxl_df = pd.DataFrame(columns=['hue', 'saturation', 'intensity', 'mean intensity', 'std intensity'])
    cur_pxl_df['hue'] = hsi[:, :, 0][::stride_width, ::stride_width].ravel()
    cur_pxl_df['saturation'] = hsi[:, :, 1][::stride_width, ::stride_width].ravel()
    cur_pxl_df['intensity'] = i_clahe[::stride_width, ::stride_width].ravel()
    cur_pxl_df['mean intensity'] = pf.pxls_mean_intensity(i_clahe, stride_width, neighbd_rad=3).ravel()
    cur_pxl_df['std intensity'] = pf.pxls_std_intensity(i_clahe, stride_width, neighbd_rad=7).ravel()
    print 'Accounted pixels: ', len(cur_pxl_df)
    
    scaler = sklearn.preprocessing.StandardScaler()
    cur_pxl_df_scaled = scaler.fit_transform(cur_pxl_df)
    
    pred = clf.predict(cur_pxl_df_scaled)
    '''
    pred_map = np.zeros_like(pxl_mask, dtype=np.uint8)
    k = 0
    for i in xrange(pred_map.shape[0]):
        for j in xrange(pred_map.shape[1]):
            if pxl_mask[i, j]:
                pred_map[i, j] = 255 * pred[k]
                if pred[k]:
                    cv2.circle(pred_map, (i, j), 8, (255, 255, 255))
                k += 1
    '''
    
    
    pred_map = pred.reshape(pxl_mask[::stride_width, ::stride_width].shape)
    show_image(pred_map)
    pred_map &= pxl_mask[::stride_width, ::stride_width]
    pred_map_full = np.zeros_like(pxl_mask, dtype=np.uint8)
    pred_map_full[::stride_width, ::stride_width] = pred_map
    pred_map_full_enl = np.zeros_like(pred_map_full, dtype=np.uint8)
    for i in xrange(pred_map_full.shape[0]):
        for j in xrange(pred_map_full.shape[1]):
            if pred_map_full[i, j]:
                #cv2.circle(pred_map_full_enl, (i, j), 3, (255, 255, 255))
                sz = stride_width        
                pred_map_full_enl[i - sz + 1:i + sz, j - sz + 1:j + sz] = 1
    
    return pred_map_full_enl


def hemorrhages_get_features_4(img, fundus_mask, threshold=0.25):
    green = img[:, :, 1]
    g_clahe = skimage.exposure.equalize_adapthist(green)
    g_med = mh.median_filter(g_clahe, Bc=np.ones((5, 5)))
    #cand = g_med < threshold
    #label = skimage.measure.label(cand)
    seeds_1 = g_med < threshold
    #seeds_2 = pf.local_minima(g_med, stride_width=5, neighbd_rad=4, mask=seeds_1 & fundus_mask)
    #show_image(seeds_2, fig_size=(7, 7))
    #label = region_growing_1(g_med, radius=1, tol1=0.05, tol2=0.05, tol3=0.05, seeds=seeds_2) + 1
    label = skimage.measure.label(seeds_1)    
    
    n_clusters = label.max() + 1

    features_df = pd.DataFrame(columns=['area', 'circularity', 'eccentricity',
                                        'std intensity', 'mean red', 'mean green'],
                               index=np.arange(n_clusters))
    features_df['area'] = cl_areas(label)
    features_df['circularity'] = cl_circularity(label)
    features_df['eccentricity'] = cl_eccentricity(label)
    intensity = img.sum(axis=2) / 3.0
    i_clahe = skimage.exposure.equalize_adapthist(intensity)
    r_clahe = skimage.exposure.equalize_adapthist(img[:, :, 0])
    # perform median filtering on i_clahe and r_clahe?
    
    for i in xrange(n_clusters):
        cl_map = (label == i)
        if cl_map.sum() == 0:
            features_df.ix[i, 'std intensity'] = 0
            features_df.ix[i, 'mean red'] = 0
            features_df.ix[i, 'mean green'] = 0
        else:
            features_df.ix[i, 'std intensity'] = i_clahe[cl_map].std()
            features_df.ix[i, 'mean red'] = r_clahe[cl_map].mean()
            features_df.ix[i, 'mean green'] = g_clahe[cl_map].mean()
    features_df = features_df.loc[(features_df.area != 0) & \
                                  (features_df.area < 80000), :]
    return label, features_df


def hemorrhages_get_features_5(img, moat, fundus_mask):
    m_clahe = skimage.exposure.equalize_adapthist(moat)
    m_med = mh.median_filter(m_clahe, Bc=np.ones((5, 5)))
    #seeds_1 = m_med < threshold
    #minima = pf.local_minima(m_med, stride_width=5, neighbd_rad=20, mask=fundus_mask)
    #minima &= fundus_mask & (m_med < threshold)
    #show_image(seeds_2, fig_size=(7, 7))
    #label = region_growing_1(m_med, radius=1, tol1=0.01, tol2=0.01, tol3=0.01, seeds=minima) + 1
    
    threshold_global_otsu = skimage.filters.threshold_otsu(m_med)
    cand = m_med <= 0.95 * threshold_global_otsu
    # erase vessels
    label = skimage.measure.label(cand)
    
    n_clusters = label.max() + 1

    features_df = pd.DataFrame(columns=['circularity', 'eccentricity',
                                        'std intensity', 'mean red', 'mean green'],
                               index=np.arange(n_clusters))
    areas = cl_areas(label)
    features_df['area'] = areas.astype(np.float64)
    features_df['circularity'] = cl_circularity(label)
    #features_df['eccentricity'] = cl_eccentricity(label)
    features_df['eccentricity'] = map(attrgetter('eccentricity'), 
                                      skimage.measure.regionprops(label + 1))
    intensity = img.sum(axis=2) / 3.0
    i_clahe = skimage.exposure.equalize_adapthist(intensity)
    r_clahe = skimage.exposure.equalize_adapthist(img[:, :, 0])
    g_clahe = skimage.exposure.equalize_adapthist(img[:, :, 1])
    # perform median filtering on i_clahe and r_clahe?
    
    for i in xrange(n_clusters):
        cl_map = (label == i)
        if cl_map.sum() == 0:
            features_df.ix[i, 'std intensity'] = 0
            features_df.ix[i, 'mean red'] = 0
            features_df.ix[i, 'mean green'] = 0
        else:
            features_df.ix[i, 'std intensity'] = i_clahe[cl_map].std()
            features_df.ix[i, 'mean red'] = r_clahe[cl_map].mean()
            features_df.ix[i, 'mean green'] = g_clahe[cl_map].mean()
    features_df = features_df.loc[(areas != 0) & \
                                  (areas < 80000), :]
    return label, features_df


def hemorrhages_ground_truth(label, true):
    covering_level = cl_mean_intensity(label, true.astype(np.float64))
    areas = cl_areas(label)
    covering_level = covering_level[(areas != 0) & (areas < 80000)]
    return covering_level


def hemorrhages_detect(img):
    moat = moat_operator(img, sigma=5.0)
    show_image(moat); plt.axis('off'); plt.savefig('../presentation/pics/hemorrhages/1.png')
    m_clahe = skimage.exposure.equalize_adapthist(moat)
    m_med = mh.median_filter(m_clahe, Bc=np.ones((5, 5)))
    show_image(m_med); plt.axis('off'); plt.savefig('../presentation/pics/hemorrhages/2.png')
    threshold_global_otsu = skimage.filters.threshold_otsu(m_med)
    print 'Otsu threshold:', threshold_global_otsu
    segmented = m_med <= 0.9 * threshold_global_otsu
    show_image(segmented); plt.axis('off'); plt.savefig('../presentation/pics/hemorrhages/3.png')
    #vessels = od.detect_vessels_3(img)
    vessels = detect_vessels_3(m_med, one_channel_img=True, gabor_perc=88.0)
    vessels = skimage.morphology.remove_small_objects(vessels, min_size=150)
    segmented[mh.dilate(vessels, Bc=np.ones((10, 10)))] = False
    show_image(segmented); plt.axis('off'); plt.savefig('../presentation/pics/hemorrhages/4.png')
    segm_label = skimage.measure.label(segmented)
    segm_eccen = np.array(map(attrgetter('eccentricity'), skimage.measure.regionprops(segm_label + 1)))
    segm_area = cl_areas(segm_label)
    #segm_circ = cf.cl_circularity(segm_label)
    segm_filtered = leave_segments_by_mask(segm_label, (segm_eccen < 0.85) & (segm_area < 30000), 
                                           replace_value=0) != 0
    return segm_filtered


def hemorrhages_detect_2(img):
    moat = moat_operator(img, sigma=5.0)
    #show_image(moat); plt.axis('off'); plt.savefig('../presentation/pics/hemorrhages/1.png')
    m_clahe = skimage.exposure.equalize_adapthist(moat)
    m_med = mh.median_filter(m_clahe, Bc=np.ones((5, 5)))
    #show_image(m_med); plt.axis('off'); plt.savefig('../presentation/pics/hemorrhages/2.png')
    threshold_global_otsu = skimage.filters.threshold_otsu(m_med)
    print 'Otsu threshold:', threshold_global_otsu
    segmented = m_med <= 0.9 * threshold_global_otsu
    #show_image(segmented); plt.axis('off'); plt.savefig('../presentation/pics/hemorrhages/3.png')
    segm_label = skimage.measure.label(segmented)
    segm_eccen = np.array(map(attrgetter('eccentricity'), skimage.measure.regionprops(segm_label + 1)))
    segm_area = cl_areas(segm_label)
    #segm_circ = cf.cl_circularity(segm_label)
    segm_filtered = leave_segments_by_mask(segm_label, (segm_eccen < 0.85) & (10 < segm_area) & (segm_area < 30000), 
                                           replace_value=0) != 0
    return segm_filtered


def optic_disk_fit(folder, n_pca_compns=6, box_shape=30):
    # All images prepared for PCA fitting must be put in special
    # folder (`folder`).
    cl_boxes_all = get_clusters_from_img_set(folder)
    pca = sklearn.decomposition.PCA(n_components=n_pca_compns)
    #boxes_pca = pca.fit_transform(cl_boxes_all)
    pca.fit(cl_boxes_all)
    return pca


def optic_disk_detect(gb, pca, box_shape=30):
    # box_shape must correspond to box_shape used to fit PCA
    t = np.percentile(gb.ravel(), 99)
    bright = gb > t
    bright = (bright * 255).astype(int)
    cl1 = bfs_clustered(bright, radius=2, tol=1)
    cl2 = erase_clusters(cl1, small_tol=0.0004 * cl1.shape[0] * cl1.shape[1],
                         large_tol=2 * cl1.size, small_replace=0)
    boxes, cl_no = get_clusters(cl2, box_shape)
    cl_num = cl1.max() + 1
    rec = pca.inverse_transform(pca.transform(boxes))
    pca_error = np.linalg.norm(boxes - rec, axis=1)
    thr = np.percentile(pca_error, 70)
    cl_no = np.array(cl_no)
    cl_to_leave = cl_no[pca_error < thr]
    if 0 not in cl_to_leave:
        cl_to_leave = np.hstack((0, cl_to_leave))
    cl3 = leave_segments(cl2, cl_to_leave)
    return cl3


def optic_disk_detect_2(img):
    hsi = rgb_to_hsi(img)
    intensity = hsi[:, :, 2].copy()
    i_sp = add_salt_and_pepper(intensity, 0.005)
    i_med = mh.median_filter(i_sp)
    i_clahe = skimage.exposure.equalize_adapthist(i_med)
    optic_disk_map = (i_clahe > 0.6) & (hsi[:, :, 1] < 0.3)
    #show_image(optic_disk_map)
    optic_disk_map = skimage.morphology.remove_small_objects(optic_disk_map, min_size=500, connectivity=2)
    optic_disk_map = mh.close_holes(mh.close(optic_disk_map, Bc=np.ones((30, 30))))
    optic_disk_map = skimage.morphology.remove_small_objects(optic_disk_map, min_size=10000, connectivity=2)
    if np.all(optic_disk_map == 0):
        print 'Disk not found'
    return optic_disk_map


def optic_disk_detect_3(img):
    '''
    Method that seems to work well with DIARETDB1 database.
    '''

    hsi = rgb_to_hsi(img)
    intensity = hsi[:, :, 2].copy()
    #plt.axis('off'); show_image(intensity)
    #i_sp = add_salt_and_pepper(intensity, 0.005)
    i_med = mh.median_filter(intensity)    # use Wiener filter instead?
    i_clahe = skimage.exposure.equalize_adapthist(i_med)
    #plt.axis('off'); show_image(i_clahe)
    seeds = (i_clahe > 0.85)
    seeds = skimage.morphology.remove_small_objects(seeds, min_size=300, connectivity=2)
    #plt.axis('off'); show_image(seeds)
    optic_disk_map = region_growing_1(i_clahe, radius=3, tol1=0.1, tol2=0.2, tol3=0.2, seeds=seeds)
    optic_disk_map += 1
    #plt.axis('off'); show_image(optic_disk_map)

    _cl_areas = cl_areas(optic_disk_map)
    print _cl_areas
    optic_disk_map = leave_segments_by_mask(optic_disk_map,
                                            (8000 < _cl_areas) & (_cl_areas < 30000))

    #optic_disk_map = skimage.morphology.remove_small_objects(optic_disk_map, min_size=500, connectivity=2)
    optic_disk_map = mh.close_holes(mh.close(optic_disk_map, Bc=np.ones((10, 10))))
    #optic_disk_map = skimage.morphology.remove_small_objects(optic_disk_map, min_size=5000, connectivity=2)
    if np.all(optic_disk_map == 0):
        print 'Disk not found'
    return optic_disk_map


def bounding_box(bin_img, bbox_radius_ratio=1.4):
    mask = np.zeros_like(bin_img, dtype=np.bool)
    nz_idx = np.where(bin_img)
    if nz_idx[0].size == 0:
        center = (bin_img.shape[0] / 2, bin_img.shape[1] / 2)
        radius = bin_img.shape[0] / 6
    else:
        center = (nz_idx[0].mean(), nz_idx[1].mean())
        radius = np.max((np.max(np.abs(nz_idx[0] - center[0])),
                         np.max(np.abs(nz_idx[1] - center[1]))))
    
    mask[(center[0] - bbox_radius_ratio * radius):\
         (center[0] + bbox_radius_ratio * radius),
         (center[1] - bbox_radius_ratio * radius):\
         (center[1] + bbox_radius_ratio * radius)] = True
    return mask


def optic_disk_mask(optic_disk_map, bbox_radius_ratio=1.4, return_center=False):
    if np.all(optic_disk_map == 0):
        if return_center:
            # in case the optic disk wasn't found,
            # then image center is returned as optic disk center
            return np.zeros_like(optic_disk_map, dtype=np.bool), \
                (optic_disk_map.shape[0] / 2, optic_disk_map.shape[1] / 2)
        return np.zeros_like(optic_disk_map, dtype=np.bool)
    
    nz_idx = np.where(optic_disk_map)
    #optic_disk_center = map_center(optic_disk_map)
    optic_disk_center = (nz_idx[0].mean(), nz_idx[1].mean())
    #dt = scipy.ndimage.distance_transform_edt(optic_disk_map)
    #optic_disk_radius = dt[optic_disk_center]

    #optic_disk_radius = ((nz_idx[0] - optic_disk_center[0]) ** 2 + (nz_idx[1] - optic_disk_center[1]) ** 2).max()
    #optic_disk_radius = np.sqrt(optic_disk_radius)

    optic_disk_radius = np.max((np.max(np.abs(nz_idx[0] - optic_disk_center[0])),
                                np.max(np.abs(nz_idx[1] - optic_disk_center[1]))))
    print 'radius:', optic_disk_radius
    print (optic_disk_center[0] - bbox_radius_ratio * optic_disk_radius,
         optic_disk_center[0] + bbox_radius_ratio * optic_disk_radius,
         optic_disk_center[1] - bbox_radius_ratio * optic_disk_radius,
         optic_disk_center[1] + bbox_radius_ratio * optic_disk_radius)
    mask = np.zeros_like(optic_disk_map, dtype=np.bool)
    mask[(optic_disk_center[0] - bbox_radius_ratio * optic_disk_radius):\
         (optic_disk_center[0] + bbox_radius_ratio * optic_disk_radius),
         (optic_disk_center[1] - bbox_radius_ratio * optic_disk_radius):\
         (optic_disk_center[1] + bbox_radius_ratio * optic_disk_radius)] = True
    if return_center:
        return mask, optic_disk_center
    return mask
