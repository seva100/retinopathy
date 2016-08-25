# -*- coding: utf-8 -*-
"""
Created on Sun Sep 06 22:10:57 2015

@author: Artem S.
"""

import os
import glob
from math import sqrt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class ImLibException(Exception):
    pass


def load_image(path):
    return np.asarray(Image.open(path)) / 255.0


def save_image(path, img):
    if img.max() <= 1.5:
        tmp = np.asarray(img * 255.0, dtype=np.uint8)
    else:
        tmp = np.asarray(img, dtype=np.uint8)
    Image.fromarray(tmp).save(path)


def show_image(img, fig_size=(10, 10)):
    plt.figure(figsize=fig_size)
    plt.imshow(img, cmap=cm.Greys_r)


def rmse(img, cleaned_img):
    return sqrt(np.mean((img - cleaned_img) ** 2))  
    # runs almost 2x times faster than sklearn.metrics.mean_square_error


def normalize_image(img):
    if len(img.shape) == 3 and img.shape[2] > 1:
        # multi-channel image
        new_img = img.copy()
        for i in range(img.shape[2]):
            new_img[:, :, i] -= new_img[:, :, i].min()
            new_img_max = new_img[:, :, i].max()
            if not np.isclose(new_img_max, 0):
                new_img[:, :, i] /= new_img_max
    else:
        # 2D (e.g. grayscale) or single-channel
        new_img = img.copy()
        new_img -= new_img.min()
        new_img_max = new_img.max()
        if not np.isclose(new_img_max, 0):
            new_img /= new_img_max
    return new_img


def load_set(folder, shuffle=False):
    img_list = sorted(glob.glob(os.path.join(folder, '*.png')) + \
                      glob.glob(os.path.join(folder, '*.jpg')) + \
                      glob.glob(os.path.join(folder, '*.jpeg')))
    if shuffle:
        np.random.shuffle(img_list)
    data = []
    filenames = []
    for img_fn in img_list:
        img = load_image(img_fn)
        data.append(img)
        filenames.append(img_fn)
    return data, filenames


def image_names_in_folder(folder):
    fn = []
    for ending in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.ppm'):
        fn.extend(glob.glob(os.path.join(folder, ending)))
    fn.sort()
    return fn


def rgb_to_grayscale(rgb_img):
    gs_img = 0.299 * rgb_img[:, :, 0] + 0.587 * rgb_img[:, :, 1] + 0.114 * rgb_img[:, :, 2]
    return gs_img


def rgb_to_hsi(rgb_img):
    r, g, b = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]
    
    # theta notation from the following article can be used:
    # Karegowda et al. "Exudates Detection in Retinal Images using Back Propagation Neural Network"
    # http://www.ijcaonline.org/volume25/number3/pxc3874062.pdf 
    #theta = np.arccos(0.5 * (2 * r - g - b)) / np.sqrt((r - g) ** 2 + (r - b) * (g - b)))
    
    # Wikipedia notation is used below.
    # Exact formulas can be found here http://www.cse.usf.edu/~mshreve/rgb-to-hsi 
    beta = np.sqrt(3) / 2.0 * (g - b)
    alpha = r - 0.5 * (g - b)
    h = np.arctan2(beta, alpha)
    s = 1.0 - 3.0 * np.minimum(r, g, b) / (r + g + b + 1e-8).astype(np.float64)
    i = (r + g + b) / 3.0
    hsi = np.empty_like(rgb_img, dtype=np.float64)
    hsi[:, :, 0] = h
    hsi[:, :, 1] = s
    hsi[:, :, 2] = i
    return hsi


def moat_operator(img, sigma=1.0):
    green = img[:, :, 1]
    fft = np.fft.fft2(green)
    vv, uu = np.meshgrid(np.arange(fft.shape[1]), np.arange(fft.shape[0]))
    spectrum = 1.0 - np.exp(-(uu ** 2 + vv ** 2) / (2.0 * sigma ** 2))
    I = fft * spectrum
    
    # since Re^2 + Im^2 = 1, it implies that sqrt(irfft^2 + iifft^2) = |ifft|
    inv = np.abs(np.fft.ifft2(I))
    moat = green - inv
    return moat


def add_salt_and_pepper(gb, prob):
    rnd = np.random.rand(gb.shape[0], gb.shape[1])
    noisy = gb.copy()
    noisy[rnd < prob] = 0
    noisy[rnd > 1 - prob] = 1
    return noisy


def pxl_distr(img):
    plt.hist(img.ravel(), bins=100)


def plot_subfigures(imgs, title=None, fig_size=None, contrast_normalize=False):
    if isinstance(imgs, list):
        imgs = np.array(imgs)
    if len(imgs.shape) == 4 and imgs.shape[0] == 1:
        imgs = imgs.reshape((imgs.shape[1], imgs.shape[2], imgs.shape[3]))
    if len(imgs.shape) == 2:
        # One picture
        if title is not None:
            plt.title(title)
        show_image(imgs)
    
    elif len(imgs.shape) == 3:
        # Multiple pictures in one row
        if fig_size is None:
            fig, axes = plt.subplots(nrows=1, ncols=imgs.shape[0])
                                     #figsize=(20, 20))
        else:
            fig, axes = plt.subplots(nrows=1, ncols=imgs.shape[0],
                                     figsize=fig_size)
        plt.gray()
        if title is not None:
            fig.suptitle(title, fontsize=12)
        for i in xrange(imgs.shape[0]):
            axes[i].axis('off')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            if contrast_normalize:
                axes[i].imshow(imgs[i])
            else:
                # Normalizing contrast for each image
                vmin, vmax = imgs[i].min(), imgs[i].max()
                axes[i].imshow(imgs[i], vmin=vmin, vmax=vmax)
            
    elif len(imgs.shape) == 4:
        # Multiple pictures in a few rows
        if fig_size is None:
            fig, axes = plt.subplots(nrows=imgs.shape[0], ncols=imgs.shape[1])
                                     #figsize=(20, 20))
        else:
            fig, axes = plt.subplots(nrows=imgs.shape[0], ncols=imgs.shape[1],
                                     figsize=fig_size)
        plt.gray()
        if title is not None:
            fig.suptitle(title, fontsize=12)
        for i in xrange(imgs.shape[0]):
            for j in xrange(imgs.shape[1]):
                axes[i][j].axis('off') 
                axes[i][j].set_xticks([])
                axes[i][j].set_yticks([])
                if contrast_normalize:
                    axes[i][j].imshow(imgs[i][j])
                else:
                    # Normalizing contrast for each image
                    vmin, vmax = imgs[i][j].min(), imgs[i][j].max()
                    axes[i][j].imshow(imgs[i][j], vmin=vmin, vmax=vmax)
    else:
        raise ImLibException("imgs array contains 3D set of images or deeper")
