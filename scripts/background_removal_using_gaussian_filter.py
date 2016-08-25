# -*- coding: utf-8 -*-


__author__ = 'Prasetia-Utama'
_method_name = "Background removal using gaussian filter"

#from scipy import misc
import numpy as np
#import glob
from scipy import ndimage

def denoise_image(image):
    x, y = image.shape
    kernel = np.array([[0.11, 0.11, 0.11], [0.11, 0.11, 0.11], [0.11, 0.11, 0.11]])
    bg = ndimage.gaussian_filter(image, sigma=6)
    bg = ndimage.convolve(bg,kernel)
    mask = image < bg - 0.149
    result = np.where(mask, image, 1.0)
    return result
    #misc.imsave('background.png', bg)
    #misc.imsave('mask.png', mask)
    #misc.imsave('result.png', result)