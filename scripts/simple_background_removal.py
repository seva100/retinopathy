"""
Simple background removal code

@author: Rangel Dokov, AjitMotra, Artem Sevastopolsky
"""

import os
import numpy as np
from math import sqrt
from scipy import signal
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import cv2
from scipy import ndimage
from imhandle import load_image, save_image, show_image

_method_name = "Background removal using median filter twice and closing filter"
kernel_size = 3
#gaussian_sigma = 2

def denoise_image(inp):
    # estimate 'background' color by a median filter
    bg = signal.medfilt2d(inp, kernel_size)
    # using local mean filter
    #kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2) 
    #bg = cv2.filter2D(inp, -1, kernel)
    # using gaussian filter
    #bg = ndimage.gaussian_filter(inp, sigma=gaussian_sigma)
    #kernel = np.full((kernel_size, kernel_size), .11)
    #bg = ndimage.convolve(bg, kernel)
    #save_image('background.png', bg)

    # compute 'foreground' mask as anything that is significantly darker than
    # the background
    #mask = inp < (bg - 0.1)
    mask = inp < (0.9 * bg)    # seems a little better than previous
    #save_image('foreground_mask.png', mask)
    
    #back = np.average(bg)
    mod = ndimage.filters.median_filter(mask, 2)    # medfilt2d doesn't support bool mask
    mod = ndimage.grey_closing(mod, size=(3, 3))

    # return the input value for all pixels in the mask or pure white otherwise
    return np.where(mask, inp, 1.0)


if __name__ == "__main__":
    inp_path = '../train/2.png'
    out_path = '../output_for_train/2.png'
    
    inp = load_image(inp_path)
    out = denoise_image(inp)
    save_image(out_path, out)
    show_image(out)
    
    '''
    def rmse(img, cleaned_img):
        return sqrt(np.mean((img - cleaned_img) ** 2))  
        # runs almost 2x times faster than sklearn.metrics.mean_square_error
    
    error = []
    for k in xrange(1, 15, 1):
        gaussian_sigma = k
        out = denoise_image(inp)
        #save_image('../output_for_train/2_{}.png'.format(k), out)
        cleaned_img = load_image(os.path.join('../train_cleaned', '2.png'))
        error.append(rmse(cleaned_img, out))
        
    plt.plot(xrange(1, 15, 1), error)
    '''