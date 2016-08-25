'''
Not currently used or rejected functions.

@author: Artem Sevastopolsky
'''


# ----------------------------------------------------
# Functions that are used for contrast normalization in subimages follow
#    [] Huiqi Li et al. - Automated Feature Extraction in Color Retinal Images by a Model Based Approach"


def hist2D(img):
    '''Returns dict of occurences of each pair (a, b) in a given 2-channel image img.'''
    if len(img.shape) != 3 or img.shape[2] != 2:
        raise Exception('hist2D() receives only 2-channel images')
    hist = defaultdict(int)
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            hist[(img[i, j, 0], img[i, j, 1])] += 1
    return hist


@jit
def weighted_mean(img, hist):
    '''Receives 2-channel img and returns its weighted mean color for every channel.'''
    mean = np.empty(2)
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            u, v = img[i, j]
            mean[0] += u * hist[(u, v)]
            mean[1] += v * hist[(u, v)]
    mean /= img.shape[0] * img.shape[1]
    return mean


n_subimgs = 8    # 8 by each side, 64 overall

av_clrs = np.empty((img_luv.shape[0], img_luv.shape[1], 2))
i_sub_size = img_luv.shape[0] // n_subimgs
j_sub_size = img_luv.shape[1] // n_subimgs
for i in xrange(0, img_luv.shape[0], i_sub_size):
    i_end = (i + i_sub_size) if i + i_sub_size < img_luv.shape[0] else img_luv.shape[0]
    for j in xrange(0, img_luv.shape[1], j_sub_size):
        j_end = (j + j_sub_size) if j + j_sub_size < img_luv.shape[1] else img_luv.shape[1]
        
        sub_img = img_luv[i:i_end, j:j_end, 1:]
        hist = hist2D(sub_img)
        mean = weighted_mean(sub_img, hist)
        #print i, j, mean
        av_clrs[i:i_end, j:j_end, :] = mean
            

diff_img = np.sqrt((img_luv[:, :, 1] - av_clrs[:, :, 0]) ** 2 + \
                   (img_luv[:, :, 2] - av_clrs[:, :, 1]) ** 2)
show_image(diff_img)

# -----------------------------------------------------------