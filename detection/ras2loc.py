# -*- coding: cp936 -*-
"""
Created on Thu Dec 11 16:41:58 2014

@author: shuaiyi
"""
import matplotlib.pyplot as plt
import os, sys
import numpy as np
import skimage
import skimage.measure as sme
from skimage.io import imread
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, label


from filter_polys import filter_poly # filter
from poly2ras import poly_ras # to raster

def plot_comparison(original, filtered, filter_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')

def ras2loc(input_raster):
    img = imread(input_raster)
    label_img = label(img)
    props = sme.regionprops(label_img)
    return props
    
if __name__ == '__main__':

#    if len( sys.argv ) < 2:
#        print "[ ERROR ]: you need to pass at least one arg -- input_raster"
#        sys.exit(1)
#   
#    ras2loc( sys.argv[1] )

    # load shp
    # shp filter
    # shp 2 ranster
    
    img = imread("segmentation/MS04/results20/MOS83.tif") # *255
    #img = skimage.img_as_bool(img)
    #convex = convex_hull_image(img)
    label_img = label(img)
    props = sme.regionprops(label_img)
    plot_comparison(img, label_img, "Label")