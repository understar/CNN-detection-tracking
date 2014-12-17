# -*- coding: cp936 -*-
"""
Created on Thu Dec 11 16:41:58 2014

@author: shuaiyi
"""
import matplotlib.pyplot as plt
import os, sys
import numpy as np
import pandas as pd
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

def ras2loc(input_raster, input_src):
    img = imread(input_raster)
    label_img = label(img)
    props = sme.regionprops(label_img, intensity_image=input_src)
    return props
    
def writeprops(props, fname):
    with open(fname, 'w') as f:
        f.writelines(['id,', 'x,', 'y,', 'area,', 'angle', '\n'])
        for r in props:
            f.writelines(["%s,"%(r.label), "%s,"%(r.centroid[0]), "%s,"%(r.centroid[1]),
                          "%s,"%(r.convex_area), "%s,"%(r.orientation) , '\n'])
    
if __name__ == '__main__':
    # load shp
    # shp filter
    # shp 2 ranster
    if len( sys.argv ) < 3:
        print "[ ERROR ]: you need to pass at least two arg -- source image -- input shp"
        sys.exit(1)
        
    src = imread(sys.argv[1])
    img = imread(sys.argv[2]) # *255
    #img = skimage.img_as_bool(img)
    #convex = convex_hull_image(img)
    label_img = label(img)
    props = sme.regionprops(label_img)
    writeprops(props, sys.argv[2][:-4] + '.csv')
    # df = pd.DataFrame.from_csv('segmentation/MA01/results5/95out_ON0062.csv')
    # plot_comparison(img, label_img, "Label")