# -*- coding: cp936 -*-
"""
Created on Thu Dec 11 16:41:58 2014

@author: shuaiyi
"""
import matplotlib.pyplot as plt
import os, sys
import numpy as np
# import pandas as pd
import skimage
import skimage.measure as sme
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.morphology import label

from skimage.morphology import erosion, dilation, opening, closing
# white_tophat, black_tophat, skeletonize

from filter_polys import filter_poly # filter
from poly2ras import poly_ras # to raster

from kitnet import DecafNet as KitNet
from kit_angle_net import DecafNet as AngleNet

_DETECTION = KitNet()
_ANGLE = AngleNet()
_WIDTH = 40

def plot_comparison(original, filtered, filter_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')

def ras2loc(input_raster, input_src):
    img = input_raster
    label_img = label(img)
    props = sme.regionprops(label_img, intensity_image=input_src)
    return props

def refilter(props, input_src):
    # TODO: 如何过滤？
    results = []
    for r in props:
        bbox = r.bbox
        oid = r.label
        area = r.area
        # TODO: 检测一副远远不够，在附近多检测几张
        cy = int((bbox[1]+bbox[3])/2)
        cx = int((bbox[0]+bbox[2])/2)        
        k_n = 5 # 5*5
        angles = []
        probs = []
        for i in np.arange(-int(k_n/2),int(k_n/2)+1):
            for j in np.arange(-int(k_n/2),int(k_n/2)+1):
                cx = cx + i
                cy = cy + j
                min_row, max_row, min_col, max_col = cx-_WIDTH/2, cx+_WIDTH/2, cy-_WIDTH/2, cy+_WIDTH/2
                # print min_row, max_row, min_col, max_col

                img = input_src[min_row:max_row, min_col:max_col, :]
                if img.shape != (_WIDTH, _WIDTH, 3):
                    continue
                else:
                    #imsave("%s_%s.png"%(cx,cy), img)
                    # print img.shape
                    img = rgb2gray(img)
                    # print img.shape
                    img = skimage.img_as_ubyte(img.reshape((_WIDTH, _WIDTH,1)))
                    angle = Cal_Angle(img)
                    if angle == None:
                        continue
                    else:
                        angles.append(angle[0])
                        probs.append(angle[1])
        if 0 != len(angles) and 0 != len(probs):
            results.append((oid, cx, cy, area, angles, img, probs))
    return results

def Cal_Angle(img, detection_net=_DETECTION, angle_net=_ANGLE):
    scores = detection_net.classify(img, False)
    is_car = detection_net.top_k_prediction(scores, 1)
    print is_car
    if is_car[1][0] == 'car':
        car_conv3 = detection_net.feature("conv3_cudanet_out")
        mid_convs = car_conv3.reshape((car_conv3.shape[0],-1))
        scores = angle_net.classify(mid_convs)
        angles = angle_net.top_k_prediction(scores, 1)
        return int(angles[1][0]), is_car[0][0]
    else:
        return None
    
def writeprops(props, fname):
    with open(fname, 'w') as f:
        f.writelines(['id,', 'x,', 'y,', 'area,', 'angle', '\n'])
        for r in props:
            #if r[6] == 1:
            f.writelines(["%s,"%r[0], "%s,"%r[1], "%s,"%r[2],
                          "%s,"%r[3], "%s"%r[4] , "%s"%r[6] ,'\n'])
            skimage.io.imsave(fname[:-4]+"_%s.png"%r[0], r[5])
    
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
    props = ras2loc(img, src)
    results = refilter(props, src)
    dir_path = os.path.split(sys.argv[2])[0]
    fname = os.path.split(sys.argv[2])[1].split('_')[1][:-4] + '.csv'
    writeprops(results, os.path.join(dir_path, fname))
    # df = pd.DataFrame.from_csv('segmentation/MA01/results5/95out_ON0062.csv')
    # plot_comparison(img, label_img, "Label")