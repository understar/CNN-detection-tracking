# -*- coding: cp936 -*-
"""
Created on Thu Dec 11 16:41:58 2014

@author: shuaiyi
"""
import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning) 
def customwarn(message, category, filename, lineno, file=None, line=None):
    pass
warnings.showwarning = customwarn

import matplotlib.pyplot as plt
import os, sys
import numpy as np
from scipy.stats import itemfreq
# import pandas as pd
import skimage
import skimage.measure as sme
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.morphology import label, square, disk
from skimage.morphology import erosion, dilation, opening, closing
# white_tophat, black_tophat, skeletonize

from filter_polys import filter_poly # filter
from poly2ras import poly_ras # to raster

from kit_angle_net import DecafNet as AngleNet

_DETECTION = None
_ANGLE = AngleNet()
_WIDTH = 40

Decaf = False
if Decaf:
    from kitnet import DecafNet as KitNet
    _DETECTION = KitNet()
else:
    #print os.getcwd()
    os.chdir("E:/2013/cuda-convnet/trunk")
    sys.path.append(".")
    #print os.getcwd()
    from show_pred import model as car_model
    _DETECTION = car_model
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

def Cal_Angle(img, detection_net=_DETECTION, angle_net=_ANGLE):
    # TODO: prob filter 0.95会不会太严格
    if Decaf:
        scores = detection_net.classify(img, False)
        is_car = detection_net.top_k_prediction(scores, 1)
        #print is_car
        if is_car[1][0] == 'car' and is_car[0][0] >= 0.9:
            car_conv3 = detection_net.feature("conv3_neuron_cudanet_out")
            mid_convs = car_conv3.reshape((car_conv3.shape[0],-1))
            scores = angle_net.classify(mid_convs)
            angles = angle_net.top_k_prediction(scores, 3)
            
            o = float(angles[1][0])-90 # *180/np.pi
            o = o if o>0 else o+360        
            return int(o), is_car[0][0]
    else:
        is_car = car_model.show_predictions(img[:,:,0])
        # print is_car
        if is_car[-1][0] == 'car' and is_car[-1][1] >= 0.9:
            mid_convs = car_model.get_features(img[:,:,0])
            scores = angle_net.classify(mid_convs)
            angles = angle_net.top_k_prediction(scores, 3)
            #print angles
            
            o = float(angles[1][0])-90 # *180/np.pi
            o = o if o>0 else o+360
            return int(o), is_car[-1][1]
    return None

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
    import progressbar
    pbar = progressbar.ProgressBar(maxval=len(props)).start()  
    cnt = 0
    
    results = []
    for r in props:
        bbox = r.bbox
        oid = r.label
        area = r.area
        # TODO: 检测一副远远不够，在附近多检测几张
        cy = int((bbox[1]+bbox[3])/2)
        cx = int((bbox[0]+bbox[2])/2)
        k_n = 5 # 5*5
        angles = np.zeros((k_n,k_n))
        probs = np.zeros((k_n,k_n))
        for i in np.arange(-int(k_n/2),int(k_n/2)+1):
            for j in np.arange(-int(k_n/2),int(k_n/2)+1):
                x = cx + i
                y = cy + j
                min_row, max_row, min_col, max_col = \
                    x-_WIDTH/2, x+_WIDTH/2, y-_WIDTH/2, y+_WIDTH/2
                # print min_row, max_row, min_col, max_col

                img = input_src[min_row:max_row, min_col:max_col, :]
                if img.shape != (_WIDTH, _WIDTH, 3):
                    continue
                else:
                    # imsave("%s_%s.png"%(cx,cy), img)
                    # print img.shape
                    img = rgb2gray(img)
                    # print img.shape
                    img = skimage.img_as_ubyte(img.reshape((_WIDTH, _WIDTH,1)))
                    angle = Cal_Angle(img)
                    if angle == None:
                        continue
                    else:
                        angles[i+int(k_n/2),j+int(k_n/2)] = angle[0]
                        probs[i+int(k_n/2),j+int(k_n/2)] = angle[1]
        # TODO: 根据检测为Car的结果所占的比例过滤? 多大比例合适
        # 确实可以过滤一部分
        ratio = float(np.count_nonzero(angles))/float(k_n*k_n)
        # print ratio
        if ratio > 0.9: #len(angles) != 0:
            results.append((oid, cx, cy, area, angles, img, probs, ratio))
        
        pbar.update(cnt+1)
        cnt += 1

    pbar.finish()
    return results

def Avg_angle(nn):
    #TODO: 通过邻域角度取众数？如何分析邻域得到角度
    # 如果有多个众数的情况？目前取第一个
    # 角度包含更多的信息，如果邻域检测不一致性较差的情况
    # 通过角度重视所占的比例不够合理，暂不复杂化问题
    freq=itemfreq(nn.ravel())
    max_count = freq.max(0)[1]
    if False: #float(max_count)/float(nn.size) < 0.5: # 如果一致的角度数量少于一半，就搞死掉
        return None
    else:
        for i in range(freq.shape[0]):
            if freq[i,1] == max_count:
                return freq[i,0]
    

def writeprops(props, fname):
    with open(fname, 'w') as f:
        f.writelines(['id,', 'x,', 'y,', 'area,', 'angle', '\n'])
        for r in props:
            #if r[6] == 1:
            angle = Avg_angle(r[4])
            if angle != None:
                f.writelines(["%s,"%r[0], "%s,"%r[1], "%s,"%r[2],
                              "%s,"%r[3], "%s"%angle ,'\n']) #, "%s"%r[6] 
                #skimage.io.imsave(fname[:-4]+"_%s_%s.png"%(r[0], 100*r[7]), r[5])
    
if __name__ == '__main__':
    # load shp
    # shp filter
    # shp 2 ranster
    if len( sys.argv ) < 3:
        print "[ ERROR ]: you need to pass at least two arg -- source image -- input multi-scale output images"
        sys.exit(1)
    
    # 加载数据    
    src = imread(sys.argv[1]) # 原始影像
    imgs = [] # 多尺度分割过滤结果
    for argc in sys.argv[2:]:
        img = imread(argc) # *255
        imgs.append(img)
    
    # 求 交集
    # 先合并成一个多通道的，再相加
    intersection = np.array(imgs).sum(0)
    delta = 255*len(imgs)
    img = (intersection >= delta).astype(np.ubyte)
    #形态学操作
    #plt.figure(); plt.imshow(src)
    #plt.figure(); open1 = opening(img, disk(1)); plt.imshow(open1)
    #plt.figure(); open3 = opening(img, disk(3)); plt.imshow(open3)
    #plt.figure(); close3 = closing(open3, disk(3)); plt.imshow(close3)
    open3 = opening(img, disk(3))
    img = open3 #closing(open3, disk(3))
    
    
    props = ras2loc(img, src)
    results = refilter(props, src)
    
    #dir_path = os.path.split(sys.argv[2])[0]
    #fname = os.path.split(sys.argv[2])[1].split('_')[1][:-4] + '.csv'
    #writeprops(results, os.path.join(dir_path, fname))
    
    writeprops(results, sys.argv[1][:-4] + '.csv')