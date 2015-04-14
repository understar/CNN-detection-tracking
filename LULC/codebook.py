# -*- coding: cp936 -*-
"""
Created on Fri Oct 10 15:51:13 2014

@author: shuaiyi
"""
import cv2
import os
import numpy as np
from numpy.random import random
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from skimage import color

# from skimage import color
# from skimage.io import imread

# codebook size
k_means_n = 400 

# detector = cv2.SURF(400, _extended=True)

#提取surf
def surf(img_path):
     # surf 阈值越小，特征点越多
    detector = cv2.SURF(350, _extended=True)
    dst = None
    if type(img_path) == type(""):
        src = cv2.imread(img_path)
        dst = cv2.cvtColor(src, cv2.cv.CV_RGB2GRAY)
    else:
        #print img_path.dtype, img_path.shape
        dst = (color.rgb2gray(img_path)*255).astype(np.uint8)
        # print dst.dtype
        # dst = cv2.cvtColor(img_path, cv2.cv.CV_RGB2GRAY)
    # print dst.dtype, dst.shape
    kp, des= detector.detect(dst, None, useProvidedKeypoints = False)
    if len(kp) != 0:
        return kp, des.reshape((len(kp),detector.descriptorSize())) # kp 行 128列
    else:
        return None, None

# 获取bow特征直方图
def computeHistograms(km, descriptors):
    code = km.predict(descriptors)
    histogram_of_words, bin_edges = np.histogram(code,
                                              bins=range(k_means_n + 1),
                                              normed=True)
    return histogram_of_words

    
allfeatures = {}

def extractFeatures(num = 500):
    i = 0
    samples = []
    for root, dirs, files in os.walk("samples"):
        for f in files:
            if f[-3:] == 'tif':
                samples.append(f)
    
    np.random.shuffle(samples) # 打乱数据
    for f in samples:
        if i < num:
            img_path = ""
            if f[0:8] == "points99":
                img_path = "samples/s_negtive/%s"%f
            else:
                img_path = "samples/s_postive/%s"%f
            if random() > 0.7:
                print img_path
                kp, des = surf(img_path)
                if kp != None:
                    allfeatures[img_path] = des
                    i = i + 1
        else:
            break
    
    return np.vstack(allfeatures.values())
    
def BoW(img_path, km):
    #print img_path.dtype
    kp, des = surf(img_path)
    if kp != None:
        return computeHistograms(km, des)
    else:
        return None
    
def build_training_dataset(km):
    X = []
    Y = []
    print "## compute the visual words histograms for each image"
    for root, dirs, files in os.walk("samples"):
        for f in files:
            img_path = ""
            label = 0
            if f[0:8] == "points99":
                img_path = "samples/s_negtive/%s"%f
                label = 1
            elif f[0:8] == "points00":
                img_path = "samples/s_postive/%s"%f
                label = 2
            else:
                continue
            
            print img_path
            kp, des = surf(img_path)
            if kp != None:
                Y.append(label)
                X.append(computeHistograms(km, des).ravel())
    return np.vstack(X), np.vstack(Y)
    
if __name__ == "__main__":
    # cal surf features
    code_train = False
    print "## computing the visual words via k-means"
    print "## extract surf features"
    if code_train:
        all_f = extractFeatures()
        # using k-mean train codebook
        print "## k-means"
        km = KMeans(n_clusters = k_means_n, random_state = 42, 
                    verbose = 1, n_jobs = -2, tol = 0.1)
        km.fit(all_f)
        joblib.dump(km, 'dict_k_means.pkl', compress = 3)
        
    km = joblib.load('dict_k_means.pkl')
    
    #codebook = km.cluster_centers_
    
    #codebook, distortion = vq.kmeans(all_f,
    #                                 200,
    #                                 thresh= 0.1)
    #with open("codebook.pkl", 'wb') as f:
    #    dump(codebook, f, protocol=HIGHEST_PROTOCOL)
    X,Y = build_training_dataset(km)
    np.save("train_BoW_x.npy", X)
    np.save("train_BoW_y.npy", Y)