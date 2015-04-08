# -*- coding: cp936 -*-
"""
Created on Tue Jan 13 21:49:47 2015

@author: shuaiyi
"""

import matplotlib.pyplot as plt
import os, sys
import numpy as np
from skimage.io import imread

# 加载 decaf
from kitnet import DecafNet as KitNet
net = KitNet()

img_list = []

def make_fig(imgs, blob_name, idx):
    fig = plt.figure()
    fig.text(.5, .95, '%s, %s'% (blob_name, idx), horizontalalignment='center') 
    img_size = 40
    bigpic = np.zeros((40 * 3 + 3 + 1, 40 * 3 + 3 + 1), dtype=np.single)
   
    for m in range(0, len(imgs)):
        i = np.ceil(m/3)
        j = m%3
        bigpic[1 + (1 + img_size) * i:1 + (1 + img_size) * i + img_size,
               1 + (1 + img_size) * j:1 + (1 + img_size) * j + img_size] = imgs[m]
            
    plt.xticks([])
    plt.yticks([])
    plt.imshow(bigpic, cmap=plt.cm.gray, interpolation='nearest')


def init(img_folder):
    """ 初始化图片文件夹
    """
    for folder in img_folder:
        for f in os.listdir(folder):
            if f.endswith('png'):
                img_list.append(os.path.join(folder, f))

def max_activation(blob_name, idx=0):
    """给定层，以及neuron的索引idx
        从img_list中随机计算若干张图片
        返回使得idx neuron激活值最大的9张图片
    """
    scores = {}
    for i in range(1000):
        j = np.random.randint(0, len(img_list))
        img = imread(img_list[j])
        img = img.reshape((img.shape[0], img.shape[1], 1))
        _ = net.classify(img, False)
        tmp = net.feature(blob_name)
        tmp = tmp.reshape((tmp.shape[0],-1))
        scores[img_list[j]] = tmp[0][idx]
        print "%s Done!"%img_list[j]

    scores = sorted(scores.items(), key=lambda x: x[1])
    return scores[-10:-1]
    
if __name__ == '__main__':
    init(["E:\\2013\\Samples-for-cuda-convnet\\samples-gray-train\\neg",
          "E:\\2013\\Samples-for-cuda-convnet\\samples-gray-train\\pos"])
    test = [("conv1_neuron_cudanet_out", 10),
            ("conv2_neuron_cudanet_out", 10),
            ("conv3_neuron_cudanet_out", 10)]
    for blob_name, idx in test:
        scores = max_activation(blob_name, idx)
        imgs = []
        for ip, _ in scores:
            imgs.append(imread(ip))
        make_fig(imgs, blob_name, idx)