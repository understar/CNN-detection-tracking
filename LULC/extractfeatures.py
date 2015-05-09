# -*- coding: cp936 -*-
"""
Created on Fri May 08 14:37:39 2015

@author: shuaiyi
"""
import numpy as np
from sklearn.externals import joblib
from PIL import Image
from spm import SPMFeature
import pickle as pkl
import argparse
import time
import logging
logging.getLogger().setLevel(logging.WARN)

"""Arguments"""
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "path to the dataset file (*.pkl)")
ap.add_argument("-c", "--clusters", type = int, default = 1000,
	help = "k-means clusters") 
ap.add_argument("-i", "--imgsize", type = int, default = 600,
	help = "image size") 

args = vars(ap.parse_args())

dataset = pkl.load(file(args["dataset"], 'rb'))

# 遍历数据集，为字典训练准备patches
all_x=[]
for k,v in dataset.items():
    print "Processing", k
    X = []
    for item in v:
        X.append(item)
    X = np.vstack(X)
    all_x.append(X)
    
all_x = np.vstack(all_x)

# 定义特征提取器，SPM，默认level为2，为1的时候就是BOF
spm = SPMFeature(clusters=args['clusters'], patch_file=None,
                 method='sift', img_size=args['imgsize'],
                 all_x=all_x, kernel_hi=False, level=2,
                 patch_num=100000)
spm.fit()

# 提取特征 
all_ftrs = {}
for k,v in dataset.items():
    print "Extracting", k
    X = []
    for item in v:
        X.append(item)
    X = np.vstack(X)
    X_ftrs = spm.transform(X)
    all_ftrs[k] = X_ftrs

# 保存
joblib.dump(all_ftrs, "RSDataset_{0}_{1}.pkl".format(args['clusters'], args['imgsize']),
            compress=3)
