# -*- coding: cp936 -*-
"""
Created on Fri Apr 24 10:06:45 2015

@author: Administrator
"""

import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from numpy.random import shuffle
import argparse
import pickle as pkl

if __name__ == "__main__":
    """Arguments"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required = True,
                    help = "path to the dataset file (*.pkl)")
    ap.add_argument("-s", "--save", required = True,
                    help = "patch for patches saving")    
    ap.add_argument("-n", "--number", required = False, type = int, default = 40000,
                    help = "number of patches")
    ap.add_argument("-p", "--size", required = False, type = int, default = 16,
                    help = "number of patches")
    args = vars(ap.parse_args())
    
    dataset = pkl.load(file(args["dataset"], 'rb'))

    X = []
    for k,v in dataset.items():
        print "Processing", k
        for item in v:
            X.append(item)
    X = np.vstack(X)
    
    # 打乱
    index = np.arange(X.shape[0])
    shuffle(index)
    
    X = X[index,:]
    Total = args['number']
    patch_size = args['size']
    print 'Extracting %s patchs...' % Total
    patches = []
    num =  Total // X.size
    for i, x in enumerate(X):
        if i % 10 == 0:
            print 'Processing', str(x[0])
        img = imread(str(x[0]))
        img = img_as_ubyte(rgb2gray(img))
        tmp = extract_patches_2d(img, (patch_size,patch_size), \
                                 max_patches=num, random_state=np.random.RandomState())
        patches.append(tmp)

    data = np.vstack(patches)
    data = data.reshape(data.shape[0], -1)
    data = np.asarray(data, 'float32')
    
    print 'Saving...'
    np.save(args['save'], data)
   
