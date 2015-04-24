# -*- coding: cp936 -*-
"""
Created on Fri Apr 24 16:05:13 2015

@author: shuaiyi
"""

# from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from skimage.io import imread
import numpy as np
import cv2

'''
import cv2
from skimage.data import lena
img = lena()
sift = cv2.SIFT()
kps = sift.detect(img)
kp.angle kp.pt kp.response kp.size
kp = cv2.KeyPoint(10,10,2)
kps, desc = sift.compute(img, [kp])

def normalizeSIFT( descriptor):
    descriptor = np.array(descriptor)
    norm = np.linalg.norm(descriptor)

    if norm > 1.0:
        descriptor /= float(norm)

    return descriptor
'''

class SiftFeature(TransformerMixin):
    def __init__(self, layer='fc6_cudanet_out', img_size=(256, 256, 3)):
        self.layer = layer
        self.size = img_size

    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        results = []
        for sample in X:
            tmp = imread(str(sample[0]))
            if tmp.shape != self.size:
                tmp = resize(tmp)
            NET.classify(tmp, True)
            results.append(NET.feature(self.layer))
        return np.vstack(results)


