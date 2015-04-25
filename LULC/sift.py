# -*- coding: cp936 -*-
"""
Created on Fri Apr 24 16:05:13 2015

@author: shuaiyi
"""

# from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np
import cv2

'''
import cv2
from skimage.data import lena
img = lena()
sift = cv2.SIFT()
kps = sift.detect(img)
kp.angle kp.pt kp.response kp.size
kp = cv2.KeyPoint(10,10,16)
kps, desc = sift.compute(img, [kp])
'''

class SiftFeature(TransformerMixin):
    """
    extract sift desc;
    input is a img patch: size = 16*16
    """ 
    def __init__(self, size=16):
        self.size = size
        self.sift = cv2.SIFT()

    
    def fit(self, X=None, y=None):
        return self
    
    def transform(self, X):
        '''
        X: array like: n_samples, n_features
        '''
        results = []
        for sample in X:
            tmp = sample.reshape(self.size, self.size)
            # 检测点，固定size，固定angle
            kp = cv2.KeyPoint(self.size//2,self.size//2,self.size, _angle=0)
            _, desc = self.sift.compute(tmp,[kp])
            desc = self.normalizeSIFT(desc)
            results.append(desc)
        return np.vstack(results)
        
    def get_params(self, deep=True):
        return {"size": self.size}
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.__setattr__(parameter, value)
        return self
        
    def normalizeSIFT(self, descriptor):
        '''
        Normal the sift: L2 norm
        '''
        descriptor = np.array(descriptor)
        norm = np.linalg.norm(descriptor)
    
        if norm > 1.0:
            descriptor /= float(norm)
    
        return descriptor
