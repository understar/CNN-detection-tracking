# -*- coding: cp936 -*-
"""
Created on Fri Apr 24 16:05:13 2015

@author: shuaiyi
"""

# from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np
import cv2

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
            tmp = np.require(sample.reshape(self.size, self.size),dtype=np.ubyte)
            # 检测点，固定size，固定angle -1必须要为-1，不然就会导致他会规划到统一方向
            kp = cv2.KeyPoint(self.size//2, self.size//2, self.size) 
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

if __name__ == "__main__":
    from skimage.data import coins
    from sklearn.feature_extraction.image import extract_patches_2d
    img = coins()
    patches = extract_patches_2d(img, (16,16),100,np.random.RandomState())
    
    patches = patches.reshape(patches.shape[0], -1)
    patches = np.asarray(patches, 'float32')
    sift = SiftFeature()
    sift.fit()
    feature = sift.transform(patches)