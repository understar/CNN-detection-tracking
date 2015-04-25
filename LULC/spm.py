# -*- coding: cp936 -*-
"""
Created on Sat Apr 25 09:59:47 2015

@author: Administrator
"""

# from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sift import SiftFeature
from SparseCode import Sparsecode, show

class SPMFeature(TransformerMixin):
    """ 
    Extract SPM Feature
        
    Parameters
    ----------
    clusters : int
      K-means n_clusters, default:1024
    
    size : int
      the size of patch, default: (256, 256, 3)
      
    method : str
      the feature extraction method, values: {'sc', 'raw', 'sift'}, 
      default: 'sc'
      
    """
    def __init__(self, clusters = 1024, size=16, method='sc'):
        self.clusters = clusters
        self.size = size
        self.method = method

    
    def fit(self, X, y=None):
        self.kmeans = MiniBatchKMeans()
        self.sift = cv2.SIFT()
        return self
    
    def transform(self, X):
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
