# -*- coding: cp936 -*-
"""
提取BoVW特征(K-means or Sparse coding)

@author: shuaiyi
"""

# from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from skimage.io import imread
import numpy as np
#import operator
import logging
logging.getLogger().setLevel(logging.ERROR)

from PIL import Image
img_size = (256,256,3)
def resize(img):
    tmp = Image.fromarray(img)
    tmp = tmp.resize(img_size[0:2])
    return np.array(tmp)

class BoVWFeature(TransformerMixin):
    """ 
    Extract BoVW Feature
        
    Parameters
    ----------
    codebook_size : int
      the size of codebook, default:1000
    
    method : str
      codebook's compute method , value: 'k-means' or 'sc'
      
    """
    def __init__(self, codebook_size=1000, method='sc'):
        self.codebook_size = codebook_size
        self.method = method

    
    def fit(self, X, y):
        # compute the codes
        return self
    
    def transform(self, X):
        """         
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, 1]
            Training vectors, where n_samples is the number of samples and
            1 is image path.
      
        Returns
        -------

          array-like = [n_samples, decaf_features]
            Class labels predicted by each classifier.
        
        """
        results = []
        for sample in X:
            tmp = imread(str(sample[0]))
            if tmp.shape != self.size:
                tmp = resize(tmp)
            NET.classify(tmp, True)
            results.append(NET.feature(self.layer))
        return np.vstack(results)
    
    def get_params(self, deep=True):
        return {"layer": self.layer}


