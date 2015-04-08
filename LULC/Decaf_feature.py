# -*- coding: cp936 -*-
"""
提取decaf特征，定义为scikit，transform样式，方便集成

@author: shuaiyi
"""

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np
#import operator
import logging
logging.getLogger().setLevel(logging.ERROR)

class DecafFeature(BaseEstimator, TransformerMixin):
    """ 
    Extract Decaf Feature
        
    Parameters
    ----------
    layer_name : str
      Decaf layer name, default:fc6_cudanet_out
    
    img_size : tuple
      the size of X, default: (256, 256, 3)
      
    """
    def __init__(self, layer_name='fc6_cudanet_out', img_size=(256, 256, 3)):
        self.layer = layer_name
        self.size = img_size
        
        from decaf.scripts.imagenet import DecafNet
        self.net = DecafNet()

  
    def transform(self, X):
        """         
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
      
        Returns
        -------

          array-like = [n_samples, decaf_features]
            Class labels predicted by each classifier.
        
        """
        results = []
        for sample in X:
            self.net.classify(sample.reshape(self.size), True)
            results.append(self.net.feature(self.layer))
        return np.vstack(results)



