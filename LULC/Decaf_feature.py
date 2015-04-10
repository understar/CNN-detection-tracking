# -*- coding: cp936 -*-
"""
提取decaf特征，定义为scikit，transform样式，方便集成

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

from decaf.scripts.imagenet import DecafNet
NET = DecafNet()

class DecafFeature(TransformerMixin):
    """ 
    Extract Decaf Feature
        
    Parameters
    ----------
    layer_name : str
      Decaf layer name, default:fc6_cudanet_out
    
    img_size : tuple
      the size of X, default: (256, 256, 3)
      
    """
    def __init__(self, layer='fc6_cudanet_out', img_size=(256, 256, 3)):
        self.layer = layer
        self.size = img_size

    
    def fit(self, X, y):
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


