# -*- coding: cp936 -*-
"""
Created on Sat Apr 25 16:37:06 2015

@author: shuaiyi
"""

from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler

class RawFeature(TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        self.minmax = MinMaxScaler()
        self.minmax.fit(X)
        return self
    
    def transform(self, X):
        '''
        X: array like: n_samples, n_features
        '''
        return self.minmax.transform(X)
        
    def get_params(self, deep=True):
        return None
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.__setattr__(parameter, value)
        return self
