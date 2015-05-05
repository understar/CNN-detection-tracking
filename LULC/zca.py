# -*- coding: cp936 -*-
"""
Created on Fri Oct 10 09:36:44 2014

@author: shuaiyi
"""
 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import as_float_array
import numpy as np
import matplotlib.pyplot as plt

class ZCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, epsilon=.01, copy=True):
        self.n_components = n_components
        self.epsilon = epsilon
        self.copy = copy
 
    def fit(self, X, y=None):
        # X = array2d(X)
        n_samples, n_features = X.shape
        X = as_float_array(X, copy=self.copy) #np.require(X, dtype=np.float32) #
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        X -= self.mean_
        X /= self.std_
        sigma = np.dot(X.T, X)/n_samples
        d, V = np.linalg.eigh(sigma)
        # u,s,v = np.linalg.svd(sigma)
        
        #eigs, eigv = eigh(np.dot(X.T, X) / n_samples + \
        #                 self.bias * np.identity(n_features))
        D = np.diag(1./np.sqrt(d + self.epsilon))
        components = np.dot(np.dot(V, D), V.T)
        self.components_ = components
        return self
 
    def transform(self, X):
        X = as_float_array(X, copy=self.copy) 
        if self.mean_ is not None and self.std_ is not None:
            X -= self.mean_
            X /= self.std_
        X_whitend = np.dot(X, self.components_)
        return X_whitend
        
def show(components, patch_size):
    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(components[:100]):
        plt.subplot(10, 10, i+1)
        plt.imshow(comp.reshape(patch_size),cmap=plt.cm.gray,
                   interpolation='none')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('Show', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    
    plt.show()
    
if __name__ == "__main__":
    from skimage.data import camera
    from sklearn.feature_extraction.image import extract_patches_2d
    img = camera()
    patches = extract_patches_2d(img, (16,16), 100, np.random.RandomState())
    
    data = patches.reshape(patches.shape[0], -1)
    data = np.asarray(data, 'float32')
    
    zca = ZCA()
    whiten = zca.fit_transform(data)
    
    show(whiten, (16,16))
    show(data, (16,16))