# -*- coding: cp936 -*-
"""
Created on Fri Apr 24 09:11:50 2015

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.decomposition import MiniBatchDictionaryLearning, RandomizedPCA #白化以及字典
from sklearn.feature_extraction.image import extract_patches_2d

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

from numpy.random import shuffle
import argparse

from sklearn.base import TransformerMixin,BaseEstimator

class Sparsecode(BaseEstimator, TransformerMixin):
    def __init__(self, patch_file=None, patch_num=10000, patch_size=(16, 16),\
                n_components=512,  alpha = 1, n_iter=1000, batch_size=100):
        self.patch_num = patch_num
        self.patch_size = patch_size
        self.patch_file = patch_file
        
        self.n_components = n_components
        self.alpha = alpha #sparsity controlling parameter
        self.n_iter = n_iter
        self.batch_size = batch_size

    
    def fit(self, X=None, y=None):
        if self.patch_file is None:
            num = self.patch_num // X.size
            data = []
            for item in X:
                img = imread(str(item[0]))
                img = img_as_ubyte(rgb2gray(img))
                #img = self.binary(img) # 二值化
                tmp = extract_patches_2d(img, self.patch_size, max_patches = num,\
                                        random_state=np.random.RandomState())
                data.append(tmp)
            
            data = np.vstack(data)
            data = data.reshape(data.shape[0], -1)
            data = np.asarray(data, 'float32')
        else:
            data = np.load(self.patch_file,'r') # load npy file
        
        # whiten
        print 'PCA Whiten...'
        self.pca = RandomizedPCA(copy=True, whiten=True)
        data = self.pca.fit_transform(data)
        
        # 0-1 scaling 都可以用preprocessing模块实现
        #data = data - np.min(data, 0)
        #data = data/(np.max(data, 0) + 0.0001) 
        
        self.dico = MiniBatchDictionaryLearning(n_components=self.n_components, \
                                           alpha=self.alpha, n_iter=self.n_iter, \
                                           batch_size =self.batch_size, verbose=True)
        self.dico.fit(data)
        return self
    
    def transform(self, X):
        X_whiten = self.pca.transform(X)
        return self.dico.transform(X_whiten)
    
    def get_params(self, deep=True):
        return {"patch_num": self.patch_num,
                "patch_size":self.patch_size,
                "alpha":self.alpha,
                "n_components":self.n_components,
                "n_iter":self.n_iter,
                "batch_size":self.batch_size}
                
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.__setattr__(parameter, value)
        return self
 

def show(components, patch_size):
    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(components[:100]):
        plt.subplot(10, 10, i+1)
        plt.imshow(comp.reshape(patch_size),cmap=plt.cm.gray,
                   interpolation='none')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('100 components extracted by SC', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    
    plt.show()
    
if __name__ == "__main__":
    """Arguments"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--patches", required = True,
                    help = "patch for patches saving")    
    args = vars(ap.parse_args())
    
    patches = args["patches"]
    sc = Sparsecode(patches, alpha=0.7,n_iter=100)
    sc.fit()
    
    print 'Show...'
    show(sc.dico.components_, (16,16))