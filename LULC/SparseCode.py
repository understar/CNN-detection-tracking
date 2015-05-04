# -*- coding: cp936 -*-
"""
Created on Fri Apr 24 09:11:50 2015

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.decomposition import SparseCoder, MiniBatchDictionaryLearning #白化以及字典
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from zca import ZCA

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

from numpy.random import shuffle
import argparse
import logging
from sklearn.base import TransformerMixin,BaseEstimator

class Sparsecode(BaseEstimator, TransformerMixin):
    def __init__(self, patch_file=None, patch_num=10000, patch_size=(16, 16),\
                n_components=256,  alpha = 1, n_iter=2000, batch_size=100):
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
            data = np.load(self.patch_file,'r+') # load npy file, 注意模式，因为后面需要修改
        
        data = np.require(data, dtype=np.float32)
        
        # Standardization
        #logging.info("Pre-processing : Standardization...")
        #self.standard = StandardScaler()
        #data = self.standard.fit_transform(data)
            
        # whiten
        #logging.info("Pre-processing : PCA Whiten...")
        #self.pca = RandomizedPCA(copy=True, whiten=True)
        #data = self.pca.fit_transform(data)
        
        # whiten
        logging.info("Pre-processing : ZCA Whiten...")
        self.zca = ZCA()
        data = self.zca.fit_transform(data)
        
        # 0-1 scaling 都可以用preprocessing模块实现
        #self.minmax = MinMaxScaler()
        #data = self.minmax.fit_transform(data)
        
        """k-means
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_components, init='k-means++', \
                                    max_iter=self.n_iter, batch_size=self.batch_size, verbose=1,\
                                    tol=0.0, max_no_improvement=100,\
                                    init_size=None, n_init=3, random_state=np.random.RandomState(0),\
                                    reassignment_ratio=0.0001)
        logging.info("Sparse coding : Phase 1 - Codebook learning (K-means).")
        self.kmeans.fit(data)
        
        logging.info("Sparse coding : Phase 2 - Define coding method (omp,lars...).")
        self.coder = SparseCoder(dictionary=self.kmeans.cluster_centers_, 
                                 transform_n_nonzero_coefs=256,
                                 transform_alpha=None, 
                                 transform_algorithm='lasso_lars',
                                 n_jobs = 1)
        """
        #'''genertic
        logging.info("Sparse coding...")
        self.coder = MiniBatchDictionaryLearning(n_components=self.n_components, \
                                           alpha=self.alpha, n_iter=self.n_iter, \
                                           batch_size =self.batch_size, verbose=True)
        self.coder.fit(data)
        #'''
        return self
    
    def transform(self, X):
        #whiten
        #X_whiten = self.pca.transform(X)
        logging.info("Compute the sparse coding of X.")
        X = np.require(X, dtype=np.float32)
        
        #TODO: 是否一定需要先fit，才能transform
        #X = self.minmax.fit_transform(X)
        
        # -mean/std and whiten
        #X = self.standard.transform(X)
        #X = self.pca.transform(X)
        
        # ZCA
        X = self.zca.transform(X)

        # MiniBatchDictionaryLearning
        # return self.dico.transform(X_whiten)
        
        # k-means
        # TODO: sparse coder method? problem...
        return self.coder.transform(X)
        
    
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
    logging.getLogger().setLevel(logging.INFO)
    """Arguments"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--patches", required = True,
                    help = "patch for patches saving")
    args = vars(ap.parse_args())
    
    patches = args["patches"]
    sc = Sparsecode(patches, n_iter=100, batch_size=200, n_components=256)
    sc.fit()
    
    logging.info('Show Compoents...')
    show(sc.coder.components_, (16,16))
    
    logging.info('Coding...')
    data = np.load(patches,'r+')[0:100]
    code = sc.transform(data)
    
    """
    img = imread(str(x_test[0,0]))
    img = img_as_ubyte(rgb2gray(img)) 
    patches = spm.extract_patches(img)
    tmp = np.array([i for x,y,i in patches])
    # tmp = tmp.astype(np.float32)
    x_t = spm.efm.transform(tmp)
    len(np.flatnonzero(x_t))
    """