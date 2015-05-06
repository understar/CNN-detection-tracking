# -*- coding: cp936 -*-
"""
Created on Sat Apr 25 09:59:47 2015
所有的影像均为灰度影像
@author: Administrator
"""

# from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib
from sklearn.feature_extraction.image import extract_patches_2d
from scipy.cluster.vq import vq
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from sift import SiftFeature
from SparseCode import Sparsecode, show
from raw import RawFeature
from PIL import Image
import logging
import time
import progressbar

class SPMFeature(TransformerMixin):
    """ 
    Extract SPM Feature
        
    Parameters
    ----------
    clusters : int
      K-means n_clusters, default:1024
    
    size : int
      the size of patch, default: 16
      
    method : str
      the feature extraction method, values: {'sc', 'raw', 'sift'}, 
      default: 'sc'
     
    level: int
      the level of Spatial Pyramid Matching, default: 2
    """
    def __init__(self, patch_file=None, patch_num=100000, level=2, clusters=1024, 
                 img_size=256, size=16, method='sc', exist=True,
                 all_x=None):
        self.patch_file = patch_file
        self.patch_num = patch_num
        self.img_size = img_size
        self.clusters = clusters
        self.size = size
        self.method = method
        self.level = level
        self.exist = exist
        self.all_x = all_x
    
    def fit(self, X=None, y=None):
        self.kmeans = MiniBatchKMeans(n_clusters=self.clusters,
                                      n_init=10, verbose=1,
                                      max_no_improvement=100,
                                      reassignment_ratio=0.01,
                                      batch_size = 300,
                                      random_state=np.random.RandomState(42))

        
        if self.patch_file is None:
            logging.info("Extract Patches for training...")
            num = self.patch_num // self.all_x.size
            data = []
            for item in self.all_x:
                img = imread(str(item[0]))
                img = img_as_ubyte(rgb2gray(img))
                if self.img_size != img.shape[0]:
                    img = self.resize(img, (self.img_size, self.img_size))
                tmp = extract_patches_2d(img, (self.size, self.size),
                                         max_patches = num,
                                         random_state=np.random.RandomState())
                data.append(tmp)
            
            data = np.vstack(data)
            data = data.reshape(data.shape[0], -1)
            data = np.asarray(data, 'float32')
        else:
            data = np.load(self.patch_file,'r+') # load npy file, 注意模式，因为后面需要修改
        
        data = np.require(data, dtype=np.float32)
        
        if self.method  == 'sift': # sift不能pickle，由于cv2的原因
            self.efm = SiftFeature(self.size)
            logging.info("Fit and Transform Feature Extraction Transformer.")
            data = self.efm.fit_transform(data)
            
        elif not self.exist:
            if self.method == 'sc': # 主要是sc需要pickle，训练太耗时
                self.efm = Sparsecode(patch_file=self.patch_file, n_components=384,
                                      alpha = 1, n_iter=500, batch_size=200)
                
                self.efm.coder.split_sign=True
                
            elif self.method == 'raw':
                self.efm = RawFeature()
            else:
                self.efm = RawFeature()
                
            logging.info("Fit and Transform Feature Extraction Transformer.")
            data = self.efm.fit_transform(data)
            
        else:
            logging.info("Loading Feature Extraction Transformer.")
            self.efm = joblib.load('efm/%s.pkl'%self.method)
            data = self.efm.transform(data)            
        
        logging.info("SPM Learning K-means Clusters.")
        self.kmeans.fit(data)
        return self
    
    def transform(self, X):
        results = []
        pbar = progressbar.ProgressBar(maxval=X.shape[0]).start()
        cnt = 0
        
        for sample in X:
            name = str(sample[0])
            logging.info("Processing %s." % name)
            img = imread(name)
            img = img_as_ubyte(rgb2gray(img)) # 目前只处理灰度图像
            if self.img_size != img.shape[0]:
                img = self.resize(img, (self.img_size, self.img_size))
            
            logging.info("[%s] 1. Extract Dense Patches (Default Step is 8)." %time.ctime())
            patches = self.extract_patches(img)
            
            logging.info("[%s] 2. Compute the feature for each patches."%time.ctime())
            tmp = np.array([i for x,y,i in patches])
            tmp = self.efm.transform(tmp)
            
            logging.info("[%s] 3. (VQ) Build Histogram for the Image At Different Levels."%time.ctime())
            img_ftrs = [(xyp[0],xyp[1],ftr) for xyp,ftr in zip(patches, tmp)]
            desc = self.buildHistogramForEachImageAtDifferentLevels(img, img_ftrs)
            results.append(desc)
            pbar.update(cnt+1)
            cnt = cnt + 1
                
        pbar.finish()
        return np.vstack(results)
        
    def extract_patches(self, arr, steps=8):
        m, n = arr.shape
        x,y = np.meshgrid(range(0,m-self.size+1,steps),
                          range(0,n-self.size+1,steps))
        xx,yy = x.flatten(),y.flatten()
        return [(i+self.size//2,j+self.size//2,arr[i:i+self.size,j:j+self.size].flatten()) for i,j in zip(xx,yy)]
        
    
    def buildHistogramForEachImageAtDifferentLevels(self, arr, ftrs, level=2):
        """
        build spatial pyramids of an image based on the attribute of level
        """
        width, height = arr.shape
        widthStep = int(width / 4)
        heightStep = int(height / 4)

        descriptors = ftrs

        # level 2, a list with size = 16 to store histograms at different location
        histogramOfLevelTwo = np.zeros((16, self.kmeans.n_clusters))
        for x, y, feature in descriptors:
            boundaryIndex = int(x / widthStep)  + int(y / heightStep) *4

            feature = feature.reshape(1, feature.size)

            codes, distance = vq(feature, self.kmeans.cluster_centers_)
            histogramOfLevelTwo[boundaryIndex][codes[0]] += 1

        # level 1, based on histograms generated on level two
        histogramOfLevelOne = np.zeros((4, self.kmeans.n_clusters))
        histogramOfLevelOne[0] = histogramOfLevelTwo[0] + histogramOfLevelTwo[1] + histogramOfLevelTwo[4] + histogramOfLevelTwo[5]
        histogramOfLevelOne[1] = histogramOfLevelTwo[2] + histogramOfLevelTwo[3] + histogramOfLevelTwo[6] + histogramOfLevelTwo[7]
        histogramOfLevelOne[2] = histogramOfLevelTwo[8] + histogramOfLevelTwo[9] + histogramOfLevelTwo[12] + histogramOfLevelTwo[13]
        histogramOfLevelOne[3] = histogramOfLevelTwo[10] + histogramOfLevelTwo[11] + histogramOfLevelTwo[14] + histogramOfLevelTwo[15]

        # level 0
        histogramOfLevelZero = histogramOfLevelOne[0] + histogramOfLevelOne[1] + histogramOfLevelOne[2] + histogramOfLevelOne[3]


        if level == 0:
            return histogramOfLevelZero

        elif level == 1:
            tempZero = histogramOfLevelZero.flatten() * 0.5
            tempOne = histogramOfLevelOne.flatten() * 0.5
            result = np.concatenate((tempZero, tempOne))
            return result

        elif level == 2:
            tempZero = histogramOfLevelZero.flatten() * 0.25
            tempOne = histogramOfLevelOne.flatten() * 0.25
            tempTwo = histogramOfLevelTwo.flatten() * 0.5
            result = np.concatenate((tempZero, tempOne, tempTwo))
            return result

        else:
            return None
        
    def resize(self, img, size):
        """
        img : numpy array
        size : new image size, (width, height)
        """
        tmp = Image.fromarray(img)
        tmp = tmp.resize(size)
        return np.array(tmp)
        
    def get_params(self, deep=True):
        return {"clusters": self.clusters,
                "method":self.method,
                "level":self.level,
                "img_size":self.img_size,
                "patch_file":self.patch_file}
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.__setattr__(parameter, value)
        return self
