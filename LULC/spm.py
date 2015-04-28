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
from scipy.cluster import vq
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from sift import SiftFeature
from SparseCode import Sparsecode, show
from raw import RawFeature

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
    def __init__(self, patch_file, level=2, clusters = 1024, size=16, method='sc'):
        self.patch_file = patch_file
        self.clusters = clusters
        self.size = size
        self.method = method
        self.level = level
    
    def fit(self, X=None, y=None):
        self.kmeans = MiniBatchKMeans(n_clusters=self.clusters, n_init=10, verbose=1)
        
        X = np.load(self.patch_file,'r+')
        
        if self.method  == 'sift':
            self.efm = SiftFeature(self.size)
        elif self.method == 'sc':
            self.efm = Sparsecode(patch_file=self.patch_file)
        elif self.method == 'raw':
            self.efm = RawFeature()
        else:
            self.efm = RawFeature()
        
        X = self.efm.fit_transform(X)
        
        print "SPM K-means..."
        self.kmeans.fit(X)
        return self
    
    def transform(self, X):
        results = []
        for sample in X:
            img = imread(str(sample[0]))
            img = img_as_ubyte(rgb2gray(img)) # 目前只处理灰度图像
            patches = self.extract_patches(img)
            tmp = np.array([i for x,y,i in patches])
            tmp = self.efm.transform(tmp)
            
            img_ftrs = [(xyp[0],xyp[1],ftr) for xyp,ftr in zip(patches, tmp)]
            
            desc = self.buildHistogramForEachImageAtDifferentLevels(img, img_ftrs)
            results.append(desc)
        return np.vstack(results)
        
    def extract_patches(self, arr, steps=8):
        m, n = arr.shape
        x,y = np.meshgrid(range(0,m-self.size+1,steps),
                          range(0,n-self.size+1,steps))
        xx,yy = x.flatten(),y.flatten()
        return [(i,j,arr[i:i+self.size,j:j+self.size].flatten()) for i,j in zip(xx,yy)]
        
    
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
        histogramOfLevelOne = np.zeros((4, self.size))
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
        
    def get_params(self, deep=True):
        return {"clusters": self.clusters,
                "method":self.method,
                "level":self.level}
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.__setattr__(parameter, value)
        return self
