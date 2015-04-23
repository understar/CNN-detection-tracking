# -*- coding: cp936 -*-
"""
Created on Sat Apr 18 10:16:27 2015

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC, SVC
from sklearn import linear_model, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.neural_network import BernoulliRBM
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage.filter import sobel, threshold_otsu
from numpy.random import shuffle

import pickle as pkl
import argparse
import time

from sklearn.base import TransformerMixin,BaseEstimator
class BoWFeature(BaseEstimator, TransformerMixin):
    def __init__(self, patch_num=10000, patch_size=(8, 8), sample_num = 300,\
                n_components=256, learning_rate=0.03, n_iter=100, batch_size=100):
        self.patch_num = patch_num
        self.patch_size = patch_size
        self.sample_num = sample_num
        
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size

    
    def fit(self, X, y=None):
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
        
        # 二值化后不需要0-1归化
        data = data - np.min(data, 0)
        data = data/(np.max(data, 0) + 0.0001)  # 0-1 scaling
        
        self.rbm = BernoulliRBM(n_components=self.n_components,\
                        learning_rate=self.learning_rate, \
                        n_iter=self.n_iter,\
                        batch_size=self.batch_size,\
                        verbose=True)
        self.rbm.fit(data)
        return self
    
    def transform(self, X):
        results = []
        for sample in X:
            img = imread(str(sample[0]))
            img = img_as_ubyte(rgb2gray(img))
            #img = self.binary(img)
            patches = extract_patches_2d(img, self.patch_size,\
                                         max_patches = self.sample_num,\
                                         random_state=np.random.RandomState())
            
            patches = patches.reshape(patches.shape[0], -1)
            patches = np.asarray(patches, 'float32')
            
            patches = patches-np.min(patches, 0)
            patches = patches/(np.max(patches, 0) + 0.0001)

            patches = self.rbm.transform(patches)
            results.append(patches.sum(axis=0))
        return np.vstack(results)
    
    def get_params(self, deep=True):
        return {"patch_num": self.patch_num,
                "sample_num":self.sample_num,
                "patch_size":self.patch_size,
                "learning_rate":self.learning_rate,
                "n_components":self.n_components,
                "n_iter":self.n_iter,
                "batch_size":self.batch_size}
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.__setattr__(parameter, value)
        return self
        
    def binary(self, img):
        edge = sobel(img)
        thresh = threshold_otsu(edge)
        edge = edge>=thresh
        return edge.astype(np.int)
 

def show(components, patch_size):
    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(components[:100]):
        plt.subplot(10, 10, i+1)
        plt.imshow(comp.reshape(patch_size),cmap=plt.cm.gray,
                   interpolation='none')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('100 components extracted by RBM', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    
    plt.show()
       
if __name__ == "__main__":
    """Arguments"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required = True,
    	help = "path to the dataset file (*.pkl)")
    ap.add_argument("-t", "--test", required = True, type = float,
	help = "size of test split")
    ap.add_argument("-s", "--search", type = int, default = 0,
    	help = "whether or not a grid search should be performed")
    args = vars(ap.parse_args())
    
    dataset = pkl.load(file(args["dataset"], 'rb'))
    # 将类别名称编码为数字
    le = LabelEncoder()
    all_labels = dataset.keys()
    le.fit(all_labels)

    # 训练样本比率
    _percent = args["test"]
    x_train=[]
    x_test=[]
    y_train=[]
    y_test=[]

    for k,v in dataset.items():
        print "Processing", k
        X = []
        Y = []
        for item in v:
            Y.append(k)
            X.append(item)
        Y = le.transform(Y)
        X = np.vstack(X)
        
        # 按比例划分训练与测试样本集
        tmp_x_train, tmp_x_test, tmp_y_train, tmp_y_test = train_test_split(X, Y, 
                                       test_size=_percent, random_state=42)
        x_train.append(tmp_x_train)
        x_test.append(tmp_x_test)
        y_train.append(tmp_y_train)
        y_test.append(tmp_y_test)
        
    x_train = np.vstack(x_train)
    x_test = np.vstack(x_test)
    y_train = np.hstack(y_train)
    y_test = np.hstack(y_test)
    
    # 打乱
    index = np.arange(y_train.shape[0])
    shuffle(index)
    
    x_train = x_train[index,:]
    y_train = y_train[index]
    
    index = np.arange(y_test.shape[0])
    shuffle(index)
    
    x_test = x_test[index,:]
    y_test = y_test[index]
    
    if args["search"] == 1:
        # initialize the RBM
        bow = BoWFeature()
        lr = LogisticRegression()
        #svm = SVC((kernel='linear', probability = True,random_state=42)
        clf = Pipeline([('bow', bow),('lr',lr)])
        
        params = {
                "lr__C": [1, 100],
                "bow__learning_rate": [0.01, 0.03],
                "bow__n_iter": [50, 100],
                "bow__n_components": [256, 512]
                }

        
        print "SEARCHING BOW+LR"
        # perform a grid search over the parameter
        # 如果没有score，没办法搜索参数
        start = time.time()
        gs = GridSearchCV(clf, params, cv=2, n_jobs = -1, verbose = 1)
        gs.fit(x_train, y_train)
     
        # print diagnostic information to the user and grab the
        # best model
        print "\ndone in %0.3fs" % (time.time() - start)
        print "best score: %0.3f" % (gs.best_score_)
        print "BOW + LOGISTIC REGRESSION PARAMETERS"
        bestParams = gs.best_estimator_.get_params()

        # loop over the parameters and print each of them out
        # so they can be manually set
        for p in sorted(params.keys()):
            print "\t %s: %f" % (p, bestParams[p])
            
        best = gs.best_estimator_
        
        print "*********************Save*******************************"
        joblib.dump(best, "classifier_rbm.pkl", compress=3)
        joblib.dump(gs, "grid_cv_rbm.pkl", compress=3)
        
    else:
        # 直接设置参数训练
        bow = BoWFeature()
        bow.patch_num=10000
        bow.patch_size=(20,20)
        bow.learning_rate=0.001
        bow.n_components=512
        bow.n_iter=100
        bow.sample_num = 1000
        
        bow.fit(x_train)
        
        svm = SVC(kernel='linear', probability = True, random_state=42)
        svm.C = 1000
        #lr = LogisticRegression()
        #lr.C = 100
        '''
        best = Pipeline([('bow', bow),('svm',svm)])
        best.fit(x_train, y_train)
        
        print "*********************Save*******************************"
        joblib.dump(best, "classifier_rbm.pkl", compress=3)
                
    print "*********************Test*******************************"
    y_test_pre = best.predict(x_test)
    cm = confusion_matrix(y_test, y_test_pre)
    from map_confusion import plot_conf
    plot_conf(cm, range(le.classes_.size), 'RSDataset.png')
    
    from sklearn.metrics import classification_report
    with file('report_rbm.txt', 'w') as f:
        report = classification_report(y_test, y_test_pre, target_names = le.classes_)
        f.writelines(report)
    '''
    show(bow.rbm.components_,(20,20))
