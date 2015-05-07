# -*- coding: cp936 -*-
"""
Created on Tue Apr 28 09:24:50 2015

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC, SVC
from sklearn import linear_model, metrics
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from numpy.random import shuffle
from spm import SPMFeature
from SparseCode import Sparsecode, show

import pickle as pkl
import argparse
import time
import logging
logging.getLogger().setLevel(logging.WARN)

""" histogram intersection kernel如何使用？

if kernelType == "HI":

        gramMatrix = histogramIntersection(trainData, trainData)
        clf = SVC(kernel='precomputed')
        clf.fit(gramMatrix, trainLabels)

        predictMatrix = histogramIntersection(testData, trainData)
        SVMResults = clf.predict(predictMatrix)
        correct = sum(1.0 * (SVMResults == testLabels))
        accuracy = correct / len(testLabels)
        print "SVM (Histogram Intersection): " +str(accuracy)+ 
              " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"
"""

"""Arguments"""
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "path to the dataset file (*.pkl)")
ap.add_argument("-p", "--patches", help = "path to the patches file (*.npy)")
ap.add_argument("-t", "--test", required = True, type = float,
	help = "size of test split")
ap.add_argument("-s", "--search", required = True, type = int, default = 0,
	help = "whether or not a grid search should be performed")
ap.add_argument("-c", "--clusters", type = int, default = 256,
	help = "k-means clusters") 
ap.add_argument("-i", "--imgsize", type = int, default = 600,
	help = "image size") 
args = vars(ap.parse_args())

dataset = pkl.load(file(args["dataset"], 'rb'))
patches = args["patches"]
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
all_x=[]
for k,v in dataset.items():
    print "Processing", k
    X = []
    Y = []
    for item in v:
        Y.append(k)
        X.append(item)
    Y = le.transform(Y)
    X = np.vstack(X)
    
    all_x.append(X)    
    
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

all_x = np.vstack(all_x)
index = np.arange(all_x.shape[0])
shuffle(index)
all_x = all_x[index]

if args["search"] == 1:
    method = 'sift'
    spm = SPMFeature(patch_file=patches, method=method, all_x=all_x, img_size=600)
    svm = SVC(kernel='linear', probability = True,random_state=42)
    clf = Pipeline([('spm', spm),('svm',svm)])
    
    params = {
            "svm__C": [0.01, 1, 100],
            "spm__clusters": [256, 512, 1024]
            }

    
    print "SEARCHING SPM+SVM"
    # perform a grid search over the parameter
    # 如果没有score，没办法搜索参数
    start = time.time()
    gs = GridSearchCV(clf, params, cv=2, n_jobs = -1, verbose = 1)
    gs.fit(x_train, y_train)
 
    # print diagnostic information to the user and grab the
    # best model
    print "\ndone in %0.3fs" % (time.time() - start)
    print "best score: %0.3f" % (gs.best_score_)
    print "SPM + SVM PARAMETERS"
    bestParams = gs.best_estimator_.get_params()

    # loop over the parameters and print each of them out
    # so they can be manually set
    for p in sorted(params.keys()):
        print "\t %s: %f" % (p, bestParams[p])
        
    best = gs.best_estimator_
    
    print "*********************Save*******************************"
    if method != 'sift':
        joblib.dump(best, "classifier_spm.pkl", compress=3)
        joblib.dump(gs, "grid_cv_spm.pkl", compress=3)
    
else:
    # 直接设置参数训练
    method = 'sift'
    spm = SPMFeature(clusters=args['clusters'], patch_file=None,
                     method=method, img_size=args['imgsize'],
                     all_x=all_x, kernel_hi=False, level=2,
                     patch_num=200000)
    if spm.kernel_hi:
        svm = SVC(kernel='precomputed', probability = True,random_state=42)
    else:
        svm = SVC(kernel='linear', probability = True,random_state=42)
    clf = Pipeline([('spm', spm),('svm',svm)])

    best = Pipeline([('spm', spm),('svm',svm)])
    best.fit(x_train, y_train)
    
    print "*********************Save*******************************"
    if method != 'sift':
        joblib.dump(best, "classifier_spm_%s.pkl"%method, compress=3)
            
print "*********************Test*******************************"
spm = best.named_steps['spm']
y_test_pre = best.predict(x_test)

from sklearn.metrics import accuracy_score
print "Accuracy:", accuracy_score(y_test, y_test_pre)
cm = confusion_matrix(y_test, y_test_pre)
np.save('RS_results/RSDataset_%s_%s_%s.npy'%(method, spm.clusters, spm.img_size), cm)
from map_confusion import plot_conf
plot_conf(conf_arr=cm, label_list=range(le.classes_.size), 
          norm=False, save_name='RS_results/RSDataset_%s_%s_%s.png'%(method, spm.clusters, spm.img_size))

from sklearn.metrics import classification_report
with file('RS_results/report_spm_%s_%s_%s.txt'%(method, spm.clusters, spm.img_size), 'w') as f:
    report = classification_report(y_test, y_test_pre, target_names = le.classes_)
    f.writelines(report)

#show(bow.rbm.components_,(20,20))