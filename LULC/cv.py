# -*- coding: cp936 -*-
"""
Created on Sat May 09 10:23:10 2015

@author: Administrator
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import cross_val_score
from spm import SPMFeature
import argparse
import time, os
import logging
logging.getLogger().setLevel(logging.WARN)

"""Arguments"""
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "path to the dataset file (*.pkl)")
ap.add_argument("-c", "--clusters", type = int, default = 1000,
	help = "k-means clusters") 
ap.add_argument("-i", "--imgsize", type = int, default = 600,
	help = "image size") 

args = vars(ap.parse_args())
fpath = "{0}_{1}_{2}.pkl".format(args['dataset'][0:-4],args['clusters'], args['imgsize'])

if not os.path.exists(fpath):
    print "Fristly, extract features to", "RSDataset_{0}_{1}.pkl".format(args['clusters'], args['imgsize'])
    cmd = "extractfeatures.py -d {0} -c {1} -i {2}".format(args['dataset'], args['clusters'], args['imgsize'])
    os.system(cmd)
else:
    print "CV..."
    all_ftrs = joblib.load(fpath, 'r+')

    # 将类别名称编码为数字
    le = LabelEncoder()
    all_labels = all_ftrs.keys()
    le.fit(all_labels)
    # 遍历数据集
    all_x=[]
    all_y=[]
    for k,v in all_ftrs.items():
        print "Loading features", k
        X = []
        Y = []
        for item in v:
            Y.append(k)
            X.append(item)
        Y = le.transform(Y)
        X = np.vstack(X)
        all_x.append(X)
        all_y.append(Y)
        
    all_x = np.vstack(all_x)
    all_y = np.hstack(all_y)
    
    cv = StratifiedShuffleSplit(y=all_y, n_iter=10, test_size=0.4)
    for c in [0.01]:
        clf = SVC(C=c, kernel='linear', probability = True,random_state=42)
        scores = cross_val_score(clf, all_x, all_y, cv=cv, verbose=1)
        print("Accuracy-%0.3f : %0.3f (+/- %0.3f)" % (c, scores.mean(), scores.std() * 2))
    if False:
        f = open("RS_results/{0}_{1}_{2}.txt".format(args['dataset'][0:-4],args['clusters'], args['imgsize']), 'w')
        f.writelines("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
        f.close()
        
        np.save("RS_results/{0}_{1}_{2}.npy".format(args['dataset'][0:-4],args['clusters'], args['imgsize']),scores)
    
    
"""
svc = svm.SVC(kernel='linear')
C_s = np.logspace(-10, 0, 10)

scores = list()
scores_std = list()
for C in C_s:
    svc.C = C
    this_scores = cross_validation.cross_val_score(svc, X, y, n_jobs=1)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))

# Do the plotting
import matplotlib.pyplot as plt
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.semilogx(C_s, scores)
plt.semilogx(C_s, np.array(scores) + np.array(scores_std), 'b--')
plt.semilogx(C_s, np.array(scores) - np.array(scores_std), 'b--')
locs, labels = plt.yticks()
plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
plt.ylabel('CV score')
plt.xlabel('Parameter C')
plt.ylim(0, 1.1)
plt.show()

"""