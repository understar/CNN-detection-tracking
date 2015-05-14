# -*- coding: cp936 -*-
"""
Created on Mon May 11 14:44:33 2015

需要绘制的图形列表：
1、随着样本尺寸整体上的ACC
2、不同类型的样本随着样本尺寸的指标变化
3、信息熵计算

@author: shuaiyi
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import recall_score, precision_score, f1_score
from spm import SPMFeature
from PIL import Image
from entropy import entropy
import argparse
import pickle as pkl
import progressbar
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
dataset = pkl.load(file(args["dataset"], 'rb'))



# 将类别名称编码为数字
le = LabelEncoder()
all_labels = dataset.keys()
le.fit(all_labels)

# TODO: 计算信息熵,熵可以统一算，下面的CV不行？
if False:
    clusters = [1000]
    imgsize = [100,150,200,250,300,350,400,450,500,550,600]
    
    print "Compute Entropy..."
    all_ents = {}
    for k,v in dataset.items():
        print "Processing", k
        for c in clusters:
            for i in imgsize:
                tmp = 0
                key = "{0}_{1}".format(c, i)
                if not all_ents.has_key(key):
                    all_ents[key] = {}
                for item in v:
                    im = Image.open(item).convert('L')
                    if im.size[0] != i:
                        im = im.resize((i, i))
                    tmp += entropy(im)
                all_ents[key][k] = tmp/len(v)
    
    print "Computing ALL Avg."
    for c in clusters:
        for i in imgsize:
            tmp = 0
            key = "{0}_{1}".format(c, i)   
            for k,v in dataset.items():
                tmp += all_ents[key][k]
            all_ents[key]['all'] = tmp/19
    
    joblib.dump(value=all_ents, filename='RSDataset_entropy.pkl', compress=3)

if False:
    c=args['clusters']
    i=args['imgsize']
    all_results = {}
    key = "{0}_{1}".format(c, i)   
    fpath = "{0}_{1}_{2}.pkl".format(args['dataset'][0:-4], c, i)
    print "CV :", c, i
    all_ftrs = joblib.load(fpath, 'r')
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
    
    precision=[]
    recall=[]
    f1=[]
    
    pbar = progressbar.ProgressBar(maxval=cv.n_iter).start()
    cnt = 0
    for train, test in cv:
        train_x, train_y, test_x, test_y = all_x[train], all_y[train], all_x[test], all_y[test]
        clf = SVC(C=1, kernel='linear', probability = True, random_state=42)
        clf.fit(train_x, train_y)
        pre_y = clf.predict(test_x)
        precision.append(precision_score(test_y, pre_y, average=None))
        recall.append(recall_score(test_y, pre_y, average=None))
        f1.append(f1_score(test_y, pre_y, average=None))
        pbar.update(cnt+1)
        cnt = cnt + 1
    
    pbar.finish()
    precision=np.vstack(precision)
    recall=np.vstack(recall)
    f1=np.vstack(f1)
    all_results = {"precision":precision,"recall":recall,"f1":f1}# 这个顺序很关键

    # saving
    joblib.dump(value={"cv":all_results, "LabelEncoder":le}, filename='RSDataset_cv_%s.pkl'%key, compress=3)
    #print("Accuracy-%0.3f : %0.3f (+/- %0.3f)" % (c, scores.mean(), scores.std() * 2))
    
if True:
    # TODO: 分析各种相关性
    clusters = [1000]
    imgsize = [100,150,200,250,300,350,400,450,500,550,600]
    
    print "Loading Entropy..."
    all_ents = joblib.load('RSDataset_entropy.pkl')
    print "Loading CV..."
    all_precision = {}
    all_recall = {}
    all_f1 = {}
    for c in clusters:
        for i in imgsize:
            key = "{0}_{1}".format(c, i)
            if not all_precision.has_key(key):
                all_precision[key] = {}
                all_recall[key] = {}
                all_f1[key] = {}
            fname = 'RSDataset_cv_%s.pkl'%key
            cv = joblib.load(fname)['cv']
            for i in range(19):
                print "Precision, Recall, F1 Loading..."
                k = str(le.inverse_transform([i])[0])
                all_precision[key][k] = cv['precision'][:,i]
                all_recall[key][k] = cv['recall'][:,i]
                all_f1[key][k] = cv['f1'][:,i]
                
    #TODO: 数据准备好，进行数据分析