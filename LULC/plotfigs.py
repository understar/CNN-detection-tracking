# -*- coding: cp936 -*-
"""
Created on Mon May 11 14:44:33 2015

需要绘制的图形列表：
1、随着样本尺寸整体上的ACC
2、不同类型的样本随着样本尺寸的指标变化
3、信息熵计算
4、挖掘本质因素，观察信息熵与各类以及与整体分类情况之间的关系，
通过投点图，观察，基本上前后都是通过投点图
from scipy import stats.pearsonr 可以计算变量之间的线性相关程度
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

from matplotlib import rcParams
# rcParams dict
rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['figure.figsize'] = 7, 5

import matplotlib.pyplot as plt
import pylab
from scipy.stats import pearsonr

def figit_ent(x, y, k):
    print "[Entropy] Corr. of %s: %s" % (k, pearsonr(x[k], y[k])[0])
    print "[Entropy] Mean:", np.mean(x[k])
    print "[F1] Mean: %s +- std: %s" % (np.mean(y[k]), np.std(y[k]))
    plt.figure()
    plt.plot(x[k], y[k], 'bo-')
    plt.title(k)
    plt.show()

def figit_size(x, y, k):
    print "[Size] Corr. of %s: %s" % (k, pearsonr(x, y[k])[0])
    plt.figure()
    plt.plot(x, y[k], 'bo-')
    plt.title(k)
    plt.show()
    
"""Arguments"""
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "path to the dataset file (*.pkl)")
ap.add_argument("-c", "--clusters", type = int, default = 1000,
	help = "k-means clusters") 
ap.add_argument("-i", "--imgsize", type = int, default = 600,
	help = "image size") 
ap.add_argument("-n", "--iter", type = int, default = 10,
	help = "n iter") 
 
args = vars(ap.parse_args())
dataset = pkl.load(file(args["dataset"], 'rb'))

# 将类别名称编码为数字
le = LabelEncoder()
all_labels = dataset.keys()
le.fit(all_labels)

#%% 计算信息熵
if False:
    clusters = [1000]
    imgsize = [100,150,200,250,300,350,400,450,500,550,600]
    
    print "Compute Entropy..."
    all_ents = {}
    for k,v in dataset.items():
        print "Processing", k
        for c in clusters:
            for i in imgsize:
                tmp = []
                key = "{0}_{1}".format(c, i)
                if not all_ents.has_key(key):
                    all_ents[key] = {}
                for item in v:
                    im = Image.open(item).convert('L')
                    if im.size[0] != i:
                        im = im.resize((i, i))
                    tmp.append(entropy(im))
                all_ents[key][k] = tmp
    
    print "Computing ALL Avg."
    for c in clusters:
        for i in imgsize:
            tmp = 0
            key = "{0}_{1}".format(c, i)   
            for k,v in dataset.items():
                tmp += np.sum(all_ents[key][k])
            all_ents[key]['all'] = tmp/(19*50)
    
    joblib.dump(value=all_ents, filename='RSDataset_entropy.pkl', compress=3)

#%% 计算单个类别的详细分类结果情况
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

    # 统计次数多一点，看看能不能平滑结果
    cv = StratifiedShuffleSplit(y=all_y, n_iter=args['iter'], test_size=0.4)
    
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
    
#%% 相关性分析及绘图
if True:
    # TODO: 分析各种相关性
    clusters = [1000]
    imgsize = [100,150,200,250,300,350,400,450,500,550,600]
    
    print "Loading Entropy..."
    all_ents = joblib.load('RSDataset_entropy.pkl')
    print "Loading CV-DETAIL..."
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
    
    print "Loading CV-All..."
    all_scores = {}
    for c in clusters:
        for i in imgsize:
            key = "{0}_{1}".format(c, i)
            fname = 'RS_results/size_acc/RSDataset_%s.npy'%key
            score = np.load(fname)
            all_scores[key] = score
    
    #TODO: 数据准备好，进行数据分析
    x_size = imgsize
    x_entropy_all = []
    y_scores_all = []
    x_entropy = {}
    x_entropy_std = {}
    y_scores_precision = {}
    y_scores_recall = {}
    y_scores_f1 = {}
    for s in x_size:
        key = "1000_{0}".format(s)
        #x_entropy_all.append(all_ents[key]['all'])
        y_scores_all.append(all_scores[key].mean())
        
        for k,v in all_ents[key].items():
            if not x_entropy.has_key(k):
                x_entropy[k] = []
                x_entropy_std[k] = []
            x_entropy[k].append(np.mean(v))
            x_entropy_std[k].append(np.std(v))
            
        for k,v in all_precision[key].items():
            if not y_scores_precision.has_key(k):
                y_scores_precision[k] = []
            y_scores_precision[k].append(v.mean())
            
        for k,v in all_recall[key].items():
            if not y_scores_recall.has_key(k):
                y_scores_recall[k] = []
            y_scores_recall[k].append(v.mean())
            
        for k,v in all_f1[key].items():
            if not y_scores_f1.has_key(k):
                y_scores_f1[k] = []
            y_scores_f1[k].append(v.mean())
            
    x_entropy_all = x_entropy['all']
    x_entropy.pop('all')
    x_entropy_std.pop('all')
        
    # 感觉熵与精度之间存在线性相关
    fit = pylab.polyfit(x_entropy_all,y_scores_all,1)
    fit_fn = pylab.poly1d(fit) 
    print "熵与总体分类精度之间的相关系数：", pearsonr(x_entropy_all, y_scores_all)
    
    from scipy.optimize import curve_fit
    def func(x,a,b):
       return a*np.log(b*x)

    popt,pcov = curve_fit(func, x_size, x_entropy_all)
    #y_fit = np.exp(popt[0]*x) 
    
    # 整体上
    plt.figure()
    plt.subplot(131)
    plt.plot(x_size, y_scores_all,'o-')
    plt.title("Size-Acc")
    
    plt.subplot(132)
    plt.plot(x_entropy_all, y_scores_all,'o-', 
             x_entropy_all, fit_fn(x_entropy_all), '--k')
    plt.title("Entropy-Acc")
    
    plt.subplot(133)
    plt.plot(x_size, x_entropy_all,'o-')
             #x_size, func(np.array(x_size), popt[0], popt[1]), '--k')
    plt.title("Size-Entropy") 
    
    plt.tight_layout()
    plt.show()
    
    # 一个简单的设想是熵才是关键觉得因素
    # 另外不同地物类型的最优尺寸肯定是不同，熵是决定因素
    # 大量的投点，来进一步观察熵的影响
    # 大量的投点，感觉上熵与单个类别的分类情况没有必然影响？？？
    x_all_entropy = []
    y_all_precision = []
    for k in x_entropy.keys():
        x_all_entropy += x_entropy[k]
        y_all_precision += y_scores_precision[k]
    
    plt.figure()
    plt.scatter(x_all_entropy, y_all_precision)
    plt.title('Entropy-Precision (ALL)')
    plt.show()
    
    #相关系数bar图
    corr_all = []
    f1_std = []
    for k in le.classes_.tolist():
        corr_all.append(pearsonr(x_entropy[k], y_scores_f1[k])[0])
        f1_std.append(np.std(y_scores_f1[k]))
    plt.figure()
    width = 0.35
    tick_marks = np.arange(len(le.classes_.tolist()))
    plt.bar(tick_marks, corr_all, width, color='r')
    plt.bar(tick_marks+width, f1_std, width, color='r')
    plt.xticks(tick_marks+width, le.classes_.tolist(), rotation=90)
    plt.axhline(0, color='black', lw=1)
    plt.ylabel('Correlation coefficient')    
    plt.tight_layout()
    plt.ylim(-1,1)
    plt.show()
    
    # TODO: 通过图示解释熵与分类精度直接的关系
    # 通过对f1指标方差（方差可以看成分类结果波动是否大的标志），进一步可以阐述为在
    # 实验熵范围内，分类结果的波动程度，也就是熵对其分类的影响程度，影响较小的情况
    # 下，此时难以表现出相关（Bridge），甚至出现负相关（Pond footballfield），也有
    # 可能表现问正相关（？）。可以解释为，当
    # 前的熵水平对于分类不足（分类精度均值较低），或者已经足够充分（分类精度平均值高）
    
    # http://matplotlib.org/users/legend_guide.html
    # http://matplotlib.org/examples/api/two_scales.html
    # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplot
    
    # 各类的情况
    # 整体上有大部分是随着熵的增加有增长趋势
    # 是不是10次太少的原因，统计力度不够???
    #for k in x_entropy.keys():
    #    figit_ent(x_entropy, y_scores_f1, k)
    #    #figit_size(x_size, y_scores_f1, k)
    figit_ent(x_entropy, y_scores_f1, 'Bridge')
    figit_ent(x_entropy, y_scores_f1, 'Pond')
    figit_ent(x_entropy, y_scores_f1, 'footballField')

#%% 绘制混淆矩阵
if False:
    # confusion matrix
    fpath = "{0}_{1}_{2}.pkl".format(args['dataset'][0:-4],args['clusters'], args['imgsize'])

    all_ftrs = joblib.load(fpath, 'r+')
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
    
    cv = StratifiedShuffleSplit(y=all_y, n_iter=1, test_size=0.4)
    for train, test in cv:
        train_x, train_y, test_x, test_y = all_x[train], all_y[train], all_x[test], all_y[test]
        clf = SVC(C=1, kernel='linear', probability = True, random_state=42)
        print "Training..."        
        clf.fit(train_x, train_y)
        y_test_pre = clf.predict(test_x)
        
        print "Confusion matrix..."
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import confusion_matrix
        print "Accuracy:", accuracy_score(test_y, y_test_pre)
        cm = confusion_matrix(test_y, y_test_pre)
        np.save('RS_results/RSDataset_%s_%s_%s.npy'%('sift', 1000, 600), cm)
        from map_confusion import plot_conf
        plot_conf(conf_arr=cm, label_list=le.classes_.tolist(), 
                  norm=False, save_name='RS_results/RSDataset_%s_%s_%s.png'%('sift', 1000, 600))
        
        from sklearn.metrics import classification_report
        with file('RS_results/report_spm_%s_%s_%s.txt'%('sift', 1000, 600), 'w') as f:
            report = classification_report(test_y, y_test_pre, target_names = le.classes_)
            f.writelines(report)