# -*- coding: cp936 -*-
"""
Created on Thu Oct 09 22:22:43 2014

@author: shuaiyi
"""

import sys
import numpy as np
import cPickle as pkl

import matplotlib.pyplot as plt
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

from numpy.random import shuffle
from sklearn.svm import LinearSVC, SVC # svm
from sklearn.externals import joblib # 保存分类器
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix

is_train = False

print "****************Loading dataset***************************"
#x_train, y_train = pkl.load(file('hog_data_train.pkl', 'rb'))
#x_test, y_test = pkl.load(file('hog_data_test.pkl', 'rb'))

x_train, y_train = np.load('x_train.npy'), np.load('y_train.npy')
x_test, y_test = np.load('x_test.npy'), np.load('y_test.npy')

clf = SVC(probability = True, class_weight='auto',random_state=np.random.RandomState())
#clf = LinearSVC() #kernel='linear', C = 10000, loss='l1', penalty='l2', random_state=42


parameters = {
    'C': (1, 10, 100, 1000),
    'kernel': ('linear', 'rbf')
    #'C': (1, 10, 20, 30, 40, 50, 60, 70, 80, 90)
    #'C': (100, 200, 300, 400, 500, 600, 700, 800, 900)
    #'C': (1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000)
    #'loss': ('l1', 'l2')
    #'penalty':('l1', 'l2')
}



if __name__ == "__main__":
    # step 1: cv (cross validation_ grid search)
    # step 2: train    
    #
    if is_train:
        gridCV = GridSearchCV(clf, parameters, n_jobs=3, verbose=True)
        print "****************Grid Search******************************"
        gridCV.fit(x_train, y_train)
        
        print "*********************Train******************************"
        # grid_cv results : {'clf__C': 5000, 'zca__bias': 0.01}
        best = gridCV.best_estimator_
        
        best.fit(x_train, y_train)
        
        print "*********************Save*******************************"
        joblib.dump(best, "classifier.pkl", compress=3)
    else:
        best = joblib.load("classifier.pkl")
        
    print "*********************Test*******************************"
    y_test_pre = best.predict(x_test)
    cm = confusion_matrix(y_test, y_test_pre)
    print "confusion matrix..."
    print cm
    
    
    print "*********************ROC********************************"
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    # print best
    y_score = best.predict_proba(x_test)
    y_test = label_binarize(y_test, classes=[0, 1, 2])
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test[:,:-1].ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Compute ROC curve and ROC area for each class
    #fpr, tpr, _ = roc_curve(gt, score[:, 1], pos_label=1)
    #roc_auc = auc(fpr, tpr)
    
    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr[0], tpr[0], label='Others ROC curve (AUC = %0.4f)' % roc_auc[0])
    plt.plot(fpr[1], tpr[1], label='Car ROC curve (AUC = %0.4f)' % roc_auc[1])
    plt.plot(fpr["micro"], tpr["micro"], label='Micro ROC curve (AUC = %0.4f)' % roc_auc["micro"])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver operating characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig("ROC.tif", dpi=300)
    plt.show()