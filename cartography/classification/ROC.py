# -*- coding: cp936 -*-
"""
Created on Thu Oct 09 22:22:43 2014

@author: shuaiyi
"""

import sys
import numpy as np
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

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
# 0代表others，1代表car
y_test_src = np.loadtxt("labels.txt", delimiter=',').astype(np.int32)
y_score = np.loadtxt("data.txt", delimiter=',')
y_test = label_binarize(y_test_src, classes=[0, 1, 2])
pred = y_score.argmax(1)

if __name__ == "__main__":
    print "*********************Confusion Matrix*******************"
    cm = confusion_matrix(y_test_src, pred)
    print "confusion matrix..."
    print cm
    
    
    print "*********************ROC********************************"

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
    # plt.tight_layout(pad=0.1) 
    #plt.plot(fpr[0], tpr[0], label='Others ROC curve (AUC = %0.4f)' % roc_auc[0])
    plt.plot(fpr[1], tpr[1], label='Vehicle ROC Curve (AUC = %0.2f)' % roc_auc[1])
    #plt.plot(fpr["micro"], tpr["micro"], label='Micro ROC curve (AUC = %0.4f)' % roc_auc["micro"])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    #plt.title('Receiver operating characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig("ROC.tif", dpi=300)
    plt.show()
    # plt.close()
