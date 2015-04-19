# -*- coding: cp936 -*-
"""
训练分类器

@author: shuaiyi
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import shuffle

from sklearn.svm import LinearSVC, SVC # svm
from sklearn.cross_validation import train_test_split #把训练样本分成训练和测试两部分
from sklearn.externals import joblib # 保存分类器
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
#from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

import BOVW
import pickle as pkl

dataset = {}
if len(sys.argv) == 1:
    print "Usage: classifier.py dataset_path"
else:
    print "Loading", sys.argv[1]
    dataset = pkl.load(file(sys.argv[1], 'rb'))

# 将类别名称编码为数字
le = LabelEncoder()
all_labels = dataset.keys()
le.fit(all_labels)

# 训练样本比率
_percent = 0.5
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

# dict learning
bovw = BOVW.BoVWFeature()
bovw.fit(x_train)

# visualization
w = bovw.dico.components_
tmp = w[150]*bovw.std + bovw.mean
imshow(tmp.reshape(8,8,3),interpolation='nearest')

x_train = bovw.transform(x_train)
x_test = bovw.transform(x_test)

# svm
svm = SVC(kernel='linear', probability = True,random_state=np.random.RandomState())

clf = Pipeline([('svm',svm)]) #('bovw', bovw),('zca',zca)

parameters = {
    'svm__C': [0.001,1]
}

gridCV = GridSearchCV(clf, parameters,n_jobs=1,verbose=True)

print "****************Grid Search******************************"
gridCV.fit(x_train, y_train)

print "*********************Train******************************"
best = gridCV.best_estimator_
best.fit(x_train, y_train)

print "*********************Save*******************************"
joblib.dump(best, "classifier_bovw.pkl", compress=3)
joblib.dump(gridCV, "grid_cv_bovw.pkl", compress=3)

# best = joblib.load(prj_name + "/classifier_svc.pkl")
        
print "*********************Test*******************************"
y_test_pre = best.predict(x_test)
cm = confusion_matrix(y_test, y_test_pre)
from map_confusion import plot_conf
plot_conf(cm, range(le.classes_.size))
#print "confusion matrix..."
#print cm

from sklearn.metrics import classification_report
with file('report_bovw.txt', 'w') as f:
    report = classification_report(y_test, y_test_pre, target_names = le.classes_)
    f.writelines(report)
    
"""
print "*********************ROC********************************"
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

print best
y_score = best.predict_proba(x_test)

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
roc_auc = auc(fpr, tpr)

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig(prj_name + "/ROC.tif", dpi=300)
plt.show()
# plt.close()
"""