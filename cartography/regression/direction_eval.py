# -*- coding: cp936 -*-
"""
Created on Thu Mar 13 10:15:28 2014
用于从prediction结果中，提取中间层输出
@author: shuaiyi
"""

import cPickle as pl
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

import sys

result = pl.load(open(sys.argv[1],'rb'))

y_true = result['labels'].ravel()
props = result['data']
y_pred = np.argmax(props,axis=1) # 获取监测结果 0 - 179
err = y_true - y_pred

# 校正179，如果abs大于160就纠正一下
err = np.array([x if abs(x)<90 else 179-abs(x) for x in err.tolist()])
r_err = err[np.abs(err)<=20]
r_y_true = y_true[np.abs(err)<=20]
r_y_pred = y_pred[np.abs(err)<=20]

print "================Error Hist========================"
plt.figure()
plt.hist(err, bins=40, range=(-20,20), normed=True, align='mid', fc = 'none')
plt.xlabel('Error (Degree)')
plt.ylabel('Frequency')
plt.title('Error Histogram')
plt.savefig("Err_hist.tif", dpi=300)
plt.show()
# np.savetxt('labels.txt', result['labels'], delimiter=',\n')
# np.savetxt('x.txt', result['data'], delimiter=',')
print "================ True ==========================="
print "True ratio:", float(r_err.size)/float(err.size)


print "================ explained_variance_score========="
"""
The explained_variance_score computes the explained variance regression score.
If \hat{y} is the estimated target output and y is the corresponding (correct) 
target output, then the explained variance is estimated as follow:
   \texttt{explained\_{}variance}(y, \hat{y}) = 1 - \frac{Var\{ y - \hat{y}\}}{Var\{y\}}
   The best possible score is 1.0, lower values are worse.
"""
from sklearn.metrics import explained_variance_score
print explained_variance_score(r_y_true, r_y_pred)  

print "================root-mean-square error=============="
#from sklearn.metrics import mean_squared_error
#print mean_squared_error(r_y_true, r_y_pred)

print "RMSE(均方根误差):", sqrt(np.sum(r_err**2)/float(r_err.size))