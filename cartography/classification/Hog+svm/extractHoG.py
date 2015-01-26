# -*- coding: cp936 -*-
"""
Created on Thu Oct 09 22:07:26 2014

@author: shuaiyi
"""
import os
import cPickle as pkl
import numpy as np
from skimage.feature import hog

def HoG_arr(arr):
    """提取列向量样本的HOG
    """
    s = np.sqrt(arr.size)
    arr = arr.reshape((s, s))
    return hog(arr, orientations=8, pixels_per_cell=(10, 10),
               cells_per_block=(2, 2), visualise=False, normalise=True)

"""加载数据"""
train_x = []; train_y = []
test_x = []; test_y = []

for db in os.listdir('data'):
    f = file(os.path.join('data', db),'rb')
    print 'Loading %s'%db
    tmp = pkl.load(f)
    if int(db.split('_')[-1]) != 8:
        train_x.append(tmp['data'])
        train_y.append(tmp['labels'])
    else:
        test_x.append(tmp['data'])
        test_y.append(tmp['labels'])
        
train_x = np.hstack(train_x)
train_y = np.hstack(train_y)
test_x = np.hstack(test_x)
test_y = np.hstack(test_y)


"""提取特征"""
train_X=[]; test_X=[]
for i in range(train_x.shape[1]):
    if i%1000 == 0:
        print "Train: Processing %sth image."%i
        #print HoG_arr(train_x[:,i])
    train_X.append(HoG_arr(train_x[:,i]))
    
for i in range(test_x.shape[1]):
    if i%1000 == 0:
        print "Test: Processing %sth image."%i
        #print HoG_arr(test_x[:,i])
    test_X.append(HoG_arr(test_x[:,i]))
    
train_X = np.vstack(train_X)
test_X = np.vstack(test_X)

"""保存特征"""
np.save('x_train.npy', train_X)
np.save('y_train.npy', train_y)
np.save('x_test.npy', test_X)
np.save('y_test.npy', test_y)
#pkl.dump((train_X, train_y), file('hog_data_train.pkl','wb'))
#pkl.dump((test_X, test_y), file('hog_data_test.pkl','wb'))