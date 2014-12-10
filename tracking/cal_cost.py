# -*- coding: cp936 -*-
"""
Created on Mon Dec 08 14:14:32 2014

@author: Administrator
"""

import numpy as np
import os, sys


# 不同检查结果之间的权重计算；
# 分为若干种类型；

# 给予KD Tree查找邻近点
from sklearn.neighbors import KDTree
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
kdt = KDTree(X, leaf_size=30, metric='euclidean')
kdt.query(X, k=2, return_distance=False)
'''
>>...
array([[0, 1],
       [1, 0],
       [2, 1],
       [3, 4],
       [4, 3],
       [5, 4]]...)
'''

    
