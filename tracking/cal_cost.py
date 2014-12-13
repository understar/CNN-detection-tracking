# -*- coding: cp936 -*-
"""
Created on Mon Dec 08 14:14:32 2014

@author: Administrator
"""

import numpy as np
import os, sys, math

from car_record import Point, Car, modulus

def gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# 搜索半径
search_r = 50

# 不同检查结果之间的权重计算；
# 分为若干种类型；

# 给予KD Tree查找邻近点
from sklearn.neighbors import KDTree
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
kdt = KDTree(X, leaf_size=30, metric='euclidean')
print kdt.query(X, k=2, return_distance=False)
'''
>>...
array([[0, 1],
       [1, 0],
       [2, 1],
       [3, 4],
       [4, 3],
       [5, 4]]...)
'''

def c_c(car_t, pt_list):
    #
    pass

def c_g(car_t, pt_t1):
    pass

def c_p(car_t, pt_t1):
    #x_t_k = car_t.curr_xy.vec()
    #v_t_k = car_t.curr_v
    #x_t_1 = pt_t1.vec()
    #k+1 = car_t.intervel
    r_hat = modulus(car_t.curr_xy.vec() + car_t.intervel * car_t.curr_v - pt_t1.vec())
    return 1 - r_hat / search_r

def c_v(car_t, pt_t1):
    v_t = car_t.curr_v
    v_t1 = (car_t.curr_xy.vec() - pt_t1.vec())/(car_t.step*car_t.interval)
    dot = np.dot(v_t, v_t1)
    t_modulus = modulus(v_t)
    t1_modulus = modulus(v_t1)
    cos_angle = dot / t_modulus / t1_modulus # cosinus of angle between x and y
    return 0.5 + 0.5*cos_angle
