# -*- coding: cp936 -*-
"""
Created on Mon Dec 08 14:14:32 2014

@author: Administrator
"""

import numpy as np
import os, sys, math

from car_record import Point, Car, modulus

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return np.pi
    return angle


def gauss2D(shape=(5,5),sigma=0.5):
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

def hist_xy(src, pt_list, k_size=5, d_num=20, a_num=20):
    """计算上下文直方图特征
        # 距离：bin_num，0-r，
        # 角度：bin_num，0-2*pi
        # 计算与x轴(1,0)的夹角    
    """

    k = gauss2D((k_size, k_size)) 
    border = int(k_size/2)
    d_nbin = [(math.ceil(search_r/d_num)*i, math.ceil(search_r/d_num)*(i+1)) for i in range(d_num+1)]
    a_nbin = [(math.ceil(2*np.pi/a_num)*i, math.ceil(2*np.pi/a_num)*(i+1)) for i in range(a_num+1)]
    result = np.zeros((len(d_nbin)+2*border, len(a_nbin)+2*border))
    for pt in pt_list:
        v = pt.vec() - src.vec()
        dist = modulus(v)
        if dist <= r:
            tmp =  np.arctan2(v[1], v[0])
            angle =  -tmp if tmp < 0 else 2*np.pi+tmp
            idx_d = int(tmp/math.ceil(2*np.pi/a_num)) + border
            idx_a = int(tmp/math.ceil(search_r/d_num))+ border
            result[(idx_d-border):(idx_d+border+1), (idx_a-border):(idx_a+border+1)] += k
    return result[border:-border,border:-border]
    
def c_c(car_t, pt_t1):
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
