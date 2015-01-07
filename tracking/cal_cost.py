# -*- coding: cp936 -*-
"""
Created on Mon Dec 08 14:14:32 2014

@author: Administrator
"""

import numpy as np
import os, sys, math

from car_record import Point, Car, modulus
from skimage.feature import match_template, peak_local_max
from skimage.color import rgb2gray

# 搜索半径
# 用于确定候选匹配的搜索半径
search_r = 60 # 单位为pixels

def unit_vector(vector):
    """ Returns the unit vector of the vector.  
        返回一个向量的单位向量，即模为1.
    """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
        计算两个向量之间的夹角
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


#==============================================================================
# def gauss2D(shape=(5,5),sigma=0.5):
#     """2D gaussian mask 返回一个2D高斯模板：：
#         should give the same result as MATLAB's
#         fspecial('gaussian',[shape],[sigma])
#     """
#     m,n = [(ss-1.)/2. for ss in shape]
#     y,x = np.ogrid[-m:m+1,-n:n+1]
#     h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
#     h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
#     sumh = h.sum()
#     if sumh != 0:
#         h /= sumh
#     return h
# 
# def hist_xy(src, pt_list, k_size=5, d_num=20, a_num=20):
#     """计算上下文直方图特征
#         # 距离：bin_num，0-r，
#         # 角度：bin_num，0-2*pi
#         # 计算与x轴(1,0)的夹角    
#     """
# 
#     k = gauss2D((k_size, k_size)) 
#     border = int(k_size/2)
#     d_nbin = [(math.ceil(search_r/d_num)*i, math.ceil(search_r/d_num)*(i+1)) for i in range(d_num+1)]
#     a_nbin = [(math.ceil(2*np.pi/a_num)*i, math.ceil(2*np.pi/a_num)*(i+1)) for i in range(a_num+1)]
#     result = np.zeros((len(d_nbin)+2*border, len(a_nbin)+2*border))
#     for pt in pt_list:
#         v = pt.vec() - src.vec()
#         dist = modulus(v)
#         if dist <= r:
#             tmp =  np.arctan2(v[1], v[0])
#             angle =  -tmp if tmp < 0 else 2*np.pi+tmp
#             idx_d = int(tmp/math.ceil(2*np.pi/a_num)) + border
#             idx_a = int(tmp/math.ceil(search_r/d_num))+ border
#             result[(idx_d-border):(idx_d+border+1), (idx_a-border):(idx_a+border+1)] += k
#     return result[border:-border,border:-border]
#     
#==============================================================================
def c_c(car_t, pt_t1):
    """计算上下文约束权重，主要是用于开始帧的匹配;
    """
    pass

def c_match(car_t, pt_t1, pt_img):
    """计算NCC归一下互相关系数，利用模板匹配，看看他们之间到底有多像！！！
    如果利用颜色信息会不会增加准确度？
    """
    if car_t.template == None or pt_img == None:
        return 0, None
    temp = rgb2gray(car_t.template)
    image = rgb2gray(pt_img)
    
    result = match_template(image, temp)
    ij = np.unravel_index(np.argmax(result), result.shape)
    y, x = ij[::-1]
    h, w = result.shape
    center = Point(w/2, h/2)
    detect = Point(x, y)
    return (1-center.dist(detect)/(math.sqrt(2)*w/2)), result #假设待匹配影像以及模板影像都是正方形

def find_pts(car_t, kdt):
    # kdt = KDTree(X, leaf_size=30, metric='euclidean')
    idx = kdt.query(car_t.curr_xy.vec(), k=10, return_distance=True)
    # 过滤阈值区 search—r
    return idx[1][idx[0]<search_r]

def c_p(car_t, pt_t1):
    """ 计算当前车辆预期结果与目标点之间Cp权重(距离变化导致的权重变化)
          如果car_t是新track，需要改写
    """
    #x_t_k = car_t.curr_xy.vec()
    #v_t_k = car_t.curr_v
    #x_t_1 = pt_t1.vec()
    #k+1 = car_t.intervel
    if False: #not car_t.is_new:
        r_hat = modulus(car_t.curr_xy.vec() + car_t.interval * car_t.curr_v - pt_t1.vec())
        return 1 - r_hat / search_r
    else:
        return 1 - car_t.curr_xy.dist(pt_t1)/(search_r) #/car_t.interval

def c_v(car_t, pt_t1):
    """ 计算当前车辆与目标点之间Cv权重(角度变化导致的权重变化)
        如果car_t是新track，需要改写
    """
    if False: #not car_t.is_new:
        v_t = car_t.curr_v
        v_t1 = (car_t.curr_xy.vec() - pt_t1.vec())/(car_t.step*car_t.interval)
        dot = np.dot(v_t, v_t1)
        t_modulus = modulus(v_t)
        t1_modulus = modulus(v_t1)
        cos_angle = dot/(t_modulus*t1_modulus) # cosinus of angle between x and y
        return 0.5 + 0.5*cos_angle
    else:
        D = 360 - car_t.curr_d
        #print D, np.cos(np.radians(D)), np.sin(np.radians(D))
        v_t = unit_vector(Point(np.cos(np.radians(D)), np.sin(np.radians(D))).vec())
        v_t1 = unit_vector(car_t.curr_xy.vec() - pt_t1.vec())
        cos_angle = np.dot(v_t, v_t1) # 如果前后点不动会导致nan值
        if np.isnan(cos_angle):
            cos_angle = 1
        # print cos_angle, v_t, v_t1
        return 0.5 + 0.5*abs(cos_angle) #不管正负方向一致就行，初始的时候是这样的

def cost(car_t, pt_t1, pt_img):
    cost1, result= c_match(car_t, pt_t1, pt_img) # 模板匹配
    cost2 = c_v(car_t, pt_t1) # 角度
    cost3 = c_p(car_t, pt_t1) # 距离
    
    # 控制cost1 2 3 之间的比重
    if cost2 > 0.5 and cost3 > 0: # 角度严重不符合，不予考虑
        alpha = 0.5
        beta = 0.2
        return (alpha*cost1+beta*cost2+(1-alpha-beta)*cost3), cost1, cost2, cost3
    else:
        return 2, cost1, cost2, cost3