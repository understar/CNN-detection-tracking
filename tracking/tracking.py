# -*- coding: cp936 -*-
"""
Created on Tue Jan 06 17:37:10 2015

@author: shuaiyi
"""
import numpy as np
import os, sys, math

from car_record import Point, Car, modulus
from cal_cost import cost, find_pts
from skimage.feature import match_template
from skimage.color import rgb2gray

from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
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


_resolution_=0.12 #MS04区域影像空间分辨率

def save_v(car_list, fname):
    with file(fname, 'w') as f:
        f.writelines(['x,y,v\n'])
        for car in car_list:
            if 0 != len(car.hist_xy):
                # 计算速度,输出单位km/h
                # v = dist / time_interval
                v = 3.6*modulus(car.curr_xy.vec()-car.hist_xy[-1].vec())*_resolution_/(car.interval*car.step)
                f.writelines(['%s,'%car.curr_xy.X, '%s,'%car.curr_xy.Y, '%s\n'%v])

def show_detection(img_path, csv):
    plt.figure()
    lines = file(csv,'r').readlines()
    img = imread(img_path)
    plt.imshow(img)
    for line in lines[1:]:
        oid, yc, xc, A, o = line.split(',')
        xc = float(xc)
        yc = float(yc)
        o = float(o)
        plt.scatter(xc, yc) # 注意绘制图片的坐标系和笛卡尔坐标系的区别
        # objs.append((xc, yc, o))
    plt.show()

def show_track(img, car):
    plt.figure()
    plt.imshow(img)
    X = []
    Y = []
    for pos in car.hist_xy:
        X.append(pos.X)
        Y.append(h-pos.Y)
    X.append(car.curr_xy.X)
    Y.append(h-car.curr_xy.Y)
    plt.plot(X, Y, '-') #marker="o", markerfacecolor="r")
    if len(X) != 0:
        plt.annotate(str(car.m_id),(X[-1], Y[-1]))
    plt.show()
    
def show_all(img, car_list):
    plt.figure()
    plt.imshow(img)
    for car in car_list:
        X = []
        Y = []
        for pos in car.hist_xy:
            X.append(pos.X)
            Y.append(h-pos.Y)
        X.append(car.curr_xy.X)
        Y.append(h-car.curr_xy.Y)
        plt.plot(X, Y, '-') #marker="o", markerfacecolor="r")
        #if len(X) != 0:
        #    plt.annotate(str(car.m_id),(X[-1], Y[-1]))
    plt.show()

def save_car(car, fname):
    with file(fname, 'w') as f:
        f.writelines(['x,y\n'])
        for pos in car.hist_xy:
            f.writelines(['%s,'%pos.X, '%s\n'%pos.Y])
        f.writelines(['%s,'%car.curr_xy.X, '%s\n'%car.curr_xy.Y])


print "================== 初始化，加载数据============================="
frames = []
dir_path = './MS04'
for f in os.listdir(dir_path):
    if f.endswith('.csv'):
        frames.append((os.path.join(dir_path, f), int(f.split('.')[0][3:])))

frames = sorted(frames, key=lambda f: f[1])



# TODO: 计算步骤
print "================== 初始化第1帧=================================="
# 1. 加载首帧检测结果，为每一个初始化一个car；
car_list = []
plt.figure()
lines = file(frames[0][0],'r').readlines()
img = imread(frames[0][0][:-4]+'.tif')
plt.imshow(img)
h, w, nc = img.shape
for line in lines[1:]:
    oid, yc, xc, A, o = line.split(',')
    # oid, xc, yc, o = line.split(',')
    xc = float(xc)
    yc = float(yc)
    o = float(o)
    if img[yc-20:yc+20,xc-20:xc+20,:].shape == (40,40,3):
        car_list.append(Car(Point(xc,h-yc), img[yc-20:yc+20,xc-20:xc+20,:], o))
        plt.scatter(xc, yc) # 注意绘制图片的坐标系和笛卡尔坐标系的区别
    # objs.append((xc, yc, o))
plt.show()

print "================== 逐帧更新===================================="
for f in frames[1:]:
    # 2.匹配首帧(不需要区别首帧，只需要区别首次匹配与非首次匹配的问题)
    lines = file(f[0],'r').readlines()
    img1 = imread(f[0][:-4]+'.tif')
    pts = []
    for line in lines[1:]:
        oid, yc, xc, A, o = line.split(',')
        # oid, xc, yc, o = line.split(',')
        xc = float(xc)
        yc = float(yc)
        o = float(o)
        target_pt = {}
        if img1[yc-30:yc+30,xc-30:xc+30,:].shape == (60,60,3):
            target_pt['loc'] = Point(xc, h-yc)
            target_pt['direction'] = o
            target_pt['img'] = img1[yc-30:yc+30,xc-30:xc+30,:]
            pts.append(target_pt)
        
    # 构建KD Tree查找邻近点
    X = []
    for pt in pts:
        X.append((pt['loc'].X, pt['loc'].Y))
    X = np.array(X)
    kdt = KDTree(X, leaf_size=30, metric='euclidean')
    
    # 增加假节点，为节点消失准备,每一个车加一个
    for car in car_list:
        target_pt = {}
        target_pt['loc'] = None
        target_pt['direction'] = None
        target_pt['img'] = None
        pts.append(target_pt)
    
    cost_arr = np.ones((len(car_list), len(pts)))*0.5
    # 计算Cost——Matrix
    for i in range(len(car_list)):
        if not car_list[i].dad:
            idx = find_pts(car_list[i], kdt)
            print idx
            for j in idx:
                #print pts[j]['loc']
                #print cost(car_list[i], pts[j]['loc'], pts[j]['img'])
                cost_arr[i,j] =  1 - cost(car_list[i], pts[j]['loc'], pts[j]['direction'], pts[j]['img'])[0]
                #print cost_arr[i,j]
    
    
    # 解算
    from munkres import Munkres, print_matrix
    
    matrix = cost_arr.tolist()
    m = Munkres()
    indexes = m.compute(matrix)
    # print_matrix(matrix, msg='Lowest cost through this matrix:')
    
    # 现在更新，没有匹配上的car假更新一下；匹配上的更新模板；位置;
    # 目标点集中如果没有被匹配上的添加新Car
    label_pts = np.zeros((len(pts)))
    for i, j in indexes:
        if pts[j]['loc'] != None and cost_arr[i,j] != 0.5:
            car_list[i].update(pts[j]['loc'], pts[j]['direction'])
            xc, yc = pts[j]['loc'].X, pts[j]['loc'].Y
            yc = h - yc
            car_list[i].template = img1[yc-20:yc+20,xc-20:xc+20,:]
            label_pts[j] = 1
        else:
            car_list[i].dummy_update()
        print '(%d, %d)->%f' % (car_list[i].m_id, j, cost_arr[i,j])
    
    for pt, label in zip(pts, label_pts):
        if label == 0 and pt['loc'] != None:
            xc, yc = pt['loc'].X, pt['loc'].Y
            yc = h - yc
            car_list.append(Car(Point(xc,h-yc), img1[yc-20:yc+20,xc-20:xc+20,:], pt['direction']))
            
    save_v(car_list, 'MS04_heatmap/v_%s'%os.path.split(f[0])[1])
    #show_all(img1, car_list)
    #raw_input("continue....?")
            
print "================== show and save =============================="
#plt.figure()
#plt.imshow(img)
#for i, j in indexes:
#    if pts[j]['loc'] != None:
#        plt.scatter( pts[j]['loc'].X, h-pts[j]['loc'].Y)
#    if pts[j]['loc'] != None and cost_arr[i,j] != 1:
#        plt.plot([car_list[i].curr_xy.X, pts[j]['loc'].X], \
#                 [h-car_list[i].curr_xy.Y, h-pts[j]['loc'].Y], \
#                 marker="o", markerfacecolor="r")
#        print '(%d, %d)->%f' % (car_list[i].m_id, j, cost_arr[i,j])

show_all(img, car_list)
    
    #raw_input("waiting...")
# plt.show()    
    
import cPickle as pkl
pkl.dump(car_list, file('track.pkl','wb'))