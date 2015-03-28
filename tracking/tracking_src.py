# -*- coding: cp936 -*-
"""
Created on Wed Jan 07 22:26:46 2015

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


def save_v(car_list, fname):
    with file(fname, 'w') as f:
        f.writelines(['x,y,v\n'])
        for car in car_list:
            if 0 != len(car.hist_xy):
                f.writelines(['%s,'%car.curr_xy.X, '%s,'%car.curr_xy.Y, \
                            '%s\n'%(modulus(car.curr_xy.vec() - \
                            car.hist_xy[-1].vec())/(car.interval*car.step))])
        

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

def show_all_syms_labels(img, car_list, syms, labels):
    plt.figure()
    plt.imshow(img)
    for car, sym, l in zip(car_list, syms, labels):
        X = []
        Y = []
        for pos in car.hist_xy:
            X.append(pos.X)
            Y.append(h-pos.Y)
        X.append(car.curr_xy.X)
        Y.append(h-car.curr_xy.Y)
        plt.plot(X, Y, sym, ms=20, label = l)#, marker="o", markerfacecolor="r")
    plt.legend()
        #if len(X) != 0:
        #    plt.annotate(str(car.m_id),(X[-1], Y[-1]))
    
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
dir_path = './MS04_src'
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
    oid, xc, yc, o = line.split(',')
    xc = float(xc)
    yc = float(yc)
    o = float(o)
    car_list.append(Car(Point(xc,h-yc), img[yc-20:yc+20,xc-20:xc+20,:], o, int(oid)))
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
        oid, xc, yc, o = line.split(',')
        xc = float(xc)
        yc = float(yc)
        o = float(o)
        target_pt = {}
        target_pt['id'] = int(oid)
        target_pt['loc'] = Point(xc, h-yc)
        target_pt['direction'] = o
        pts.append(target_pt)
    
    # 现在更新，没有匹配上的car假更新一下；匹配上的更新模板；位置;
    # 目标点集中如果没有被匹配上的添加新Car
    label_pts = np.zeros((len(pts)))
    label_car = np.zeros((len(car_list)))
    for i in range(len(car_list)):
        for j in range(len(pts)):
            if pts[j]['id'] == car_list[i].oid:
                car_list[i].update(pts[j]['loc'], pts[j]['direction'])
                xc, yc = pts[j]['loc'].X, pts[j]['loc'].Y
                yc = h - yc
                car_list[i].template = img1[yc-20:yc+20,xc-20:xc+20,:]
                label_pts[j] = 1
    
    for pt, label in zip(pts, label_pts):
        if label == 0 and pt['loc'] != None:
            xc, yc = pt['loc'].X, pt['loc'].Y
            yc = h - yc
            car_list.append(Car(Point(xc,h-yc), img1[yc-20:yc+20,xc-20:xc+20,:], pt['direction'], pt['id']))
            
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
    
import cPickle as pkl
pkl.dump(car_list, file('track_src.pkl','wb'))