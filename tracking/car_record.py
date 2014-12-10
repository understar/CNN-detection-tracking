# -*- coding: cp936 -*-

from __future__ import division
import unittest
import numpy as np
import math

'''
Created on Mon Dec 08 16:00:57 2014

@author: Administrator

一条追踪记录包括什么？
1 标识ID
2 历史数据（位置数据，当前速度，平均速度，当前方向）
3 是否为新追踪
4 是否已经结束

单元测试
'''

class Point:
    def __init__(self, x, y):
        self.X = x
        self.Y = y
        
    def dist(self, pt2):
        return math.sqrt((self.X - pt2.X)**2 + (self.Y - pt2.Y)**2)
        
    def __repr__(self):
        return '(%s,%s)'%(self.X, self.Y)

# Tracking Record
class Car:
    hist_xy = list()
    hist_v = list()
    curr_v = 0
    curr_d = None
    curr_xy = None
    m_id = None
    def __init__(self, _id, _loc, step_t = 2.0):
        self.m_id = _id
        self.curr_xy = _loc
        self.step = step_t
        self.is_new = True
    
    def update(self, t1_xy, direction):
        self.curr_d = direction
        # 更新速度
        self.curr_v = self.curr_xy.dist(t1_xy)/self.step
        if not self.is_new:
            self.hist_v.append(self.curr_v)
        else:
            self.is_new = False
        
        self.hist_xy.append(self.curr_xy)
        self.curr_xy = t1_xy
     
    def cost(self, t1_all):
        # 分为4种类型计算每一个点的cost
        # 如果self.history不为空，属于type 1,2
        # 如果self.history为空，属于type 3
        # 如果候选t1_all为空，属于type 4
        pass
     
    def __repr__(self):
        return "Now : Location (%s, %s), Speed (%s m/s), Direction (%s)." % \
            (self.curr_xy.X, self.curr_xy.Y, self.curr_v, self.curr_d)

# 单元测试
class TestCar(unittest.TestCase):
    def setUp(self):
        self.m_car = Car(0, Point(0,0))

    def test_update(self):
        self.m_car.update(Point(3,4),45)
        self.m_car.update(Point(6,8),45)
        self.assertEqual(2.5, self.m_car.curr_v)
       
if __name__ == '__main__':
    #m_car = Car(0, Point(0,0))
    #m_car.update(Point(3,4),45)
    #m_car.update(Point(6,8),45)
    unittest.main()
    #print "Test!"
   

        
        