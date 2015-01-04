# -*- coding: cp936 -*-
"""

@author: shuaiyi
"""
import matplotlib.pyplot as plt
import os, sys
import numpy as np
from skimage.io import imread

from kit_angle_net import DecafNet as AngleNet
_DETECTION = None
_ANGLE = AngleNet()
_WIDTH = 40

Decaf = False
if Decaf:
    from kitnet import DecafNet as KitNet
    _DETECTION = KitNet()
else:
    os.chdir("E:/2013/cuda-convnet/trunk")
    from show_pred import model as car_model
    _DETECTION = car_model
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

def Cal_Angle(img, detection_net=_DETECTION, angle_net=_ANGLE):
    if Decaf:
        scores = detection_net.classify(img, False)
        is_car = detection_net.top_k_prediction(scores, 1)
        #print is_car
        if is_car[1][0] == 'car':
            car_conv3 = detection_net.feature("conv3_neuron_cudanet_out")
            mid_convs = car_conv3.reshape((car_conv3.shape[0],-1))
            scores = angle_net.classify(mid_convs)
            angles = angle_net.top_k_prediction(scores, 3)
            print angles
            
            o = float(angles[1][0])-90 # *180/np.pi
            o = o if o>0 else o+360        
            return int(o)
    else:
        scores = car_model.show_predictions(img[:,:,0])
        print scores
        if scores[-1][0] == 'car':
            mid_convs = car_model.get_features(img[:,:,0])
            scores = angle_net.classify(mid_convs)
            angles = angle_net.top_k_prediction(scores, 3)
            print angles
            
            o = float(angles[1][0])-90 # *180/np.pi
            o = o if o>0 else o+360        
            return int(o)
    return None
    
if __name__ == '__main__':
    #kit_net = KitNet()
    #car = smalldata.car()
    ## print car.shape
    #car = car.reshape((40,40,1))
    #scores = kit_net.classify(car)
    #print 'Is car ? prediction:', kit_net.top_k_prediction(scores, 1)
    #
    #car_conv3 = kit_net.feature("conv3_cudanet_out")
    #mid_convs = car_conv3.reshape((car_conv3.shape[0],-1))
    #
    #net = DecafNet()
    #scores = net.classify(mid_convs)
    #print 'Direction ? prediction:', net.top_k_prediction(scores, 5)
    
    with open("./segmentation/Test/angle.txt", 'w') as f:
        dir_path = "./segmentation/Test/angle"
        for img_path in os.listdir(dir_path):
            print "Processing ... %s" % img_path
            img = imread(os.path.join(dir_path,img_path))
            img = img.reshape((img.shape[0], img.shape[1], 1))
            angle = Cal_Angle(img)
            if angle != None:
                f.writelines([img_path, '\t', str(angle), '\n'])