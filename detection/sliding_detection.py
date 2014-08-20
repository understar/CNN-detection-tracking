# -*- coding: cp936 -*-
"""
Created on Mon Aug 18 19:46:56 2014

@author: Administrator
"""
import logging
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
from kitnet import DecafNet
from skimage.color import rgb2gray

WIN_SIZE = 40

class sliding(object):
    def __init__(self, net_file=None, meta_file=None):
        logging.info("Loading DecafNet...")
        self.net = DecafNet(net_file,meta_file)
        logging.info("Loading default image...")
        self.img = io.imread("test.png")
        
    def run_slidding(self, img=None, ssize=20, mask=None):
        if img==None:
            logging.info("Loading default image (test.png-cross2)")
            self.img = io.imread("test.png")
        else:
            self.img=io.imread(img)
        self.gray = rgb2gray(self.img)
        
        if mask==None:
            logging.info("Search all image...")
            mask = np.ones(self.gray.shape)
        else:
            logging.info("Using mask...")
            
        logging.info("Using sliding step size %s" % ssize)
        logging.info("Image info:" + str(self.gray.shape))
        row, col = self.gray.shape
        
        logging.info("Sliding the window to detect vehicles...")
        result={}
        for x in np.arange(0, row, ssize):
            for y in np.arange(0, col, ssize):
                if mask[x,y]==1:
                    logging.info("Detect row %s, col %s"%(x, y))
                    temp = np.zeros((WIN_SIZE,WIN_SIZE),dtype=np.uint)
                    try:
                        temp = self.gray[x:(x+WIN_SIZE), y:(y+WIN_SIZE)]
                        temp = temp.reshape((WIN_SIZE,WIN_SIZE,1))          
                        scores = self.net.classify(temp)
                        result[(x,y)]=self.net.top_k_prediction(scores, 1)                    
                    except:
                        continue
        return result
        
    def show_detection(self, result=None):
        if result==None:
            logging.info("Run_sliding first...")
            return 0
        else:
            plt.subplot(111)
            plt.imshow(self.img)
            for k,v in result.items():
                #print v
                if v[1]==['car']: # v=(scores[array], labels[list])
                    plt.scatter(k[1]+WIN_SIZE/2, k[0]+WIN_SIZE/2, marker='+', s=100, linewidths=2, c="g", cmap=plt.cm.coolwarm)
                
            plt.show()
        return 1
                
        
if __name__=="__main__":
    logging.getLogger().setLevel(logging.INFO)
    detector = sliding()
    # TODO: 支持mask检测，如何得到好的Mask？
    result = detector.run_slidding(ssize=3)
    detector.show_detection(result)