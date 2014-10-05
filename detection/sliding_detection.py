# -*- coding: cp936 -*-
"""
Created on Mon Aug 18 19:46:56 2014

@author: Administrator
"""
import logging
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from kitnet import DecafNet
from skimage.color import rgb2gray

WIN_SIZE = 40
NET = None
IMG = None
GRAY = None

def load_net(net_file=None, meta_file=None):
    global NET, IMG, GRAY
    logging.info("Loading DecafNet...")
    NET = DecafNet(net_file,meta_file)
    logging.info("Loading default image...")
    #global IMG = io.imread("test.png")

def detect(loc):
    # global NET, IMG, GRAY, WIN_SIZE
    # print loc
    x, y, NET, GRAY, WIN_SIZE = loc
    logging.info("Detect row %s, col %s" % (x, y))
    temp = np.zeros((WIN_SIZE,WIN_SIZE),dtype=np.uint)
    try:
        temp = GRAY[x:(x+WIN_SIZE), y:(y+WIN_SIZE)]
        temp = temp.reshape((WIN_SIZE,WIN_SIZE,1))          
        scores = NET.classify(temp)
        return NET.top_k_prediction(scores, 1)                    
    except:
        return None

def run_slidding(img=None, ssize=20, mask=None):
    global NET, IMG, GRAY, WIN_SIZE
    if img==None:
        logging.info("Loading default image (test.png-cross2)")
        IMG = io.imread("test.png")
    else:
        IMG=io.imread(img)
    GRAY = rgb2gray(IMG)
    
    if mask==None:
        logging.info("Search all image...")
        mask = np.ones(GRAY.shape)
    else:
        logging.info("Using mask...")
        
    logging.info("Using sliding step size %s" % ssize)
    logging.info("Image info:" + str(GRAY.shape))
    row, col = GRAY.shape
    
    logging.info("Sliding the window to detect vehicles...")
    # logging.info("Single thread.")
    # 不仅没有mpi支持，而且还是单线程，太慢了；
    result={}
    locs=[]
    for x in np.arange(0, row - WIN_SIZE + 1, ssize):
        for y in np.arange(0, col - WIN_SIZE + 1, ssize):
            if mask[x,y]==1:
                locs.append((x,y,NET, GRAY, WIN_SIZE))
    #logging.info("Multiprocessing...")
    #with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
    #    for loc, res in zip(locs, executor.map(detect, locs)):
    #        if res != None:
    #            result[loc[0:2]] = res
    logging.info("Normal sliding window detection.")
    #并行不是必须的电脑资源就这么多
    for loc in locs:
        res = detect(loc)
        if res != None:
            result[loc[0:2]] = res
    return result
    
def show_detection(result=None):
    global NET, IMG, GRAY
    if len(result) == 0:
        logging.info("Run_sliding first...")
        return 0
    else:
        plt.subplot(111)
        plt.imshow(IMG)
        for k,v in result.items():
            #print v
            if v[1]==['car']: # v=(scores[array], labels[list])
                plt.scatter(k[1]+WIN_SIZE/2, k[0]+WIN_SIZE/2, marker='+', \
                            s=100, linewidths=2, c="g", cmap=plt.cm.coolwarm)
            
        plt.show()
    return 1
                
        
if __name__=="__main__":
    logging.getLogger().setLevel(logging.INFO)
    # detector = sliding()
    load_net()
    # TODO: 支持mask检测，如何得到好的Mask？
    result = run_slidding(img="dark.png",ssize=5)
    show_detection(result)
    
    import pickle as pkl
    pkl.dump({"result":result}, open("dark.pkl","wb"))