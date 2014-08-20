# coding: cp936
import logging
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

THRE=0.8

def is_strong(x):
    global THRE
    if np.sum(x>THRE)/x.size <0.6:
        return 0
    else:
        center = (x.shape[0]-1)/2
        return x[center]

def is_max(x):
    global THRE
    if x.max() < THRE:
        return 0
    else:
        center = (x.shape[0]-1)/2
        if x.max()==x[center]: 
            return x[center]
        else:
            return 0
#: 一方面非最大（非最大不合适，会打散目标，只进行阈值化），另一方面去除孤立虚假响应
# TODO: 大一点的step滑动尺寸，会不会更容易处理？
# TODO: 使用形态学处理？
def not_max(r_map, thre=0.9 ,k=3):
    assert (k%2)!=0, "Odd number needed for k!"
    global THRE
    THRE = thre
    return ndimage.generic_filter(r_map, is_max, size=(k,k),mode='nearest')
 
""" Eliminate the pixel if its neighborhood dosen't strong enough.
"""   
def group(r_map, thre=0.9, k=5):
    assert (k%2)!=0, "Odd number needed for k!"
    global THRE
    THRE = thre
    return ndimage.generic_filter(r_map, is_strong, size=(k,k),mode='nearest')
    
def main(r_map_arr=None):
    if r_map_arr==None:
        logging.info("Loading default response map (test.npy).")
        r_map_arr=np.load('test.npy')
    
    logging.info("Using default not-max suppression k=3.")
    logging.info("Using default thre value 0.8")
    result1=not_max(r_map_arr)
    
    logging.info("Using group method k=3.")
    logging.info("Using default thre value 0.9")
    result2=group(r_map_arr)
    #fig = plt.figure()
    plt.subplot(131)
    plt.imshow(r_map_arr)
    plt.subplot(132)
    plt.imshow(result1)
    plt.subplot(133)
    plt.imshow(result2)
    
if __name__=="__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
