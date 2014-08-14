# coding: cp936
import logging
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

THRE=0.8

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

def not_max(r_map, thre=0.8 ,k=3):
    assert (k%2)!=0, "Odd number needed for k!"
    global THRE
    THRE = thre
    return ndimage.generic_filter(r_map, is_max, size=(k,k),mode='nearest')
    
def main(r_map_arr=None):
    if r_map_arr==None:
        logging.info("Loading default response map (test.npy).")
        r_map_arr=np.load('test.npy')
    
    logging.info("Using default not-max suppression k=3.")
    logging.info("Using default thre value 0.8")
    result=not_max(r_map_arr)
    
    #fig = plt.figure()
    plt.subplot(121)
    plt.imshow(r_map_arr)
    plt.subplot(122)
    plt.imshow(result)
    #fig.colorbar()
    
if __name__=="__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
