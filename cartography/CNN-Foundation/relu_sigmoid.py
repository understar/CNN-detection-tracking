# -*- coding: cp936 -*-
"""
Created on Thu Oct 09 22:22:43 2014

@author: shuaiyi
"""

import sys
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
# rcParams dict
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['figure.figsize'] = 7, 6

def relu(x):
    return np.max([x, 0])

def sigmoid(x):
    return 1.0/(1.0+np.e**(-x))

if __name__ == "__main__":
    x = np.arange(-10,10,0.1)
    np_relu = np.vectorize(relu)
    y = np_relu(x)
    ax1 = plt.subplot(121)
    plt.plot(x,y,'b-')
    #plt.title("LeRU")
    ax1.set_xlabel('(a) LeRU')
    
    np_sigmoid = np.vectorize(sigmoid)
    y = np_sigmoid(x)
    ax2 = plt.subplot(122)
    plt.plot(x,y,'b-')
    #plt.title("Sigmoid")
    ax2.set_xlabel('(b) Sigmoid')
