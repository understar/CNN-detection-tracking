# -*- coding: cp936 -*-
"""
Cuda-convnet conv3 T-SNE可视化
@author: shuaiyi
"""

import os
from skimage import io
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
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

import matplotlib.cm as cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox 
import random

def visualization(sample_list, feature_arr):
    model = TSNE(n_components=2, random_state=np.random.RandomState())
    print "t-SNE..."
    f_tsne = model.fit_transform(feature_arr)
    
    ax = plt.subplot(111)
    X = []
    for i in range(len(sample_list)):
        if random.randint(0,10) >= 6:
            img_path = "./angle_neg/%s" % sample_list[i].rstrip()
            xy = (f_tsne[i][0], f_tsne[i][1])
            X.append(xy)
            arr_sam = io.imread(img_path) 
        
            imagebox = OffsetImage(arr_sam, zoom=0.5, cmap=cm.gray) 
            ab = AnnotationBbox(imagebox, xy, 
                                xycoords='data', 
                                pad=0, 
                                )
            ax.add_artist(ab) 

    X = np.array(X)
    
    #ax.grid(True) #打开格网
    #ax.set_xlim(-40, 40)
    #ax.set_ylim(-40, 40)
    plt.scatter(X[:,0], X[:,1], 7)
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.title('t-SNE Visualization')
    plt.savefig("tSNE.tif", dpi=300)
    plt.show()
       
if __name__ == "__main__":
    samples_list = file("images.txt").readlines() 
    feature_arr = np.loadtxt("features.txt",delimiter=',')
    visualization(samples_list, feature_arr)