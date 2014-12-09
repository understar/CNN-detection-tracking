# -*- coding: cp936 -*-
"""
Created on Tue Dec 09 16:14:53 2014

@author: shuaiyi
"""

import skimage.segmentation as seg
import skimage.io as io
import matplotlib.pyplot as plt

img = io.imread('test.png')

ax = plt.figure()

segments_slic = seg.slic(img, n_segments=1000, compactness=10, sigma=1)
plt.imshow(seg.visualize_boundaries(img, segments_slic))

#for fun
#def f(n):
#    if n == 1:
#        return 1000*1.047
#    return (f(n-1)+1000)*1.047