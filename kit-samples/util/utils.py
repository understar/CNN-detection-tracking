# -*- coding: cp936-*-
__author__ = 'shuaiyi'

import cv2

def save_samples(path, samples_dict):
    
    for k,v in samples_dict.items():
        print 'Saving...%s' % k
        cv2.imwrite('%s//%s.png'%(path, k), v)