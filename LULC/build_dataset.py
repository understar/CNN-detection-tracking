# -*- coding: cp936 -*-
"""
用于从按文件夹组织的样本中构建训练数据集
数据集格式：
字典，键为类型名称，值为列表，存储样本numpy数组
@author: shuaiyi
"""

import numpy as np
import os, sys
# from sklearn.externals import joblib
import pickle as pkl
from skimage.io import imread
from PIL import Image

img_size = (256,256)

def resize(img):
    tmp = Image.fromarray(img)
    tmp = tmp.resize(img_size)
    return np.array(tmp)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print 'Usage: build_dataset.py root_fold_path'
    else:
        root = sys.argv[1]
        dataset = {}
        for d in os.listdir(root):
            catelog = d
            images = []
            for img in os.listdir(os.path.join(root, d)):
                img_path = os.path.join(root, d, img) 
                print 'Processing' ,img_path
                tmp = imread(img_path)
                if tmp.shape != (256,256,3):
                    #print 'Resizing',img_path,tmp.shape
                    tmp = resize(tmp)
                    #print tmp.shape
                images.append(tmp)
                
            dataset[catelog] = images
        
        print 'Saving...'
        save_file = os.path.split(root)[-1] + '.pkl'
        pkl.dump(dataset, file(save_file,'wb'))
        #joblib.dump(dataset ,save_file)
                
    