# -*- coding: cp936 -*-
"""
Created on Thu Mar 13 10:15:28 2014
用于从prediction结果中，提取中间层输出
@author: shuaiyi
"""

import cPickle as pl
import numpy as np
import sys

result = pl.load(open(sys.argv[1],'rb'))
np.savetxt('labels.txt', result['labels'], delimiter=',')
np.savetxt('data.txt', result['data'], delimiter=',')