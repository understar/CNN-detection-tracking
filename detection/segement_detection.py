# -*- coding: cp936 -*-
"""
Created on Fri Oct 17 19:51:20 2014
程序逻辑：
可选：添加字段标识是否为车辆；
1、遍历每一个region；取center xy
2、获取offset；
3、以中心点取取正方形图像；
4、分类，为字段赋值

注意的问题：
保证样本的采集的正确性; xy轴不要搞错了; 样本正确

多尺度分割内容：
我们需要破碎的分割，但是可以先不破碎分割的情况下，快速去除不需要的内容；
在判断为车辆的区域合并后，再破碎分割

@author: shuaiyi
"""
import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning) 
#自定义警告处理函数，将所有警告屏蔽掉
def customwarn(message, category, filename, lineno, file=None, line=None):
    pass

warnings.showwarning = customwarn

import os, sys, math
import cv2 #使用cv2的resize，实现图像的缩放
import skimage.io as io #读写图像
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.util import img_as_ubyte
from sklearn.externals import joblib

import progressbar # 进度条
#import logging
#logging.getLogger()

try:
    from osgeo import ogr, gdal
except:
    import ogr, gdal

#from optparse import OptionParser
#
#parser = OptionParser()
#parser.add_option("-f", "--file", dest="image filename",
#                  help="source image file", metavar="FILE")
#parser.add_option("-q", "--quiet",
#                  action="store_false", dest="verbose", default=True,
#                  help="don't print status messages to stdout")
#
#(options, args) = parser.parse_args()


"""基于坐标值经纬度，以及栅格的信息，计算影像坐标
"""
def offset(ds, x, y):
    # get georeference info
    transform = ds.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]

    # compute pixel offset
    xOffset = int((x - xOrigin) / pixelWidth)
    yOffset = int((y - yOrigin) / pixelHeight)
    return (xOffset, yOffset)

"""判断shp中是否包含某一字段
"""
def is_exist(Layer,field_name):
    layerDefinition = Layer.GetLayerDefn()
    for i in range(layerDefinition.GetFieldCount()):
        if layerDefinition.GetFieldDefn(i).GetName() == field_name:
            return True
    return False

"""得到单个feature所在的raster范围影像数据
"""
def getRegion(r, f): # raster feature
    geo = f.GetGeometryRef()
    c_pt = geo.Centroid()
    cx, cy = offset(r, c_pt.GetX(), c_pt.GetY())
    
    l_x, r_x, d_y, u_y = geo.GetEnvelope() # 记录的是上下Y坐标，以及左右x坐标
    env_w = r_x - l_x
    env_h = u_y - d_y
    max_len = math.sqrt(env_w**2 + env_h**2)
    env_area = env_w * env_h
    
    # sample size : 40
    w = h = 40
    in_size = 40

    #print cx , cy, w, h
    if cx - w/2 >= 0 and cx + w/2 <= r.RasterXSize \
    and cy - h/2 >= 0 and cy + h/2  <= r.RasterYSize:
        cx = cx - w/2
        cy = cy - h/2
    else:
        # 边缘区域
        if cx - w/2 < 0:
            cx = 0
        elif cx + w/2 > r.RasterXSize:
            cx = r.RasterXSize - w - 5 #
        else:
            cx = cx - w/2
            
        if cy - h/2 < 0:
            cy = 0
        elif cy + h/2 > r.RasterYSize:
            cy = r.RasterYSize - h - 5 #
        else:
            cy = cy - h/2
        
    img = r.ReadAsArray(cx ,cy , w, h)
    img = img.swapaxes(0,2).swapaxes(0,1)
    img = rgb2gray(img)
    img = resize(img, (in_size, in_size), mode='wrap')
    return img_as_ubyte(img.reshape((in_size, in_size,1))), (cx,cy,w,h), max_len, env_area
    #io.imsave("segementation/%s_%s_%s_%s.png" % \
    #         (lu_offset_x, lu_offset_y, w, h), img)
    #tmp = cv2.imread("segementation/%s_%s_%s_%s.png" % \
    #                (lu_offset_x, lu_offset_y, w, h))
    #return cv2.resize(tmp, (256,256), interpolation=cv2.INTER_LINEAR)
    #return resize(img, (256,256))

# 加载 decaf 和 classifier
from kitnet import DecafNet
net = DecafNet()

# 读取栅格图像
gdal.AllRegister()

if len(sys.argv) != 5:
    print "Usage: segement_detection.py path_image_folder path_shp_folder start end"
    sys.exit()
else:
    image_folder = sys.argv[1]
    shp_folder = sys.argv[2]
    start = int(sys.argv[3])
    end = int(sys.argv[4])

for i in range(start, end):
    g_raster = gdal.Open(image_folder+'/MOS%s.tif'%i, gdal.GA_ReadOnly) # 与分割文件对应的原始栅格
    
    print "Processing image " + image_folder+'/MOS%s.tif'%i
    # 读取分割结果 shp 文件
    driver = ogr.GetDriverByName('ESRI Shapefile')
    os.chdir(shp_folder)
    fn = "MOS%s.shp"%i
    dataSource = driver.Open(fn, 1) # 需要读写
    os.chdir(os.path.dirname(__file__))
    if dataSource is None: 
        print 'Could not open ' + fn
        sys.exit(1) #exit with an error code
    
    layer = dataSource.GetLayer(0)   
    
    # 添加字段 slide 
    # 如果已经存在就不再添加
    if not is_exist(layer, "car"):
        fieldDefn = ogr.FieldDefn('car', ogr.OFTInteger)
        layer.CreateField(fieldDefn)
        
    if not is_exist(layer, 'value'):
        fieldDefn = ogr.FieldDefn('value', ogr.OFTReal)
        layer.CreateField(fieldDefn)
    
    if not is_exist(layer, 'env'):
        fieldDefn = ogr.FieldDefn('env', ogr.OFTString)
        layer.CreateField(fieldDefn)
    
    if not is_exist(layer, 'maxlen'):
        fieldDefn = ogr.FieldDefn('maxlen', ogr.OFTReal)
        layer.CreateField(fieldDefn)
        
    if not is_exist(layer, 'envarea'):
        fieldDefn = ogr.FieldDefn('envarea', ogr.OFTReal)
        layer.CreateField(fieldDefn)
    
    numFeatures = layer.GetFeatureCount()
    print 'Total region count:', numFeatures
    
    #test
    img = None
    TEST = False
    if TEST == True:
        feature = layer.GetNextFeature()
        img, env, maxlen, envarea = getRegion(g_raster, feature)
        scores = net.classify(img, False)
        is_car = net.top_k_prediction(scores, 2)
        if is_car[1][0] == 'car':
            print "Woh...a car..."
        raw_input("enter any character break:")
        break
    else:
        # loop through the regions and predict them
        pbar = progressbar.ProgressBar(maxval=numFeatures).start()
        
        cnt = 0
        feature = layer.GetNextFeature()
        while feature:
            # 获取对应的图像样本
            img, env, maxlen, envarea= getRegion(g_raster, feature)

            scores = net.classify(img, False)
            is_car = net.top_k_prediction(scores, 2)
            # print type(is_car[0][0])
            if is_car[1][0] == 'car':
                feature.SetField("car", 1)
                feature.SetField("value", float(is_car[0][0]))
            else:
                feature.SetField("car", 0)
                feature.SetField("value", float(is_car[0][1]))
            # 全部输出
            
            feature.SetField("env", "%s,%s,%s,%s" % env)
            feature.SetField("maxlen", maxlen)
            feature.SetField("envarea", envarea)
            
            layer.SetFeature(feature) # 这一步可以用于保存修改
            pbar.update(cnt+1)
            cnt = cnt + 1
            feature = layer.GetNextFeature()
                
        pbar.finish()
    
    # close the data source
    dataSource.Destroy()
