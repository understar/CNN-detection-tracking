# -*- coding: cp936 -*-
import os, sys
from osgeo import gdal, ogr
import matplotlib.pyplot as plt
import numpy as np

def poly_ras(in_shp, src_shp):
    # Define pixel_size and NoData value of new raster
    pixel_size = 1
    NoData_value = 0
    
    driver = ogr.GetDriverByName('ESRI Shapefile')
    # 打开原shp
    source_ds = driver.Open(src_shp, 0)
    if source_ds is None: 
        print 'Could not open ' + src_shp
        sys.exit(1) #exit with an error code
    
    source_layer = source_ds.GetLayer(0) 
    
    # Open the data source and read in the extent
    # TODO: 注意shp被过滤过，可能会导致extent变化
    x_min, x_max, y_min, y_max = source_layer.GetExtent()
    
    # Create the destination data source
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)
    
    # 打开过滤后的shp
    source_ds = driver.Open(in_shp, 0)
    if source_ds is None: 
        print 'Could not open ' + in_shp
        sys.exit(1) #exit with an error code
    
    source_layer = source_ds.GetLayer(0) 
    
    raster_fn = os.path.join( os.path.split(in_shp)[0] , os.path.split(in_shp)[1][0:-4] + ".tif")
    
    target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, x_res, y_res, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(NoData_value)
    
    # Rasterize
    gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[255])
    
    # Read as array
    return band.ReadAsArray()
    
if __name__ == '__main__':
    if len( sys.argv ) < 3:
        print "[ ERROR ]: you need to pass at least 2 arg -- input shp -- input source shp"
        sys.exit(1)
#    try:
#        for f in os.listdir(sys.argv[1]):
#            if f[-3:] == "shp":
#                poly_ras( os.path.join(sys.argv[1], f) )
#    except:
    poly_ras(sys.argv[1], sys.argv[2])