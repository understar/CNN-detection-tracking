# -*- coding: cp936 -*-
"""
Created on Thu Dec 11 15:37:10 2014
拷贝shp文件
在拷贝的过程中可以，加入过滤，通过过滤可以变成一个detection过程
@author: shuaiyi
"""

from osgeo import ogr
import os, sys

def filter_poly( inShapefile, field_name_target=[], prob_filter=0.8 ):
    # Get the input Layer
    # inShapefile = "~/DATA/SHAPES/KC_ADMIN/parcel_address/parcel_address.shp"
    inDriver = ogr.GetDriverByName("ESRI Shapefile")
    inDataSource = inDriver.Open(inShapefile, 0)
    inLayer = inDataSource.GetLayer()
    
    # TODO: 此处为过滤语句,如何方便指导，方便调试
    # (Car = 1) AND (Value >= 0.8)
    inLayer.SetAttributeFilter("(Car = 1) AND (value >= %s)"%prob_filter)

    # Create the output LayerS
    outShapefile = os.path.join( os.path.split( inShapefile )[0], 
                                "%sout_"%int(prob_filter*100) + os.path.split(inShapefile)[1] )
    outDriver = ogr.GetDriverByName("ESRI Shapefile")

    # Remove output shapefile if it already exists
    if os.path.exists(outShapefile):
        outDriver.DeleteDataSource(outShapefile)

    # Create the output shapefile
    outDataSource = outDriver.CreateDataSource(outShapefile)
    out_lyr_name = os.path.splitext( os.path.split( outShapefile )[1] )[0]
    outLayer = outDataSource.CreateLayer( out_lyr_name, geom_type=ogr.wkbMultiPolygon )

    # Add input Layer Fields to the output Layer if it is the one we want
    inLayerDefn = inLayer.GetLayerDefn()
    for i in range(0, inLayerDefn.GetFieldCount()):
        fieldDefn = inLayerDefn.GetFieldDefn(i)
        fieldName = fieldDefn.GetName()
        
        # 仅仅保留需要的字段
        if fieldName not in field_name_target:
            continue
        outLayer.CreateField(fieldDefn)

    # Get the output Layer's Feature Definition
    outLayerDefn = outLayer.GetLayerDefn()

    # Add features to the ouput Layer
    for inFeature in inLayer:
        # Create output Feature
        outFeature = ogr.Feature(outLayerDefn)

        # Add field values from input Layer
        for i in range(0, outLayerDefn.GetFieldCount()):
            fieldDefn = outLayerDefn.GetFieldDefn(i)
            fieldName = fieldDefn.GetName()
            if fieldName not in field_name_target:
                continue

            outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(),
                inFeature.GetField(i))

        # Set geometry as centroid
        geom = inFeature.GetGeometryRef()
        outFeature.SetGeometry(geom.Clone())
        # Add new feature to output Layer
        outLayer.CreateFeature(outFeature)

    # Close DataSources
    inDataSource.Destroy()
    outDataSource.Destroy()

if __name__ == '__main__':
    if len( sys.argv ) < 3:
        print "[ ERROR ]: you need to pass at least two arg --input shp  --prob filter"
        sys.exit(1)

    filter_poly( sys.argv[1], prob_filter=float(sys.argv[2]) )