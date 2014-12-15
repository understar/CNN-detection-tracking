:通过分割进行检测，参数为图片文件夹及shp文件夹，及shp编号（有必要修正一下）
rem segement_detection.py ./segmentation/MS04 ./segmentation/MS04/results5 83 86
segement_detection.py ./segmentation/MS04 ./segmentation/MS04/results10 83 86
segement_detection.py ./segmentation/MS04 ./segmentation/MS04/results15 83 86
segement_detection.py ./segmentation/MS04 ./segmentation/MS04/results20 83 86

:设置过滤语句，导出需要的分割结果
rem filter_polys.py ./segmentation/MS04/results20/MOS83.shp value

:将分割结果转换为图像，方便进一步基于形态学的处理
rem poly2ras.py ./segmentation/MS04/results20/out_MOS83.shp
pause