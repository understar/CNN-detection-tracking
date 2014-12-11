REM segement_detection.py ./segmentation/MS04 ./segmentation/MS04/results5 83 86
REM segement_detection.py ./segmentation/MS04 ./segmentation/MS04/results10 83 86
REM segement_detection.py ./segmentation/MS04 ./segmentation/MS04/results15 83 86
REM segement_detection.py ./segmentation/MS04 ./segmentation/MS04/results20 83 86
rem filter_polys.py ./segmentation/MS04/results20/MOS83.shp value
poly2ras.py ./segmentation/MS04/results20/out_MOS83.shp
pause