set start=62
set end=99
set delta=95
set prj_name=MA
set img_pre=ON00

:通过分割进行检测，参数为图片文件夹及shp文件夹，及shp编号（有必要修正一下）
REM segement_detection.py ./segmentation/MA01 ./segmentation/MA01/results5
REM segement_detection.py ./segmentation/MA01 ./segmentation/MA01/results10
REM segement_detection.py ./segmentation/MA01 ./segmentation/MA01/results15
REM segement_detection.py ./segmentation/MA01 ./segmentation/MA01/results20

REM segement_detection.py ./segmentation/MS04 ./segmentation/MS04/results5
REM segement_detection.py ./segmentation/MS04 ./segmentation/MS04/results10
REM segement_detection.py ./segmentation/MS04 ./segmentation/MS04/results15
REM segement_detection.py ./segmentation/MS04 ./segmentation/MS04/results20

:设置过滤语句，导出需要的分割结果
REM for /l %%i in (62,1,68) do filter_polys.py ./segmentation/MA01/results5/ON00%%i.shp 0.95

REM for /l %%i in (83,1,111) do filter_polys.py ./segmentation/MS04/results15/MOS%%i.shp 0.95
REM for /l %%i in (83,1,111) do filter_polys.py ./segmentation/MS04/results20/MOS%%i.shp 0.95
REM for /l %%i in (83,1,111) do filter_polys.py ./segmentation/MS04/results15/MOS%%i.shp 0.95
REM for /l %%i in (83,1,111) do filter_polys.py ./segmentation/MS04/results20/MOS%%i.shp 0.95

:将分割结果转换为图像，方便进一步基于形态学的处理
REM for /l %%i in (62,1,68) do poly2ras.py ./segmentation/MA01/results5/95out_ON00%%i.shp ./segmentation/MA01/results5/ON00%%i.shp

REM for /l %%i in (83,1,111) do poly2ras.py ./segmentation/MS04/results10/95out_MOS%%i.shp ./segmentation/MS04/results10/MOS%%i.shp
REM for /l %%i in (83,1,111) do poly2ras.py ./segmentation/MS04/results15/95out_MOS%%i.shp ./segmentation/MS04/results15/MOS%%i.shp
REM for /l %%i in (83,1,111) do poly2ras.py ./segmentation/MS04/results20/95out_MOS%%i.shp ./segmentation/MS04/results20/MOS%%i.shp

:将分割结果导出位置，面积及方向等
REM for /l %%i in (62,1,68) do ras2loc.py ./segmentation/MA01/ON00%%i.tif ./segmentation/MA01/results5/95out_ON00%%i.tif

for /l %%i in (85,1,111) do ras2loc.py ./segmentation/MS04/MOS%%i.tif ./segmentation/MS04/results5/95out_MOS%%i.tif
REM ./segmentation/MS04/results10/95out_MOS%%i.tif ./segmentation/MS04/results15/95out_MOS%%i.tif ./segmentation/MS04/results20/95out_MOS%%i.tif

REM ras2loc.py ./segmentation/MS04/MOS83.tif ./segmentation/MS04/results5/95out_MOS83.tif ./segmentation/MS04/results10/95out_MOS83.tif ./segmentation/MS04/results15/95out_MOS83.tif ./segmentation/MS04/results20/95out_MOS83.tif
pause