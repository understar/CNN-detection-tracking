from skimage.segmentation import felzenszwalb
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

img = io.imread('cross2.png')

plt.subplots_adjust(left=0.25, bottom=0.25)


plt.subplot(121)
plt.imshow(img)

axcolor = 'lightgoldenrodyellow'
axSigma = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
axScale  = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
axMins  = plt.axes([0.25, 0.2, 0.65, 0.03], axisbg=axcolor)

sSigma = Slider(axSigma, 'Sigma', 0.0, 1, valinit=0.8)
sScale = Slider(axScale, 'Scale', 1, 1000, valinit=200)
sMins = Slider(axMins, 'Min-size', 1, 100, valinit=50)

def update(val):
    sigma = sSigma.val
    scale = int(sScale.val)
    min_size = int(sMins.val)
    result = felzenszwalb(img, sigma=sigma,scale = scale, min_size=min_size)
    plt.subplot(122)
    plt.imshow(result)
    #plt.colorbar()

sSigma.on_changed(update)
sScale.on_changed(update)
sMins.on_changed(update)