# coding: cp936
from __future__ import print_function

import matplotlib.pyplot as plt
from matplotlib import rcParams
# rcParams dict
rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['figure.figsize'] = 7, 5

import numpy as np
from scipy import ndimage as nd

from skimage import io
from skimage.util import img_as_float
from skimage.filter import gabor_kernel

test = img_as_float(io.imread('conv_test.png',True))

def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    # 仅仅使用实部
    return nd.convolve(image, np.real(kernel), mode='wrap')
    #return np.sqrt(nd.convolve(image, np.real(kernel), mode='wrap')**2 +
    #               nd.convolve(image, np.imag(kernel), mode='wrap')**2)

# Plot a selection of the filter bank kernels and their responses.
results = []
kernel_params = []
for theta in range(4):
    theta = theta / 4. * np.pi
    frequency = 0.1
    kernel = gabor_kernel(frequency, theta=theta)
    # print(kernel)
    params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
    kernel_params.append(params)
    # Save kernel and the power image for each image
    results.append((kernel, power(test, kernel)))

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(5, 4))
plt.gray()

# fig.suptitle('Convolutional Layer Using Gabor', fontsize=11)

axes[0][0].axis('off')

# Plot Source image
axes[1][0].imshow(test)
axes[1][0].set_title("Input image", fontsize=11)
axes[1][0].axis('off')

for label, (kernel, power), idx in zip(kernel_params, results, range(1,5)):
    # Plot Gabor kernel
    ax = axes[0,idx]
    ax.imshow(np.real(kernel), interpolation='nearest')
    ax.set_title(label, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot Gabor responses with the contrast normalized for each filter
    ax = axes[1,idx]
    vmin = np.min(power)
    vmax = np.max(power)
    ax.imshow(power, vmin=vmin, vmax=vmax)
    ax.axis('off')

plt.show()


''' pylearn2 CNORM 代码'''
class CrossChannelNormalization(object):
    """
    See "ImageNet Classification with Deep Convolutional Neural Networks"
    Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton
    NIPS 2012
    Section 3.3, Local Response Normalization
    .. todo::
        WRITEME properly
    f(c01b)_[i,j,k,l] = c01b[i,j,k,l] / scale[i,j,k,l]
    scale[i,j,k,l] = (k + sqr(c01b)[clip(i-n/2):clip(i+n/2),j,k,l].sum())^beta
    clip(i) = T.clip(i, 0, c01b.shape[0]-1)
    Parameters
    ----------
    alpha : WRITEME
    k : WRITEME
    beta : WRITEME
    n : WRITEME
    """

    def __init__(self, alpha = 1e-4, k=2, beta=0.75, n=5):
        self.__dict__.update(locals())
        del self.self

        if n % 2 == 0:
            raise NotImplementedError("Only works with odd n for now")

    def __call__(self, c01b):
        """
        .. todo::
            WRITEME
        """
        half = self.n // 2 #等价于self.n % 2
        sq = T.sqr(c01b)
        ch, r, c, b = c01b.shape
        extra_channels = T.alloc(0., ch + 2*half, r, c, b)
        sq = T.set_subtensor(extra_channels[half:half+ch,:,:,:], sq)
        scale = self.k
        for i in xrange(self.n):
            scale += self.alpha * sq[i:i+ch,:,:,:]

        scale = scale ** self.beta
        return c01b / scale