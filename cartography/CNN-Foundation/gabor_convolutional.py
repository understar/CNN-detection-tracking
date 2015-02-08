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
