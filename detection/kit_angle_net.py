#coding:cp936
"""
implements a wrapper over the kitnet classifier trained by Zhangshuaiyi 
using the cuda convnet code.
目前支持Layertype：
conv、fc、cmrnorm、neuron、pool、softmax

"""
import cPickle as pickle
from decaf.util import translator, transform
import logging
import numpy as np
import os,sys

#sys.path.append(os.path.dirname(__file__))

# angle training model
_KITNET_FILE = 'kitanglenet.epoch41'
_META_FILE = 'anglebatches.meta41'

# This is a legacy flag specifying if the network is trained with vertically
# flipped images, which does not hurt performance but requires us to flip
# the input image first.
_KITNET_FLIP = True

# Kit mid (conv3) size
INPUT_DIM = 2048

class DecafNet(object):
    """A wrapper that returns the decafnet interface to classify images."""
    def __init__(self, net_file=None, meta_file=None):
        """Initializes DecafNet.

        Input:
            net_file: the trained network file.(normally,['model_state']['layers'])
            meta_file: the meta information for images.
        """
        logging.info('Initializing decafnet...')
        try:
            if not net_file:
                # use the internal decafnet file.
                net_file = _KITNET_FILE
            if not meta_file:
                # use the internal meta file.
                meta_file = _META_FILE
            cuda_decafnet = pickle.load(open(net_file,'rb'))['model_state']['layers']
            meta = pickle.load(open(meta_file,'rb'))
        except IOError:
            raise RuntimeError('Cannot find DecafNet files.')
        # First, translate the network
        self._net = translator.translate_cuda_network(
            cuda_decafnet, {'data': (INPUT_DIM, 1)})
        # Then, get the labels and image means.
        self.label_names = meta['label_names']
        self._data_mean =  meta['data_mean']
        logging.info('Kitnet initialized.')
        return

    def classify_direct(self, images):
        """Performs the classification directly, assuming that the input
        images are already of the right form.

        Input:
            images: a numpy array of size (num x 2048 x 1), dtype
                float32, c_contiguous, and has the mean subtracted and the
                image flipped if necessary.
        Output:
            scores: a numpy array of size (num x 360) containing the
                predicted scores for the 360 classes.
        """
        return self._net.predict(data=images)['probs_cudanet_out']

    @staticmethod
    def oversample(image, center_only=False):
        """Oversamples an image. Currently the indices are hard coded to the
        source and the flipped one, a total of 2 images.

        Input:
            image: an image of size (2048 x 1) and has data type uint8.
            center_only: if True, only return the center image (source).
        Output:
            images: the output of size (2 x 2048 x 1)
        """
        if center_only:
            return np.ascontiguousarray(image[np.newaxis,0:INPUT_DIM,0:INPUT_DIM,1], dtype=np.float32)
        else:
            # TODO: 可以考虑增加多种变换版本 40*40 ，可以在外部重写，旋转版本、缩放版本
            images = np.empty((2, INPUT_DIM, INPUT_DIM, 1),
                              dtype=np.float32)
            images[0] = image
            # flipped version
            images[1] = images[0,::-1]
            return images
    
    def classify(self, image, center_only=False):
        """Classifies an input image.
        
        Input:
            image: an image of 1 channels and has data type uint8. Only the
                center region will be used for classification.
        Output:
            scores: a numpy vector of size 2 containing the
                predicted scores for the 2 classes.
        """
        # first, extract the 40*40 center.
        image = transform.scale_and_extract(transform.as_rgb(image), 40)
        # convert to [0,255] float32
        image = image.astype(np.float32) * 255.
        if _KITNET_FLIP:
            # Flip the image if necessary, maintaining the c_contiguous order
            image = image[::-1, :].copy()
        # subtract the mean
        image -= self._data_mean
        # oversample the images
        images = DecafNet.oversample(image, center_only)
        predictions = self.classify_direct(images)# 预测一副影像的多个变化版本，取均值
        return predictions.mean(0)

    def top_k_prediction(self, scores, k):
        """Returns the top k predictions as well as their names as strings.
        
        Input:
            scores: a numpy vector of size 2 containing the
                predicted scores for the 2 classes.
        Output:
            indices: the top k prediction indices.
            names: the top k prediction names.
        """
        indices = scores.argsort()
        return (scores[indices[:-(k+1):-1]],
                [self.label_names[i] for i in indices[:-(k+1):-1]])

    def feature(self, blob_name):
        """Returns the feature of a specific blob.
        Input:
            blob_name: the name of the blob requested.
        Output:
            array: the numpy array storing the feature.
        """
        # We will copy the feature matrix in case further calls overwrite
        # it.
        return self._net.feature(blob_name).copy()


if __name__ == '__main__':
    """A simple demo showing how to run decafnet."""
    from decaf.util import smalldata, visualize
    logging.getLogger().setLevel(logging.INFO)
    net = DecafNet()
    #car = smalldata.car()
    # print car.shape
    #car = car.reshape((INPUT_DIM,INPUT_DIM,1))
    #scores = net.classify(car)
    #print 'prediction:', net.top_k_prediction(scores, 1)
    # TODO: fix pydot error syntax error in line 2 near ','
    # TODO: next step: setup the deepviz to visualize
    # visualize.draw_net_to_file(net._net, "decafnet.png") #bug!
    # print 'Network structure written to decafnet.png'
