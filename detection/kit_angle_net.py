#coding:cp936
"""
implements a wrapper over the kitnet classifier trained by Zhangshuaiyi 
using the cuda convnet code.
目前支持Layertype：
conv、fc、cmrnorm、neuron、pool、softmax

"""
import cPickle as pickle
from decaf.util import translator, transform
from skimage.io import imread
import logging
import numpy as np
import os,sys

#sys.path.append(os.path.dirname(__file__))

# angle training model
_KITNET_FILE = '400.906'#'kitanglenet.epoch350'#'kitanglenet.41'
_META_FILE = '400.906.meta'

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
            scores: a numpy array of size (num x 360) "E:/2013/cuda-convnet/trunk"containing the
                predicted scores for the 360 classes.
        """
        return self._net.predict(data=images)['probs_cudanet_out'] #'fc1_cudanet_out'
    
    def classify(self, images):
        """Classifies an input image.
        
        Input:
            image: an image of 1 channels and has data type uint8. Only the
                center region will be used for classification.
        Output:
            scores: a numpy vector of size 2 containing the
                predicted scores for the 2 classes.
        """
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
    
    if len(sys.argv) == 1:
        car = smalldata.car()
    else:
        print "Using " + sys.argv[1]
        car = imread(sys.argv[1])
    
    Decaf = False
    if Decaf:
        from kitnet import DecafNet as KitNet
        kit_net = KitNet()
    
        # print car.shape
        car = car.reshape((40,40,1))
        scores = kit_net.classify(car)
        print 'Is car ? prediction:', kit_net.top_k_prediction(scores, 1)
        
        car_conv3 = kit_net.feature("conv3_neuron_cudanet_out") #conv3_cudanet_out
        mid_convs = car_conv3.reshape((car_conv3.shape[0],-1))
    else:
        os.chdir("E:/2013/cuda-convnet/trunk")
        # sys.path.append("E:/2013/cuda-convnet/trunk")
        from show_pred import model as car_model
        scores = car_model.show_predictions(car)
        print 'Is car ? prediction:', scores[-1]
        
        mid_convs = car_model.get_features(car)
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # visualize.draw_net_to_file(net._net, "decafnet.png") #bug!
    # print 'Network structure written to decafnet.png'
    net = DecafNet()
    scores = net.classify(mid_convs)
    print 'Direction ? prediction:',  net.top_k_prediction(scores, 5)#scores*180 
    