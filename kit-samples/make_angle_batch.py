__author__ = 'shuaiyi'

'''make cuda-convnet batches from images in the input dir'''

import os
import sys
import numpy as np
import cPickle as pickle
from PIL import Image

_size = 40
channels = 1

def process(image):
    image = np.array(image)           # size x size x 3
    if channels > 1:
        image = np.rollaxis(image, 2)     # 3 x size x size
        image = image.reshape(-1)         # ...
    elif channels == 1:
        image = image.reshape(-1)
    return image


def get_batch_path(output_dir, number):
    filename = "data_batch_{}".format(number)
    return os.path.join(output_dir, filename)


def get_empty_batch():
    return np.zeros(( _size*_size*channels, 0 ), dtype=np.uint8)


def write_batch(path, batch, labels, filenames, batch_label):
    print "writing {}...\n".format(path)
    # labels = [0 for x in range(batch.shape[1])]  # Labels is 1 when it is car else is 0
    d = {'labels': labels, 'data': batch, 'filenames':filenames, 'batch_label':batch_label}
    pickle.dump(d, open(path, "wb"))

if __name__ == '__main__':
    if not os.path.exists('batchs'):
        os.mkdir('batchs')
  
    input_dir_pos = 'samples/pos'
    output_dir = 'batchs'

    batch_counter = 905
    batch_size = 5000

    print "reading file names..."
    pos_names = [d for d in os.listdir(input_dir_pos) if d.endswith('.png')]
    np.random.shuffle(pos_names)

    current_batch = get_empty_batch()
    counter = 0
    labels = []
    filenames = []
    batch_mean = np.zeros((current_batch.shape))

    for pos_img in pos_names:
        
        image_pos = Image.open(os.path.join(input_dir_pos,pos_img))
        # split result is str...
        image_label = int(pos_img.split('_')[2])
        labels.append(image_label)
        filenames.append(pos_img)
        try:
            image_pos = process(image_pos)
        except ValueError:
            print "problem with image {}".format(pos_img)
            sys.exit(1)

        image_pos = image_pos.reshape(-1, 1)
        current_batch = np.hstack(( current_batch, image_pos ))

        if current_batch.shape[1] == batch_size:
            batch_label = 'This is batch %s' % batch_counter
            batch_path = get_batch_path(output_dir, batch_counter)
            write_batch(batch_path, current_batch, labels, filenames, batch_label)

            batch_counter += 1
            current_batch = get_empty_batch()
            labels = []
            filenames = []

        counter += 1
        if counter % 1000 == 0:
            print pos_img