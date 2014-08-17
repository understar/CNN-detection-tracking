__author__ = 'shuaiyi'

'''make cuda-convnet batches from images in the input dir'''

import os
import shutil
import sys
import numpy as np
import cPickle as pickle
from PIL import Image
from PIL import ImageOps

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
    
    
def write_meta(path, data_mean, label_names, num_vis, num_cases_per_batch):
    """
    num_vis: number of the dim of case
    """
    out_file = open(path, 'wb')
    # dic = {'label_names':['others','car']}
    dic = {'data_mean':data_mean, 'label_names':label_names, 'num_vis':num_vis, 'num_cases_per_batch':num_cases_per_batch}
    pickle.dump(dic, out_file)
    out_file.close()

if __name__ == '__main__':
    if os.path.exists('batchs'):
        shutil.rmtree('batchs')
        os.mkdir('batchs')   
    else:
        os.mkdir('batchs')
  
    input_dir_neg = 'samples/neg'
    input_dir_pos = 'samples/pos'
    output_dir = 'batchs'

    batch_counter = 1
    batch_size = 5000

    print "reading file names..."
    pos_names = [d for d in os.listdir(input_dir_pos) if d.endswith('.png')]
    neg_names = [d for d in os.listdir(input_dir_neg) if d.endswith('.png')]
    np.random.shuffle(pos_names)
    np.random.shuffle(neg_names)

#    if batch_counter > 7:
#        omit_batches = batch_counter - 7
#        omit_images = omit_batches * batch_size
#        names = names[omit_images:]
#        print "omiting {} images".format(omit_images)

    current_batch = get_empty_batch()
    counter = 0
    labels = []
    filenames = []
    batch_mean = np.zeros((current_batch.shape))

    for pos_img, neg_img in zip(pos_names,neg_names):
        
        image_pos = Image.open(os.path.join(input_dir_pos,pos_img))
        # split result is str...
        image_label = 1
        labels.append(image_label)
        filenames.append(pos_img)
        try:
            image_pos = process(image_pos)
        except ValueError:
            print "problem with image {}".format(pos_img)
            sys.exit(1)

        image_pos = image_pos.reshape(-1, 1)
        current_batch = np.hstack(( current_batch, image_pos ))
        
        image_neg = Image.open(os.path.join(input_dir_neg,neg_img))
        # split result is str...
        image_label = 0
        labels.append(image_label)
        filenames.append(neg_img)
        try:
            image_neg = process(image_neg)
        except ValueError:
            print "problem with image {}".format(neg_img)
            sys.exit(1)

        image_neg = image_neg.reshape(-1, 1)
        current_batch = np.hstack(( current_batch, image_neg ))

        if current_batch.shape[1] == batch_size:
            batch_label = 'This is batch %s' % batch_counter
            batch_path = get_batch_path(output_dir, batch_counter)
            write_batch(batch_path, current_batch, labels, filenames, batch_label)

            batch_counter += 1
            current_batch = get_empty_batch()
            labels = []
            filenames = []

        counter += 2
        if counter % 1000 == 0:
            print pos_img
            
    for batch in os.listdir(output_dir):  # batchs 
        batch_data = pickle.load(open(os.path.join(output_dir,batch),'rb'))['data']
        mean = batch_data.mean(1).reshape((-1,1))
        batch_mean = np.hstack(( batch_mean, mean ))
    data_mean = batch_mean.mean(1).reshape((-1,1))
    label_names = ['others','car']
    num_vis = data_mean.shape[0]
    write_meta(os.path.join(output_dir, 'batches.meta'), data_mean, label_names, num_vis, batch_size)
