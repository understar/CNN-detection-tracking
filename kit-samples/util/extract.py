# -*- coding: cp936-*-
__author__ = 'shuaiyi'
"""采集带旋转的图像样本
"""
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np

'''控制样本的尺寸'''
SAMPLE_SIZE = 40

def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in radians). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    ''' Calculate Rotation Matrix '''
    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    ''' Calculate Translation Matrix '''
    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the transform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if (width > image_size[0]):
        width = image_size[0]

    if (height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

# todo: Code the generate samples
def get_sample(region_name, img, car, num):
    """
    Car is dict, num is Number of samples from one car;
    return samples dict;
    通过名字来存储样本的信息，包括角度信息。
    需要根据图片的分辨率，来确定相应的尺寸信息。
    """
    # todo 是否可以直接使用中心，取外包络矩形的中心。
    xc = int(car['xc'])
    yc = int(car['yc'])
    try:
        srcbox = img[yc - SAMPLE_SIZE:yc + SAMPLE_SIZE, xc - SAMPLE_SIZE:xc + SAMPLE_SIZE]
        samples = {}
        
        for i in range(num):
            angleidx = np.random.randint(0, 360)
            box_rotated = rotate_image(srcbox, angleidx)  #*math.pi/180)
            # figure();imshow(box_rotated);plt.title(angleidx)
            sample = crop_around_center(box_rotated, SAMPLE_SIZE, SAMPLE_SIZE)#crop
            o_sample = car['o'] - angleidx if (car['o'] - angleidx)>=0 else car['o'] - angleidx + 360
            samples['%s_%s_%s_%s'%(region_name, car['id'], int(o_sample), i)] = sample
        
        return samples
    except:
        return {}


def gen_neg_samples(region_name, frame_id, img, cars, num):
    """
    Extract neg sample from img;
    neg sample doesn't contain car.
    """
    samples = {}
    for i in range(num):
        while True:
            xc = np.random.randint(img.shape[1])
            yc = np.random.randint(img.shape[0])
            if not is_contain(xc, yc, cars):
                break
        try:
            srcbox = img[yc - SAMPLE_SIZE:yc + SAMPLE_SIZE, 
                         xc - SAMPLE_SIZE:xc + SAMPLE_SIZE]
            angleidx = np.random.randint(0, 360)
            box_rotated = rotate_image(srcbox, angleidx)  #*math.pi/180)
            # figure();imshow(box_rotated);plt.title(angleidx)
            sample = crop_around_center(box_rotated, SAMPLE_SIZE, SAMPLE_SIZE)#crop
            samples['%s_%s_%s_%s'%(region_name, frame_id, i, 999)] = sample
        except:
            continue
    
    return samples

def is_contain(xc, yc, cars):
    dists = [ math.sqrt((xc-car['xc'])**2+(yc-car['yc'])**2) for car in cars]
    return False if min(dists)>SAMPLE_SIZE else True

if __name__ == '__main__':
    img = cv2.imread('E://2013//samples-for-cuda-convnet//training//StuttgartCrossroad01//MOS55.png', 1)
    car = {'xc': 247.9, 'yc': 437.65, 'o': 304.862, 'w': 9.1, 'h': 23.5, 'id': u'1'}
    samples = get_sample('StuttgartCrossroad01', img, car, 5)
    
    for k,v in samples.items():
        plt.subplot(1,5,int(k.split('_')[3])+1)
        plt.imshow(v)
        plt.title(k.split('_')[2])
    
    print 'done!'
