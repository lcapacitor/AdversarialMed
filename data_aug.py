import os
import cv2
import random
import argparse
import numpy as np

def img_rotate_counter_clk(img, range=10.0):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    scale = 1.0
    angle = random.randrange(-range, 0)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    img_ro = cv2.warpAffine(img, M, (h, w))
    return img_ro

def img_rotate_clk(img, range=10.0):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    scale = 1.0
    angle = random.randrange(0, range)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    img_ro = cv2.warpAffine(img, M, (h, w))
    return img_ro

def img_hflip(img):
    img_hf = img[:, ::-1]
    return img_hf

def img_vflip(img):
    img_vf = img[::-1, :]
    return img_vf

def img_shift(img, max_amt=0.05):
    from scipy import ndimage
    new_img = np.copy(img)
    shape = new_img.shape
    max_x = int(shape[0] * max_amt)
    max_y = int(shape[1] * max_amt)
    x = np.random.randint(low=-max_x, high=max_x)
    y = np.random.randint(low=-max_y, high=max_y)
    return ndimage.interpolation.shift(new_img,shift=[x,y,0])

def imgAug(img_dir, out_img_num):

    if not os.path.exists(img_dir):
        print("Error: the given directory {} doesn't exist, please check".format(img_dir))
        return 0

    filenames = os.listdir(img_dir)
    random.shuffle(filenames)

    print ('Data augmentation under {}...'.format(img_dir))

    while len(os.listdir(img_dir)) < out_img_num:

        for filename in filenames:
            if len(os.listdir(img_dir)) >= out_img_num:
                break

            path = os.path.join(img_dir, filename)
            im = cv2.imread(path)

            f_i = random.randint(0, len(aug_funcs)-1)
            aug_func = aug_funcs[f_i]

            aug_im = aug_func(im)
            i_name = 'aug{}_'.format(f_i) + filename
            out_name = os.path.join(img_dir, i_name)
            cv2.imwrite(out_name, aug_im)


def main(args):
    img_dir = args.img_dir
    out_img_num = args.number
    imgAug(img_dir, out_img_num)


if __name__ == '__main__':
    aug_funcs = [img_rotate_counter_clk, img_rotate_clk, img_shift, img_vflip, img_hflip]
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True, help='path to the images to augment, e.g. img/0')
    parser.add_argument('--number', type=int, required=True, help='number of images required after augment')
    args = parser.parse_args()
    main(args)
