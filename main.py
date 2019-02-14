import gzip
# Read or generate p2h, a dictionary of image name to image id (picture to hash)
import pickle
import platform
import random
# Suppress annoying stderr output when importing keras.
import sys
from lap import lapjv
from math import sqrt
# Determine the size of each image
from os.path import isfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as pil_image
from scipy.misc import imread, imsave, imresize
from pandas import read_csv
from scipy.ndimage import affine_transform
from tqdm import tqdm
import time
import os
import os.path
import cv2
# from pretrainedmodels.models import bninception, resnet18
import torch as torch
from torch import optim
import torch.nn.functional as F
import h5py
from timeit import default_timer as timer
import argparse
from os import listdir
from os.path import isfile, join

from setting import mode
from models import SiameseNetVer2, SiameseNet, SubBlock

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Rotate, IAAAffine,
    IAASuperpixels, RGBShift, ChannelShuffle, RandomGamma, ToGray, InvertImg, ElasticTransform
)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--name', default='', type=str, help='')
parser.add_argument('--gpu', default='0', type=str, help='')
parser.add_argument('--epochs', default=1000, type=int, help='')
parser.add_argument('--batch', default=50, type=int, help='')
parser.add_argument('--img_size', default=384, type=int, help='')
parser.add_argument('--channel', default=3, type=int, help='')
parser.add_argument('--path_train', default='../DATASET/humpback_whale/size768/train/', type=str, help='')
parser.add_argument('--path_train_mask', default='../DATASET/humpback_whale/size768/train_mask/', type=str, help='')
parser.add_argument('--path_test', default='../DATASET/humpback_whale/size768/test/', type=str, help='')
parser.add_argument('--path_test_mask', default='../DATASET/humpback_whale/size768/test_mask/', type=str, help='')
parser.add_argument('--checkpoint', default='', type=str, help='')
parser.add_argument('--mode', default='classic', type=str, help='')
parser.add_argument('--learn', default=1, type=int, help='')
parser.add_argument('--random_score', default=0, type=int, help='')
parser.add_argument('--ampl', default=1000.0, type=float, help='')
parser.add_argument('--norm_zero_one', default=False, type=bool, help='')
parser.add_argument('--model_mode', default='train', type=str, help='')


args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

DATA = '../DATA/humpback_whale_siamese_torch/'
TRAIN_DF = '../DATASET/humpback_whale/all/train.csv'
SUB_Df = '../DATASET/humpback_whale/all/sample_submission.csv'
P2H = DATA + 'metadata/p2h.pickle'
P2SIZE = DATA + 'metadata/p2size.pickle'
BB_DF = DATA + 'metadata/bounding_boxes.csv'

TRAIN = args.path_train
TRAIN_MASK = args.path_train_mask
TEST = args.path_test
TEST_MASK = args.path_test_mask
batch = args.batch
num_epochs = args.epochs
channel = args.channel
img_size = args.img_size
mode = args.mode

epsilon = 1e-10
img_base = 384
features_size = 512 if img_size == 384 else 2048
# img_shape = (384, 384, channel)  # The image shape used by the model
img_shape = (img_size, img_size, channel)
anisotropy = 2.15  # The horizontal compression ratio
crop_margin = 0.05  # The margin added around the bounding box to compensate for bounding box inaccuracy

train_dict = dict([(image_name, widx) for _, image_name, widx in read_csv(TRAIN_DF).to_records()])
submit = [image_name for _, image_name, _ in read_csv(SUB_Df).to_records()]
all_image_names = list(train_dict.keys()) + submit
p2bb = pd.read_csv(BB_DF).set_index("Image")


assert img_size == 384 or img_size == 768 or img_size == 1024
assert channel == 1 or channel == 3


def expand_path(image_name):
    if isfile(TRAIN + image_name):
        return TRAIN + image_name
    if isfile(TRAIN_MASK + image_name):
        return TRAIN_MASK + image_name
    if isfile(TEST + image_name):
        return TEST + image_name
    if isfile(TEST_MASK + image_name):
        return TEST_MASK + image_name
    return image_name


def get_image_sizes():
    sizes = {}
    for p in tqdm(all_image_names):
        size = pil_image.open(expand_path(p)).size
        sizes[p] = size
    return sizes


p2size = get_image_sizes()


def strong_aug(p=0.9):
    return Compose([
        OneOf([
            IAAAdditiveGaussianNoise(scale=(0.01 * 255, 0.05 * 255), p=1.0),
            GaussNoise(var_limit=(20, 120), p=1.0),
            RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=0.9),
        RandomBrightnessContrast(p=1.0),
        OneOf([
            # MotionBlur(p=1.0),
            # MedianBlur(blur_limit=3, p=1.0),
            Blur(blur_limit=5, p=1.0),
            IAASharpen(p=1.0),
            # IAAEmboss(p=1.0),
            # IAASuperpixels(n_segments=10, p_replace=0.05, p=1.0),
        ], p=0.9),
        OneOf([
            CLAHE(clip_limit=8, p=1.0),
            RGBShift(p=1.0),
            ChannelShuffle(p=1.0),
            HueSaturationValue(p=1.0),
            # ToGray(p=1.0),
        ], p=0.9),
        # OneOf([
        #     OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, p=1.0),
        #     # GridDistortion(border_mode=cv2.BORDER_CONSTANT, p=1.0),
        #     IAAPiecewiseAffine(nb_rows=4, nb_cols=4, p=1.0),
        #     IAAPerspective(scale=(0.05, 0.075), p=1.0),
        #     # IAAAffine(mode='constant', p=1.0),
        #     ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=alpha_affine,
        #                      border_mode=cv2.BORDER_CONSTANT,
        #                      p=1.0),
        # ], p=0.9),
    ], p=p)


def elastic_transform():
    alpha = 1
    sigma = 1
    alpha_affine = 35
    return ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=alpha_affine, p=1.0, border_mode=cv2.BORDER_CONSTANT)


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t)/60
        hr = t // 60
        min_ = t % 60
        return '%2d hr %02d min' % (hr, min_)
    elif mode == 'sec':
        t = int(t)
        min_ = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min_, sec)
    else:
        raise NotImplementedError


def make_crop_samples():
    n = 0
    for image_name in tqdm(train_dict, total=len(train_dict)):
        n += 1
        img = read_cropped_image(image_name, False)
        imsave(f'../DATASET/humpback_whale/crop_samples/{image_name}', img)
        for i in range(3):
            img = read_cropped_image(image_name, True)
            imsave(f'../DATASET/humpback_whale/crop_samples/{image_name}'.replace('.jpg', f'-aug{i}.jpg'), img)
        if n >= 10:
            break


def print_model_arch():
    net = SiameseNet(channel, args.img_size)
    print(net)
    print('\n')
    param_dict = {}
    for name, param in net.named_parameters():
        name_layer = name.replace('.weight', '').replace('.bias', '')
        nn = 1
        for s in list(param.size()):
            nn = nn * s
        if name_layer in param_dict:
            param_dict[name_layer] += nn
        else:
            param_dict[name_layer] = nn

    for k in param_dict:
        print(f'{k} : {param_dict[k]}')


def print_model_arch2():
    def conv_layer(layer):
        return str(type(layer)).find('Conv2d') > 0

    def maxpool_layer(layer):
        return str(type(layer)).find('MaxPool2d') > 0

    def conv_output_shape(layer, h, w):
        new_h = h
        new_w = w
        if conv_layer(layer):
            new_h = round((w - layer.kernel_size[0] + layer.padding[0] * 2) / layer.stride[0] + 1)
            new_w = round((w - layer.kernel_size[1] + layer.padding[1] * 2) / layer.stride[1] + 1)
        return (None, new_h, new_w, layer.out_channels)

    def maxpool_output_shape(layer, h, w):
        if type(layer.stride) == int:
            new_h = int(h / layer.stride)
            new_w = int(w / layer.stride)
        elif len(layer.stride) == 2:
            new_h = int(h / layer.stride[0])
            new_w = int(w / layer.stride[1])
        return (None, new_h, new_w, 0)

    def print_layer(n, layer, name, h, w):
        shape = ''

        pre = ''
        for _ in range(n):
            pre += '   '

        if conv_layer(layer):
            shape = conv_output_shape(layer, h, w)
            b, h, w, c = shape
        elif maxpool_layer(layer):
            shape = maxpool_output_shape(layer, h, w)
            b, h, w, c = shape

        name_ = name.replace('-', '.')
        print(f'{pre}{name_} : {shape}')

        return h, w

    def print_module(n, layer1, h, w):
        for layer_name2 in layer1._modules:
            layer2 = layer1._modules[layer_name2]
            h, w = print_layer(n, layer2, layer_name2, h, w)
            if len(layer2._modules) > 0:
                h, w = print_module(n+1, layer2, h, w)
        return h, w

    net = SiameseNet(channel, args.img_size)
    print_module(0, net, 384, 384)


def set_weights_from_keras(model):
    pyt_state_dict = model.state_dict()

    for key in pyt_state_dict.keys():
        layer_name = key.replace('-', '.')
        path = f'./pytorch_weights/{layer_name}.npy'
        if (isfile(path)):
            shape_before = pyt_state_dict[key].shape
            weight = torch.tensor(np.load(path))
            if shape_before == weight.shape:
                pyt_state_dict[key] = weight
                # shape_after = pyt_state_dict[key].shape
                # print(f'{shape_before} : {shape_after}')
        # else:
        #     print(f'file not found: {layer_name}')

    model.load_state_dict(pyt_state_dict)
    # torch.save(model.state_dict(), './checkpoints/pytorch/baseline-ep60-gray.model')


def get_prepared_data():
    h2ws_ = {}
    new_whale = 'new_whale'
    for image_name, w in train_dict.items():
        if w != new_whale:  # Use only identified whales
            if image_name not in h2ws_:
                h2ws_[image_name] = []
            if image_name not in h2ws_[image_name]:
                h2ws_[image_name].append(w)
    for image_name, ws in h2ws_.items():
        if len(ws) > 1:
            h2ws_[image_name] = sorted(ws)

    # For each whale, find the unambiguous images ids.
    w2hs_ = {}
    for image_name, ws in h2ws_.items():
        if len(ws) == 1:  # Use only unambiguous pictures
            w = ws[0]
            if w not in w2hs_:
                w2hs_[w] = []
            if image_name not in w2hs_[w]:
                w2hs_[w].append(image_name)
    for w, hs in w2hs_.items():
        if len(hs) > 1:
            w2hs_[w] = sorted(hs)

    train_ = []  # A list of training image ids
    for hs in w2hs_.values():
        if len(hs) > 1:
            train_ += hs
    random.shuffle(train_)
    train_set_ = set(train_)

    w2ts_ = {}  # Associate the image ids from train to each whale id.
    for w, hs in w2hs_.items():
        for image_name in hs:
            if image_name in train_set_:
                if w not in w2ts_:
                    w2ts_[w] = []
                if image_name not in w2ts_[w]:
                    w2ts_[w].append(image_name)
    for w, ts in w2ts_.items():
        w2ts_[w] = np.array(ts)

    t2i_ = {}  # The position in train of each training image id
    for i, t in enumerate(train_):
        t2i_[t] = i

    return h2ws_, w2hs_, train_, w2ts_, t2i_, train_set_


def encode(img, mask):
    for w in range(mask.shape[1]):
        if mask[:, w].max() > 0:
            start = -1
            end = -1
            for i in range(mask.shape[0]):
                if mask[i, w] > 0:
                    start = i
                    break
            for i in range(1, mask.shape[0]):
                if mask[mask.shape[0]-i, w] > 0:
                    end = mask.shape[0]-i
                    break
            if start != end:
                # img[:, w, :] = imresize(img[start:end, w+1, :], (img.shape[0], 1, 3))
                img[:, w] = imresize(img[start:end, w:w+1], (img.shape[0], 1))[:, 0]
    return img


def code_vertical(img, mask):
    for h in range(mask.shape[0]):
        if mask[h, :].max() > 0:
            start = -1
            end = -1
            for i in range(mask.shape[1]):
                if mask[h, i] > 0:
                    start = i
                    break
            for i in range(1, mask.shape[1]):
                if mask[h, mask.shape[1]-i] > 0:
                    end = mask.shape[1]-i
                    break
            if start != end:
                # img[h, :, :] = imresize(img[h:h+1, start:end, :], (1, img.shape[1], 3))
                img[h, :] = imresize(img[h:h + 1, start:end], (1, img.shape[1]))[0]
    return img


def size_normalization(img, msk):
    start_h = 0
    end_h = img.shape[0]
    for h in range(1, img.shape[0]):
        if img[h, :].max() > 0 and start_h == 0:
            start_h = h - 1
        if img[img.shape[0]-h-1, :].max() > 0 and end_h == img.shape[0]:
            end_h = img.shape[0]-h

    start_w = 0
    end_w = img.shape[1]
    for w in range(1, img.shape[1]):
        if img[:, w].max() > 0 and start_w == 0:
            start_w = w - 1
        if img[:, img.shape[1]-w-1].max() > 0 and end_w == img.shape[1]:
            end_w = img.shape[1]-w

    img = imresize(img[start_h:end_h, start_w:end_w], img.shape)
    msk = imresize(msk[start_h:end_h, start_w:end_w], msk.shape)
    return img, msk


def read_raw_image(p):
    if channel == 1:
        # img = pil_image.open(expand_path(p))
        img = imread(expand_path(p), mode='L')
    else:
        img = imread(expand_path(p), mode='RGB')
    return img


def read_cropped_image(p, augment, image=None, norm_zero_one=False):
    """
    @param p : the name of the picture to read
    @param augment: True/False if data augmentation should be performed
    @param image:
    @return a numpy array with the transformed image
    """
    size_x, size_y = p2size[p]

    # Determine the region of the original image we want to capture based on the bounding box.
    row = p2bb.loc[p]
    x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']
    dx = x1 - x0
    dy = y1 - y0
    x0 -= dx * crop_margin
    x1 += dx * crop_margin + 1
    y0 -= dy * crop_margin
    y1 += dy * crop_margin + 1
    if x0 < 0:
        x0 = 0
    if x1 > size_x:
        x1 = size_x
    if y0 < 0:
        y0 = 0
    if y1 > size_y:
        y1 = size_y
    dx = x1 - x0
    dy = y1 - y0
    if dx > dy * anisotropy:
        dy = 0.5 * (dx / anisotropy - dy)
        y0 -= dy
        y1 += dy
    else:
        dx = 0.5 * (dy * anisotropy - dx)
        x0 -= dx
        x1 += dx

    # Generate the transformation matrix
    trans = np.array([[1, 0, -0.5 * img_shape[0]], [0, 1, -0.5 * img_shape[1]], [0, 0, 1]])
    trans = np.dot(np.array([[(y1 - y0) / img_shape[0], 0, 0], [0, (x1 - x0) / img_shape[1], 0], [0, 0, 1]]), trans)
    if augment:
        trans = np.dot(build_transform(
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(0.8, 1.0),
            random.uniform(0.8, 1.0),
            random.uniform(-0.05 * (y1 - y0), 0.05 * (y1 - y0)),
            random.uniform(-0.05 * (x1 - x0), 0.05 * (x1 - x0))
        ), trans)
    trans = np.dot(np.array([[1, 0, 0.5 * (y1 + y0)], [0, 1, 0.5 * (x1 + x0)], [0, 0, 1]]), trans)

    # Read the image, transform to black and white and comvert to numpy array
    if image is None:
        img = read_raw_image(p)
    else:
        img = image
    if channel == 3:
        img = img.astype(np.float32)
        assert len(img.shape) == 3 and img.shape[2] == 3

    # Apply affine transformation
    if channel == 1:
        matrix = trans[:2, :2]
        offset = trans[:2, 2]
        output_shape = img_shape[:-1]
    else:
        matrix = np.zeros((3, 3))
        matrix[2, 2] = 1
        matrix[:2, :2] = trans[:2, :2]
        offset = np.zeros(3)
        offset[:2] = trans[:2, 2]
        output_shape = img_shape

    msk = None
    if mode == 'none':
        nothing_to_do = True
    elif mode == 'classic':
        img = affine_transform(img, matrix, offset, output_shape=output_shape, order=1, mode='constant',
                               cval=np.average(img))
        data = {"image": img}
        augmented = strong_aug()(**data)
        img = augmented['image']
    else:
        msk = imread(TRAIN_MASK + p.replace('jpg', 'png'))

        # img_msk = np.stack((img, msk), axis=2)
        # data = {'image': img_msk}
        # aug_data = elastic_transform()(**data)
        # # img, msk = aug_data['image'], aug_data['mask']
        # img_msk = aug_data['image']
        # img, msk = img_msk[:, :, 0], img_msk[:, :, 1]

        # img_msk = np.stack((img, msk, msk), axis=2)
        # img_msk = affine_transform(img_msk, matrix, offset, output_shape=img_msk.shape, order=1, mode='constant')
        # img, msk = img_msk[:, :, 0], img_msk[:, :, 1]

        # img = img - img.min()
        # img = (img / img.max() * 255).astype(np.uint8)
        img[~msk.astype(np.bool)] = 0

        img, msk = size_normalization(img, msk)

        if mode == 'size_normalization':
            ok = True
        elif mode == 'horizontal_encoding':
            img = encode(img.copy(), msk)
        elif mode == 'vertical_encoding':
            img = code_vertical(img.copy(), msk)

    show_samples = False
    if show_samples:
        if msk is None:
            msk = img
        delm = np.ones((img.shape[0], 5), dtype=np.uint8) * 255
        all_img = np.concatenate((img, delm, msk), axis=1)
        imsave('../DATA/humpback_whale_siamese_torch/train_samples/' + p.replace('.jpg', f'-{random.randint(100, 999)}.jpg'), all_img)

    if norm_zero_one:
        # Normalize to [0, 1]
        for c in range(img.shape[2]):
            img[:, :, c] -= img[:, :, c].min()
            img[:, :, c] /= img[:, :, c].max()
    else:
        # Normalize to zero mean and unit variance
        if channel == 1:
            img = img.reshape(img_shape).astype(np.float32)
            img -= np.mean(img, keepdims=True)
            img /= np.std(img, keepdims=True) + epsilon
        else:
            for c in range(img.shape[2]):
                img[:, :, c] -= np.mean(img[:, :, c], keepdims=True)
                img[:, :, c] /= np.std(img[:, :, c], keepdims=True) + epsilon
    return img


def read_for_training_tmp(p, augmentation=False):
    """
    Read and preprocess an image with data augmentation (random transform).
    """
    read_mode = 'L' if channel == 1 else 'RGB'
    img = imread(expand_path(p), mode=read_mode)
    msk = None

    if mode == 'background':
        data = {'image': img}
    elif mode == 'instance':
        msk = imread(expand_path(p.replace('.jpg', '.png')))
        data = {'image': img, 'mask': msk}

    if augmentation and mode != 'code':
        data_aug = strong_aug()(**data)
        img = data_aug['image']
        if 'mask' in data_aug:
            msk = data_aug['mask']

    if mode == 'instance':
        img[~msk.astype(np.bool)] = 0
        img, msk = size_normalization(img, msk)

    # normilization
    if len(img.shape) == 2:
        img = img.reshape(img_shape).astype(np.float32)
        img -= np.mean(img, keepdims=True)
        img /= np.std(img, keepdims=True) + epsilon
    else:
        for c in range(img.shape[2]):
            img[:, :, c] -= np.mean(img[:, :, c], keepdims=True)
            img[:, :, c] /= np.std(img[:, :, c], keepdims=True) + epsilon

    if len(img.shape) == 2:
        img = img.reshape(*img.shape, 1)

    return img
    # return read_cropped_image(p, True)


def read_for_training(p, norm_zero_one=False):
    """
    Read and preprocess an image with data augmentation (random transform).
    """
    return read_cropped_image(p, True, norm_zero_one=norm_zero_one)


def read_for_validation(p, norm_zero_one=False):
    """
    Read and preprocess an image without data augmentation (use for testing).
    """
    return read_cropped_image(p, False, norm_zero_one=norm_zero_one)


def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    """
    Build a transformation matrix with the specified characteristics.
    """
    rotation = np.deg2rad(rotation)
    shear = np.deg2rad(shear)
    rotation_matrix = np.array(
        [[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, height_shift], [0, 1, width_shift], [0, 0, 1]])
    shear_matrix = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    zoom_matrix = np.array([[1.0 / height_zoom, 0, 0], [0, 1.0 / width_zoom, 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])
    return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))


class TrainingData(torch.utils.data.Dataset):
    def __init__(self, score_, w2ts_, train_, t2i_, steps=1000, batch_size=32, mode='standart', norm_zero_one=False):
        """
        @param score the cost matrix for the picture matching
        @param steps the number of epoch we are planning with this score matrix
        """
        super(TrainingData, self).__init__()
        self.score = -score_  # Maximizing the score is the same as minimuzing -score.
        self.steps = steps
        self.batch_size = batch_size
        self.w2ts = w2ts_
        self.train = train_
        self.t2i = t2i_
        self.mode = mode
        self.norm_zero_one = norm_zero_one
        for ts in self.w2ts.values():
            idxs = [self.t2i[t] for t in ts]
            for i in idxs:
                for j in idxs:
                    self.score[
                        i, j] = 10000.0  # Set a large value for matching whales -- eliminates this potential pairing
        self.on_epoch_end()

    def __getitem__(self, index):
        # tic = time.time()
        start = self.batch_size * index
        end = min(start + self.batch_size, len(self.match) + len(self.unmatch))
        size = end - start
        assert size > 0
        a = np.zeros((size,) + img_shape, dtype=np.float32)
        b = np.zeros((size,) + img_shape, dtype=np.float32)
        c = np.zeros((size, 1), dtype=np.float32)
        j = start // 2
        for i in range(0, size, 2):
            a[i, :, :, :] = read_for_training(self.match[j][0], self.norm_zero_one)
            b[i, :, :, :] = read_for_training(self.match[j][1], self.norm_zero_one)
            c[i, 0] = 1  # This is a match
            a[i + 1, :, :, :] = read_for_training(self.unmatch[j][0], self.norm_zero_one)
            b[i + 1, :, :, :] = read_for_training(self.unmatch[j][1], self.norm_zero_one)
            c[i + 1, 0] = 0  # Different whales
            j += 1
        a = np.rollaxis(a, 3, 1)
        b = np.rollaxis(b, 3, 1)
        # toc = time.time()
        # print(f'{toc - tic}')
        return [a, b], c

    def on_epoch_end(self):
        if self.steps <= 0:
            return  # Skip this on the last epoch.
        self.steps -= 1
        self.match = []
        self.unmatch = []
        start = timer()
        _, _, x = lapjv(self.score)  # Solve the linear assignment problem
        print('Solve the linear assignment problem: {}'.format(time_to_str((timer() - start), 'sec')))
        y = np.arange(len(x), dtype=np.int32)

        # Compute a derangement for matching whales
        for ts in self.w2ts.values():
            d = ts.copy()
            while True:
                random.shuffle(d)
                if not np.any(ts == d):
                    break
            for ab in zip(ts, d):
                self.match.append(ab)

        # Construct unmatched whale pairs from the LAP solution.
        for i, j in zip(x, y):
            if i == j:
                print(self.score)
                print(x)
                print(y)
                print(i, j)
            assert i != j
            self.unmatch.append((self.train[i], self.train[j]))

        # Force a different choice for an eventual next epoch.
        self.score[x, y] = 10000.0
        self.score[y, x] = 10000.0
        random.shuffle(self.match)
        random.shuffle(self.unmatch)
        # print(len(self.match), len(train), len(self.unmatch), len(train))
        assert len(self.match) == len(self.train) and len(self.unmatch) == len(self.train)

    def __len__(self):
        return (len(self.match) + len(self.unmatch) + self.batch_size - 1) // self.batch_size


class FeatureGen(torch.utils.data.Dataset):
    # A Keras generator to evaluate only the BRANCH MODEL
    def __init__(self, data, batch_size=64, verbose=1, norm_zero_one=False):
        super(FeatureGen, self).__init__()
        self.data = data
        self.batch_size = batch_size
        self.verbose = verbose
        self.norm_zero_one = norm_zero_one
        if self.verbose > 0:
            self.progress = tqdm(total=len(self), desc='Features')

    def __getitem__(self, index):
        start = self.batch_size * index
        size = min(len(self.data) - start, self.batch_size)
        a = np.zeros((size,) + img_shape, dtype=np.float32)
        for i in range(size):
            a[i, :, :, :] = read_for_validation(self.data[start + i], self.norm_zero_one)
        if self.verbose > 0:
            self.progress.update()
            if self.progress.n >= len(self):
                self.progress.close()
        a = np.rollaxis(a, 3, 1)
        return a

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size


class ScoreGen(torch.utils.data.Dataset):
    def __init__(self, x, y=None, batch_size=2048, verbose=1):
        super(ScoreGen, self).__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.verbose = verbose
        if y is None:
            self.y = self.x
            self.ix, self.iy = np.triu_indices(x.shape[0], 1)
        else:
            self.iy, self.ix = np.indices((y.shape[0], x.shape[0]))
            self.ix = self.ix.reshape((self.ix.size,))
            self.iy = self.iy.reshape((self.iy.size,))
        self.subbatch = (len(self.x) + self.batch_size - 1) // self.batch_size
        if self.verbose > 0:
            self.progress = tqdm(total=len(self), desc='Scores')

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.ix))
        a = self.y[self.iy[start:end], :]
        b = self.x[self.ix[start:end], :]
        if self.verbose > 0:
            self.progress.update()
            if self.progress.n >= len(self):
                self.progress.close()
        return [a, b]

    def __len__(self):
        return (len(self.ix) + self.batch_size - 1) // self.batch_size


def score_reshape(score, x, y=None):
    """
    Tranformed the packed matrix 'score' into a square matrix.
    @param score the packed matrix
    @param x the first image feature tensor
    @param y the second image feature tensor if different from x
    @result the square matrix
    """
    if y is None:
        # When y is None, score is a packed upper triangular matrix.
        # Unpack, and transpose to form the symmetrical lower triangular matrix.
        m = np.zeros((x.shape[0], x.shape[0]), dtype=np.float32)
        m[np.triu_indices(x.shape[0], 1)] = score.squeeze()
        m += m.transpose()
    else:
        m = np.zeros((y.shape[0], x.shape[0]), dtype=np.float32)
        iy, ix = np.indices((y.shape[0], x.shape[0]))
        ix = ix.reshape((ix.size,))
        iy = iy.reshape((iy.size,))
        m[iy, ix] = score.squeeze()
    return m


def compute_score(model, device, train_ids, batch_size):
    """
    Compute the score matrix by scoring every pictures from the training set against every other picture O(n^2).
    """
    feature_loader_train = torch.utils.data.DataLoader(FeatureGen(train_ids, batch_size, 0), num_workers=6)
    features = np.zeros((len(train_ids), features_size))

    model.eval()
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(feature_loader_train), total=len(feature_loader_train), desc='Features'):
            data = data[0].to(device)

            output = model(data, mode='branch')
            start = batch_idx * batch_size
            end = start + batch_size
            features[start:end] = output.cpu().data.numpy()

        batch_size = 1024 * 4
        score_gen = torch.utils.data.DataLoader(ScoreGen(features, batch_size=batch_size, verbose=0), num_workers=6)
        score = np.zeros((len(score_gen.dataset.ix), 1))
        for batch_idx, data in tqdm(enumerate(score_gen), total=len(score_gen), desc='Score   '):
            for i in range(len(data)):
                data[i] = data[i][0].type(torch.FloatTensor).to(device)

            output = model(data, mode='head')
            start = batch_idx * batch_size
            end = min(start + batch_size, len(score_gen.dataset.ix))
            score[start:end] = output.cpu().data.numpy()

        score = score_reshape(score, features)
        # np.save('../DATA/humpback_whale_siamese_torch/scores_valid/score_valid.npy', score)
        # np.save('./temp/feature.npy', features)

    # train_df = pd.DataFrame()
    # train_df['Image'] = train_ids
    # train_df.to_csv('../DATA/humpback_whale_siamese_torch/scores_valid/train_ids.csv')

    return score


def compute_features(model, device, ids, batch_size):
    feature_loader_train = torch.utils.data.DataLoader(FeatureGen(ids, batch_size, 0), num_workers=6)
    features = np.zeros((len(ids), features_size))

    model.eval()
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(feature_loader_train), total=len(feature_loader_train), desc='Features'):
            data = data[0].to(device)

            output = model(data, mode='branch')
            start = batch_idx * batch_size
            end = start + batch_size
            features[start:end] = output.cpu().data.numpy()

    return features


def compute_score_matrix(model, device, features, batch_size):
    if len(features) == 1:
        score_gen = torch.utils.data.DataLoader(ScoreGen(features[0], batch_size=batch_size, verbose=0), num_workers=6)
        score_mat = np.zeros((len(features[0]), 1))
    elif len(features) == 2:
        score_gen = torch.utils.data.DataLoader(ScoreGen(features[0], features[1], batch_size=batch_size, verbose=0), num_workers=6)
        score_mat = np.zeros((len(features[0]) * len(features[1]), 1))

    for batch_idx, data in tqdm(enumerate(score_gen), total=len(score_gen), desc='Score   '):
        for i in range(len(data)):
            data[i] = data[i][0].type(torch.FloatTensor).to(device)

        output = model(data, mode='head')
        start = batch_idx * batch_size
        end = min(start + batch_size, len(score_gen.dataset.ix))
        score_mat[start:end] = output.cpu().data.numpy()

    score_mat = score_reshape(score_mat, features[1], features[0])
    return score_mat


def prepare_submission(threshold, filename, score, known, h2ws):
    """
    Generate a Kaggle submission file.
    @param threshold the score given to 'new_whale'
    @param filename the submission file name
    @param score
    @param known
    @param h2ws
    """
    vtop = 0
    vhigh = 0
    pos = [0, 0, 0, 0, 0, 0]
    with open(filename, 'wt', newline='\n') as f:
        f.write('Image,Id\n')
        for i, p in enumerate(tqdm(submit)):
            t = []
            s = set()
            a = score[i, :]
            for j in list(reversed(np.argsort(a))):
                h = known[j]
                if a[j] < threshold and 'new_whale' not in s:
                    pos[len(t)] += 1
                    s.add('new_whale')
                    t.append('new_whale')
                    if len(t) == 5:
                        break
                for w in h2ws[h]:
                    assert w != 'new_whale'
                    if w not in s:
                        if a[j] > 1.0:
                            vtop += 1
                        elif a[j] >= threshold:
                            vhigh += 1
                        s.add(w)
                        t.append(w)
                        if len(t) == 5:
                            break
                if len(t) == 5:
                    break
            if 'new_whale' not in s:
                pos[5] += 1
            assert len(t) == 5 and len(s) == 5
            f.write(p + ',' + ' '.join(t[:5]) + '\n')
    return vtop, vhigh, pos


def test(model, device, test_loader):
    model.eval()

    with torch.no_grad():
        accurate_labels = 0
        all_labels = 0
        loss = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            for i in range(len(data)):
                data[i] = data[i].to(device)

            output_positive = model(data[:2])
            output_negative = model(data[0:3:2])

            target = target.type(torch.LongTensor).to(device)
            target_positive = torch.squeeze(target[:, 0])
            target_negative = torch.squeeze(target[:, 1])

            loss_positive = F.cross_entropy(output_positive, target_positive)
            loss_negative = F.cross_entropy(output_negative, target_negative)

            loss = loss + loss_positive + loss_negative

            accurate_labels_positive = torch.sum(torch.argmax(output_positive, dim=1) == target_positive).cpu()
            accurate_labels_negative = torch.sum(torch.argmax(output_negative, dim=1) == target_negative).cpu()

            accurate_labels = accurate_labels + accurate_labels_positive + accurate_labels_negative
            all_labels = all_labels + len(target_positive) + len(target_negative)

        accuracy = 100. * accurate_labels / all_labels
        print('Test accuracy: {}/{} ({:.3f}%)\tLoss: {:.6f}'.format(accurate_labels, all_labels, accuracy, loss))


def train(model, device, train_loader, epoch, optimizer, start):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        for i in range(len(data)):
            data[i] = data[i].to(device)

        optimizer.zero_grad()

        output = model([data[0][0], data[1][0]])
        target_ = target[0]
        target_ = target_.type(torch.FloatTensor)
        target_ = torch.squeeze(target_[:, 0])
        target_ = target_.to(device)

        output_ = torch.squeeze(output[:, 0])

        loss = F.binary_cross_entropy(output_, target_)
        # loss = F.cross_entropy(output, target_)

        loss.backward()

        optimizer.step()
        if batch_idx % 10 == 0:
            print('\r', end='', flush=True)
            message = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t time: {}'.format(
                epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), loss.item(), time_to_str((timer() - start), 'min'))
            print(message, end='', flush=True)

    print('\n', end='', flush=True)


### PREDICTION ###


def predict(model, device, batch_size, file_out, score_file='', norm_zero_one=False):
    # Find elements from training sets not 'new_whale'
    tic = time.time()
    h2ws = {}
    for image_name, w in train_dict.items():
        if w != 'new_whale':  # Use only identified whales
            if image_name not in h2ws:
                h2ws[image_name] = []
            if w not in h2ws[image_name]:
                h2ws[image_name].append(w)
    known = sorted(list(h2ws.keys()))

    # Dictionary of picture indices
    h2i = {}
    for i, h in enumerate(known):
        h2i[h] = i

    feature_loader_known = torch.utils.data.DataLoader(FeatureGen(known, batch_size, 0, norm_zero_one=norm_zero_one), num_workers=10)
    feature_loader_submit = torch.utils.data.DataLoader(FeatureGen(submit, batch_size, 0, norm_zero_one=norm_zero_one), num_workers=10)

    feature_known = np.zeros((len(known), features_size))
    feature_submit = np.zeros((len(submit), features_size))
    score = np.zeros((len(known) * len(submit), 1))

    model.eval()
    with torch.no_grad():

        for batch_idx, data in tqdm(enumerate(feature_loader_known), total=len(feature_loader_known), desc='Features known'):
            data = data[0].to(device)

            output = model(data, mode='branch')
            start = batch_idx * batch_size
            end = start + batch_size
            feature_known[start:end] = output.cpu().data.numpy()

        for batch_idx, data in tqdm(enumerate(feature_loader_submit), total=len(feature_loader_submit), desc='Features submit'):
            data = data[0].to(device)

            output = model(data, mode='branch')
            start = batch_idx * batch_size
            end = start + batch_size
            feature_submit[start:end] = output.cpu().data.numpy()

        batch_size = 1024 * 4
        score_gen = torch.utils.data.DataLoader(ScoreGen(feature_known, feature_submit, batch_size=batch_size, verbose=0), num_workers=6)
        for batch_idx, data in tqdm(enumerate(score_gen), total=len(score_gen), desc='Score'):
            for i in range(len(data)):
                data[i] = data[i][0].type(torch.FloatTensor).to(device)

            output = model(data, mode='head')
            start = batch_idx * batch_size
            end = start + batch_size
            score[start:end] = output.cpu().data.numpy()

    threshold = 0.99

    score = score_reshape(score, feature_known, feature_submit)
    if score_file != '':
        torch.save({'score_matrix': score, 'known': known, 'submit': submit, 'threshold': threshold}, score_file)

    # Generate the subsmission file.
    prepare_submission(threshold, file_out, score, known, h2ws)
    toc = time.time()
    print("Submission time: ", (toc - tic) / 60.)

    return score


def prediction():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    list_dirs = [args.name]

    for checkpoint_dir in list_dirs:
        mypath = f'../DATA/humpback_whale_siamese_torch/checkpoints/{checkpoint_dir}/'
        files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        files.sort()

        submit_dir = f'../DATA/humpback_whale_siamese_torch/submissions/{checkpoint_dir}/'
        score_dir = f'../DATA/humpback_whale_siamese_torch/scores/{checkpoint_dir}/'
        os.makedirs(submit_dir, exist_ok=True)
        os.makedirs(score_dir, exist_ok=True)

        for n, file in enumerate(files):
            sub_file = join(submit_dir, file.replace('.pt', '.csv'))
            score_file = join(score_dir, file)
            if isfile(score_file):
                continue
            print(file)

            checkpoint = torch.load(mypath + file)
            model = SiameseNet(checkpoint['channel'], checkpoint['features_size'])
            model.load_state_dict(checkpoint['state_dict'])
            model.to(device)
            predict(model, device, args.batch, sub_file, score_file, checkpoint['norm_zero_one'])


def make_submission_from_score(threshold=0.94):
    h2ws, w2hs, train_ids, w2ts, t2i, train_set = get_prepared_data()

    checkpoint = torch.load('../DATA/humpback_whale_siamese_torch/scores/all-files101.pt')
    score = checkpoint['score_matrix']
    known = checkpoint['known']
    # submit = checkpoint['submit']
    file_out = f'../DATA/humpback_whale_siamese_torch/submissions/all-files101-th({threshold}).scv'
    prepare_submission(threshold, file_out, score, known, h2ws)


### VALIDATION ###


def validation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    list_dirs = [args.name]

    for my_dir in list_dirs:
        my_path = f'../DATA/humpback_whale_siamese_torch/checkpoints/{my_dir}/'
        files = [f for f in listdir(my_path) if isfile(join(my_path, f))]
        files.sort()

        score_dir = f'../DATA/humpback_whale_siamese_torch/scores_valid/{my_dir}/'
        os.makedirs(score_dir, exist_ok=True)

        for n, file in enumerate(files):
            score_file = join(score_dir, file)
            if isfile(score_file):
                continue
            print(file)

            checkpoint = torch.load(my_path + file)
            model = SiameseNet(checkpoint['channel'], checkpoint['features_size'])
            model.load_state_dict(checkpoint['state_dict'])
            model.to(device)
            validate(model, device, args.batch, score_file, checkpoint['norm_zero_one'])


def validate(model, device, batch_size, file_out, norm_zero_one):
    train_df = pd.read_csv(TRAIN_DF)

    whale2image = {}
    for image, wid in zip(train_df['Image'], train_df['Id']):
        if wid not in whale2image:
            whale2image[wid] = []
        whale2image[wid].append(image)

    known = []
    for wid in whale2image:
        if wid != 'new_whale' and len(whale2image[wid]) > 1:
            known += whale2image[wid]

    unknown = known + whale2image['new_whale']

    known.sort()
    unknown.sort()

    feature_loader_known = torch.utils.data.DataLoader(
        FeatureGen(known, batch_size, 0, norm_zero_one=norm_zero_one),
        num_workers=10)
    feature_loader_unknown = torch.utils.data.DataLoader(
        FeatureGen(unknown, batch_size, 0, norm_zero_one=norm_zero_one),
        num_workers=10)

    feature_known = np.zeros((len(known), features_size))
    feature_unknown = np.zeros((len(unknown), features_size))
    score = np.zeros((len(known) * len(unknown), 1))

    model.eval()
    with torch.no_grad():

        for batch_idx, data in tqdm(enumerate(feature_loader_known), total=len(feature_loader_known),
                                    desc='Features known  '):
            data = data[0].to(device)

            output = model(data, mode='branch')
            start = batch_idx * batch_size
            end = start + batch_size
            feature_known[start:end] = output.cpu().data.numpy()

        for batch_idx, data in tqdm(enumerate(feature_loader_unknown), total=len(feature_loader_unknown),
                                    desc='Features unknown'):
            data = data[0].to(device)

            output = model(data, mode='branch')
            start = batch_idx * batch_size
            end = start + batch_size
            feature_unknown[start:end] = output.cpu().data.numpy()

        batch_size = 1024 * 4
        score_gen = torch.utils.data.DataLoader(
            ScoreGen(feature_known, feature_unknown, batch_size=batch_size, verbose=0), num_workers=6)
        for batch_idx, data in tqdm(enumerate(score_gen), total=len(score_gen), desc='Score'):
            for i in range(len(data)):
                data[i] = data[i][0].type(torch.FloatTensor).to(device)

            output = model(data, mode='head')
            start = batch_idx * batch_size
            end = start + batch_size
            score[start:end] = output.cpu().data.numpy()

    score = score_reshape(score, feature_known, feature_unknown)

    torch.save({'score_matrix': score, 'known': known, 'submit': unknown, 'threshold': 0}, file_out)


def validate_splits(model, device, batch_size, name):
    train_df = pd.read_csv(TRAIN_DF)

    whale2image = {}
    for image, idx in zip(train_df['Image'], train_df['Id']):
        if idx in whale2image:
            whale2image[idx].append(image)
        else:
            whale2image[idx] = []
            whale2image[idx].append(image)

    splits = [[], []]
    new_whales = whale2image['new_whale']
    total_sum = 0
    only_sum = 0
    total = 0
    for n, idx in enumerate(whale2image):
        if idx == 'new_whale':
            continue
        lengh = len(whale2image[idx])
        total += lengh
        if lengh > 1:
            total_sum += lengh
            random.shuffle(whale2image[idx])
            sep_idx = int(lengh / 2)
            for i in range(lengh):
                if i < sep_idx:
                    splits[n % 2].append(whale2image[idx][i])
                else:
                    splits[(n + 1) % 2].append(whale2image[idx][i])
        else:
            only_sum += lengh

    for s in range(len(splits)):
        print(f'validation split: {s}')
        known = splits[s % 2]
        predc = splits[(s + 1) % 2] + new_whales

        random.shuffle(known)
        random.shuffle(predc)

        feature_loader_known = torch.utils.data.DataLoader(FeatureGen(known, batch_size, 0), num_workers=10)
        feature_loader_predc = torch.utils.data.DataLoader(FeatureGen(predc, batch_size, 0), num_workers=10)

        feature_known = np.zeros((len(known), features_size))
        feature_predc = np.zeros((len(predc), features_size))
        score = np.zeros((len(known) * len(predc), 1))

        model.eval()
        with torch.no_grad():

            for batch_idx, data in tqdm(enumerate(feature_loader_known), total=len(feature_loader_known),
                                        desc='Features known'):
                data = data[0].to(device)

                output = model(data, mode='branch')
                start = batch_idx * batch_size
                end = start + batch_size
                feature_known[start:end] = output.cpu().data.numpy()

            for batch_idx, data in tqdm(enumerate(feature_loader_predc), total=len(feature_loader_predc),
                                        desc='Features submit'):
                data = data[0].to(device)

                output = model(data, mode='branch')
                start = batch_idx * batch_size
                end = start + batch_size
                feature_predc[start:end] = output.cpu().data.numpy()

            batch_size = 1024 * 48
            score_gen = torch.utils.data.DataLoader(
                ScoreGen(feature_known, feature_predc, batch_size=batch_size, verbose=0), num_workers=6)
            for batch_idx, data in tqdm(enumerate(score_gen), total=len(score_gen), desc='Score'):
                for i in range(len(data)):
                    data[i] = data[i][0].type(torch.FloatTensor).to(device)

                output = model(data, mode='head')
                start = batch_idx * batch_size
                end = start + batch_size
                score[start:end] = output.cpu().data.numpy()

        score = score_reshape(score, feature_known, feature_predc)

        torch.save({'score': score, 'known': known, 'predc': predc}, f'../DATA/humpback_whale_siamese_torch/scores_valid/{name}-split{s}.pt')


def validation_score(score_matrix, ids1, ids2):
    threshold = 1.0

    train_df = pd.read_csv(TRAIN_DF)
    image2whale = {}
    for image, whale in zip(train_df['Image'], train_df['Id']):
        image2whale[image] = whale

    score = 0
    for i, image_true in tqdm(enumerate(ids1), total=len(ids1), desc='Valid score'):
        s = set()
        predicted = []
        a = score_matrix[i, :]
        arg_sort = list(reversed(np.argsort(a)))
        for j in arg_sort:
            image = ids2[j]
            whale = image2whale[image]
            if a[j] < threshold and 'new_whale' not in s:
                s.add('new_whale')
                predicted.append('new_whale')
                if len(predicted) == 5:
                    break
            if whale not in s:
                s.add(whale)
                predicted.append(whale)
                if len(predicted) == 5:
                    break
            if len(predicted) == 5:
                break
        for n, t in enumerate(predicted):
            if image2whale[image_true] == t:
                score += 1 / (n + 1)

    return score / len(ids1)


def get_new_whales():
    train_df = pd.read_csv(TRAIN_DF)
    new_ws = []
    for image_name, ids in zip(train_df['Image'], train_df['Id']):
        if ids == 'new_whale':
            new_ws.append(image_name)
    return new_ws


def calculate_valid_score():
    h2ws, w2hs, train_ids, w2ts, t2i, train_set = get_prepared_data()
    ids2 = train_ids
    ids2.sort()
    ids1 = ids2 + get_new_whales()
    ids1.sort()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dir_in = '../DATA/humpback_whale_siamese_torch/checkpoints/exp384-ch1-t1-pt/'
    dir_out = '../DATA/humpback_whale_siamese_torch/scores_valid/exp384-ch1-t1-nw/'
    os.makedirs(dir_out, exist_ok=True)

    files = listdir(dir_in)
    files.sort()

    for n, file in enumerate(files):
        if n > 0:
            continue
        model = SiameseNet()
        state = torch.load(join(dir_in, file))
        model.load_state_dict(state['state_dict'])

        model.to(device)

        # feature1 = compute_features(model, device, ids1, 50)
        # feature2 = compute_features(model, device, ids2, 50)
        # score_mat = compute_score_matrix(model, device, [feature1, feature2], 1024 * 4)

        # valid_score = 0.0
        # np.save(join(dir_out, file.replace('.pt', f'-val_score{round(valid_score, 4)}.npy')), score_mat)
        score_mat = np.load('../DATA/humpback_whale_siamese_torch/scores_valid/exp384-ch1-t1-nw/exp384-ch1-t1-ep002-val_score0.0.npy')
        valid_score = validation_score(score_mat, ids1, ids2)

        print(f'{file} : {valid_score}')


###


def load_state_dict(src, dst, name='', save_path=''):
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    for key in src_state.keys():
        if key in dst_state and src_state[key].shape == dst_state[key].shape:
            dst_state[key] = src_state[key]
        elif key == 'conv2d_1.weight':
            for i in range(src_state[key].shape[0]):
                for j in range(dst_state[key].shape[1]):
                    dst_state[key][i, j, :, :] = src_state[key][i, 0, :, :]
        else:
            print(key)

    for key in src_state.keys():
        if key in dst_state:
            print(f'{key} : {np.mean(src_state[key].cpu().data.numpy()) == np.mean(dst_state[key].cpu().data.numpy())}')
        else:
            print(f'{key} : {False}')

    if save_path != '':
        dst.load_state_dict(dst_state)
        state = {"epoch": 0,
                 "model_name": name,
                 "state_dict": dst.state_dict(),
                 "optimizer": {}}
        torch.save(state, save_path)


def main():
    if False:
        mydir = '../DATA/humpback_whale_siamese_torch/checkpoints/exp768-ch1-t1/'
        mydir_out = '../DATA/humpback_whale_siamese_torch/checkpoints/exp768-ch1-t1-pt/'
        files = listdir(mydir)
        for file in files:
            print(file)
            model = torch.load(join(mydir, file))
            epoch = int(file[8:11])
            state = {"epoch": epoch,
                     "model_name": 'exp384-ch1-t2',
                     "state_dict": model.state_dict(),
                     "optimizer": {}}
            file_name = f'exp384-ch1-t2-ep{str(epoch).zfill(3)}.pt'
            torch.save(state, join(mydir_out, file_name))

    do_learn = args.learn
    save_frequency = 5

    checkpoint_dir = DATA + f'./checkpoints/{args.name}/'
    submission_dir = DATA + f'./submissions/{args.name}/'
    score_dir = DATA + f'./scores/{args.name}/'
    score_valid_dir = DATA + f'./scores_valid/{args.name}/'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)
    os.makedirs(score_dir, exist_ok=True)
    os.makedirs(score_valid_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer_state = None

    if args.checkpoint == '':
        model = SiameseNet(channel, features_size)
        start_epoch = 0
    else:
        # src = torch.load(args.checkpoint)
        # model = SiameseNetVer2(channel, args.img_size)
        # load_state_dict(src, model, name=args.name, save_path='../DATA/humpback_whale_siamese_torch/exp768-ch1-t2-init.tar')
        # model = SiameseNetVer2(channel, args.img_size)
        state = torch.load(args.checkpoint)
        model = SiameseNet(state['channel'], state['features_size'])
        model.load_state_dict(state['state_dict'])
        start_epoch = state['epoch'] + 1
        if len(state['optimizer']) != 0:
            optimizer_state = state['optimizer']

    model.to(device)

    weight_decay = 0.0001
    batch_size = batch
    start = timer()

    if do_learn:  # training mode
        optimizer = optim.Adam(model.parameters(), lr=64e-5, weight_decay=weight_decay)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

        h2ws, w2hs, train_ids, w2ts, t2i, train_set = get_prepared_data()

        first = True

        for epoch in range(start_epoch, num_epochs):
            if first or (epoch - 1) % save_frequency == 0:
                first = False
                if args.random_score:
                    score_mat = np.random.random_sample(size=(len(train_ids), len(train_ids)))
                    # score_mat = np.load('./temp/score.npy')
                else:
                    score_mat = compute_score(model, device, train_ids, batch_size)
                    valid_score = validation_score(score_mat, train_ids, train_ids)
                    print(f'Validation score on {epoch} epoch: {valid_score}')
                    score_valid_file = score_valid_dir + '{}-ep{:03}.pt'.format(args.name, epoch)
                    torch.save({'score_matrix': score_mat,
                            'epoch': epoch,
                            "model_name": args.name,
                            'model_state': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'train_ids': train_ids,
                            'valid_score': valid_score}, score_valid_file)

                score_mat = score_mat + args.ampl * np.random.random_sample(size=score_mat.shape)
                train_loader = torch.utils.data.DataLoader(
                    TrainingData(score_mat, w2ts, train_ids, t2i, steps=100, batch_size=batch_size,
                                 norm_zero_one=args.norm_zero_one), num_workers=6)

            train(model, device, train_loader, epoch, optimizer, start)
            # test(model, device, test_loader)
            
            if epoch % save_frequency == 0:
                state_file = checkpoint_dir + '{}-ep{:03}.pt'.format(args.name, epoch)
                submt_file = submission_dir + '{}-ep{:03}.csv'.format(args.name, epoch)
                score_file = score_dir + '{}-ep{:03}.pt'.format(args.name, epoch)

                state = {"epoch": epoch,
                         "model_name": args.name,
                         "state_dict": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         'channel': channel,
                         'features_size': features_size,
                         'image_size': img_size,
                         'norm_zero_one': args.norm_zero_one}
                torch.save(state, state_file)
                # predict(model, device, batch_size, submt_file, score_file)

    else:  # prediction
        print('Start predicting...')
        predict(model, device, batch_size, submission_dir + '{}-ep{:03}.csv'.format(args.name, num_epochs))
        # print('Start validation...')
        # checkpoint_split = args.checkpoint.split('/')
        # valid_name = checkpoint_split[len(checkpoint_split) - 1].replace('.model', '')
        # validate(model, device, batch_size, valid_name)


if __name__ == "__main__":
    if args.model_mode == 'train':
        print('Stat training')
        main()
    elif args.model_mode == 'validate':
        print('Stat validation')
        validation()
    elif args.model_mode == 'predict':
        print('Stat prediction')
        prediction()

    # make_submission_from_score(0.93)
    # make_submission_from_score(0.92)
    # make_submission_from_score(0.91)
    # make_submission_from_score(0.90)
