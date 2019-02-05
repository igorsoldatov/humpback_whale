# from main import read_cropped_image, read_raw_image
import pandas as pd
import numpy as np
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from scipy.misc import imread, imsave, imresize
import time
import matplotlib.pyplot as plt
from skimage.morphology import label
from shutil import copyfile
from scipy import ndimage


def bounding_boxes_shape():
    BB_DF = './metadata/bounding_boxes.csv'
    p2bb = pd.read_csv(BB_DF).set_index("Image")
    for n, row in tqdm(p2bb.iterrows(), total=len(p2bb)):
        h = row['y1'] - row['y0']
        w = row['x1'] - row['x0']
        print(f'{h} : {w}')


def get_border_mask(mask):
    h, w = mask.shape
    mask = mask.astype(np.bool).astype(np.uint8)
    mask[:h-1, :] = mask[:h-1, :] + mask[1:, :]
    mask[:, :w-1] = mask[:, :w-1] + mask[:, 1:]

    border = np.zeros(mask.shape, dtype=np.uint8)
    border[mask == 1] = 255
    return border


def code_horizontal_rgb(img, mask):
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
                img[:, w, :] = imresize(img[start:end, w+1, :], (img.shape[0], 1, 3))
    return img


def code_vertical_rgb(img, mask):
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
                img[h, :, :] = imresize(img[h:h+1, start:end, :], (1, img.shape[1], 3))
    return img


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


def remove_small_blobs(img, threshold=3000):
    blobs = label(img)
    if blobs.max() > 1:
        # print(blobs.max())
        for i in range(1, blobs.max() + 1):
            blob_sum = blobs[blobs == i].sum() / i
            if blob_sum <= threshold:
                img[blobs == i] = 0
            else:
                print(f'{blob_sum}')
    return img


def show_masks():
    path_masks = f'../DATASET/humpback_whale/masks/'
    path_image = f'../DATASET/humpback_whale/all/train/'
    files = [f for f in listdir(path_masks) if isfile(join(path_masks, f))]
    bord = None
    for name in files:
        mask = imread(path_masks + name, mode='L')
        mask = remove_small_blobs(mask)
        imag = imread(path_image + name.replace('_mask.png', '.jpg'), mode='RGB')
        imag[mask.astype(np.bool) == False] = 0
        # c = code_vertical(imag, mask)
        # mask = get_border_mask(mask)
        if bord is None:
            bord = plt.imshow(mask)
        else:
            bord.set_data(mask)
        plt.pause(1.)
        plt.draw()


def resize_dataset():
    path_in = f'../DATASET/humpback_whale/all/test/'
    path_out = f'../DATASET/humpback_whale/size1024/test/'
    files = [f for f in listdir(path_in) if isfile(join(path_in, f))]
    bord = None
    for file in tqdm(files, total=len(files)):
        image = imread(path_in + file, mode='RGB')
        image = read_cropped_image(file, False, image)
        image = image.astype(np.uint8)
        imsave(path_out + file, image)
        # if bord is None:
        #     bord = plt.imshow(image)
        # else:
        #     bord.set_data(image)
        # plt.pause(1.)
        # plt.draw()


def remove_small_blobs_mask():
    path_in = f'../DATASET/humpback_whale/size768/train_mask/'
    files = [f for f in listdir(path_in) if isfile(join(path_in, f))]
    for file in tqdm(files, total=len(files)):
        image = imread(path_in + file)
        # image = remove_small_blobs(image)
        image = ndimage.binary_fill_holes(image.astype(np.bool))
        image = image.astype(np.uint8) * 255
        imsave(path_in + file, image)


def make_encode_images():
    path_image = f'../DATASET/humpback_whale/size768/train/'
    path_mask = f'../DATASET/humpback_whale/size768/train_mask/'
    path_out = f'../DATASET/humpback_whale/size768/train_code/'
    files = [f for f in listdir(path_image) if isfile(join(path_image, f))]
    files.sort(reverse=False)
    # n = 0
    for i in range(int(len(files) / 2), len(files)):
        file = files[i]
        if not isfile(path_out + file):
            image = imread(path_image + file, mode='RGB')
            mask = imread(path_mask + file.replace('.jpg', '.png'))

            image[~mask.astype(np.bool)] = 0
            image, mask = size_normalization(image, mask)

            code = encode(image, mask)
            imsave(path_out + file, code)

            # n += 1
            # if n >= 10:
            #     break


def make_size_norm_images():
    path_image = f'../DATASET/humpback_whale/size384/train/'
    path_mask = f'../DATASET/humpback_whale/size384/train_mask/'
    path_out = f'../DATASET/humpback_whale/size384/train_size_norm/'
    files = [f for f in listdir(path_image) if isfile(join(path_image, f))]
    for file in tqdm(files, total=len(files)):
        image = imread(path_image + file, mode='L')
        mask = imread(path_mask + file.replace('.jpg', '.png'))

        image[~mask.astype(np.bool)] = 0
        image, mask = size_normalization(image, mask)

        # code = code_horizontal(image, mask)
        imsave(path_out + file, image)


def sample_unique_whales():
    idxs = {}
    train_labels = pd.read_csv('../DATASET/humpback_whale/all/train.csv')
    for name, idx in tqdm(zip(train_labels['Image'], train_labels['Id']), total=len(train_labels)):
        if idx in idxs:
            idxs[idx].append(name)
        else:
            idxs[idx] = []
    for idx in idxs:
        if len(idxs[idx]) == 1:
            name = idxs[idx][0]
            src = '../DATASET/humpback_whale/all/train/'
            dst = '../DATASET/humpback_whale/train_unique/'
            copyfile(src + name, dst + name)


make_encode_images()
