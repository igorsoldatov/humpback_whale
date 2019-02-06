from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import torch as torch
import numpy as np
import pandas as pd


TRAIN_DF = '../DATASET/humpback_whale/all/train.csv'


def validation_score(score_matrix, ids1, ids2):
    DIR_TRAIN = '../DATASET/humpback_whale/size768/train/'
    DIR_ERROR = '../DATA/humpback_whale_siamese_torch/validation_errors/'
    threshold = 0.50

    train_df = pd.read_csv(TRAIN_DF)
    image2whale = {}
    whale2images = {}
    for image, whale in zip(train_df['Image'], train_df['Id']):
        image2whale[image] = whale
        if whale not in whale2images:
            whale2images[whale] = []
        whale2images[whale].append(image)

    nws = 0  # new whales
    errors = 0
    total_score = 0
    stat_count = []
    for c in range(100):
        stat_count.append(0)
    mismatch = []
    for m in range(6):
        mismatch.append(0)

    for i, image_true in enumerate(ids1):
        s = set()
        predicted = []
        a = score_matrix[i, :]
        arg_sort = list(reversed(np.argsort(a)))
        for j in arg_sort:
            image = ids2[j]
            whale = image2whale[image]
            if a[j] < threshold and 'new_whale' not in s:
                if len(predicted) == 0:
                    nws += 1
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

        score = 0
        m = 0
        for n, t in enumerate(predicted):
            if image2whale[image_true] == t:
                score += 1 / (n + 1)
                m = n + 1
                break

        mismatch[m] += 1

        total_score += score
        if score == 0:
            errors += 1
            # copyfile(join(DIR_TRAIN, image_true), join(DIR_ERROR, image_true))
            # whale_true = image2whale[image_true]
            # whale_len = len(whale2images[whale_true])
            # stat_count[whale_len] += 1
            # print(f'{image_true} : {whale_true} : {whale_len}')
            # if whale_len == 3:
            #     ws = whale2images[whale_true]
            #     img_plt = imread(join(DIR_TRAIN, image_true), mode='RGB')
            #     # img1 = imread(join(DIR_TRAIN, ws[0]), mode='RGB')
            #     # img2 = imread(join(DIR_TRAIN, ws[1]), mode='RGB')
            #     # img_plt = np.concatenate((img1, img2), axis=1)
            #     cv2.imshow('image', img_plt)
            #     cv2.waitKey(0)

    for c in range(100):
        if stat_count[c] > 0:
            print(f'{c} : {stat_count[c]}')

    total_score /= len(ids1)

    return total_score, mismatch


def validation():
    score_dir = '../DATA/humpback_whale_siamese_torch/scores_valid/exp768-ch3-t1/'
    files = listdir(score_dir)
    files.sort()
    for file in files:
        score_dict = torch.load(join(score_dir, file))
        score, mismatch = validation_score(score_dict['score_matrix'], score_dict['train_ids'], score_dict['train_ids'])
        print(f'File: {file}, Mismatch: {mismatch}, Score: {score}')


validation()