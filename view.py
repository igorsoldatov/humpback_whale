import cv2
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from scipy.misc import imread, imsave, imresize
from shutil import copyfile
from scipy import ndimage
from skimage.morphology import label
from albumentations import (
    Rotate,
    Compose,
    ElasticTransform,
    RandomBrightnessContrast,
)

TRAIN_DF = '../DATASET/humpback_whale/all/train.csv'
SUB_Df = '../DATASET/humpback_whale/all/sample_submission.csv'
TRAIN = '../DATASET/humpback_whale/size768/train/'
TRAIN_MASK = '../DATASET/humpback_whale/size768/train_mask/'
TEST = '../DATASET/humpback_whale/size768/test/'
TEST_MASK = '../DATASET/humpback_whale/size768/test_mask/'
mode = 'code'


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


def strong_aug(p=1.0):
    return Compose([
        RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        Rotate((-30, 30), p=1.0, border_mode=cv2.BORDER_CONSTANT)
        # ElasticTransform(alpha=600, sigma=25,  alpha_affine=0, border_mode=cv2.BORDER_CONSTANT, p=1.0)
    ], p=p)


def read_for_training(p, augmentation=False):
    """
    Read and preprocess an image with data augmentation (random transform).
    """
    img = imread(TRAIN + p, mode='RGB')
    msk = img

    if mode == 'background':
        data = {'image': img}
    elif mode == 'instance' or mode == 'code':
        msk = imread(TRAIN_MASK + p.replace('.jpg', '.png'))
        data = {'image': img, 'mask': msk}

    if augmentation:
        data_aug = strong_aug()(**data)
        img = data_aug['image']
        if 'mask' in data_aug:
            msk = data_aug['mask']

    if mode == 'instance' or mode == 'code':
        img[~msk.astype(np.bool)] = 0
        img, msk = size_normalization(img, msk)
        if mode == 'code':
            img = encode(img, msk)

    return img, msk


# draw -----------------------------------
def image_show(name, image, resize=1):
    H,W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))


def draw_shadow_text(img, text, pt,  fontScale, color, thickness, color1=None, thickness1=None):
    if color1 is None:
        color1 = (0, 0, 0)
    if thickness1 is None:
        thickness1 = thickness+2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pt, font, fontScale, color1, thickness1, cv2.LINE_AA)
    cv2.putText(img, text, pt, font, fontScale, color,  thickness,  cv2.LINE_AA)


def mask_overlay(image, mask, color=(0, 255, 0)):
    """
    Helper function to visualize mask on the top of the car
    """
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]
    return img


def remove_small_blobs(img, threshold=3000):
    blobs = label(img)
    if blobs.max() > 1:
        for i in range(1, blobs.max() + 1):
            blob_sum = blobs[blobs == i].sum() / i
            if blob_sum <= threshold:
                img[blobs == i] = 0
    return img


def explore_images():
    image_path = f'../DATASET/humpback_whale/size768/train/'
    mask_path = f'../DATASET/humpback_whale/size768/train_mask/'
    files = [f for f in listdir(image_path) if isfile(join(image_path, f))]

    n = 1

    while n <= len(files):
        file = files[n]
        img_name = image_path + file
        msk_name = mask_path + file.replace('.jpg', '.png')
        img = imread(img_name)
        msk = imread(msk_name)
        img_msk = mask_overlay(img, msk.astype(np.bool))

        blobs = label(msk)
        if blobs.max() > 1:
            # print(blobs.max())
            b = []
            for i in range(1, blobs.max() + 1):
                blob_sum = blobs[blobs == i].sum() / i
                b.append(blob_sum)

            b.sort()
            bs = ', '.join([str(i) for i in b])

            # img, msk = read_for_training(file, True)
            # if len(img.shape) != len(msk.shape):
            #     msk = np.stack((msk, msk, msk), axis=2)
            # img_msk = np.concatenate((img, msk), axis=1)

            draw_shadow_text(img_msk, f'{n}/{len(files)} : {file} : {bs}',  (5, 15), 0.5, (255, 255, 255), 1)

            cv2.imshow('image-mask', img_msk)

            key = cv2.waitKey(0)
            if key % 256 == 27:
                break
            elif key % 256 == 83:
                n += 1
            elif key % 256 == 81:
                n -= 1
            elif key % 256 == 32:
                copyfile(img_name, '../DATASET/humpback_whale/size768/train_error/image/' + file)
                copyfile(msk_name, '../DATASET/humpback_whale/size768/train_error/mask/' + file.replace('.jpg', '.png'))
            elif key % 256 == 102:
                msk = ndimage.binary_fill_holes(msk.astype(np.bool))
                img_msk = mask_overlay(img, msk)
                draw_shadow_text(img_msk, f'{n}/{len(files)} : {file}', (5, 15), 0.5, (255, 255, 255), 1)
                cv2.imshow('image-mask', img_msk)
                key = cv2.waitKey(0)
                n += 1
            elif key % 256 == 114:
                msk = remove_small_blobs(msk, b[0] + 1)
                imsave(msk_name, msk)
                img_msk = mask_overlay(img, msk.astype(np.bool))
                draw_shadow_text(img_msk, f'{n}/{len(files)} : {file}', (5, 15), 0.5, (255, 255, 255), 1)
                cv2.imshow('image-mask', img_msk)
                key = cv2.waitKey(0)

            # print(key % 256)
        else:
            n += 1

        # print(key % 256)


def view_and_copy():
    path_in = '../DATA/humpback_whale_siamese_torch/validation_errors/'
    srs = '../DATASET/humpback_whale/all/train/'
    dst = '../DATA/humpback_whale_siamese_torch/for_best_annotation'

    files = [f for f in listdir(path_in) if isfile(join(path_in, f))]
    n = 0
    while n < len(files):
        file = files[n]
        img_name = path_in + file
        img = imread(img_name)

        draw_shadow_text(img, f'{n}/{len(files)}',  (5, 15), 0.5, (255, 255, 255), 1)

        cv2.imshow('image', img)

        key = cv2.waitKey(0)
        if key % 256 == 27:
            break
        elif key % 256 == 83:
            n += 1
        elif key % 256 == 81:
            n -= 1
        elif key % 256 == 32:
            copyfile(join(srs, file), join(dst, file))


def explore_prediction():
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

    submit = [image_name for _, image_name, _ in pd.read_csv(SUB_Df).to_records()]
    train_dict = dict([(image_name, widx) for _, image_name, widx in pd.read_csv(TRAIN_DF).to_records()])

    h2ws = {}
    w2i ={}
    for image_name, w in train_dict.items():
        if w != 'new_whale':  # Use only identified whales
            if image_name not in h2ws:
                h2ws[image_name] = []
            if w not in h2ws[image_name]:
                h2ws[image_name].append(w)
            if w in w2i:
                w2i[w].append(image_name)
            else:
                w2i[w] = []
                w2i[w].append(image_name)
    known = sorted(list(h2ws.keys()))

    hand_result_df = pd.read_csv('./hand_result.csv')
    hand_result = hand_result_df['test'].tolist()
    target_df = pd.read_csv(
        '../DATA/humpback_whale_siamese_torch/submissions/ensembling/ensemble-size768+384-files84-first2.csv')
    target = [idx.split(' ') for idx in target_df['Id']]

    dir_name = 'exp768-ch1-t1'
    mypath = f'../DATA/humpback_whale_siamese_torch/scores/{dir_name}/'
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    score = np.load(join(mypath, files[0]))
    for i in range(1, len(files)):
        score_path = join(mypath, files[i])
        score += np.load(score_path)

    threshold = 0
    for th in range(1):
        # threshold -= 0.01
        vtop = 0
        vhigh = 0
        pos = [0, 0, 0, 0, 0, 0]
        result_whales = []
        result_images = []
        result_scores = []

        print(f'test : train : whale : count_examples : score')

        for i, p in enumerate(submit):
            if p in hand_result:
                continue
            t = []
            images = []
            scores = []
            s = set()
            a = score[i, :]
            args = list(reversed(np.argsort(a)))
            for j in args:
                image_name = known[j]
                if a[j] < threshold and 'new_whale' not in s:
                    pos[len(t)] += 1
                    s.add('new_whale')
                    t.append('new_whale')
                    if len(t) == 5:
                        break
                for w in h2ws[image_name]:
                    assert w != 'new_whale'
                    if w not in s:
                        if a[j] > 1.0:
                            vtop += 1
                        elif a[j] >= threshold:
                            vhigh += 1
                        s.add(w)
                        t.append(w)
                        images.append(image_name)
                        scores.append(a[j])
                        if len(t) == 5:
                            break
                if len(t) == 5:
                    break
            if 'new_whale' not in s:
                pos[5] += 1
            result_whales.append(t)
            result_images.append(images)
            result_scores.append(scores)

            test_img = imread(expand_path(p))
            draw_shadow_text(test_img, f'{i}/{len(submit)}', (5, 15), 0.5, (255, 255, 255), 1)
            draw_shadow_text(test_img, f'name: {p}', (5, 35), 0.5, (255, 255, 255), 1)
            n = 0
            while True:
                image_name = images[n]
                count_examples = len(w2i[t[n]])
                whale_score = scores[n] / len(files)
                img = imread(expand_path(image_name))
                draw_shadow_text(img, f'{n + 1}', (5, 15), 0.5, (255, 255, 255), 1)
                draw_shadow_text(img, f'name: {image_name}', (5, 35), 0.5, (255, 255, 255), 1)
                draw_shadow_text(img, f'whale: {t[n]} / {count_examples}', (5, 55), 0.5, (255, 255, 255), 1)
                draw_shadow_text(img, f'score: {whale_score}', (5, 75), 0.5, (255, 255, 255), 1)

                plot_img = np.concatenate((test_img, img), axis=1)
                cv2.imshow('prediction', plot_img)
                key = cv2.waitKey(0) % 256
                print(key)
                if key == 83:
                    n += 1 if n < len(images) - 1 else -4
                    continue
                elif key == 81:
                    n -= 1 if n > 0 else -4
                    continue
                elif key == 82:
                    break
                elif key == 84:
                    print(f'{p} : {image_name} : {t[n]} : {count_examples} : {whale_score}')
                elif key == 110:
                    print(f'{p} : {image_name} : new_whale : {count_examples} : {whale_score}')


def validation_score_0(score_matrix, train_ids):
    train_df = pd.read_csv(TRAIN_DF)
    image2whale = {}
    for image, whale in zip(train_df['Image'], train_df['Id']):
        image2whale[image] = whale

    score = 0
    for i, idx in tqdm(enumerate(train_ids), total=len(train_ids), desc='Valid score'):
        a = score_matrix[i, :]
        args = list(reversed(np.argsort(a)))
        for n in range(5):
            if image2whale[idx] == image2whale[train_ids[args[n]]]:
                score += 1 / (n + 1)
                break

    return score


def validation_score(score_matrix, ids1, ids2):
    DIR_TRAIN = '../DATASET/humpback_whale/size768/train/'
    DIR_ERROR = '../DATA/humpback_whale_siamese_torch/validation_errors/'
    threshold = 0.90

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

    print('\n\n')
    for c in range(100):
        if stat_count[c] > 0:
            print(f'{c} : {stat_count[c]}')

    total_score /= len(ids1)

    print(mismatch)
    print(len(ids1))

    return total_score, nws, errors


def validation():
    score_matrix = np.load('../DATA/humpback_whale_siamese_torch/scores_valid/exp384-ch1-t1-pt/exp384-ch1-t1-ep176-val_score0.9016.npy')
    train_ids = pd.read_csv('../DATA/humpback_whale_siamese_torch/scores_valid/train_ids.csv')['Image'].tolist()
    train_ids.sort()

    score, nws, err = validation_score(score_matrix, train_ids, train_ids)

    print(f'Validation score: {score}, {nws}, {err}')


def test_something():
    train_df = pd.read_csv(TRAIN_DF)
    w2i = {}
    for image_name, ids in zip(train_df['Image'], train_df['Id']):
        if ids not in w2i:
            w2i[ids] = []
        w2i[ids].append(image_name)

    count2 = 0
    count = []
    for i in range(11):
        count.append(0)

    for w in w2i:
        if w == 'new_whale':
            continue
        for i in range(1, 11):
            if len(w2i[w]) > i:
                count[i] += len(w2i[w])
        if len(w2i[w]) > 2:
            count2 += len(w2i[w])

    print(f'Whale with examples more than: {count}')
    print(f'Whale with examples more than: {count2}')


def work_with_bb():
    TRAIN_PATH = '/home/igor/kaggle/DATASET/humpback_whale/all/train/'
    old_bb = pd.read_csv('/home/igor/kaggle/DATA/humpback_whale_siamese_torch/metadata/bounding_boxes.csv')
    new_bb = pd.read_csv('/home/igor/kaggle/DATA/humpback_whale_siamese_torch/metadata/bb_valid_err.csv')
    bb = new_bb['filename'].tolist()
    list_x0 = []
    list_x1 = []
    list_y0 = []
    list_y1 = []
    for n, row in old_bb.iterrows():
        if row['Image'] in bb:
            new_row = new_bb[new_bb['filename'] == row['Image']]
            row_data = json.loads(new_row['region_shape_attributes'].values[0])
            x0 = row_data['x']
            x1 = x0 + row_data['width']
            y0 = row_data['y']
            y1 = y0 + row_data['height']
            row['x0'] = x0
            row['x1'] = x1
            row['y0'] = y0
            row['y1'] = y1
            # img = imread(join(TRAIN_PATH, row['Image']), mode='RGB')
            # cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            # cv2.rectangle(img, (row['x0'], row['y0']), (row['x1'], row['y1']), (0, 0, 255), 2)
            # cv2.imshow('image', img)
            # key = cv2.waitKey(0)
            list_x0.append(x0)
            list_x1.append(x1)
            list_y0.append(y0)
            list_y1.append(y1)
        else:
            list_x0.append(row['x0'])
            list_x1.append(row['x1'])
            list_y0.append(row['y0'])
            list_y1.append(row['y1'])
    old_bb['x0'] = list_x0
    old_bb['x1'] = list_x1
    old_bb['y0'] = list_y0
    old_bb['y1'] = list_y1
    old_bb.to_csv('/home/igor/kaggle/DATA/humpback_whale_siamese_torch/metadata/bounding_boxes_new.csv', header=True, index=False)


validation()
