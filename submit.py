import pandas as pd
import numpy as np
import os
import glob
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from itertools import groupby
from operator import itemgetter
from scipy.misc import imread, imsave, imresize
from shutil import copyfile
from scipy.misc import imread, imsave, imresize
# from main import SUB_Df, TRAIN_DF
from pylab import *


TRAIN_DF = '../DATASET/humpback_whale/all/train.csv'
SUB_Df = '../DATASET/humpback_whale/all/sample_submission.csv'


def known_whales():
    train_labels = pd.read_csv('../DATASET/humpback_whale/all/train.csv')
    unique_whales = set(train_labels['Id'].tolist())
    print(f'Unique whales in train: {len(unique_whales)}')


def error_analysis():
    size = 384
    submission = pd.read_csv('../DATASET/humpback_whale/all/sample_submission.csv')
    mypath = f'./submissions/torch-gray768+384/'
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    all = {}

    for file in tqdm(files, total=len(files)):
        sub = pd.read_csv(join(mypath, file))
        for name, idx in zip(sub['Image'], sub['Id']):
            if name in all:
                all[name].append(idx.split(' ')[0])
                all[name].append(idx.split(' ')[1])
            else:
                all[name] = []

    sum_pred_type = []
    for i in range(int(len(files) / 10) + 1):
        sum_pred_type.append(0)

    images = submission['Image'].tolist()
    err_type = []
    uniqe_idx = set()

    for name in images:
        all[name].sort()
        a = [{'name': key, 'count': len(list(group))} for key, group in groupby(all[name])]
        b = sorted(a, key=itemgetter('count'), reverse=True)
        pred_type = int(b[0]['count'] / 10)

        err_type.append(pred_type)
        sum_pred_type[pred_type] += 1
        for i in range(min(5, len(b))):
            pred_name = b[i]['name']
            uniqe_idx.add(pred_name)

        src = f'../DATASET/humpback_whale/all/test/'
        if pred_type <= 2:
            dst = f'../DATASET/humpback_whale/errores/{pred_type}/'
            os.makedirs(dst, exist_ok=True)
            copyfile(src + name, dst + name)

    print(sum_pred_type)
    print(f'Unique idx: {len(uniqe_idx)}')

    errores = pd.DataFrame()
    errores['Image'] = images
    errores['Class'] = err_type
    errores.to_csv(f'./analysis/analysis-gray768+384-files{len(files)}-first{2}.csv', header=True, index=False)


def show_result(images, bord):
    PATH = '../DATASET/humpback_whale/size384/'
    TRAIN = PATH + 'train/'
    TEST = PATH + 'test/'
    img_cat = []
    img_cat.append(imread(TEST + images[0]))
    for i in range(1, 5):
        img_cat.append(imread(TRAIN + images[i]))
    cat = np.concatenate(img_cat, axis=1)
    if bord is None:
        bord = plt.imshow(cat)
    else:
        bord.set_data(cat)
    plt.pause(20.)
    plt.draw()


def prepare_submission(threshold, filename, score, known, h2ws):
    """
    Generate a Kaggle submission file.
    @param threshold the score given to 'new_whale'
    @param filename the submission file name
    @param score
    @param known
    @param h2ws
    """
    submit = [image_name for _, image_name, _ in pd.read_csv(SUB_Df).to_records()]
    bord = None

    vtop = 0
    vhigh = 0
    pos = [0, 0, 0, 0, 0, 0]
    with open(filename, 'wt', newline='\n') as f:
        f.write('Image,Id\n')
        for i, p in enumerate(tqdm(submit)):
            t = []
            s = set()
            a = score[i, :]
            args = list(reversed(np.argsort(a)))

            # param = [p]
            # for n in range(5):
            #     param.append(known[args[n]])
            # show_result(param, bord)

            for j in args:
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


def make_prediction_from_score():
    train_dict = dict([(image_name, widx) for _, image_name, widx in pd.read_csv(TRAIN_DF).to_records()])
    h2ws = {}
    for image_name, w in train_dict.items():
        if w != 'new_whale':  # Use only identified whales
            if image_name not in h2ws:
                h2ws[image_name] = []
            if w not in h2ws[image_name]:
                h2ws[image_name].append(w)
    known = sorted(list(h2ws.keys()))

    mypath = f'./scores/horiz-code384-00/'
    file_out = './submission_tmp.scv'
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for file in files:
        print(file)
        score = np.load(join(mypath, file))
        prepare_submission(0.99, file_out, score, known, h2ws)
        break


def split_submissions():
    sub_file = './submissions/ensembling/ensemble-size768+384-files84-first2.csv'
    sub = pd.read_csv(sub_file)
    sub_to = pd.read_csv(sub_file)
    for n in range(5):
        predicted = []
        for idx in tqdm(sub['Id'], total=len(sub)):
            ids = idx.split(' ')[n]
            predicted.append(' '.join([ids for i in range(5)]))
        sub_to['Id'] = predicted
        sub_to.to_csv(sub_file.replace('.csv', f'-{n}.csv'), header=True, index=False)


def get_submission_score(base, submit):
    base_sub = pd.read_csv(base)
    score_sub = pd.read_csv(submit)
    score = 0
    for target, predict in zip(base_sub['Id'], score_sub['Id']):
        t = target.split(' ')[0]
        for n, p in enumerate(predict.split(' ')):
            if t == p:
                score += 1 / (n + 1)
                break

    return score / len(base_sub)


def scoreing_submissions():
    base = '../DATA/humpback_whale_siamese_torch/submissions/ensembling/ensemble-size768+384-files84-first2.csv'
    mypath = f'../DATA/humpback_whale_siamese_torch/submissions/exp768-ch1-t3/'
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for file in files:
        score = get_submission_score(base, mypath + file)
        print(f'{score} : {file}')


def make_ensemble():
    base = '../DATA/humpback_whale_siamese_torch/submissions/ensembling/ensemble-size768+384-files84-first2.csv'
    train_labels = pd.read_csv('../DATASET/humpback_whale/all/train.csv')
    unique_whales = set(train_labels['Id'].tolist())

    dir_name = 'exp768-ch1-t1'
    submission = pd.read_csv('../DATASET/humpback_whale/all/sample_submission.csv')
    mypath = f'../DATA/humpback_whale_siamese_torch/submissions/{dir_name}/'
    # mypath = f'../DATA/humpback_whale_siamese_torch/submissions/**/*.csv'
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    all = {}

    f_count = 0
    threshold = 0.60
    # for file in glob.iglob(mypath, recursive=True):
    for file in tqdm(files, total=len(files)):
        score = get_submission_score(base, join(mypath, file))
        print(f'{file} : {score}')
        # if score < threshold:
        #     continue
        f_count += 1
        sub = pd.read_csv(join(mypath, file))
        for name, idx in zip(sub['Image'], sub['Id']):
            if name in all:
                all[name].append(idx.split(' ')[0])
                all[name].append(idx.split(' ')[1])
            else:
                all[name] = []

    ensamble = {}
    predicted_n = []
    for n in range(5):
        predicted_n.append(set())

    for m, row in submission.iterrows():
        name = row['Image']
        all[name].sort()
        a = [{'name': key, 'count': len(list(group))} for key, group in groupby(all[name])]
        b = sorted(a, key=itemgetter('count'), reverse=True)
        ensamble[name] = b
        for n in range(min(5, len(b))):
            predicted_n[n].add(b[n]['name'])

    for name in ensamble:
        whale = ensamble[name][1]['name']
        count = ensamble[name][0]['count']
        if whale not in predicted_n[0] and count < 2:
            tmp = ensamble[name][0]
            ensamble[name][0] = ensamble[name][1]
            ensamble[name][1] = tmp

    unique_check = set()
    for m, row in submission.iterrows():
        b = ensamble[row['Image']]
        predicted = ' '.join([(b[i]['name'] if i < len(b) else b[len(b) - 1]['name']) for i in range(5)])
        row['Id'] = predicted
        unique_check.add(b[0]['name'])

    print(f'Unique whales in predict: {len(unique_check)}')

    # submission.to_csv(f'./submissions/ensembling/ensemble-size768+384-files{len(files)}-first{2}.csv', header=True, index=False)
    submission.to_csv(f'../DATA/humpback_whale_siamese_torch/submissions/ensembling/{dir_name}-files{f_count}-first{2}-unique{len(unique_check)}-score({threshold}).csv', header=True,
                      index=False)


def make_score_ensemble():
    def scoreing(pred, targ):
        score = 0
        for n in range(len(pred)):
            for m in range(5):
                if targ[n][0] == pred[n][m]:
                    score += 1 / (m + 1)
                    break
        return score / len(pred)

    submit = [image_name for _, image_name, _ in pd.read_csv(SUB_Df).to_records()]
    train_dict = dict([(image_name, widx) for _, image_name, widx in pd.read_csv(TRAIN_DF).to_records()])

    h2ws = {}
    for image_name, w in train_dict.items():
        if w != 'new_whale':  # Use only identified whales
            if image_name not in h2ws:
                h2ws[image_name] = []
            if w not in h2ws[image_name]:
                h2ws[image_name].append(w)
    known = sorted(list(h2ws.keys()))

    target_df = pd.read_csv('../DATA/humpback_whale_siamese_torch/submissions/ensembling/ensemble-size768+384-files84-first2.csv')
    target = [idx.split(' ') for idx in target_df['Id']]

    dir_name = 'exp768-ch1-t1'
    mypath = f'../DATA/humpback_whale_siamese_torch/scores/{dir_name}/'
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    score = np.load(join(mypath, files[0]))
    for i in range(1, len(files)):
        score_path = join(mypath, files[i])
        score += np.load(score_path)

    threshold = len(files)
    for th in range(100):
        threshold -= 0.01
        vtop = 0
        vhigh = 0
        pos = [0, 0, 0, 0, 0, 0]
        result_whales = []
        result_images = []
        result_scores = []
        for i, p in enumerate(submit):
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

        score_pred = scoreing(result_whales, target)
        print(f'{str(round(threshold, 6)).zfill(6)} : {score_pred}')


make_ensemble()
