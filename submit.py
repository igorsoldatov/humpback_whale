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
import operator
import torch
from PIL import Image as pil_image


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
    sub_file = '../DATA/humpback_whale_siamese_torch/submissions/analysis_errors/new_whale_err_total.csv'
    dir_out = '../DATA/humpback_whale_siamese_torch/submissions/analysis_errors/'

    sub = pd.read_csv(sub_file)
    sub_to = pd.read_csv(sub_file)
    for n in range(5):
        predicted = []
        for idx in tqdm(sub['Id'], total=len(sub)):
            ids = idx.split(' ')[n]
            predicted.append(' '.join([ids for i in range(5)]))
        sub_to['Id'] = predicted
        sub_to.to_csv(join(dir_out, f'{n}.csv'), header=True, index=False)


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


def make_ensemble(dir_names, sub_name):
    base = '/media/igor/D44EC2E74EC2C20A/humpback_whale_siamese_torch/submissions/hand/' \
           'all-files101-th(0.92)-PL(0.911)-cH.scv'
    train_labels = pd.read_csv('../DATASET/humpback_whale/all/train.csv')
    unique_whales = set(train_labels['Id'].tolist())

    submission = pd.read_csv('../DATASET/humpback_whale/all/sample_submission.csv')

    files = []
    for dir_name in dir_names:
        my_path = f'../DATA/humpback_whale_siamese_torch/submissions/{dir_name}/'
        files += [join(my_path, f) for f in listdir(my_path) if isfile(join(my_path, f))]
    all = {}

    f_count = 0
    threshold = 0.99
    # for file in glob.iglob(mypath, recursive=True):
    for file in files:
        score = get_submission_score(base, file)
        # print(f'{file} : {score}')
        # if score < threshold:
        #     continue
        f_count += 1
        sub = pd.read_csv(file)
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

    unique_check_first = set()
    unique_check = set()
    for m, row in submission.iterrows():
        b = ensamble[row['Image']]
        predicted = ' '.join([(b[i]['name'] if i < len(b) else b[len(b) - 1]['name']) for i in range(5)])
        row['Id'] = predicted

        unique_check_first.add(b[0]['name'])
        for i in range(5):
            if i < len(b):
                unique_check.add(b[i]['name'])

    print(f'Unique whales in first row: {len(unique_check_first)}, Unique whales total: {len(unique_check)}')

    submit_file = f'{sub_name}-files{f_count}-first{2}-unique{len(unique_check_first)}-score({threshold}).csv'
    submission.to_csv(f'../DATA/humpback_whale_siamese_torch/submissions/{submit_file}', header=True, index=False)


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


def analysis_errors():
    sub = pd.read_csv('../DATA/humpback_whale_siamese_torch/submissions/ensembling/'
                      'best-files183-first2-unique3517-score(0.99)-PL(0.895).csv')

    stat = {}
    for target in sub['Id']:
        for t in target.split(' '):
            if t in stat:
                stat[t] += 1
            else:
                stat[t] = 1
    sorted_stat = sorted(stat.items(), key=operator.itemgetter(1), reverse=False)
    filler = sorted_stat[0][0]

    new_whale_err = []
    known_err = []
    for target in sub['Id']:
        arr_target = target.split(' ')
        if arr_target[0] != 'new_whale':
            new_whale_err.append(' '.join([filler for _ in range(5)]))
            known_err.append(' '.join(['new_whale' for _ in range(5)]))
        else:
            new_whale_err.append(' '.join(arr_target[1:] + [filler]))
            known_err.append(' '.join([filler for _ in range(5)]))

    sub['Id'] = new_whale_err
    sub.to_csv('../DATA/humpback_whale_siamese_torch/submissions/analysis_errors/new_whale_err_total.csv',
               header=True, index=False)
    sub['Id'] = known_err
    sub.to_csv('../DATA/humpback_whale_siamese_torch/submissions/analysis_errors/known_err_total.csv',
               header=True, index=False)


def compile_model_and_hand_prediction():
    model_predict = pd.read_csv('../DATA/humpback_whale_siamese_torch/submissions/all-files101-th(0.92)-PL(0.898).scv')
    hand_predict = pd.read_csv('./hand_result_new.csv')

    hand_result = {}
    for row in hand_predict.iterrows():
        image_name = row[1]['Image']
        wid = row[1]['Whale']
        hand_result[image_name] = wid

    new_predict = []
    err_count = 0
    for image_name, wids in zip(model_predict['Image'], model_predict['Id']):
        wids_arr = wids.split(' ')
        wid = wids_arr[0]
        if image_name in hand_result and wid != hand_result[image_name]:
            err_count += 1
            new_predict.append(' '.join([hand_result[image_name]] + wids_arr[:4]))
            print(f'{wid} : {hand_result[image_name]}')
        else:
            new_predict.append(wids)

    print(f'Errors: {err_count}')
    model_predict['Id'] = new_predict
    model_predict.to_csv('../DATA/humpback_whale_siamese_torch/submissions/all-files101-th(0.92)-PL(0.898)-cH.scv',
                         header=True, index=False)


def make_submission_from_hand():
    model_predict = pd.read_csv('../DATA/humpback_whale_siamese_torch/submissions/ensembling/'
                                'best-files183-first2-unique3517-score(0.99)-PL(0.895).csv')
    hand_predict = pd.read_csv('./hand_result_new.csv')

    hand_result = {}
    for row in hand_predict.iterrows():
        image_name = row[1]['Image']
        wid = row[1]['Whale']
        confidance = row[1]['Confidence']
        if confidance == 'H':
            hand_result[image_name] = wid

    print(f'confidance: {len(hand_result)}')

    new_predict = []
    for image_name, wids in zip(model_predict['Image'], model_predict['Id']):
        if image_name in hand_result:
            new_predict.append(' '.join([hand_result[image_name] for i in range(5)]))
        else:
            new_predict.append(' '.join(['w_dec7ffd' for i in range(5)]))

    model_predict['Id'] = new_predict
    model_predict.to_csv('../DATA/humpback_whale_siamese_torch/submissions/h1115_cH.scv',
                         header=True, index=False)


def submission_analysis(sub_file):
    first_unique_stat = {}
    total_unique_stat = {}

    sub = pd.read_csv(sub_file)
    for wid in sub['Id']:
        ws = wid.split(' ')

        if ws[0] not in first_unique_stat:
            first_unique_stat[ws[0]] = 0
        first_unique_stat[ws[0]] += 1

        for w in ws:
            if w not in total_unique_stat:
                total_unique_stat[w] = 0
            total_unique_stat[w] += 1

    print(f'First unique: {len(first_unique_stat)}, Total unique: {len(total_unique_stat)}')

    wids = []
    count_first = []
    count_total = []
    for k in total_unique_stat:
        wids.append(k)
        count_total.append(total_unique_stat[k])
        if k in first_unique_stat:
            count_first.append(first_unique_stat[k])
        else:
            count_first.append(0)
    stat_df = pd.DataFrame()
    stat_df['Whale'] = wids
    stat_df['Total'] = count_total
    stat_df['First'] = count_first
    stat_df.to_csv(sub_file.replace('.csv', '-stat.csv'), header=True, index=False)


########################################################################################################################


def validation_score(score_matrix, ids1, ids2, threshold=0.99):
    train_df = pd.read_csv(TRAIN_DF)
    image2whale = {}
    for image, whale in zip(train_df['Image'], train_df['Id']):
        image2whale[image] = whale

    score = 0
    new_whales = 0
    for i, image_true in enumerate(ids1):
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
        if predicted[0] == 'new_whale':
            new_whales += 1

    return score / len(ids1), new_whales


def validation_compile_score_choose_threshold():
    my_dir = '../DATA/humpback_whale_siamese_torch/scores_valid/exp384-ch3-t1/'
    files = listdir(my_dir)
    score_dict = torch.load(join(my_dir, files[0]))
    score_matrix = score_dict['score_matrix']
    for i in range(1, len(files)):
        score_dict_tmp = torch.load(join(my_dir, files[i]))
        score_matrix += score_dict_tmp['score_matrix']
    score_matrix /= len(files)

    threshold = 0.90
    for _ in range(10):
        score, new_whales = validation_score(score_matrix, score_dict['submit'], score_dict['known'], threshold)
        print(f'Validation score: {round(score, 6)}, Threshold: {round(threshold, 4)}, New whales: {new_whales}')
        threshold += 0.01


def compute_validation_score(my_dir):
    my_path = f'/media/igor/D44EC2E74EC2C20A/humpback_whale_siamese_torch/scores_valid/{my_dir}/'
    thresh = 0.5
    files = listdir(my_path)
    for file in files:
        sdict = torch.load(join(my_path, file))
        score, nw = validation_score(sdict['score_matrix'], sdict['train_ids'], sdict['train_ids'], thresh)
        print(f'{file} score: {round(score, 6)}, Threshold: {round(thresh, 4)}, New whales: {nw}')


########################################################################################################################


def gray_train_stat():
    TRAIN_PATH = '../DATASET/humpback_whale/all/train/'
    train_df = pd.read_csv('../DATASET/humpback_whale/all/train.csv')
    stat = {}
    for file, wid in zip(train_df['Image'], train_df['Id']):
        ch = pil_image.open(join(TRAIN_PATH, file)).layers
        if wid not in stat:
            stat[wid] = []
        stat[wid].append(file)
    for wid in stat:
        print(f'{wid}: {len(stat[wid])}')


# gray_train_stat()

# make_ensemble(['exp384-ch1-t1-00'], 'exp384-ch1-t1')
# make_ensemble(['exp384-ch3-t1'], 'exp384-ch3-t1')
# make_ensemble(['exp768-ch1-t1'], 'exp768-ch1-t1')
# make_ensemble(['exp768-ch3-t1'], 'exp768-ch3-t1')
# make_ensemble(['exp384-ch1-t1', 'exp384-ch3-t1', 'exp768-ch1-t1', 'exp768-ch3-t1'], 'exp-ch-t1')


# submission_analysis('../DATA/humpback_whale_siamese_torch/submissions/exp384-ch1-t1-files54-first2-unique3676-score(0.99).csv')


# analysis_errors()
# split_submissions()
# compile_model_and_hand_prediction()
# make_submission_from_hand()


compute_validation_score('exp768-ch3-t1')
