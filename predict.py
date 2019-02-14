from main import SiameseNet, SubBlock, predict
from os import listdir
from os.path import isfile, join
import torch
import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prediction():
    list_dirs = ['exp384-ch3-t1', 'exp384-ch3-t4']
    # list_dirs = ['exp768-ch1-t1', 'exp768-ch3-t1']

    for checkpoint_dir in list_dirs:
        mypath = f'../DATA/humpback_whale_siamese_torch/checkpoints/{checkpoint_dir}/'
        files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        files.sort()

        submit_dir = f'../DATA/humpback_whale_siamese_torch/submissions/{checkpoint_dir}/'
        score_dir = f'../DATA/humpback_whale_siamese_torch/scores/{checkpoint_dir}/'
        os.makedirs(submit_dir, exist_ok=True)
        os.makedirs(score_dir, exist_ok=True)

        for file in tqdm(files, total=len(files)):
            sub_file = join(submit_dir, file.replace('.pt', '.csv'))
            score_file = join(score_dir, file)
            if isfile(score_file):
                continue
            print(file)

            checkpoint = torch.load(mypath + file)
            model = SiameseNet(checkpoint['channel'], checkpoint['features_size'])
            model.load_state_dict(checkpoint['state_dict'])
            model.to(device)
            predict(model, device, 16, sub_file, score_file, checkpoint['norm_zero_one'])


prediction()


