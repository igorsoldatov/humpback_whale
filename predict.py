from main import SiameseNet, SubBlock, predict
from os import listdir
from os.path import isfile, join
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mydir = 'horiz-code384-00'
mypath = f'./checkpoints/{mydir}/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
files.sort(reverse=False)

os.makedirs(f'./submissions/{mydir}/', exist_ok=True)

for file in files:
    sub_file = f'./submissions/{mydir}/' + file.replace('.model', '.csv')
    score_file = f'./scores/{mydir}/' + file.replace('.model', '.npy')
    if isfile(score_file):
        continue
    print(file)
    model = torch.load(mypath + file)
    predict(model, device, 16, sub_file, score_file)
