from __future__ import print_function, division

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
import math
import pandas as pd

from PIL import Image


import cv2


import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import Parameter
import torchvision
from torchvision import datasets, models, transforms

import cv2
from PIL import Image
import argparse
from pathlib import Path
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save",action="store_true")
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # 디바이스 설정

    conf = get_config(False)
    model_path = "2021-08-07-03-49_accuracy:0_step:346037_None.pth"

    mtcnn = MTCNN()
    print('mtcnn loaded')

    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, model_path, True, True)
    else:
        learner.load_state(conf, model_path, True, True)
    learner.model.eval()
    print('learner loaded')

    # model_ft = model.bottleneck_IR_SE()
    # model_ft.load_state_dict(torch.load(''))
    # model = mtcnn


    #model_file = torch.load("./model_save.pt")# 사전학습 모델 다운 후 알맞은 경로 지정
    #model = ResNet(IRBlock, [3, 4, 6, 3], use_se=False, im_size=112).to(device) # 모델 정의
    #model.load_state_dict(model_file) # 사전 학습된 모델의 weight 로 업데이트
    learner.model.eval() # 모델을 평가 모드 설정

    import pandas as pd
    submission = pd.read_csv("/home/leo/Desktop/inha_challenge/inha_data/sample_submission.csv")

    left_test_paths = list()
    right_test_paths = list()

    for i in range(len(submission)):
        left_test_paths.append(submission['face_images'][i].split()[0])
        right_test_paths.append(submission['face_images'][i].split()[1])

    # 이미지 데이터 전처리 정의

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    # Left Side Image Processing

    left_test = list()
    for left_test_path in left_test_paths:
        img = Image.open("/home/leo/Desktop/inha_challenge/inha_data/test/" + left_test_path + '.jpg').convert("RGB")# 경로 설정 유의(ex .inha/test)
        img = data_transform(img) # 이미지 데이터 전처리
        left_test.append(img) 
    left_test = torch.stack(left_test)
    #print(left_test.size()) # torch.Size([6000, 3, 112, 112])

    left_infer_result_list = list()
    with torch.no_grad():
        '''
        메모리 부족으로 6,000개 (배치) 한번에 입력으로 넣지 않고 1,000개 씩 입력으로 줌
        '''
        batch_size = 500
        for i in range(0, 12):
            i = i * batch_size
            tmp_left_input = left_test[i:i+batch_size]
            #print(tmp_input.size()) # torch.Size([1000, 3, 112, 112])
            left_infer_result = learner.model(tmp_left_input.to(device))
            #print(left_infer_result.size()) # torch.Size([1000, 512])
            left_infer_result_list.append(left_infer_result)

        left_infer_result_list = torch.stack(left_infer_result_list, dim=0).view(-1, 512)
        #print(left_infer_result_list.size()) # torch.Size([6000, 512])


    # Right Side Image Processing

    right_test = list()
    for right_test_path in right_test_paths:
        img = Image.open("/home/leo/Desktop/inha_challenge/inha_data/test/" + right_test_path + '.jpg').convert("RGB") # 경로 설정 유의 (ex. inha/test)
        img = data_transform(img)# 이미지 데이터 전처리
        right_test.append(img)
    right_test = torch.stack(right_test)
    #print(right_test.size()) # torch.Size([6000, 3, 112, 112])

    right_infer_result_list = list()
    with torch.no_grad():
        '''
        메모리 부족으로 6,000개 (배치) 한번에 입력으로 넣지 않고 1,000개 씩 입력으로 줌
        '''
        batch_size = 500
        for i in range(0, 12):
            i = i * batch_size
            tmp_right_input = right_test[i:i+batch_size]
            #print(tmp_input.size()) # torch.Size([1000, 3, 112, 112])
            right_infer_result = learner.model(tmp_right_input.to(device))
            #print(left_infer_result.size()) # torch.Size([1000, 512])
            right_infer_result_list.append(right_infer_result)

        right_infer_result_list = torch.stack(right_infer_result_list, dim=0).view(-1, 512)
        #print(right_infer_result_list.size()) # torch.Size([6000, 512])


    def cos_sim(a, b):
        return F.cosine_similarity(a, b)

    cosin_similarity = cos_sim(left_infer_result_list, right_infer_result_list)


    submission = pd.read_csv("/home/leo/Desktop/inha_challenge/inha_data/sample_submission.csv") 
    submission['answer'] = cosin_similarity.tolist()
    #submission.loc['answer'] = submission['answer']
    submission.to_csv('/home/leo/Desktop/inha_challenge/inha_data/submission.csv', index=False)
