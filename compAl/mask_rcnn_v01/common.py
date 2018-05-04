import os
from datetime import datetime
PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
IDENTIFIER   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

#numerical libs
import math
import numpy as np
import random
import PIL
import cv2

import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Qt4Agg')
#matplotlib.use('Qt5Agg')

# torch libs
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel

# std libs
import collections
import copy
import numbers
import inspect
import shutil
from timeit import default_timer as timer

import csv
import pandas as pd
import pickle
import glob
import sys
from distutils.dir_util import copy_tree
import time
import matplotlib.pyplot as plt

import skimage
import skimage.color
import skimage.morphology
from scipy import ndimage
import configparser

# config related handling
def run_once(func):
    ''' a declare wrapper function to call only once, use @run_once declare keyword '''
    def wrapper(*args, **kwargs):
        if 'result' not in wrapper.__dict__:
            wrapper.result = func(*args, **kwargs)
        return wrapper.result
    return wrapper

@run_once
def read_config():
    conf = configparser.ConfigParser()
    candidates = ['config_default.ini', 'config.ini']
    conf.read(candidates)
    return conf

config = read_config() # keep the line as top as possible

# edit settings here
#data and results forder
ROOT_DIR = '.'
#ROOT_DIR = '../media/alvin/disk_D/dataset/Bowl/mask_rcnn_data'
#DATA_DIR    = ROOT_DIR + '/data'  #'/media/root/5453d6d1-e517-4659-a3a8-d0a878ba4b60/data/kaggle/science2018/data' #
#DATA_DIR    = '../stage2_testing_data'
DATA_DIR    = config['param'].get('img_folder')
#RESULTS_DIR = ROOT_DIR + '/results'
RESULTS_DIR = './results'
TASK_NAME = '/' + config['train'].get('log_name')  # task output folder, subfolder of RESULTS_DIR.
TASK_OUTDIR = RESULTS_DIR + TASK_NAME

##Input excel file
#for train.py
# move to train.py, COMMENT_CSV_PATH = DATA_DIR +'/__download__/stage1_train.csv'
# EXCEL_PATH = DATA_DIR + '/split/stage1_train_sort_comment_external.xlsx' #for train.py
# EXCEL_PATH = DATA_DIR + '/split/stage1_train_fix_v4_external.xlsx'
# Train_CSV_SHEET = '0320'
# Train_CSV_SHEET = 'test'

#for submit.py
# UNUSED!! EXCEL_PATH_TEST = DATA_DIR + '/split/Annotate_stage_1_test_list.xlsx'
# UNUSED!! using csv, using file name instead. TEST_CSV_SHEET = 'stage1_test_2clusters_sort_all'
# move to submit.py, TEST_CSV_PATH = DATA_DIR +'/split/test.csv'

# PRETRAIN_FILE = ROOT_DIR + '/resnet-50.t7'
# PRETRAIN_FILE = None

##Training
#checkpoint file for training
# Unused!! usning helper.load_ckpt() and ckpt_path() to automaticly check, INITIAL_CP_FILE = RESULTS_DIR + TASK_NAME + '/checkpoint/' + config['train'].get('resume_ckpt')
# INITIAL_CP_FILE = None
# move to learn_rate@config_default.ini, LEARNING_RATE = 0.001
# move to iter_accum@config_default.ini, LEARNING_ITER_ACCUM = 4
# move to n_batch@config_default.ini, LREARNING_BATCH = 4

# move to classes_map@config_default.ini, SELECT_CATEGORY = 'major_category' # 'sub_category', 'major_category'
# REMOVE!! automatically calculate. NUM_CLASSES = 5
# move to classes_map@config_default.ini, LABEL_MAP =  {'Histology' : 1, 'Fluorescence' : 2, 'Brightfield' : 3, 'Cloud' : 4}
# move to classes_map@config_default.ini, LABEL_MAP_MAJOR =  {'Histology' : 1, 'Fluorescence' : 2, 'Brightfield' : 3}
# move to classes_map@config_default.ini, LABEL_MAP_SUB =  {  'HE' : 1, 'Fluorescence' : 2, 'Brightfield' : 3, 'Cloud' : 4, 'Drosophilidae' : 5, 'IHC' : 6 }

##Prediction
#checkpoint file for prediction
# move to submit.py, using ckpt_path() and predict_model_ckpt instead, PREDICT_MODEL = '235_model.pth'
# PREDICT_CP_FILE = TASK_OUTDIR + '/checkpoint/' + PREDICT_MODEL
# UNUSED!! SUB_TASK_NAME = TASK_NAME
# SUB_TASK_NAME = '/V10_final_continue'
# UNUSED!! using argument, run_submit('train') instead. SUBMIT_MODE = 'train' #  'test' for kaggle submit
# always save overlay images, SUBMIT_IMAGE = 'yes'
# using get_submit_dir() instead. SUBMIT_FOLDER = '/' + PREDICT_MODEL.split('_')[0] +'_'+SUBMIT_MODE+'_'
##--------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def np_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def np_sigmoid(x):
  return 1 / (1 + np.exp(-x))


#---------------------------------------------------------------------------------
print('@%s:  ' % os.path.basename(__file__))

if 1:
    SEED = 35202 #1510302253  #int(time.time()) #
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print ('\tset random seed')
    print ('\t\tSEED=%d'%SEED)

if 1:
    torch.backends.cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = True
    print ('\tset cuda environment')
    print ('\t\ttorch.__version__              =', torch.__version__)
    print ('\t\ttorch.version.cuda             =', torch.version.cuda)
    print ('\t\ttorch.backends.cudnn.version() =', torch.backends.cudnn.version())
    try:
        print ('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =',os.environ['CUDA_VISIBLE_DEVICES'])
        NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    except Exception:
        print ('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =','None')
        NUM_CUDA_DEVICES = 1

    print ('\t\ttorch.cuda.device_count()      =', torch.cuda.device_count())
    print ('\t\ttorch.cuda.current_device()    =', torch.cuda.current_device())


print('')

#---------------------------------------------------------------------------------