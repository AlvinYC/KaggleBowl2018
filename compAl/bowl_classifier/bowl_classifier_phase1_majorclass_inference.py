

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
#from skimage.io import imread, imshow, imread_collection, concatenate_images
from PIL import Image, ImageOps
import numpy as np
import time
import re
import os
import configparser
import argparse
import sys
#from mask-rcnn-V01.common import *

from dataset import KaggleDataset,Compose
np.set_printoptions(precision=8,suppress=True)

def read_config():
    conf = configparser.ConfigParser()
    candidates = ['config_default.ini', 'config.ini']
    conf.read(candidates)
    return conf

def get_force_general_info():
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument('force')
        parser.add_argument('threshold')
        args = parser.parse_args()
        force_general = True
        threshold     = args.threshold
    else:
        force_general = False
        threshold     = 0.9
    return force_general, threshold

#print(args.force)
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
             nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 3),
        )
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = self.softmax(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

config = read_config() # keep the line as top as possible
force_general, threshold = get_force_general_info()

print('force_general = ' + str(force_general))
############################################################################################################################
#  in inference procedure, we only need input two information to this program
#  1. image folder: in inference mode, testing data only contain image, no other information
#  2. model:        what's model you wanna use in this inference
#############################################################################################################################
TEST_IMG_DIR   = config['phase1'].get('TEST_IMG_DIR')               # [INPUT] need assign, image folder which is kaggle format
MODEL_IN_PATH  = config['phase1'].get('MODEL_IN_PATH')              # [INPUT] need assign, what model you wanna use
confidence_th  = float(threshold)
if force_general == True:
    CSV_OUT_PATH = config['phase1'].get('CSV_OUT_PATH_FORCE')       # [OUTPUT] CSV file
else:
    CSV_OUT_PATH = config['phase1'].get('CSV_OUT_PATH')             # [OUTPUT] CSV file
majorlabel     =  ['Histology', 'Fluorescence','Brightfield']
num_class      = len(majorlabel)

if not os.path.isdir(TEST_IMG_DIR):
    print(TEST_IMG_DIR + ' is not existed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    exit(0)

fn = open(CSV_OUT_PATH, "w")
#test_ids = os.listdir(TEST_IMG_DIR)
#test_ids = list(filter(lambda x: re.search('.png',x), test_ids))
test_ids = next(os.walk(TEST_IMG_DIR))[1]
test_ids =  [re.sub('\.png','',x) for x in test_ids]
print('test_ids = ' + str(len(test_ids)))
CSV_HEADER  = "source,zip_file,image_id,width,height,total_masks,"                          # info 01
CSV_HEADER += "probability_H,probability_F,probability_B,Major_Confidence,"                 # info 02
CSV_HEADER += "probability_HE,probability_IHC,Sub_Confidence,"                              # info 02
CSV_HEADER += "pred_major_category,pred_sub_category,major_category,sub_category,done\n"    # info 03
Image_id_idx   = CSV_HEADER.strip().split(',').index('image_id')            #info_02 , will be updated
PH_idx         = CSV_HEADER.strip().split(',').index('probability_H')       #info_02 , will be updated
PF_idx         = CSV_HEADER.strip().split(',').index('probability_F')       #info_02 , will be updated
PB_idx         = CSV_HEADER.strip().split(',').index('probability_B')       #info_02 , will be updated
Major_Conf_idx = CSV_HEADER.strip().split(',').index('Major_Confidence')    #info_02 , will be updated
PHE_idx        = CSV_HEADER.strip().split(',').index('probability_HE')      #info_02
PIHC_idx       = CSV_HEADER.strip().split(',').index('probability_IHC')     #info_02
Sub_Conf_idx   = CSV_HEADER.strip().split(',').index('Sub_Confidence')      #info_02 , 
Pred_major_idx = CSV_HEADER.strip().split(',').index('pred_major_category') #info_03 , will be updated
Pred_sub_idx   = CSV_HEADER.strip().split(',').index('pred_sub_category')   #info_03
M_Category_idx = CSV_HEADER.strip().split(',').index('major_category')      #info_03 , will be updated
S_Category_idx = CSV_HEADER.strip().split(',').index('sub_category')        #info_03
done_idx       = CSV_HEADER.strip().split(',').index('done')                #info_03

#   ____    _____   _____      _      _   _   _       _____           _____      _      ____    _       _____ 
#  |  _ \  | ____| |  ___|    / \    | | | | | |     |_   _|         |_   _|    / \    | __ )  | |     | ____|
#  | | | | |  _|   | |_      / _ \   | | | | | |       | |    _____    | |     / _ \   |  _ \  | |     |  _|  
#  | |_| | | |___  |  _|    / ___ \  | |_| | | |___    | |   |_____|   | |    / ___ \  | |_) | | |___  | |___ 
#  |____/  |_____| |_|     /_/   \_\  \___/  |_____|   |_|             |_|   /_/   \_\ |____/  |_____| |_____|
#                                                                                                           

fn.write(CSV_HEADER)
filenames = []
f_content = {}
for n, id_ in enumerate(test_ids):
    id_ = re.sub('\.png','',id_)
    img_path = TEST_IMG_DIR + '/' + id_ + '/images/' + id_ + '.png' 
    #img      = imread(img_path)[:,:,:3]
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    width,height = img.size
    #height   = img.shape[0]
    #width    = img.shape[1]
    filenames.append(id_)
    info_01  = 'Kaggle_Stage2,zip_name,'+str(id_)+','+str(width)+','+str(height)+','+str(0)+','
    info_02  = 'None,None,None,None,None,None,None,'
    info_03  = 'None,None,None,None,0\n'
    f_content[id_] = info_01 + info_02 + info_03
    # write this defaut CSV as KaggleDataset's input
    # print('id_ ' + id_ + '\t' + f_content[id_]) # debug
    fn.write(f_content[id_])    
fn.close()



#   _   _   ____    ____       _      _____   _____            ____   ____   __     __
#  | | | | |  _ \  |  _ \     / \    |_   _| | ____|          / ___| / ___|  \ \   / /
#  | | | | | |_) | | | | |   / _ \     | |   |  _|    _____  | |     \___ \   \ \ / / 
#  | |_| | |  __/  | |_| |  / ___ \    | |   | |___  |_____| | |___   ___) |   \ V /  
#   \___/  |_|     |____/  /_/   \_\   |_|   |_____|          \____| |____/     \_/   
#     


CSV_FILE         = re.sub('\.csv','',CSV_OUT_PATH)
dataset_test     = KaggleDataset(CSV_FILE,transform=Compose(), img_folder= TEST_IMG_DIR, resize_scale=[128,128])
confidence_alert = 0

valid_idx = range(dataset_test.__len__())
valid_loader = DataLoader(dataset_test, sampler=SubsetRandomSampler(valid_idx),batch_size=4,num_workers=2)

# network
net = VGG('VGG16')
print(net)
net.cuda()
net.eval()
net.load_state_dict(torch.load(MODEL_IN_PATH))

invert_majorlabel = {v:k for k,v in dataset_test.majorlabels.items()}

for i, data in enumerate(valid_loader, 0):
    inputs    = data['image']
    uids      = data['uid']
    labels    = data['label']
    sublabels = data['sublabel']
    images    = Variable(inputs.cuda())
    output    = net(images)
    _, predicted = torch.max(output.data, 1)
    output_cpu = output.data.cpu().numpy()
    output_cpu = np.exp(output_cpu) # if net contain LogSoftmax layer
    predicted_cpu = predicted.cpu().numpy()
    for j in range(len(uids)):
        output_score = np.array2string(output_cpu[j,:],precision=8, separator='\t', formatter={'float_kind':lambda x:'%0.8f' %x})
        max_score  = sorted(output_cpu[j,:],reverse=True)[0]
        sec_score  = sorted(output_cpu[j,:],reverse=True)[1]
        dist_score = max_score - sec_score
        if dist_score < confidence_th:
            confidence_alert += 1
        show_confi_alert = ' (*)' if dist_score < confidence_th else ''
        output_score = re.sub('\[|\]|\ ','',output_score)
        print('{:65s}'.format(uids[j]) + '\t' + str(predicted_cpu[j])+'\t'+output_score+'\t'+str(dist_score) +'\t'+ show_confi_alert)

        content_list = f_content[str(uids[j])].split(',')
        content_list[PH_idx]         = output_score.split('\t')[0]
        content_list[PF_idx]         = output_score.split('\t')[1]
        content_list[PB_idx]         = output_score.split('\t')[2]
        content_list[Pred_major_idx] = invert_majorlabel[predicted_cpu[j]]
        if (force_general == True) & (dist_score < confidence_th):
            content_list[M_Category_idx] = 'GEN'
            #print('dist_score ' + str(dist_score) + ' ==> confidence_th ' + str(confidence_th))
        else:
            content_list[M_Category_idx] = invert_majorlabel[predicted_cpu[j]]
        content_list[Major_Conf_idx] = str('{:0.8f}'.format(dist_score))
        content_list[done_idx]       = '0' if dist_score < confidence_th else '1'

        f_content[str(uids[j])] = ','.join(content_list) + '\n'
        
#    ___    _   _   _____   ____    _   _   _____            ____   ____   __     __
#   / _ \  | | | | |_   _| |  _ \  | | | | |_   _|          / ___| / ___|  \ \   / /
#  | | | | | | | |   | |   | |_) | | | | |   | |    _____  | |     \___ \   \ \ / / 
#  | |_| | | |_| |   | |   |  __/  | |_| |   | |   |_____| | |___   ___) |   \ V /  
#   \___/   \___/    |_|   |_|      \___/    |_|            \____| |____/     \_/   
#                                                                                 

# sorting output csv by 'done' (imply confidence level < therhold) and major category
content_list = [f_content[each_file].strip().split(',') for each_file in filenames] # split to list for each line
#content_list.sort(key=lambda row: (row[done_idx],row[M_Category_idx]))              # sorting
content_list.sort(key=lambda row: (row[M_Category_idx],row[Major_Conf_idx]))              # sorting
content_list = [','.join(x) for x in content_list]                                  # merge to string for each line(list)
fn = open(CSV_OUT_PATH, "w")
fn.write(CSV_HEADER)
fn.write('\n'.join(content_list))
fn.close()
print('confidence_alert = ' + str(confidence_alert))    
print('Finished evaluation')

