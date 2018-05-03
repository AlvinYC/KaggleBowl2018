

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
import argparse
import numpy as np
import time
import sys
import re
import os

from dataset import KaggleDataset,Compose
np.set_printoptions(precision=8,suppress=True)

#  ###################################################################################################################
#  ####################`..#######`..#########`....##########`.....#########`........#####`..##########################
#  ####################`.#`..###`...#######`..####`..#######`..###`..######`..###########`..##########################
#  ####################`..#`..#`#`..#####`..########`..#####`..####`..#####`..###########`..##########################
#  ####################`..##`..##`..#####`..########`..#####`..####`..#####`......#######`..##########################
#  ####################`..###`.##`..#####`..########`..#####`..####`..#####`..###########`..##########################
#  ####################`..#######`..#######`..#####`..######`..###`..######`..###########`..##########################
#  ####################`..#######`..#########`....##########`.....#########`........#####`........####################
#  ###################################################################################################################

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
        #self.classifier = nn.Linear(8192,2)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            #nn.Linear(512 * 4 * 4, 3),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2),
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

def prepare_model_info_net():
    net = VGG('VGG16')
    #print(net)
    net.cuda()
    return net

def prepare_model_info_other(net):
    criterion        = nn.CrossEntropyLoss()
    init_lr          = 0.0001
    optimizer        = optim.SGD(net.parameters(), lr=init_lr, momentum=0.9) 
    epoch_checkpoint = [10,30,50,70]
    return optimizer, epoch_checkpoint, criterion, init_lr

#  ################################################################################################################
#  ##########`..#####`..#####`...........#####`..#####`..###########`..#####`...........#####`..######`..##########
#  ##########`..#####`..##########`..#########`..#####`..###########`..##########`..##########`..####`..###########
#  ##########`..#####`..##########`..#########`..#####`..###########`..##########`..###########`..#`..#############
#  ##########`..#####`..##########`..#########`..#####`..###########`..##########`..#############`..###############
#  ##########`..#####`..##########`..#########`..#####`..###########`..##########`..#############`..###############
#  ##########`..#####`..##########`..#########`..#####`..###########`..##########`..#############`..###############
#  ############`.....#############`..#########`..#####`........#####`..##########`..#############`..###############
#  ################################################################################################################

def adjust_lr(init_lr, optimizer, epoch, method=None):
    threshold_epoch = 50
    lr = init_lr
    if  method == 'dual':
        # just use 2 learning rate(lr)
        lr = init_lr * 0.1 if epoch>=threshold_epoch else init_lr
    elif method == 'decade':
        # decade 1/10 lr every 10 epcoh
        lr = init_lr * (0.1 ** (((epoch-threshold_epoch)//10)+1)) if epoch>=threshold_epoch else init_lr        
    elif method == 'triple':
        # use 3 learning rate (contaion 2 threshol: epoch 30/50)
        # lr decade 1/10 when met first epoch threshold
        # lr decade 1/20 when met second epoch threshold
        if epoch >= 50:
            lr = init_lr / 20
        elif epoch >= 30:
            lr = init_lr / 10
        else:
            lr = init_lr
    else:
        # otherwise: use defaut init_lr
        lr = init_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def cv_kn2digit(v):
    if v.lower() in ('no','n','0'):
        return 'no'
    elif v.lower() in ('loocv','max'):
        return 'loocv'
    elif re.search('\d+',v.lower()):
        return v
    else:
        raise argparse.ArgumentTypeError('[no, max, loocv, 1, 2, 3, ...n] expected.')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


#  #############################################################################################################
#  ###############`..######################`..###################################################`..############
#  ###############`..######################`..###################################################`..############
#  ###############`..########`..#########`.`.#`.########`..##########`....#########`..#########`.`.#`.##########
#  ###########`..#`..######`..##`..########`..########`..##`..######`..##########`.###`..########`..############
#  ##########`.###`..#####`..###`..########`..#######`..###`..########`...######`.....#`..#######`..############
#  ##########`.###`..#####`..###`..########`..#######`..###`..##########`..#####`.###############`..############
#  ###########`..#`..#######`..#`...########`..########`..#`...#####`..#`..#######`....###########`..###########
#  #############################################################################################################

#def prepare_data(args, category=None):
def prepare_dataset(args, category=None):
    if args.mode == 'train':
        train_csv     = re.sub('\.csv$','',args.in_train_csv)
        dataset_train = KaggleDataset(train_csv,transform=Compose(), img_folder= args.in_train_img, category=category, resize_scale=[128,128])
        return dataset_train

    if args.mode == 'valid':
        valid_csv     = re.sub('\.csv$','',args.in_valid_csv)
        dataset_valid = KaggleDataset(valid_csv,transform=Compose(), img_folder= args.in_valid_img, category=category, resize_scale=[128,128])
        return dataset_valid

    if args.mode == 'test':
        test_csv      = re.sub('\.csv$','',args.in_test_csv)
        dataset_test  = KaggleDataset(test_csv,transform=Compose(), img_folder= args.in_test_img, category=category, resize_scale=[128,128])
        return dataset_test

def prepare_dataloader(args, dataset, dataset_test, K_fold=None, K_idx=None):
    batch_size    = 4
    train_loader  = []
    valid_loader  = []    
    test_loader   = []    
    if args.mode == 'train':
        if re.search('\d+',str(K_fold)):
            train_idx, valid_idx = dataset_train.split(K_fold,K_idx)
            train_loader  = DataLoader(dataset_train, sampler=SubsetRandomSampler(train_idx),batch_size=batch_size,num_workers=2)
            valid_loader  = DataLoader(dataset_train, sampler=SubsetRandomSampler(valid_idx),batch_size=batch_size,num_workers=2)
        else: #  args.mode = 'no', there is NO validation loader
            train_idx     = range(dataset_train.__len__())
            train_loader  = DataLoader(dataset_train, sampler=SubsetRandomSampler(train_idx),batch_size=batch_size,num_workers=2)

    if args.mode == 'valid':
            valid_idx     = range(dataset_valid.__len__())
            valid_loader  = DataLoader(dataset_valid,  sampler=SubsetRandomSampler(valid_idx),batch_size=batch_size,num_workers=2)

    if args.mode == 'test':
        testing_idx  = range(dataset_test.__len__())
        test_loader  = DataLoader(dataset_test,  sampler=SubsetRandomSampler(testing_idx),batch_size=batch_size,num_workers=2)

    if args.mode == 'train':
        print('database_train ' + '{:>5d}'.format(len(dataset_train) if dataset_train is not None else 0) + ' w/ batch size ' + str(batch_size))
        print('total batch = ' + '{:>4d}'.format(int(np.ceil(len(dataset_train)/batch_size))))
    if args.mode == 'valid':
        print('database_valid ' + '{:>5d}'.format(len(dataset_valid) if dataset_valid is not None else 0) + ' w/ batch size ' + str(batch_size))
        print('total batch = ' + '{:>4d}'.format(int(np.ceil(len(dataset_valid)/batch_size))))
    #print('database_test  ' + '{:>5d}'.format(len(dataset_test ) if dataset_test is not None else 0) + ' w/ batch size ' + str(batch_size))
    
    print('train batch = '  + '{:>4d}'.format(len(train_loader)))
    print('valid batch = '  + '{:>4d}'.format(len(valid_loader)))
    #print('test  batch = '  + '{:>4d}'.format(len(test_loader)))

    return train_loader, valid_loader, test_loader

    #  ##################################################################################################################
    #  ####################`...#`......#####`.......###############`.############`..#####`...#####`..####################
    #  #########################`..#########`..####`..############`.#..##########`..#####`.#`..###`..####################
    #  #########################`..#########`..####`..###########`.##`..#########`..#####`..#`..##`..####################
    #  #########################`..#########`.#`..##############`..###`..########`..#####`..##`..#`..####################
    #  #########################`..#########`..##`..###########`......#`..#######`..#####`..###`.#`..####################
    #  #########################`..#########`..####`..########`..#######`..######`..#####`..####`.#..####################
    #  #########################`..#########`..######`..#####`..#########`..#####`..#####`..######`..####################
    #  ##################################################################################################################

def training_model(args,train_loader,kth=None,labelname=None):
    # 'kth' is only used to concate output file name, network training procedure would not use it
    net = prepare_model_info_net()
    optimizer, epoch_checkpoint, criterion, init_lr = prepare_model_info_other(net)

    for epoch in range(epoch_checkpoint[-1]):
        adjust_lr(init_lr, optimizer, epoch, method='triple')
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs = data['image']
            labels = data['sublabel']
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            optimizer.zero_grad()
            output = net(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data.item()

            iterataion_num = 10
            if i % iterataion_num == (iterataion_num-1):
                print('[%4d, %5d] loss: %.5f lr: %.6f' %(epoch + 1, i + 1, running_loss / iterataion_num,optimizer.param_groups[0]['lr']))
                running_loss = 0.0

        if (epoch+1) in epoch_checkpoint:
            MODEL_OUT_PATH  = args.ou_model_path + '/' + args.task
            MODEL_OUT_PATH += '/' + args.task + '_K'+'{:03d}'.format(0 if kth == None else kth)+'_epoch'+'{:03d}'.format(epoch+1)
            print('MODEL_OUT_PATH ' + MODEL_OUT_PATH)
            torch.save(net.state_dict(),MODEL_OUT_PATH)

#  #####################################################################################################################
#  ####################`..#########`..###########`.############`..###########`..#####`.....#############################
#  #####################`..#######`..###########`.#..##########`..###########`..#####`..###`..##########################
#  ######################`..#####`..###########`.##`..#########`..###########`..#####`..####`..#########################
#  #######################`..###`..###########`..###`..########`..###########`..#####`..####`..#########################
#  ########################`..#`..###########`......#`..#######`..###########`..#####`..####`..#########################
#  #########################`....###########`..#######`..######`..###########`..#####`..###`..##########################
#  ##########################`..###########`..#########`..#####`........#####`..#####`.....#############################
#  #####################################################################################################################

def validing_model(args,valid_loader,kth=None,labelname=None):
    net = prepare_model_info_net()
    optimizer, epoch_checkpoint, criterion, init_lr = prepare_model_info_other(net)
    invert_labelname = {v:k for k,v in labelname.items()} 

    class_correct = list(0. for i in range(len(labelname)))
    class_total = list(0. for i in range(len(labelname)))    

    #for epoch in epoch_checkpoint:
    #MODEL_IN_PATH = args.ou_model_path + '/' + args.task + '/' +  args.task + '_K'+'{:03d}'.format(kth)+'_epoch'+'{:03d}'.format(epoch)

    net.load_state_dict(torch.load(args.valid_model))

    #CSV_OUT_PATH = args.ou_model_path + '/' + args.task + '/' +  args.task + '_K'+'{:03d}'.format(kth)+'_epoch'+'{:03d}'.format(epoch) + '.csv'
    if args.mode == 'train':
        CSV_OUT_PATH = args.ou_model_path + '/' + args.task + '/valid_result_'+ re.sub('.*/','',args.valid_model) + '.csv'
    if args.mode == 'valid':
        CSV_OUT_PATH = './valid_result_'+ re.sub('.*/','',args.valid_model) + '.csv'
    print('CSV_OUT_PATH: ' + CSV_OUT_PATH)
    fn = open(CSV_OUT_PATH, "w")
    fn.write("uid\tground_truth\tprediected\tcorrect\tHE\tIHC\tconfidence\n")

    correct       = 0
    total         = 0
    class_total   = [0] * len(labelname) 
    class_correct = [0] * len(labelname) 
    confidence_level_alert = 0

    for i, data in enumerate(valid_loader, 0):
        inputs = data['image']
        labels = data['sublabel']
        uids   = data['uid']
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        images = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        output = net(images)

        _, predicted = torch.max(output.data, 1)
        c = predicted.eq(labels.data).squeeze()
        
        output_cpu = output.data.cpu().numpy()
        output_cpu = np.exp(output_cpu) # if net contain LogSoftmax layer

        predicted_cpu = predicted.cpu().numpy()
        for j in range(len(uids)):
            label = int(labels[j])
            class_correct[label] += int(c[j])
            class_total[label] += 1
            total += 1
            correct += int(c[j])
            
            output_score = np.array2string(output_cpu[j,:],precision=8, separator='\t', formatter={'float_kind':lambda x:'%0.8f' %x})
            max_score  = sorted(output_cpu[j,:],reverse=True)[0]
            sec_score  = sorted(output_cpu[j,:],reverse=True)[1]
            dist_score = max_score - sec_score
            if dist_score < 0.4:
                confidence_level_alert += 1

            output_score = re.sub('\[|\]|\ ','',output_score)
            
            print('{:65s}'.format(uids[j]) + '\t' + str(label) + '\t' + str(predicted_cpu[j])+'\t'+str(c[j].cpu().numpy())+'\t'+output_score)
            fn.write('{:65s}'.format(uids[j])+'\t'+str(label)+'\t'+str(predicted_cpu[j])+'\t'+str(c[j].cpu().numpy())+'\t')
            fn.write(output_score + '\t' + str(dist_score) +'\n')
        
    print('{:12s}'.format('correct = ') + '{:>4d}'.format(correct))
    print('{:12s}'.format('total   = ') + '{:>4d}'.format(total))
    print('{:20s}'.format('Total test accuracy    = ') + '{:>6.2f}%'.format(100 * int(correct) / total))
    print('{:20s}'.format('confidence_level < 0.4 = ') + '{:>3d}'.format(confidence_level_alert))
    fn.write('{:12s}'.format('correct = ') + '{:>4d}'.format(correct) + '\n')
    fn.write('{:12s}'.format('total   = ') + '{:>4d}'.format(total) + '\n')
    fn.write('{:12s}'.format('Total test accuracy    = ') + '{:>6.2f}'.format(100 * int(correct) / total) + '\n') 
    fn.write('{:12s}'.format('confidence_level < 0.4 = ') + str(confidence_level_alert) + '\n')

    for i in range(len(labelname)):
        class_accurcy = float(0) if int(class_total[i]) == 0 else 100 * int(class_correct[i]) // int(class_total[i])
        print('Accuracy of {:12s} = {:>6.2f}% on {:>4.0f} , {:>4.0f}'.format(invert_labelname[i], class_accurcy, class_correct[i], class_total[i]))
        fn.write('Accuracy of {:12s} = {:>6.2f}% on {:>4.0f} , {:>4.0f}\n'.format(invert_labelname[i], class_accurcy, class_correct[i], class_total[i]))
    print('\n\n')
    fn.close()

#  ##################################################################################################################
#  #####`...#`......#####`........#######`..#..#######`...#`......#####`..#####`...#####`..########`....#############
#  ##########`..#########`..###########`..####`..##########`..#########`..#####`.#`..###`..######`.####`..###########
#  ##########`..#########`..############`..################`..#########`..#####`..#`..##`..#####`..##################
#  ##########`..#########`......##########`..##############`..#########`..#####`..##`..#`..#####`..##################
#  ##########`..#########`..#################`..###########`..#########`..#####`..###`.#`..#####`..###`....##########
#  ##########`..#########`..###########`..####`..##########`..#########`..#####`..####`.#..######`..####`.###########
#  ##########`..#########`........#######`..#..############`..#########`..#####`..######`..#######`.....#############
#  ##################################################################################################################


# call bowl_classifier_phase1_majorclass_inference.py to do testing preocedure


#  #######################################################################################################
#  ####################`..#######`..###########`.############`..#####`...#####`..#########################
#  ####################`.#`..###`...##########`.#..##########`..#####`.#`..###`..#########################
#  ####################`..#`..#`#`..#########`.##`..#########`..#####`..#`..##`..#########################
#  ####################`..##`..##`..########`..###`..########`..#####`..##`..#`..#########################
#  ####################`..###`.##`..#######`......#`..#######`..#####`..###`.#`..#########################
#  ####################`..#######`..######`..#######`..######`..#####`..####`.#..#########################
#  ####################`..#######`..#####`..#########`..#####`..#####`..######`..#########################
#  #######################################################################################################
if __name__ == '__main__':
    model_save_path = './model_save'        # [output] symbolic link to your desired output path
    #task_name       = 'Major3_P128x128_VGG16_Softmax_triple_lr_wo_drop2_alldata_direct_out_K00_EPOCH90' # [output] folder under ./model_save
    task_name       = 'test_histology' # [output] folder under ./model_save
    img_path_train  = './bowl_stage1_data/stage1_train_fix_v4_external'    # [input] training image folder
    csv_path_train  = './stage1_train_fix_v4_external.csv' # [input] option, if there is csv, we can do category filter, else select all
    #img_path_test   = './bowl_stage1_data/stage1_test'     # [input] testing  image folder
    #csv_path_test   = './stage1_test_v4_ordered.csv'       # [input] option, if there is csv, we can do category filter, else select all
    img_path_valid  = './bowl_stage1_data/stage1_train_fix_v4_external'    # [input] training image folder
    csv_path_valid  = './stage1_train_fix_v4_external.csv' # [input] option, if there is csv, we can do category filter, else select all
    valid_mode      = './model_save/' + task_name + '/test_K000_epoch003'

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action='store', choices=['train', 'valid','test'], help='train mode or testing mode')
    parser.add_argument('--task', action='store_true', help='give a name to create folder under ./model_save')
    parser.add_argument('--in_train_img', action='store_true', help='image path(kaggle format) for training data')
    parser.add_argument('--in_train_csv', action='store_true', help='image path(kaggle format) for training data')
    parser.add_argument('--ou_model_path', action='store_true', help='where is the ./model_save')
    parser.add_argument('--cv', action='store', type=cv_kn2digit, help='cross validation option, {no, loocv, max, kn: n=1...k}')
    parser.add_argument('--cv_once', action='store', type=str2bool, choices=[0, 1], help='Do K-fold CV but only do once')
    parser.add_argument('--in_valid_img', action='store', help='image path(kaggle format) for validation data')
    parser.add_argument('--in_valid_csv', action='store', help='image path(kaggle format) for validation data')
    parser.add_argument('--valid_model',  action='store', help='model for validtion data')
    epoch_checkpoint = [10,30,50,70]
    parser.set_defaults(mode= 'train',
                        task= task_name,
                        in_train_img = img_path_train,
                        in_train_csv = csv_path_train,
                        ou_model_path= model_save_path,
                        cv           = 'no',
                        cv_once      = False,
                        in_valid_img = img_path_valid,
                        in_valid_csv = csv_path_valid,
                        valid_model  = valid_mode
                        )
    args = parser.parse_args()
    #=============================================================
    if args.mode == 'train':
        # arg parameter check
        if not os.path.exists(args.in_train_img): 
            print('folder not exist: '+img_path_train) 
            exit(-1)
        if not os.path.exists(model_save_path): 
            print('folder not exist, auto-generate it for '+model_save_path)
            os.mkdir(model_save_path) 
        #generate task folder under ./model_save when trainging mode
        output_task_folder = model_save_path+'/'+args.task
        if not os.path.exists(output_task_folder): os.mkdir(output_task_folder)
        #generate dataset for training data and valid_date in training mode
        #if cv (cross validation option) is not 'no', valid_loader is not None
        #in this stage, it don't provide dataloader, dataloader will be prepared under k-fold loop
        dataset_train = prepare_dataset(args,category='Histology')
        print('argv.cv = ' + str(args.cv))
        print('epoch_checkpoint = ' + str())
        if re.search('\d+|loocv',args.cv):
            maxK = dataset_train.__len__() if args.cv == 'loocv' else min(int(args.cv),dataset_train.__len__())
            # k_fold cross validation with kth = 0 ... maxK
            start_time = time.time()
            for kth in range(maxK):
                if kth != None: print(''.join(['*']*50) + ' K = ' + '{:02d}'.format(kth) + ' ' + ''.join(['*']*50))
                train_loader, valid_loader, _ = prepare_dataloader(args,dataset_train, dataset_test=None, K_fold=maxK, K_idx=kth)
                training_model(args, train_loader, kth=kth, labelname=dataset_train.sublabels)
                for epoch in epoch_checkpoint:
                    args.valid_model = args.ou_model_path + '/' + args.task + '/' + args.task + '_K{:03d}_epoch{:03d}'.format(kth,epoch)
                    print('args.valid_model = ' + args.valid_model)
                    validing_model(args, valid_loader, kth=kth, labelname=dataset_train.sublabels)

                if args.cv_once == True:
                    print('\n************* args.cv_once is TRUE, we only do K_fold with 1st fold, skip the other fold\n')
                    break
                
            print('Finished\n--- %s seconds ---' % (time.time() - start_time))
        if args.cv == 'no':
            train_loader, _, _ = prepare_dataloader(args,dataset_train, dataset_test=None, K_fold='no',K_idx=None)
            training_model(args, train_loader, kth=None, labelname=dataset_train.sublabels)
            
    if args.mode == 'valid':
        if not os.path.isdir(args.in_valid_img):  sys.exit('no such folder: '+ args.in_valid_img) 
        if not os.path.exists(args.in_valid_csv): sys.exit('no such file : ' + args.in_valid_csv)
        if not os.path.exists(args.valid_model):  sys.exit('no such model : '+ args.valid_model)            
            
        dataset_valid      = prepare_dataset(args,category='Histology')
        _, valid_loader, _ = prepare_dataloader(args,dataset_valid, dataset_test=None, K_fold=None, K_idx=None)
        validing_model(args, valid_loader, kth=None, labelname=dataset_valid.majorlabels)
        
    if args.mode == 'test':
        print('\n *******  please use bow_classifier_phase1_majorclass_inference.py under compAl to do inference  ******* \n')
        exit(-1)





