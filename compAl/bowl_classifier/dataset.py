
# coding: utf-8

# In[1]:


import os
import random
import json

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as tx

from PIL import Image, ImageOps
from skimage.io import imread
from skimage import filters, img_as_ubyte
from skimage.morphology import remove_small_objects, dilation, erosion
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.filters import gaussian_filter
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
import matplotlib.pyplot as plt
# Ignore skimage convertion warnings
import warnings
warnings.filterwarnings("ignore")


class KaggleDataset(Dataset):
    """Kaggle dataset."""

    def __init__(self, root, transform=None, cache=None, category=None, source=None, img_folder=None,resize_scale=[512,512]):
        """
        Args:
            root_dir (string): Directory of data (train or test).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root        = root
        self.transform   = transform
        self.majorlabels = {'Histology': 0, 'Fluorescence':1, 'Brightfield':2}
        self.sublabels   = {'Histology': 0, 'Fluorescence': 0, 'Brightfield': 0, 'Cloud':0 , 'HE': 0, 'IHC': 1, 'Drosophilidae': 0} # 6
        self.img_folder  = img_folder
        self.resize_scale= resize_scale
        print('root ' + root)
        if os.path.isfile(root+'.csv'):
            df = pd.read_csv(root+'.csv')
                               
            if (source != None) & ('source' in df.columns):
                ok = df['source'] == source
                df = df[ok]
               
            if category != None:
                ok = df['major_category'] == category
                df = df[ok]

            # TCGA_ClorNorm is same as TCGA, should be removed to avoid look answer in traing data
            if 'source' in df.columns:
                ok = df['source'] != 'TCGA_ColorNorm'
                df = df[ok]                  
            else:
                df['source'] = pd.Series('Test', index=df.index)
            self.ids = list(df['image_id'])
            self.label = [self.majorlabels[x] if x != 'None' else 'None' for x in list(df['major_category'])]
            self.sublabel = [self.sublabels[x] if x != 'None' else 'None' for x in list(df['sub_category'])]


            self.source = list(df['source'])

        else:
            self.ids = next(os.walk(root))[1]
        #self.ids.sort()
        self.cache = cache

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        try:
            uid = self.ids[idx]
            label = self.label[idx]
            sublabel = self.sublabel[idx]
            source = self.source[idx]
        except:
            raise IndexError()

        if self.cache is not None and uid in self.cache:
            sample = self.cache[uid]
        else:
            #img_name = os.path.join('../mix_kaggle_external_data/data/stage1_train_fix_v4_external', uid, 'images', uid + '.png')
            img_name = os.path.join(self.img_folder, uid, 'images', uid + '.png')
            image = Image.open(img_name)
            # ignore alpha channel if any, because they are constant in all training set
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # resize image for model
            image = image.resize(self.resize_scale, Image.ANTIALIAS)
            # overlay masks to single mask
            w, h = image.size

            sample = {'image': image,
                      'label': label,
                      'sublabel': sublabel,
                      'uid': uid,
                      'source': source}
            
            if self.cache is not None:
                self.cache[uid] = sample
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def split(self,K_fold=0):
        # get list of dataset index
        n = len(self.ids)
        indices = list(range(n))
        # random shuffle the list
        
        s = random.getstate()
        random.seed(10)
        random.shuffle(indices)
        random.setstate(s)
        
        # return splitted lists
        split = int(np.floor(0.1 * n))
        #return indices[split:], indices[:split]    
        kfold_beg = K_fold * split
        kfold_end = kfold_beg + split
         
        Testing_idx = indices[kfold_beg:kfold_end]
        Training_idx = [x for x in range(n) if x not in Testing_idx]
        
        return Training_idx,Testing_idx

    def split_loocv(self,K_fold=0):   
        # get list of dataset index
        n = len(self.ids)
        indices = list(range(n))
        # random shuffle the list
        s = random.getstate()
        random.seed(10)
        random.shuffle(indices)
        random.setstate(s)
        
        Testing_idx = [indices[K_fold]]
        print('K_fold = '+str(K_fold)+'\tTesting_idx = '+str(Testing_idx))
        Training_idx = [x for x in range(n) if x not in Testing_idx]
        
        return Training_idx,Testing_idx        

    def over_sampling(self, train_idx=None, desired_source='Kaggle', desired_class_name='Brightfield', m_time=None):
        over_sample_idx = []
        class_label = self.majorlabels[desired_class_name]

        for idx in train_idx:
            uid = self.ids[idx]
            label = self.label[idx]
            sublabel = self.sublabel[idx]
            source = self.source[idx]
            #print('%3d self.source[idx] = %10s, self.label[idx] = %d, self.uid[idx] = %s' %(idx,source,label,uid))
            if (source == desired_source) & (label == class_label):
                over_sample_idx.append(idx)
        over_sample_idx = over_sample_idx * m_time
        train_idx += over_sample_idx
        return train_idx


class Compose():
    def __init__(self, augment=False, padding=False, tensor=True):
        self.tensor = tensor
        self.augment = augment
        self.padding = padding


    def __call__(self, sample):
        image, label, sublabel, uid, source = sample['image'], sample['label'],sample['sublabel'], sample['uid'], sample['source']
 
        # perform ToTensor()
        if self.tensor:
            #image, label, uid = [tx.to_tensor(x) for x in (image, label, uid)]
            image = tx.to_tensor(image)
            
            label = label
            sublabel = sublabel 
            
            # perform Normalize()
            #image = tx.normalize(image, self.mean, self.std)
            #image = tx.normalize(image, 0.5, 0.5)

        # prepare a shadow copy of composed data to avoid screwup cached data
        x = sample.copy()
        x['image'], x['label'],x['sublabel'], x['uid'], x['source'] = image, label, sublabel, uid, source
        #x['image'], x['sublabel'], x['uid'] = image, sublabel, uid

        return x


if __name__ == '__main__':
    # over sampling testing
    dataset = KaggleDataset('./stage1_train_fix_v4_external',transform=Compose(), resize_scale=[128,128])
    train_idx, valid_idx = dataset.split()
    print('train_idx len ' + str(len(train_idx)))
    print('train_idx = '+str(train_idx))
    dataset.over_sampling(train_idx,'Kaggle','Brightfield',m_time=2)
    print('train_idx len ' + str(len(train_idx)))
    train_loader = DataLoader(dataset, sampler=SubsetRandomSampler(train_idx),batch_size=4,num_workers=2) 
       
    print('{:<3s} {:<15s} {:>5s} {:<65s}'.format('id','source','sublabel', 'uid'))  
    count = 0
    for i, data in enumerate(train_loader, 0):
        inputs = data['image']
        labels = data['label']
        sublabels = data['sublabel']
        uids   = data['uid']
        sources = data['source']
        #print('valid len = '+ str(len(labels)))
        for j in range(len(inputs)):
            if (sources[j] == 'Kaggle') & (labels[j].cpu().numpy()==2) :
                print('{:<3d} {:<15s} {:>5d} {:<65s}'.format(i,sources[j], labels[j].cpu().numpy(), uids[j]))
                count += 1       
    print('all count for Kaggle BrightField is ' + str(count))

    # LOOCV testing
    #dataset = KaggleDataset('./stage1_train_fix_v4_external',transform=Compose(), category='Brightfield',resize_scale=[128,128])
    #dataset = KaggleDataset('./stage1_train_fix_v4_external',transform=Compose(),category='Histology',resize_scale=[128,128])


'''   
    for k in range(dataset.__len__()):
    #dataset = KaggleDataset('./stage1_train_fix_v4_external',transform=Compose())
    #dataset = KaggleDataset('./stage1_train_fix_v4_external',transform=Compose(), category='Histology',resize_scale=[128,128])
    #train_idx, valid_idx = dataset.split(K_fold=k)
        train_idx, valid_idx = dataset.split_loocv(K_fold=k)
        #print('train_idx len() = ' + str(len(train_idx)) + ',valid_idx len() =' + str(len(valid_idx)) )
        print('\ntrain_idx = ' + str(train_idx) + ',valid_idx len() =' + str(valid_idx[0]))
        #print('valid_idx len() =' + str(valid_idx[0]))
        train_loader = DataLoader(dataset, sampler=SubsetRandomSampler(train_idx),batch_size=4,num_workers=2)
        test_loader = DataLoader(dataset, sampler=SubsetRandomSampler(valid_idx),batch_size=4,num_workers=2)
        print('{:<3s} {:<15s} {:>5s} {:<65s}'.format('id','source','sublabel', 'uid'))   
        for i, data in enumerate(test_loader, 0):
            inputs = data['image']
            labels = data['label']
            sublabels = data['sublabel']
            uids   = data['uid']
            sources = data['source']
            print('valid len = '+ str(len(labels)))
            for j in range(len(inputs)):
                if sources[j] != None:
                    print('{:<3d} {:<15s} {:>5d} {:<65s}'.format(i,sources[j], sublabels[j].cpu().numpy(), uids[j]))       
        break
'''
'''   
    print('datalen ' + str(dataset.__len__()))
    print('(Train_idx, Valid_idx ( %d, %d) -> %d' %(len(train_idx),len(valid_idx), (len(train_idx)+len(valid_idx))))
    train_idx = dataset.over_sampling(train_idx,'Brightfield',m_time=3)
    print('( %d, %d) -> %d' %(len(train_idx),len(valid_idx), (len(train_idx)+len(valid_idx))))

    trainloader = torch.utils.data.DataLoader(dataset,batch_size=4, shuffle=True, num_workers=2)

    print('{:<3s} {:<15s} {:>5s} {:<65s}'.format('id','source','sublabel', 'uid'))   
    for i, data in enumerate(trainloader, 0):
        inputs = data['image']
        labels = data['label']
        sublabels = data['sublabel']
        uids   = data['uid']
        sources = data['source']

        for j in range(len(inputs)):
            if sources[j] != None:
                print('{:<3d} {:<15s} {:>5d} {:<65s}'.format(i,sources[j], sublabels[j].cpu().numpy(), uids[j]))

'''
'''
    print('idx ' + str(idx))
    sample = dataset[idx]

    invert_majorlabel = {v:k for k,v in dataset.majorlabels.items()}
    invert_sublabel = {v:k for k,v in dataset.sublabels.items()}
    print('sample[uid] = ' + sample['uid'] )
    print('sample[label] = ' + str(sample['label']) + '('+ invert_majorlabel[sample['label']] +')')
    print('sample[sublabel] = ' + str(sample['sublabel']) + '('+ invert_sublabel[sample['sublabel']]+')')
'''

'''
    trainloader = torch.utils.data.DataLoader(dataset,batch_size=4, shuffle=True, num_workers=2)
    for i, data in enumerate(trainloader, 0):
        inputs = data['image']
        labels = data['label']
        sublabels = data['sublabel']
        uids   = data['uid']
        source = data['source']
        #print(uids)
        break
    print('trainloader')
    print('uid = ' + uids[0])
    print('label = ' + str(int(labels[0])))
    print('sublabels = ' + str(int(sublabels[0])))
    plt.figure(figsize=(8,6))
    img = inputs[0].numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()
'''

'''
train_idx, valid_idx = dataset.split()
train_loader = DataLoader(dataset, sampler=SubsetRandomSampler(train_idx),batch_size=4,num_workers=2)
valid_loader = DataLoader(dataset, sampler=SubsetRandomSampler(valid_idx),batch_size=4,num_workers=2)
print(len(dataset))
print(len(train_loader))
print(len(valid_loader))




for i, data in enumerate(train_loader, 0):
    inputs = data['image']
    labels = data['label']
    uids   = data['uid']
    sublabels = data['sublabel']
    print(uids)
    break



inputs[0].numpy()

plt.figure(figsize=(8,6))
img = inputs[0].numpy()
plt.imshow(np.transpose(img, (1, 2, 0)))
'''
