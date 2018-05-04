import os
import re
import configparser
import argparse
import sys
def read_config():
    conf = configparser.ConfigParser()
    candidates = ['config_setting.ini']
    conf.read(candidates)
    return conf

def get_category_setting(config, Policy, req_type):
    if req_type == 'Fluorescence':
        model_type = config[Policy].get('MODEL_Fluorescence')
        model_path = config[Policy].get('MODEL_F_PATH')
    if req_type == 'Brightfield':
        model_type = config[Policy].get('MODEL_Brightfield')
        model_path = config[Policy].get('MODEL_B_PATH')
    if req_type == 'HE':
        model_type = config[Policy].get('MODEL_HE')
        model_path = config[Policy].get('MODEL_HE_PATH')
    if req_type == 'IHC':    
        model_type = config[Policy].get('MODEL_IHC')
        model_path = config[Policy].get('MODEL_IHC_PATH')
    if req_type == 'GEN':    
        model_type = config[Policy].get('MODEL_GEN')
        model_path = config[Policy].get('MODEL_GEN_PATH')        
    return model_type, model_path
        

def update_unet_config_ini(category,sub_category,model_path):
    config_path = '../DSB2018-cam-ex5/config.ini'
    f_content = []
    with open(config_path) as fn: f_content = fn.read().splitlines()
    f_content = ['model_used = '+ model_path if re.search('^model_used',x) else x for x in f_content]
    if category != 'None':
        f_content = ['category = '+ category if re.search('^category',x) else x for x in f_content]
        f_content = ['sub_category = None' if re.search('^sub_category',x) else x for x in f_content]
    if sub_category != 'None':
        f_content = ['category = None' if re.search('^category',x) else x for x in f_content]
        f_content = ['sub_category = '+ sub_category if re.search('^sub_category',x) else x for x in f_content]
        
    fn = open(config_path, "w")
    fn.write('\n'.join(f_content))
    #print('\n'.join(f_content))
    fn.close()
 
def update_mask_rcnn_config_ini(category,sub_category,model_path):
    config_path = '../mask_rcnn_v01/config.ini'
    f_content = []
    with open(config_path) as fn: f_content = fn.read().splitlines()
    f_content = ['model_used      = '+ model_path if re.search('^model_used',x) else x for x in f_content]
    if category != 'None':
        f_content = ['category        = '+ category if re.search('^category',x) else x for x in f_content]
        f_content = ['sub_category    = None' if re.search('^sub_category',x) else x for x in f_content]
    if sub_category != 'None':
        f_content = ['category        = None' if re.search('^category',x) else x for x in f_content]
        f_content = ['sub_category    = '+ sub_category if re.search('^sub_category',x) else x for x in f_content]
        
    fn = open(config_path, "w")
    fn.write('\n'.join(f_content))
    #print('\n'.join(f_content))
    fn.close() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('Policy')
    parser.add_argument('Category')

    args = parser.parse_args()   
    config = read_config() # keep the line as top as possible
    # load config information to get 
    # model_type: UNET/MASK_RCNN
    # model_path: UNET :        ../DSB2018-cam-ex5/checkpoint/XXXXX.pkl
    #             MASK_RCNN :   ../mask_rcnn_v01/model_save/XXXXX.pkl
    model_type, model_path = get_category_setting(config,args.Policy,args.Category)

    category_is_major = 1 if re.search('Fluorescence|Brightfield|GEN',args.Category) else 0
    if category_is_major == 1: 
        category = args.Category
        sub_category = 'None'
    else:
        category = 'None'
        sub_category = args.Category
    #print('category = ' + category)
    #print('sub_category = ' + sub_category)
    if model_type == 'UNET':
        update_unet_config_ini(category,sub_category,model_path)
       
    if model_type == 'MASK_RCNN':
        update_mask_rcnn_config_ini(category,sub_category,model_path)
    
    print(model_type)
    sys.exit(0)
   
