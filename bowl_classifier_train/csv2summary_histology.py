
import os
import re

MODEL_SAVE_ROOT   = './model_save/'
TASK_SET          = 'SubMajor2_stage1_final_Fixed_EPOCH90'  # *** should copy from bowl_calssifier_kaggle_major2.py ***
MODEL_NAME_HEADER = re.sub('EPOCH\d+.*','',TASK_SET)        # MODEL_NAME_HEADER = Major3_LINEAR_Softmax_K10_
MODEL_NAME_EPOCH  = re.sub(MODEL_NAME_HEADER,'',TASK_SET)   # MODEL_NAME_EPOCH  =                           EPOCH50

MODEL_SAVE_PATH = MODEL_SAVE_ROOT + TASK_SET

csv_ids = next(os.walk(MODEL_SAVE_PATH))[2]
csv_ids = list(filter(lambda x: re.search('.*csv$',x),csv_ids))

summary_dict = {}
conf_th = 0.9

for csv_item in sorted(csv_ids):
    #print(csv_item)
    fname = MODEL_SAVE_PATH + '/' + csv_item
    lines = [line.rstrip('\n') for line in open(fname)]
    
    narrow_line = list(filter(lambda x: re.search('correct 	= (.*)',x),lines))[0]
    all_correct = re.search('correct 	= (.*)',narrow_line).group(1)
    narrow_line = list(filter(lambda x: re.search('total 	= (.*)',x),lines))[0]
    all_train   = re.search('total 	= (.*)',narrow_line).group(1)
    narrow_line = list(filter(lambda x: re.search('Total test accuracy 	=(.*) \%',x),lines))[0]
    all_percent = re.search('Total test accuracy 	=(.*) \%',narrow_line).group(1)    
    narrow_line = list(filter(lambda x: (len(x.split('\t'))>=7) & (x.split('\t')[-1]!='confidence'), lines))                   # make sure raw is data format
    narrow_line_pred = list(filter(lambda x: x.split('\t')[3]=='0', narrow_line))        # mask sure raw is predict error
    narrow_line_conf = list(filter(lambda x: float(x.split('\t')[6])<conf_th, narrow_line))  # mask sure row is lower confidence

    pred_error  = '\n'.join(narrow_line_pred)
    narrow_line = list(filter(lambda x: float(x.split('\t')[6])<conf_th, narrow_line_pred))  # mask sure row is both predict error and lower confidence
    confidence  = str(len(narrow_line)) +'/'+str(len(narrow_line_conf))
    # P: percent = C/T * 100
    # C: correct number
    # T: total number       
    narrow_line = list(filter(lambda x: re.search('Accuracy of    HE',x),lines))[0]
    HP,HC,HT    = re.search('=(.*) \% on (.*), (.*)',narrow_line).group(1,2,3)
    narrow_line = list(filter(lambda x: re.search('Accuracy of   IHC',x),lines))[0]
    FP,FC,FT    = re.search('=(.*) \% on (.*), (.*)',narrow_line).group(1,2,3)
    
    csv_dict = {}
    csv_dict['all_correct'] = all_correct
    csv_dict['all_train'] = all_train
    csv_dict['all_percent'] = all_percent
    csv_dict['confidence'] = confidence
    csv_dict['pred_error'] = pred_error
    csv_dict['HP'] = HP # HE percent
    csv_dict['HC'] = HC # HE correct
    csv_dict['HT'] = HT # HE total
    csv_dict['FP'] = FP # IHC percent
    csv_dict['FC'] = FC # IHC correct
    csv_dict['FT'] = FT # IHC total

    summary_dict[csv_item] = csv_dict

report_file1 = MODEL_SAVE_PATH + '/' + 'summary_report.txt'
fn = open(report_file1, "w")

########################################################################################
#############################  ###########    ##########################################
##########################  ###  #######    ############################################
#########################  ######  ###    ##############################################
###########################  ##  ###    ################################################
#############################  ###    ####  ############################################
################################    ####  ###  #########################################
##############################    ####  #######  #######################################
############################    ########  ###  #########################################
##########################    ############  ############################################
########################################################################################
########################################################################################
fn.write('\n\n\n')
fn.write(''.join(['*']*150)+'\n')
fn.write('{:^150s}'.format('predicted error(%) for each epoch')+'\n')
fn.write(''.join(['*']*150)+'\n\n')
########################################################################################
# header
########################################################################################
K_fold_name       = [re.sub('_epoch\d+\.csv','',x) for x in sorted(summary_dict.keys())]# Major3_L2M2_K10_EPOCH50_K08_epoch050.csv
K_fold_name       = sorted(set(K_fold_name))                                            # Major3_L2M2_K10_EPOCH50_K08
short_k_fold_name = [re.sub(MODEL_NAME_HEADER,'',x) for x in K_fold_name]               #                 EPOCH50_K08

fn.write('{:10s}'.format(' '))
fn.write('|'.join(['{:^12s}'.format(x) for x in short_k_fold_name]))
fn.write('\n')

########################################################################################
# each epoch
########################################################################################
epoch_list = [re.sub(MODEL_NAME_HEADER+MODEL_NAME_EPOCH+'_K\d+_|\.csv','',x) for x in sorted(summary_dict.keys())]
epoch_list = sorted(set(epoch_list))                                         #                           _epoch050

for epoch_each in epoch_list:
    #print(epoch_each)
    fn.write('{:10s}'.format(epoch_each))
    this_line = []
    for each_k_fold in K_fold_name:
        CSV_NAME = each_k_fold + '_' + epoch_each + '.csv'
        this_line.append(summary_dict[CSV_NAME]['all_percent'])
    this_line = [ x+'%' for x in this_line]
    fn.write('|'.join(['{:^12s}'.format(x) for x in this_line]))
    fn.write('\n')

########################################################################################
#########  ####  ####        ####        ##############   ###########    ###############
#########  ####  ####  ##########  ######  ##########  ###  #######    #################
#########  ####  ####  ##########  ######  ########  ######  ####    ###################
#########  ####  ####        ####  ####  ############  ###  ###    #####################
#########        ####  ##########      ################   ##    ###  ###################
#########  ####  ####  ##########  ####  #################    ###  ###  ################
#########  ####  ####  ##########  ######  #############    ###  #######  ##############
#########  ####  ####  ##########  ######  ###########    #######  ###  ################
#########  ####  ####  ##########        ###########    ###########  ###################
########################################################################################
########################################################################################

fn.write('\n\n\n')
fn.write(''.join(['*']*150)+'\n')
fn.write('{:^150s}'.format('(H&E, IHC) predicted accuracy(%) for each epoch')+'\n')
fn.write(''.join(['*']*150)+'\n\n')
########################################################################################
# header1
########################################################################################
K_fold_name       = [re.sub('_epoch\d+\.csv','',x) for x in sorted(summary_dict.keys())]# Major3_L2M2_K10_EPOCH50_K08_epoch050.csv
K_fold_name       = sorted(set(K_fold_name))                                            # Major3_L2M2_K10_EPOCH50_K08
short_k_fold_name = [re.sub(MODEL_NAME_HEADER,'',x) for x in K_fold_name]               #                 EPOCH50_K08

fn.write('{:10s}'.format(' '))
fn.write('|'.join(['{:^12s}'.format(x) for x in short_k_fold_name]))
fn.write('\n')
########################################################################################
# header2
########################################################################################
fn.write('{:10s}'.format(' '))
category_header = ['HE','IHC']
category_header = ['{:^6s}'.format(x) for x in category_header]
sigle_unit = ''.join(category_header)
thie_line  = [sigle_unit] * len(K_fold_name)
fn.write('|'.join(['{:^12s}'.format(x) for x in thie_line]))
fn.write('\n')


########################################################################################
# category number
########################################################################################
fn.write('{:10s}'.format(' '))
this_line = []
for each_k_fold in K_fold_name:
    CSV_NAME = each_k_fold + '_' + epoch_each + '.csv'
    detail = [summary_dict[CSV_NAME]['HT'], summary_dict[CSV_NAME]['FT']]
    detail = ['{:^6s}'.format(x) for x in detail]
    this_row = ''.join(detail)
    this_line.append(this_row)
    print(CSV_NAME + '\t' + this_row)
    #this_line.append(''.join('{:>5s}'.format(x) for x in detail))
fn.write('|'.join(['{:^12s}'.format(x) for x in this_line]))
fn.write('\n')


########################################################################################
# each epoch
########################################################################################
epoch_list = [re.sub(MODEL_NAME_HEADER+MODEL_NAME_EPOCH+'_K\d+_|\.csv','',x) for x in sorted(summary_dict.keys())]
epoch_list = sorted(set(epoch_list))                                         #                           _epoch050

for epoch_each in epoch_list:
    print(epoch_each)
    fn.write('{:10s}'.format(epoch_each))
    this_line = []
    for each_k_fold in K_fold_name:
        CSV_NAME = each_k_fold + '_' + epoch_each + '.csv'
        detail = [summary_dict[CSV_NAME]['HP'], summary_dict[CSV_NAME]['FP']]
        detail = [re.sub('\.\d+','',x) for x in detail ]
        detail = [x+'%' for x in detail ]
        detail = ['{:>6s}'.format(x) for x in detail]
        this_row = ''.join(detail)
        this_line.append(this_row)
        print(CSV_NAME + '\t' + this_row)
        #this_line.append(''.join('{:>5s}'.format(x) for x in detail))
    fn.write('|'.join(['{:^12s}'.format(x) for x in this_line]))
    fn.write('\n')

########################################################################################
################################   ######   ############################################
###############################   ######   #############################################
#########################                          #####################################
#############################   #######   ##############################################
############################   #######   ###############################################
###########################   #######   ################################################
###################                              #######################################
#########################   ########   #################################################
########################   ########   ##################################################
#######################   ########   ###################################################
########################################################################################

fn.write('\n\n\n')
fn.write(''.join(['*']*150)+'\n')
fn.write('{:^150s}'.format('(H&E, IHC) predicted error counter for each epoch')+'\n')
fn.write(''.join(['*']*150)+'\n\n')
########################################################################################
# header1
########################################################################################
K_fold_name       = [re.sub('_epoch\d+\.csv','',x) for x in sorted(summary_dict.keys())]# Major3_L2M2_K10_EPOCH50_K08_epoch050.csv
K_fold_name       = sorted(set(K_fold_name))                                            # Major3_L2M2_K10_EPOCH50_K08
short_k_fold_name = [re.sub(MODEL_NAME_HEADER,'',x) for x in K_fold_name]               #                 EPOCH50_K08

fn.write('{:10s}'.format(' '))
fn.write('|'.join(['{:^12s}'.format(x) for x in short_k_fold_name]))
fn.write('\n')
########################################################################################
# header2
########################################################################################
fn.write('{:10s}'.format(' '))
category_header = ['HE','IHC']
category_header = ['{:^6s}'.format(x) for x in category_header]
sigle_unit = ''.join(category_header)
thie_line  = [sigle_unit] * len(K_fold_name)
fn.write('|'.join(['{:^12s}'.format(x) for x in thie_line]))
fn.write('\n')


########################################################################################
# category number
########################################################################################
fn.write('{:10s}'.format(' '))
this_line = []
for each_k_fold in K_fold_name:
    CSV_NAME = each_k_fold + '_' + epoch_each + '.csv'
    detail = [summary_dict[CSV_NAME]['HT'], summary_dict[CSV_NAME]['FT']]
    detail = ['{:^6s}'.format(x) for x in detail]
    this_row = ''.join(detail)
    this_line.append(this_row)
    print(CSV_NAME + '\t' + this_row)
    #this_line.append(''.join('{:>5s}'.format(x) for x in detail))
fn.write('|'.join(['{:^12s}'.format(x) for x in this_line]))
fn.write('\n')


########################################################################################
# each epoch
########################################################################################
epoch_list = [re.sub(MODEL_NAME_HEADER+MODEL_NAME_EPOCH+'_K\d+_|\.csv','',x) for x in sorted(summary_dict.keys())]
epoch_list = sorted(set(epoch_list))                                         #                           _epoch050

for epoch_each in epoch_list:
    print(epoch_each)
    fn.write('{:10s}'.format(epoch_each))
    this_line = []
    epoch_pred_error_num = 0
    for each_k_fold in K_fold_name:
        CSV_NAME = each_k_fold + '_' + epoch_each + '.csv'
        detail[0] = str(int(summary_dict[CSV_NAME]['HT'])-int(summary_dict[CSV_NAME]['HC']))
        detail[1] = str(int(summary_dict[CSV_NAME]['FT'])-int(summary_dict[CSV_NAME]['FC']))
        epoch_pred_error_num += int(detail[0]) + int(detail[1])
        detail = ['{:^6s}'.format(x) for x in detail]
        this_row = ''.join(detail)
        this_line.append(this_row)
        print(CSV_NAME + '\t' + this_row)
        #this_line.append(''.join('{:>5s}'.format(x) for x in detail))
    
    fn.write('|'.join(['{:^12s}'.format(x) for x in this_line]))
    fn.write(' ('+str(epoch_pred_error_num)+')')
    fn.write('\n')

########################################################################################
###################################    ######    #######################################
#################################       ###       ######################################
###############################          #         #####################################
##############################                      ####################################
##############################                      ####################################
###############################                    #####################################
#################################                #######################################
###################################            #########################################
#####################################        ###########################################
######################################     #############################################
######################################## ###############################################
fn.write('\n\n\n')
fn.write(''.join(['*']*150)+'\n')
show_string = 'predicted error and confidence level < '+str(conf_th)+' ( (pred_error & conf_level < '+str(conf_th)+')/ (conf_level < '+ str(conf_th)+')'
fn.write('{:^150s}'.format(show_string)+'\n')
fn.write(''.join(['*']*150)+'\n\n')
########################################################################################
# header
########################################################################################
K_fold_name       = [re.sub('_epoch\d+\.csv','',x) for x in sorted(summary_dict.keys())]# Major3_L2M2_K10_EPOCH50_K08_epoch050.csv
K_fold_name       = sorted(set(K_fold_name))                                            # Major3_L2M2_K10_EPOCH50_K08
short_k_fold_name = [re.sub(MODEL_NAME_HEADER,'',x) for x in K_fold_name]               #                 EPOCH50_K08

fn.write('{:10s}'.format(' '))
fn.write('|'.join(['{:^12s}'.format(x) for x in short_k_fold_name]))
fn.write('\n')

########################################################################################
# each epoch
########################################################################################
epoch_list = [re.sub(MODEL_NAME_HEADER+MODEL_NAME_EPOCH+'_K\d+_|\.csv','',x) for x in sorted(summary_dict.keys())]
epoch_list = sorted(set(epoch_list))                                         #                           _epoch050

for epoch_each in epoch_list:
    #print(epoch_each)
    fn.write('{:10s}'.format(epoch_each))
    this_line = []
    this_line_denominator = []
    this_line_numerator =[]    
    for each_k_fold in K_fold_name:
        CSV_NAME = each_k_fold + '_' + epoch_each + '.csv'
        this_line.append(summary_dict[CSV_NAME]['confidence'])
        this_line_denominator.append(re.sub('.*/','',summary_dict[CSV_NAME]['confidence']))
        this_line_numerator.append(re.sub('/.*','',summary_dict[CSV_NAME]['confidence']))
    this_line = [ x for x in this_line]
    fn.write('|'.join(['{:^12s}'.format(str(x)) for x in this_line]))
    print('\t(%d/%d)\t' %(sum(list(map(int,this_line_numerator))),sum(list(map(int,this_line_denominator)))))
    
    fn.write('\t(%d/%d)\t' %(sum(list(map(int,this_line_numerator))),sum(list(map(int,this_line_denominator)))))
    fn.write('\n')
fn.write('\n')

########################################################################################
# show all predicted error file name on each epoch
########################################################################################
fn.write('\n\n\n')
fn.write(''.join(['*']*150)+'\n')
fn.write('【predicted error for each epoch】\n')
fn.write(''.join(['*']*150)+'\n\n')

for epoch_each in epoch_list:
    #print(epoch_each)
    fn.write('{:10s}'.format(epoch_each)+'\n')
    this_line = []

    for each_k_fold in K_fold_name:
        CSV_NAME = each_k_fold + '_' + epoch_each + '.csv'
        print('CSV_NAME '+ CSV_NAME)
        fn.write('['+re.sub('.*/|\.csv','',CSV_NAME)+']\n')
        #this_line.append(summary_dict[CSV_NAME]['conf_detail'])
        fn.write(summary_dict[CSV_NAME]['pred_error']+'\n\n')
    fn.write('\n\n')        

fn.close()

