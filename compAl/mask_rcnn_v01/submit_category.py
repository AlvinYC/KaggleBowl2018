import os, sys
sys.path.append(os.path.dirname(__file__))
import json
import cv2
import collections
import ntpath
from pathlib import Path
from torch.nn.modules.module import _addindent

from common import *
from net.metric import run_length_encode
from net.resnet50_mask_rcnn.configuration import Configuration
from net.resnet50_mask_rcnn.model import MaskRcnnNet
from dataset import augment
from dataset.reader import ScienceDataset, multi_mask_to_color_overlay
from net.resnet50_mask_rcnn.draw import multi_mask_to_contour_overlay
from utility.file import Logger, backup_project_as_zip
from utility.draw import draw_rcnn_detection_nms
from utility.helper import rle_decode, dsb_iou_metric2, label_masks,\
                            ckpt_path, load_ckpt, revert, evaluate_IoU,\
                            rle2png_fullresolutionWithContour, filter_small,\
                            filter_fiber
import train

def submit_collate(batch):
    batch_size = len(batch)
    #for b in range(batch_size): print (batch[b][0].size())
    inputs    = torch.stack([batch[b][0]for b in range(batch_size)], 0)
    images    =             [batch[b][1]for b in range(batch_size)]
    indices   =             [batch[b][2]for b in range(batch_size)]
    id        =             [batch[b][3]for b in range(batch_size)]
    return [inputs, images, indices, id]

def get_submit_dir(evaluate_mode):
    out_dir = TASK_OUTDIR
    predict_model_ckpt = Path(ckpt_path(out_dir)).stem
    #submit_folder = '/' + predict_model_ckpt + '_' + evaluate_mode
    #submit_folder = '/'
    #submit_dir = out_dir + '/submit' + submit_folder
    #submit_dir =  './results' + submit_folder
    submit_dir =  './results'
    return submit_dir

#--------------------------------------------------------------
def run_submit(evaluate_mode):
    c = config['submit']
    n_worker = c.getint('n_worker')
    data_src = json.loads(c.get('data_src'))
    data_major = json.loads(c.get('data_major'))
    data_sub = json.loads(c.get('data_sub'))
    cc = config['maskrcnn']
    class_map = json.loads(cc.get('classes_map'))
    # generate metafiles such as /npys and /overlays
    out_dir = TASK_OUTDIR
    submit_dir = get_submit_dir(evaluate_mode)
    # initial_checkpoint = PREDICT_CP_FILE

    os.makedirs(submit_dir +'/overlays', exist_ok=True)
    os.makedirs(submit_dir +'/npys', exist_ok=True)
    #os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    #os.makedirs(out_dir +'/backup', exist_ok=True)
    #backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.evaluate.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')

    ## net ------------------------------
    cfg = Configuration()
    net = MaskRcnnNet(cfg).cuda()
    epoch = load_ckpt(out_dir, net)
    if epoch == 0:
        print("Aborted: checkpoint not found!")
        return
    '''
    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
    '''
    # print(torch_summarize(net))
    log.write('%s\n\n'%(type(net)))
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    if evaluate_mode == 'test':
        #output_csv_path = DATA_DIR +'/split/test.csv'
        #output_csv_path = '../bowl_classifier/stage2_test.csv'
        output_csv_path = config['param'].get('CSV_PATH')
        print('output_csv_path ==> ' + output_csv_path)
        print(config['param'].get('category'))
        print(config['param'].get('sub_category'))
        test_csv = pd.read_csv(output_csv_path)
        if config['param'].get('category') != 'None':
            test_csv = test_csv[test_csv['major_category']==config['param'].get('category')]
        if config['param'].get('sub_category') != 'None':
            test_csv = test_csv[test_csv['sub_category']==config['param'].get('sub_category')]
        if (config['param'].get('category') != 'None') & (config['param'].get('sub_category') != 'None'):
            print('[compAI error], dont supprt filter both major category and sub category')
        #test_csv = test_csv[test_csv['major_category']=='Histology']
        
    print(output_csv_path)

    print(test_csv.head())
    test_dataset = ScienceDataset(
                                test_csv, mode='test',
                                # 'train1_ids_gray_only1_500', mode='test',
                                #'valid1_ids_gray_only1_43', mode='test',
                                #'debug1_ids_gray_only_10', mode='test',
                                # 'test1_ids_gray2_53', mode='test',
                                transform = augment.submit_augment)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler = SequentialSampler(test_dataset),
                        batch_size  = 1,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn  = submit_collate)

    log.write('\ttest_dataset.split = %s\n'%(output_csv_path))
    log.write('\tlen(test_dataset)  = %d\n'%(len(test_dataset)))
    log.write('\n')

    ## start evaluation here! ##############################################
    log.write('** start evaluation here! **\n')
    start = timer()
    pred_masks = []
    true_masks = []
    IoU = []
    label_counts = []
    predicts_counts = []
    predict_image_labels =[]
    total_prediction  =[]
    confidence = []
    test_num  = len(test_loader.dataset)
    for i, (inputs, images, indices, ids) in enumerate(test_loader, 0):
#        print('\rpredicting: %10d/%d (%0.0f %%)  %0.2f min'%(i, test_num-1, 100*i/(test_num-1),
#                         (timer() - start) / 60), end='',flush=True)
        print('\rpredicting: %10d/%d (%0.0f %%)  %0.2f min'%(i+1, test_num, 100*i/(test_num),
                         (timer() - start) / 60), end='',flush=True)

        time.sleep(0.01)
        net.set_mode('test')

        with torch.no_grad():
            inputs = Variable(inputs).cuda()
            net(inputs )
            revert(net, images) #unpad, undo test-time augment etc ....

        ##save results ---------------------------------------
        batch_size = len(indices)
        assert(batch_size==1)  #note current version support batch_size==1 for variable size input
                               #to use batch_size>1, need to fix code for net.windows, etc

        batch_size,C,H,W = inputs.size()
        inputs = inputs.data.cpu().numpy()

        window          = net.rpn_window
        rpn_probs_flat   = net.rpn_probs_flat.data.cpu().numpy()
        rpn_logits_flat = net.rpn_logits_flat.data.cpu().numpy()
        rpn_deltas_flat = net.rpn_deltas_flat.data.cpu().numpy()
        detections = net.detections
        rpn_proposals = net.rpn_proposals
        # print ('detections shape', detections.shape)
        masks      = net.masks
        keeps       = net.keeps
        category_labels  = net.category_labels
        label_sorteds = net.label_sorted
        # print ('masks shape', len(masks))
        # print ('batch_size', batch_size)
        for b in range(batch_size):
            #image0 = (inputs[b].transpose((1,2,0))*255).astype(np.uint8)

            image  = images[b]
            height,width = image.shape[:2]
            # print ('hw', height, width)
            mask   = masks[b]
            keep   = keeps[b]
            category_label = category_labels[b]
            # label_sorted = np.asarray(list(label_sorted[b]))
            label_sorted = label_sorteds[b]
            # print ('sum_label',sum_label )
            sum_label = 0
            for i in range(int((len(label_sorted)/2))):
                sum_label = sum_label + label_sorted[i*2+1]
            # category_label = []
            # print ('category_label', category_label)
            if category_label == []:
                category_image = 'NAN'
                nms_label_count = 0
            else:
                category_image = [key for key,value in class_map.items() if value == category_label][0]
                nms_label_count  = label_sorted[1]
            pred_masks.append(mask)
            predict_image_labels.append(category_image)
            if evaluate_mode == 'train':
                IoU_one, label_counts_one, predicts_counts_one = evaluate_IoU(mask, true_mask)
                IoU.append(IoU_one)
                label_counts.append(label_counts_one)
                predicts_counts.append(predicts_counts_one)
            confidence_one   =  round(nms_label_count / (sum_label+0.0000001), 4)
            confidence.append(confidence_one)
            prob  = rpn_probs_flat[b]
            delta = rpn_deltas_flat[b]
            # detection = detections[b]
            image_rcnn_detection_nms = draw_rcnn_detection_nms(image, detections, threshold=0.1)
            # image_rpn_proposal_before_nms = draw_rpn_proposal_before_nms(image,prob,delta,window,0.995)
            image_rpn_detection_nms = draw_rcnn_detection_nms(image, rpn_proposals, threshold=0.1)

            contour_overlay  = multi_mask_to_contour_overlay(cfg, mask, detections,keep,  image, color=[0,255,0])
            color_overlay    = multi_mask_to_color_overlay(mask, color='summer')
            if evaluate_mode == 'train':
                color_overlay_true    = multi_mask_to_color_overlay(true_mask, image, color='summer')
            color_overlay    = multi_mask_to_color_overlay(mask, color='summer')
            color1_overlay   = multi_mask_to_contour_overlay(cfg, mask, detections, keep, color_overlay, color=[255,255,255])
            image_rcnn_detection_nms = image_rcnn_detection_nms[:height,:width]
            # image_rpn_proposal_before_nms = image_rpn_proposal_before_nms[:height,:width]
            if evaluate_mode == 'train':
                #all = np.hstack((image,contour_overlay, image_rpn_detection_nms, image_rcnn_detection_nms, image_rpn_detection_nms, color1_overlay, color_overlay_true))
                all = np.hstack((image,image_rpn_detection_nms, image_rcnn_detection_nms, image_rpn_detection_nms, contour_overlay, color_overlay_true))
            else:
                all = np.hstack((image, color1_overlay, image_rpn_detection_nms, image_rcnn_detection_nms, contour_overlay))

            # --------------------------------------------
            id = test_dataset.ids[indices[b]]
            name =id.split('/')[-1]

            #draw_shadow_text(overlay_mask, 'mask',  (5,15),0.5, (255,255,255), 1)
            np.save(submit_dir + '/npys/%s.npy'%(name),mask)
            #cv2.imwrite(out_dir +'/submit/npys/%s.png'%(name),color_overlay)

            # always save overlay images
            cv2.imwrite(submit_dir +'/overlays/%s.png'%(name),all)

            #psd
            os.makedirs(submit_dir +'/psds/%s'%name, exist_ok=True)
            cv2.imwrite(submit_dir +'/psds/%s/%s.png'%(name,name),image)
            cv2.imwrite(submit_dir +'/submit/psds/%s/%s.mask.png'%(name,name),color_overlay)
            cv2.imwrite(submit_dir +'/submit/psds/%s/%s.contour.png'%(name,name),contour_overlay)

            # image_show('all',all)
            # image_show('image',image)
            # image_show('multi_mask_overlay',multi_mask_overlay)
            # image_show('contour_overlay',contour_overlay)
            # cv2.waitKey(1)

    assert(test_num == len(test_loader.sampler))
    log.write('initial_checkpoint  = %s\n'%(Path(ckpt_path(out_dir)).name))
    log.write('test_num  = %d\n'%(test_num))
    log.write('\n')
    if evaluate_mode == 'train':
        ids = test_csv['image_id']
        label_column = config['train'].get('label_column')
        major_category = test_csv['major_category']
        sub_category = test_csv['sub_category']
        # answer = []
        answer = predict_image_labels == major_category
        print ('answer', answer)
        print ('predict_image_labels', predict_image_labels)
        df_predict = pd.DataFrame({'image_id' : ids, 'pred_mask': pred_masks , 'true_mask': true_masks,
                                   'major_category' : major_category, 'sub_category' : sub_category,'IoU' : IoU ,
                                   'label_counts' : label_counts, 'predicts_counts' : predicts_counts,
                                   'predict_category': predict_image_labels, 'yes_or_no': answer, 'confidence': confidence})
        # df_predict= df_predict.assign(label_counts=0)
        # df_predict= df_predict.assign(predicts_counts=0)
        # df_predict= df_predict.assign(ap=0)
        # for i in range(df_predict.shape[0]):
        #     df_predict.loc[i, ['ap', 'label_counts', 'predicts_counts']]= evaluate_water(df_predict.loc[:,'pred_mask'].values.tolist()[i], df_predict.loc[:,'true_mask'].values.tolist()[i])
        IoU_mean = df_predict['IoU']
        IoU_his = df_predict.loc[df_predict['major_category']=='Histology', ['IoU']]
        IoU_flo = df_predict.loc[df_predict['major_category']=='Fluorescence', ['IoU']]
        IoU_bri = df_predict.loc[df_predict['major_category']=='Brightfield', ['IoU']]
        # print ('Major Category IoU:\n')
        # print ('IoU(%d):'%len(IoU_mean),IoU_mean.mean())
        # print ('IoU_Histology(%d):'%len(IoU_his),IoU_his.mean().values[0])
        # print ('IoU_Fluorescence(%d):'%len(IoU_flo),IoU_flo.mean().values[0])
        # print ('IoU_Brightfield(%d):'%len(IoU_bri),IoU_bri.mean().values[0])
        log.write('Major Category IoU:\n')
        log.write('IoU(%d):%s\n'%(len(IoU_mean),IoU_mean.mean()))
        log.write('IoU_Histology(%d):%s\n'%(len(IoU_his),IoU_his.mean().values[0]))
        log.write('IoU_Fluorescence(%d):%s\n'%(len(IoU_flo),IoU_flo.mean().values[0]))
        log.write('IoU_Brightfield(%d):%s\n'%(len(IoU_bri),IoU_bri.mean().values[0]))
        log.write('\n')
        IoU_he = df_predict.loc[df_predict['sub_category']=='HE', ['IoU']]
        IoU_flo_sub = df_predict.loc[df_predict['sub_category']=='Fluorescence', ['IoU']]
        IoU_bri_sub = df_predict.loc[df_predict['sub_category']=='Brightfield', ['IoU']]
        IoU_clo_sub = df_predict.loc[df_predict['sub_category']=='Cloud', ['IoU']]
        IoU_dro_sub = df_predict.loc[df_predict['sub_category']=='Drosophilidae', ['IoU']]
        IoU_ihc_sub = df_predict.loc[df_predict['sub_category']=='IHC', ['IoU']]
        # print ('Sub Category IoU:\n')
        log.write('Sub Category IoU:\n')
        # print ('IoU_he(%d):'%len(IoU_he),IoU_he.mean().values[0])
        # print ('IoU_Fluorescence(%d):'%len(IoU_flo_sub),IoU_flo_sub.mean().values[0])
        # print ('IoU_Brightfield(%d):'%len(IoU_bri_sub),IoU_bri_sub.mean().values[0])
        # print ('IoU_Cloud(%d):'%len(IoU_clo_sub),IoU_clo_sub.mean().values[0])
        # print ('IoU_Drosophilidae(%d):'%len(IoU_dro_sub),IoU_dro_sub.mean().values[0])
        # print ('IoU_IHC(%d):'%len(IoU_ihc_sub),IoU_ihc_sub.mean().values[0])
        log.write('IoU_he(%d):%s\n'%(len(IoU_he),IoU_he.mean().values[0]))
        log.write('IoU_Fluorescence(%d):%s\n'%(len(IoU_flo_sub),IoU_flo_sub.mean().values[0]))
        log.write('IoU_Brightfield(%d):%s\n'%(len(IoU_bri_sub),IoU_bri_sub.mean().values[0]))
        log.write('IoU_Cloud(%d):%s\n'%(len(IoU_clo_sub),IoU_clo_sub.mean().values[0]))
        log.write('IoU_Drosophilidae(%d):%s\n'%(len(IoU_dro_sub),IoU_dro_sub.mean().values[0]))
        log.write('IoU_IHC(%d):%s\n'%(len(IoU_ihc_sub),IoU_ihc_sub.mean().values[0]))
        log.write('\n')

        df_predict = df_predict.drop(['pred_mask', 'true_mask'], axis=1)
        # print (df_predict)

        prediction_csv_file = submit_dir + '/prediction.csv'
        print('prediction_csv_file ==> '+prediction_csv_file)
        df_predict.to_csv(prediction_csv_file)
    else:
        ids = test_csv['image_id']
        df_predict = pd.DataFrame({'image_id' : ids, 'predict_image_labels': predict_image_labels, 'confidence': confidence})
        prediction_csv_file = submit_dir + '/prediction.csv'
        df_predict.to_csv(prediction_csv_file)
        

def run_npy_to_sumbit_csv(evaluate_mode, submit_subset=None):
    print('=============')
    task_name = config['train'].get('log_name')
    submit_dir = get_submit_dir(evaluate_mode)
    npy_dir = Path(submit_dir) / 'npys'
    visualization_dir = submit_dir + '/visualization/'
    os.makedirs(visualization_dir, exist_ok=True)

    if submit_subset is None:
        rle_file = submit_dir + '/submission_' + evaluate_mode + '_' + task_name + '.csv'
    else:
        rle_file = submit_dir + '/submission_' + evaluate_mode + '_' + task_name + ntpath.basename(submit_subset) + '.csv'
    print('Output submission file :', rle_file)

    ## start -----------------------------
    all_num=0
    cvs_ImageId = []
    cvs_EncodedPixels = []

    output_csv_path = config['param'].get('CSV_PATH')
    test_csv = pd.read_csv(output_csv_path)

    test_csv = pd.read_csv(output_csv_path)
    if config['param'].get('category') != 'None':
        test_csv = test_csv[test_csv['major_category']==config['param'].get('category')]
    if config['param'].get('sub_category') != 'None':
        test_csv = test_csv[test_csv['sub_category']==config['param'].get('sub_category')]
    if (config['param'].get('category') != 'None') & (config['param'].get('sub_category') != 'None'):
        print('[compAI error], dont supprt filter both major category and sub category')

    ids = test_csv['image_id'].values.tolist()
    if submit_subset is not None:
        subset_csv = test_csv[ test_csv['sub_category'] == submit_subset ].reset_index()
        ids = subset_csv['image_id'].values.tolist()
        print('Use subset list', submit_subset, ',', len(subset_csv))

    check_names = []
    for npy_file in npy_dir.glob('*.npy'):
        uid = npy_file.stem
        check_names.append(uid)
        multi_mask = np.load(npy_file).astype(int)

        #<todo> ---------------------------------
        #post-processing here
        multi_mask = filter_small(multi_mask, 20)
        multi_mask = filter_fiber(multi_mask)
        #<todo> ---------------------------------

        num = int( multi_mask.max())
        for m in range(num):
            rle = run_length_encode(multi_mask==m+1)
            cvs_ImageId.append(uid)
            cvs_EncodedPixels.append(rle)

        all_num += num
        #<debug> ------------------------------------
        #print(all_num, num)  ##GT is 4152?

    print('Total Number of masks in',len(check_names),'images: ',all_num)  ##GT is 4152?
    null_count = 0
    if evaluate_mode == 'test':
        ALL_TEST_IMAGE_ID = test_csv.image_id
        for t in ALL_TEST_IMAGE_ID:
            if t not in check_names:
                null_count = null_count + 1
                #print('Write null for ',t,'(',null_count,'): part of 65 stage1_test images')
                cvs_ImageId.append(t)
                cvs_EncodedPixels.append('') #null
        print('Check',len(check_names),'images for submission.  Write null for',null_count,'images in kaggle stage1_test')

    df = pd.DataFrame({ 'ImageId' : cvs_ImageId , 'EncodedPixels' : cvs_EncodedPixels})
    df.to_csv(rle_file, index=False, columns=['ImageId', 'EncodedPixels'])
    print('Saving',rle_file)
    print('=============')

    # create full resolution images if submit_submit == None
    if submit_subset is None:
        print("Create full resolution images...")
        '''
        if evaluate_mode == 'train':
            pred_dir = DATA_DIR + '/__download__/stage1_train/'
        else:
            pred_dir = DATA_DIR + '/__download__/stage1_test/'
        '''    
        pred_dir = config['param'].get('img_folder')
        fullres_dir = visualization_dir + 'fullres/'
        rle2png_fullresolutionWithContour(rle_file, pred_dir, fullres_dir)


def generate_html(evaluate_mode, submit_subset=None):
    print('+++++++++++++++++++++++')
    if(submit_subset == None):
        print("Please specify 'copy' or a list of image_id in the submit_subset argument.")
        return;

    task_name = config['train'].get('log_name')
    out_dir = TASK_OUTDIR
    predict_model_ckpt = Path(ckpt_path(out_dir)).stem
    submit_dir = get_submit_dir(evaluate_mode)
    visualization_dir = submit_dir + '/visualization/'
    original_dir = visualization_dir + '/original/'
    overlays_src_dir = submit_dir + '/overlays/'
    overlays_des_dir = visualization_dir + '/overlays/'
    original_source = './data/__download__/stage1_%s/' % evaluate_mode
    v4_src_dir = './results/stage1_train_fix_v4_overlay/'
    v4_des_dir = visualization_dir + '/stage1_train_fix_v4_overlay/'

    if submit_subset == "zip":
        backup_project_as_zip(visualization_dir, submit_dir +'visualization_%s_%s.html.zip' % (task_name,predict_model_ckpt+'_'+evaluate_mode))
        return

    if submit_subset == "copy":
        print("Copy over original and overlay images...")
        os.makedirs(overlays_des_dir, exist_ok=True)
        cmd_cp_overlay = "cp %s/* %s" %(overlays_src_dir, overlays_des_dir)
        print("Running %s" % cmd_cp_overlay)
        os.system(cmd_cp_overlay)

        os.makedirs(original_dir, exist_ok=True)
        cmd_cp_original = "cp %s/*/images/*.png %s" %(original_source, original_dir)
        print("Running %s" % cmd_cp_original)
        os.system(cmd_cp_original)

        if evaluate_mode == 'train':
            # copy over overlay images with v4 annotation
            os.makedirs(v4_des_dir, exist_ok=True)
            cmd_cp_v4 = "cp %s/* %s" %(v4_src_dir, v4_des_dir)
            print("Running %s" % cmd_cp_v4)
            os.system(cmd_cp_v4)

        print("Done!\n")
    else:
        print("Generate %s html with submit_subset:"%evaluate_mode, submit_subset)
        if evaluate_mode == 'train':
            visualization_dir = submit_dir + '/visualization/'
            IoU_csv_file = visualization_dir + 'fullres/IoU_results.csv'
            IoU_csv_table = pd.read_csv(IoU_csv_file, index_col=None)
            annotation_table = pd.read_csv(Path(DATA_DIR) / '__download__' / 'stage1_train.csv', index_col=None)
        else:
            annotation_table = pd.read_csv(Path(DATA_DIR) / '__download__' / 'stage1_test.csv', index_col=None)
        # print(annotation_table.head())

        # read in the submit_subset
        df = pd.read_csv(submit_subset, header=None, index_col=None)
        df.columns = ['image_id']
        ids = df['image_id'].drop_duplicates().values.tolist()

        html_file = 'visualization_%s_%s.html' % (task_name,predict_model_ckpt +'_'+ntpath.basename(submit_subset))
        html_dir = visualization_dir + html_file
        html_dir = Path(html_dir)

        with html_dir.open('w', newline='') as result:
            result.write('<!DOCTYPE html>\n')
            result.write('<html>\n')
            result.write('<h2>Visualization of %s - %d images</h2>\n'%(ntpath.basename(submit_subset), len(ids)))
            result.write('<body>\n')
            result.write('<style>table, th, td {border: 1px solid black;}</style>\n')

            for idx in range(0, len(ids)):
                i = annotation_table.image_id == ids[idx]
                mc = annotation_table.loc[i,'major_category'].to_string(index=False)
                sc = annotation_table.loc[i,'sub_category'].to_string(index=False)
                tm = annotation_table.loc[i,'total_masks'].to_string(index=False)
                width = annotation_table.loc[i,'width'].to_string(index=False)
                height = annotation_table.loc[i,'height'].to_string(index=False)

                result.write('<table>\n')
                result.write('<tr><th>index</th><th>image_id</th><th>major_category</th><th>sub_category</th><th>total_masks</th><th>width</th><th>height</th></tr>\n')
                result.write('<tr><th>%d/%d</th><th>%s</th><th>%s</th><th>%s</th><th>%s</th><th>%s</th><th>%s</th></tr>\n'%(idx+1,len(ids),ids[idx],mc,sc,tm,width,height))

                result.write('</table>\n')
                result.write('<BR>\n')

                if evaluate_mode == 'train':
                    IoU = IoU_csv_table.loc[IoU_csv_table.image_id == ids[idx],"Average"].to_string(index=False)
                    result.write('IoU using train_v4 annotation:%s\n'%IoU)

                if evaluate_mode == 'test':
                    IoU = annotation_table.loc[i,'IoU'].to_string(index=False)
                    result.write('caunet 0.465 IoU:%s\n'%IoU)

                result.write('<table>\n')
                result.write('<tr><th>overlays</th></tr>\n')
                result.write('<tr><th><img src="overlays/%s.png"></th></tr>\n'%ids[idx])
                result.write('</table>\n')
                result.write('<hr>\n')
                result.write('<table>\n')
                result.write('<tr><th>full resolution overlay</th><th>original</th></tr>\n')
                result.write('<tr><th><img src="fullres/%s.png"></th><th><img src="original/%s.png"></th></tr>\n'%(ids[idx],ids[idx]))
                result.write('</table>\n')
                if evaluate_mode == 'train':
                    result.write('<hr>\n')
                    result.write('<table>\n')
                    result.write('<tr><th>train_v4 masks</th></tr>\n')
                    result.write('<tr><th><img src="stage1_train_fix_v4_overlay/%s.png"></th></tr>\n'%ids[idx])
                    result.write('</table>\n')

                result.write('<hr>\n')
                result.write('<hr>\n')
                result.write('<BR>\n')

                result.write('</body>\n')
                result.write('</html>\n')

            print(html_dir,"is created for visualization.\n")
            print('-----------------------')
################################### main #######################################
def main(argv):
        # python submit_category.py arg1 arg2
        ## not used yet    Path(argv[1]).expanduser()
        ## not used yet    Path(argv[2]).expanduser()

    if 1: #1
        # generate metafiles such as /npys and /overlays
        run_submit('test')
        #run_submit('train')

    if 1:#1
        # read in /npys and generate RLE csv file and full resolution images for generate_html visualization
        run_npy_to_sumbit_csv('test')


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    main(sys.argv)
    print('\nSuccess!')
