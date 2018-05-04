import os, sys
sys.path.append(os.path.dirname(__file__))
import collections
from torch.nn.modules.module import _addindent
from train import *

ALL_TEST_IMAGE_ID =[
    '0114f484a16c152baa2d82fdd43740880a762c93f436c8988ac461c5c9dbe7d5',
    '0999dab07b11bc85fb8464fc36c947fbd8b5d6ec49817361cb780659ca805eac',
    '0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732',
    '0e132f71c8b4875c3c2dd7a22997468a3e842b46aa9bd47cf7b0e8b7d63f0925',
    '0ed3555a4bd48046d3b63d8baf03a5aa97e523aa483aaa07459e7afa39fb96c6',
    '0f1f896d9ae5a04752d3239c690402c022db4d72c0d2c087d73380896f72c466',
    '1747f62148a919c8feb6d607faeebdf504b5e2ad42b6b1710b1189c37ebcdb2c',
    '17b9bf4356db24967c4677b8376ac38f826de73a88b93a8d73a8b452e399cdff',
    '1879f4f4f05e2bada0ffeb46c128b8df7a79b14c84f38c3e216a69653495153b',
    '191b2b2205f2f5cc9da04702c5d422bc249faf8bca1107af792da63cccfba829',
    '1962d0c5faf3e85cda80e0578e0cb7aca50826d781620e5c1c4cc586bc69f81a',
    '1cdbfee1951356e7b0a215073828695fe1ead5f8b1add119b6645d2fdc8d844e',
    '1d9eacb3161f1e2b45550389ecf7c535c7199c6b44b1c6a46303f7b965e508f1',
    '1ef68e93964c2d9230100c1347c328f6385a7bc027879dc3d4c055e6fe80cb3c',
    '259b35151d4a7a5ffdd7ab7f171b142db8cfe40beeee67277fac6adca4d042c4',
    '295682d9eb5acb5c1976a460c085734bfaf38482b0a3f02591c2bfdcd4128549',
    '31f1fbe85b8899258ea5bcf5f93f7ac8238660c386aeab40649c715bd2e38a0a',
    '336d3e4105766f8ad328a7ee9571e743f376f8cbcf6a969ca7e353fe3235c523',
    '38f5cfb55fc8b048e82a5c895b25fefae7a70c71ab9990c535d1030637bf6a1f',
    '3c4c675825f7509877bc10497f498c9a2e3433bf922bd870914a2eb21a54fd26',
    '432f367a4c5b5674de2e2977744d10289a064e5704b21af6607b4975be47c580',
    '43a71aeb641faa18742cb826772a8566c6c947d7050f9ab15459de6cc2b3b6af',
    '44afae184c89e6ba55985b4d341acc1ae1e8b6ef96312064e0e6e630e022b078',
    '4727d94c6a57ed484270fdd8bbc6e3d5f2f15d5476794a4e37a40f2309a091e2',
    '472b1c5ff988dadc209faea92499bc07f305208dbda29d16262b3d543ac91c71',
    '4be73d68f433869188fe5e7f09c7f681ed51003da6aa5d19ce368726d8e271ee',
    '4f949bd8d914bbfa06f40d6a0e2b5b75c38bf53dbcbafc48c97f105bee4f8fac',
    '505bc0a3928d8aef5ce441c5a611fdd32e1e8eccdc15cc3a52b88030acb50f81',
    '519dc0d672d1c295fc69b629af8721ccb1a1f136d1976685a68487e62547ffe0',
    '51c70bb8a299943b27f8b354571272692d8f2705036a1a9562156c76da5f025b',
    '52b267e20519174e3ce1e1994b5d677804b16bc670aa5f6ffb6344a0fdf63fde',
    '53df5150ee56253fe5bc91a9230d377bb21f1300f443ba45a758bcb01a15c0e4',
    '550450e4bff4036fd671decdc5d42fec23578198d6a2fd79179c4368b9d6da18',
    '5cee644e5ffbef1ba021c7f389b33bafd3b1841f04d3edd7922d5084c2c4e0c7',
    '648c8ffa496e1716017906d0bf135debfc93386ae86aa3d4adbda9a505985fd9',
    '697a05c6fe4a07c601d46da80885645ad574ea19b47ee795ccff216c9f1f1808',
    '699f2992cd71e2e28cf45f81347ff22e76b37541ce88087742884cd0e9aadc68',
    '78a981bd27ba0c65a9169548665a17bda9f49050d0d3893a6567d1eb92cd003d',
    '7bdb668e6127b7eafc837a883f0648002bd063c736f55a4f673e787250a3fb04',
    '7f4cbe0b36b5d09466476a7d4e01f4f976c67872d549f4ff47b3e1e3a2b403af',
    '8922a6ac8fd0258ec27738ca101867169b20d90a60fc84f93df77acd5bf7c80b',
    '8b59819fbc92eefe45b1db95c0cc3a467ddcfc755684c7f2ba2f6ccb9ad740ab',
    '912a679e4b9b1d1a75170254fd675b8c24b664d80ad7ea7e460241a23535a406',
    '9ab2d381f90b485a68b82bc07f94397a0373e3215ad20935a958738e55f3cfc2',
    '9f17aea854db13015d19b34cb2022cfdeda44133323fcd6bb3545f7b9404d8ab',
    'a4816cc1fb76cb3c5e481186833fc0ae9cf426a1406a2607e974e65e9cddba4f',
    'a984e7fb886aa02e29d112766d3ce26a4f78eac540ce7bbdbd42af2761928f6d',
    'ab298b962a63e4be9582513aaa84a5e270adba5fd2b16a50e59540524f63c3b8',
    'ade080c6618cbbb0a25680cf847f312b5e19b22bfe1cafec0436987ebe5b1e7e',
    'b83d1d77935b6cfd44105b54600ffc4b6bd82de57dec65571bcb117fa8398ba3',
    'bdc789019cee8ddfae20d5f769299993b4b330b2d38d1218646cf89e77fbbd4d',
    'c8e79ff4ac55f4b772057de28e539727b7f4f2a3de73bf7a082a0ace86d609eb',
    'ca20076870e8fb604e61802605a9ac45419c82dd3e23404c56c4869f9502a5ef',
    'd616d323a9eeb9da1b66f8d5df671d63c092c9919cb2c0b223e29c63257c944d',
    'd6eb7ce7723e2f6dc13b90b41a29ded27dbd815bad633fdf582447c686018896',
    'd8d4bf68a76e4e4c5f21de7ac613451f7115a04db686151e78b8ec0b6a22022b',
    'da6c593410340b19bb212b9f6d274f95b08c0fc8f2570cd66bc5ed42c560acab',
    'dab46d798d29aff2e99c23f47ed3064f5cafb1644629b015c95a2dd2ee593bb4',
    'df40099c6306ca1f47fcc8a62e2fa39486d4e223177afdc51b2ad189691802d8',
    'e17b7aedd251a016c01ef9158e6e4aa940d9f1b35942d86028dc1222192a9258',
    'eea70a7948d25a9a791dbcb39228af4ea4049fe5ebdee9c04884be8cca3da835',
    'f0d0ab13ff53adc3c4d57e95a5f83d80b06f2cbc0bf002b52cf7b496612e0ce4',
    'f5effed21f671bbf4551ecebb7fe95f3be1cf09c16a60afe64d2f0b95be9d1eb',
    'fac507fa4d1649e8b24c195d990f1fc3ca3633d917839e1751a9d412a14ab5e3',
    'fe9adb627a6f45747c5a8223b671774791ededf9364f6544be487c540107fa4f',
]


## overwrite functions ###
def revert(net, images):
    #undo test-time-augmentation (e.g. unpad or scale back to input image size, etc)

    def torch_clip_proposals (proposals, index, width, height):
        boxes = torch.stack((
             proposals[index,0],
             proposals[index,1].clamp(0, width  - 1),
             proposals[index,2].clamp(0, height - 1),
             proposals[index,3].clamp(0, width  - 1),
             proposals[index,4].clamp(0, height - 1),
             proposals[index,5],
             proposals[index,6],
        ), 1)
        return proposals

    # ----

    batch_size = len(images)
    for b in range(batch_size):
        image  = images[b]
        height,width  = image.shape[:2]


        # net.rpn_logits_flat  <todo>
        # net.rpn_deltas_flat  <todo>
        # net.rpn_window       <todo>
        # net.rpn_proposals    <todo>

        # net.rcnn_logits
        # net.rcnn_deltas
        # net.rcnn_proposals <todo>

        # mask --
        # net.mask_logits
        if len(net.detections)!=0: 
            index = (net.detections[:,0]==b).nonzero().view(-1)
            net.detections = torch_clip_proposals (net.detections, index, width, height)

        net.masks[b] = net.masks[b][:height,:width]

    return net, image



#-----------------------------------------------------------------------------------
def submit_augment(image, index, id):
    pad_image = pad_to_factor(image, factor=16)
    input = torch.from_numpy(pad_image.transpose((2,0,1))).float().div(255)
    return input, image, index, id


def submit_collate(batch):

    batch_size = len(batch)
    #for b in range(batch_size): print (batch[b][0].size())
    inputs    = torch.stack([batch[b][0]for b in range(batch_size)], 0)
    images    =             [batch[b][1]for b in range(batch_size)]
    indices   =             [batch[b][2]for b in range(batch_size)]
    id        =             [batch[b][3]for b in range(batch_size)]
    return [inputs, images, indices, id]

def summary_model(input_size, model):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx+1)
            summary[m_key] = collections.OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = -1
            summary[m_key]['output_shape'] = list(output.size())
            summary[m_key]['output_shape'][0] = -1

            params = 0
            if hasattr(module, 'weight'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                if module.weight.requires_grad:
                    summary[m_key]['trainable'] = True
                else:
                    summary[m_key]['trainable'] = False
            if hasattr(module, 'bias'):
                params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params
            
        if not isinstance(module, nn.Sequential) and \
            not isinstance(module, nn.ModuleList) and \
            not (module == model):
            hooks.append(module.register_forward_hook(hook))
    
    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(1,*in_size)) for in_size in input_size]
    else:
        x = Variable(torch.rand(1,*input_size))

    # create properties
    summary = collections.OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    model(x)
    # remove these hooks
    for h in hooks:
        h.remove()

    return summary

def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'   

    tmpstr = tmpstr + ')'
    return tmpstr

#--------------------------------------------------------------
def run_submit():
    out_dir = RESULTS_DIR + SUB_TASK_NAME
    submit_dir = out_dir + '/submit' + SUBMIT_FOLDER
    initial_checkpoint = PREDICT_CP_FILE

    # out_dir  = RESULTS_DIR + '/mask-rcnn-50-gray500-02'
    # initial_checkpoint = \
    #     RESULTS_DIR + '/mask-rcnn-50-gray500-02/checkpoint/00016500_model.pth'
    #     ##
    # print (initial_checkpoint)
    ## setup  ---------------------------
    os.makedirs(submit_dir +'/overlays', exist_ok=True)
    os.makedirs(submit_dir +'/npys', exist_ok=True)
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.%s.zip'%IDENTIFIER)

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

    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
    # print(torch_summarize(net))
    log.write('%s\n\n'%(type(net)))
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    
    evaluate_mode = SUBMIT_MODE
    save_image = SUBMIT_IMAGE
    if evaluate_mode == 'test':
        input_path = EXCEL_PATH_TEST
        output_csv_path = TEST_CSV_PATH
        sheet_name = TEST_CSV_SHEET
        csv_from_excel(input_path, output_csv_path, sheet_name)
        test_csv = pd.read_csv(output_csv_path)
    else:
        # input_path = EXCEL_PATH
        output_csv_path = COMMENT_CSV_PATH
        # sheet_name = 'validation_test'
        # sheet_name = Train_CSV_SHEET
        # csv_from_excel(input_path, output_csv_path, sheet_name)
        # runs the csv_from_excel function:
        test_csv = pd.read_csv(output_csv_path)
        test_csv = test_csv[test_csv['source'].isin(SOURCE_LIST)].reset_index()
        # test_csv = test_csv[test_csv['category']=='Histology']

    test_dataset = ScienceDataset(
                                test_csv, mode='test',
                                #'train1_ids_gray_only1_500', mode='test',
                                #'valid1_ids_gray_only1_43', mode='test',
                                #'debug1_ids_gray_only_10', mode='test',
                                # 'test1_ids_gray2_53', mode='test',
                                transform = submit_augment)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler = SequentialSampler(test_dataset),
                        batch_size  = 1,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn  = submit_collate)


    log.write('\ttest_dataset.split = %s\n'%(COMMENT_CSV_PATH))
    log.write('\tlen(test_dataset)  = %d\n'%(len(test_dataset)))
    log.write('\n')





    ## start evaluation here! ##############################################
    log.write('** start evaluation here! **\n')
    start = timer()
    pred_masks = []
    true_masks = []
    ap = []
    label_counts = []
    predicts_counts = []

    test_num  = len(test_loader.dataset)
    for i, (inputs, images, indices, ids) in enumerate(test_loader, 0):
        print('\rpredicting: %10d/%d (%0.0f %%)  %0.2f min'%(i, test_num-1, 100*i/(test_num-1),
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
        # print ('detections shape', detections.shape)
        masks      = net.masks
        keeps       = net.keep
        # print ('batch_size', batch_size)
        for b in range(batch_size):
            #image0 = (inputs[b].transpose((1,2,0))*255).astype(np.uint8)
            if evaluate_mode == 'train':
                path_mask = glob.glob(os.path.join(DATA_DIR + '/image/', '*', 'multi_masks/', ids[b] +'.npy'))
                true_mask = np.load( path_mask[0]).astype(np.int32)
                true_masks.append(true_mask)
            
            image  = images[b]
            height,width = image.shape[:2]
            # print ('hw', height, width)
            mask   = masks[b]
            keep   = keeps[b]
            pred_masks.append(mask)

            if evaluate_mode == 'train': 
                ap_one, label_counts_one, predicts_counts_one = evaluate_ap(mask, true_mask)
                ap.append(ap_one)
                label_counts.append(label_counts_one)
                predicts_counts.append(predicts_counts_one)
            prob  = rpn_probs_flat[b]
            delta = rpn_deltas_flat[b]
            # detection = detections[b]
            image_rcnn_detection_nms = draw_rcnn_detection_nms(image, detections, threshold=0.1)
            # image_rpn_proposal_before_nms = draw_rpn_proposal_before_nms(image,prob,delta,window,0.995)

            contour_overlay  = multi_mask_to_contour_overlay(cfg, mask, detections,keep,  image, color=[0,255,0])
            color_overlay    = multi_mask_to_color_overlay(mask, color='summer')
            if evaluate_mode == 'train':
                color_overlay_true    = multi_mask_to_color_overlay(true_mask, image, color='summer')
            color_overlay    = multi_mask_to_color_overlay(mask, color='summer')
            color1_overlay   = multi_mask_to_contour_overlay(cfg, mask, detections, keep, color_overlay, color=[255,255,255])
            image_rcnn_detection_nms = image_rcnn_detection_nms[:height,:width]
            # image_rpn_proposal_before_nms = image_rpn_proposal_before_nms[:height,:width]
            if evaluate_mode == 'train':
                all = np.hstack((image,contour_overlay, image_rcnn_detection_nms, color1_overlay, color_overlay_true))
            else:
                all = np.hstack((image,contour_overlay,color1_overlay, image_rcnn_detection_nms))


            # --------------------------------------------
            id = test_dataset.ids[indices[b]]
            name =id.split('/')[-1]

            #draw_shadow_text(overlay_mask, 'mask',  (5,15),0.5, (255,255,255), 1)
            np.save(submit_dir + '/npys/%s.npy'%(name),mask)
            #cv2.imwrite(out_dir +'/submit/npys/%s.png'%(name),color_overlay)
##image
            if save_image == 'yes':
                if evaluate_mode == 'train':
                    cv2.imwrite(submit_dir +'/overlays/%s_%s.png'%(name,ap[i]),all)  
                else:
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


    log.write('initial_checkpoint  = %s\n'%(initial_checkpoint))
    log.write('test_num  = %d\n'%(test_num))
    log.write('\n')
    if evaluate_mode == 'train':
        ids = test_csv['image_id']
        major_category = test_csv['major_category']
        sub_category = test_csv['sub_category']
        source = test_csv['source']
        df_predict = pd.DataFrame({'image_id' : ids, 'pred_mask': pred_masks , 'true_mask': true_masks, 'source' : source,
                                   'major_category' : major_category, 'sub_category' : sub_category, 'ap' : ap , 
                                   'label_counts' : label_counts,'predicts_counts' : predicts_counts })
        # df_predict= df_predict.assign(label_counts=0)
        # df_predict= df_predict.assign(predicts_counts=0)
        # df_predict= df_predict.assign(ap=0)
        # for i in range(df_predict.shape[0]):
        #     df_predict.loc[i, ['ap', 'label_counts', 'predicts_counts']]= evaluate_water(df_predict.loc[:,'pred_mask'].values.tolist()[i], df_predict.loc[:,'true_mask'].values.tolist()[i])
        ap_mean = df_predict['ap'].mean()
        print ('AP:',ap_mean)
        ap_his = df_predict.loc[df_predict['major_category']=='Histology', ['ap']].mean().values[0]
        ap_flo_major = df_predict.loc[df_predict['major_category']=='Fluorescence', ['ap']].mean().values[0]
        ap_bri_major = df_predict.loc[df_predict['major_category']=='Brightfield', ['ap']].mean().values[0]
        print ('Major category:')
        print ('ap_Histology:',ap_his)
        print ('ap_Fluorescence:',ap_flo_major)
        print ('ap_Brightfield:',ap_bri_major)
        ap_he  = df_predict.loc[df_predict['sub_category']=='HE', ['ap']].mean().values[0]
        ap_flo_sub = df_predict.loc[df_predict['sub_category']=='Fluorescence', ['ap']].mean().values[0]
        ap_bri_sub = df_predict.loc[df_predict['sub_category']=='Brightfield', ['ap']].mean().values[0]
        ap_clo = df_predict.loc[df_predict['sub_category']=='Cloud', ['ap']].mean().values[0]
        ap_dro = df_predict.loc[df_predict['sub_category']=='Drosophilidae', ['ap']].mean().values[0]
        ap_ihc = df_predict.loc[df_predict['sub_category']=='IHC', ['ap']].mean().values[0]
        print ('Sub category:')
        print ('ap_HE:',ap_he)
        print ('ap_Fluorescence:',ap_flo_sub)
        print ('ap_Brightfield:',ap_bri_sub)
        print ('ap_Cloud:',ap_clo)
        print ('ap_Drosophilidae:',ap_dro)
        print ('ap_IHC:',ap_ihc)
        ap_kag  = df_predict.loc[df_predict['source']=='Kaggle', ['ap']].mean().values[0]
        ap_bbb  = df_predict.loc[df_predict['source']=='BBBC006', ['ap']].mean().values[0]
        ap_cel  = df_predict.loc[df_predict['source']=='celltracking', ['ap']].mean().values[0]
        ap_gnf  = df_predict.loc[df_predict['source']=='ISBI09_gnf', ['ap']].mean().values[0]
        ap_ic100  = df_predict.loc[df_predict['source']=='ISBI09_ic100', ['ap']].mean().values[0]
        ap_isbi  = df_predict.loc[df_predict['source']=='ISBI14', ['ap']].mean().values[0]
        ap_tcga  = df_predict.loc[df_predict['source']=='TCGA', ['ap']].mean().values[0]
        ap_tcgac  = df_predict.loc[df_predict['source']=='TCGA_ColorNorm', ['ap']].mean().values[0]
        print ('Source category:')
        print ('ap_Kaggle:',ap_kag)
        print ('ap_BBBC006:',ap_bbb)
        print ('ap_celltracking:',ap_cel)
        print ('ap_ISBI09_gnf:',ap_gnf)
        print ('ISBI09_ic100:',ap_ic100)
        print ('ap_ISBI14:',ap_isbi)
        print ('ap_TCGA:',ap_tcga)
        print ('ap_TCGA_ColorNorm:',ap_tcgac)

        log.write('Mean AP  = %s\n'%(ap_mean))
        log.write('Histology AP   = %s\n'%(ap_his))
        log.write('Fluorescence AP  = %s\n'%(ap_flo_major))
        log.write('Brightfield AP  = %s\n'%(ap_bri_major))

        log.write('HE AP  = %s\n'%(ap_he))
        log.write('Fluorescence AP  = %s\n'%(ap_flo_sub))
        log.write('Brightfield AP  = %s\n'%(ap_bri_sub))
        log.write('Cloud AP  = %s\n'%(ap_clo))
        log.write('Drosophilidae AP  = %s\n'%(ap_dro))
        log.write('IHC AP  = %s\n'%(ap_ihc))

        log.write('Kaggle AP  = %s\n'%(ap_kag))
        log.write('BBBC006 AP  = %s\n'%(ap_bbb))
        log.write('celltracking AP  = %s\n'%(ap_cel))
        log.write('ISBI09_gnf AP  = %s\n'%(ap_gnf))
        log.write('ISBI09_ic100 AP  = %s\n'%(ap_ic100))
        log.write('ISBI14 AP  = %s\n'%(ap_isbi))
        log.write('TCGA AP  = %s\n'%(ap_tcga))
        log.write('TCGA_ColorNorm AP  = %s\n'%(ap_tcgac))


        log.write('\n')
        df_predict = df_predict.drop(['pred_mask', 'true_mask'], axis=1)
        # print (df_predict)
        ids_histology = test_csv.loc[test_csv[SELECT_CATEGORY]=='Histology', ['image_id']]
        # print (ids_histology)

        prediction_csv_file = submit_dir + '/prediction.csv'
        df_predict.to_csv(prediction_csv_file)

    # ids_flouresence = 
    # ids_brightfield = 

##-----------------------------------------------------------------------------------------------------
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn

def evaluate_ap (y_pred, labels):
    #     y_pred = offset(y_pred)
#     labels = resize(labels, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#     print (labels.shape)
#     y_pred = (y_pred > 0.5)
#     y_pred = label(y_pred)
#     labels = label(labels)
    true_objects = len(np.unique(labels))
    # print (true_objects)
    pred_objects = len(np.unique(y_pred))
    # print (pred_objects)
    # Compute intersection between all objects
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]
    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    # Compute union
    union = area_true + area_pred - intersection
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9
    # Compute the intersection over union
    iou = intersection / union
    # Precision helper function
    # Loop over IoU thresholds
    prec = []
    score = 0
    # print("Number of true objects:", true_objects)
    # print("Number of predicted objects:", pred_objects)

    # print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        p = tp / (tp + fp + fn)
        # print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        score_t = p/t
        score = score+score_t
        prec.append(p)
    ap = round(np.mean(prec),3)
    # print("AP\t-\t-\t-\t{:1.3f}".format(ap))
    return ap, true_objects, pred_objects

## post process #######################################################################################
def filter_small(multi_mask, threshold):
    num_masks = int(multi_mask.max())

    j=0
    for i in range(num_masks):
        thresh = (multi_mask==(i+1))

        area = thresh.sum()
        if area < threshold:
            multi_mask[thresh]=0
        else:
            multi_mask[thresh]=(j+1)
            j = j+1

    return multi_mask

def shrink_by_one(multi_mask):

    multi_mask1=np.zeros(multi_mask.shape,np.int32)

    num = int( multi_mask.max())
    for m in range(num):
        mask  =  multi_mask==m+1
        contour = mask_to_inner_contour(mask)
        thresh  = thresh & (~contour)
        multi_mask1 [thresh] = m+1
    return multi_mask1

def run_npy_to_sumbit_csv():

    image_dir   = '/home/sychen/code/kaggle/data/image/stage1_test/images'
    out_dir = RESULTS_DIR + SUB_TASK_NAME
    submit_dir = out_dir + '/submit' + SUBMIT_FOLDER
        # '/root/share/project/kaggle/science2018/results/mask-rcnn-50-gray500-02/submit'


    npy_dir = submit_dir  + '/npys'
    csv_file = submit_dir + '/submission.csv'
    
    ## start -----------------------------
    all_num=0
    cvs_ImageId = [];
    cvs_EncodedPixels = [];

    npy_files = glob.glob(npy_dir + '/*.npy')
    for npy_file in npy_files:
        name = npy_file.split('/')[-1].replace('.npy','')

        multi_mask = np.load(npy_file)

        #<todo> ---------------------------------
        #post process here
        multi_mask = filter_small(multi_mask, 8)
        #<todo> ---------------------------------

        num = int( multi_mask.max())
        for m in range(num):
            rle = run_length_encode(multi_mask==m+1)
            cvs_ImageId.append(name)
            cvs_EncodedPixels.append(rle)
        all_num += num

        #<debug> ------------------------------------
        # print(all_num, num)  ##GT is 4152?
        # image_file = image_dir +'/%s.png'%name
        # image = cv2.imread(image_file)
        # color_overlay   = multi_mask_to_color_overlay(multi_mask)
        # color1_overlay  = multi_mask_to_contour_overlay(multi_mask, detections, color_overlay)
        # contour_overlay = multi_mask_to_contour_overlay(multi_mask, detections, image, [0,255,0])
        # all = np.hstack((image, contour_overlay, color1_overlay)).astype(np.uint8)
        # image_show('all',all)
        # cv2.waitKey(1)
    #exit(0)
    # submission csv  ----------------------------

    # kaggle submission requires all test image to be listed!
    for t in ALL_TEST_IMAGE_ID:
        cvs_ImageId.append(t)
        cvs_EncodedPixels.append('') #null

    df = pd.DataFrame({ 'ImageId' : cvs_ImageId , 'EncodedPixels' : cvs_EncodedPixels})
    df.to_csv(csv_file, index=False, columns=['ImageId', 'EncodedPixels'])

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    run_submit()
    run_npy_to_sumbit_csv()

    print('\nsucess!')