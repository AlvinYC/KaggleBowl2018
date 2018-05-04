import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'3,2' #'3,2,1,0'
import argparse
import json
from tensorboardX import SummaryWriter

from common import *
from utility.file   import *
from dataset.reader import *
from net.rate   import adjust_learning_rate, get_learning_rate
from net.metric import *
from dataset import augment
from utility.helper import save_ckpt, load_ckpt

# -------------------------------------------------------------------------------------
from net.resnet50_mask_rcnn.draw  import *
from net.resnet50_mask_rcnn.model import *

# move to dataset.augment, WIDTH, HEIGHT = 128,128
# move to dataset.augment, WIDTH, HEIGHT = 192,192\
# move to dataset.augment, WIDTH, HEIGHT = 256, 256
COMMENT_CSV_PATH = DATA_DIR +'/__download__/stage1_train.csv'
# -------------------------------------------------------------------------------------


def train_collate(batch):

    batch_size = len(batch)
    #for b in range(batch_size): print (batch[b][0].size())
    inputs    = torch.stack([batch[b][0]for b in range(batch_size)], 0)
    boxes     =             [batch[b][1]for b in range(batch_size)]
    labels    =             [batch[b][2]for b in range(batch_size)]
    instances =             [batch[b][3]for b in range(batch_size)]
    metas     =             [batch[b][4]for b in range(batch_size)]
    indices   =             [batch[b][5]for b in range(batch_size)]

    return [inputs, boxes, labels, instances, metas, indices]



### training ##############################################################
def evaluate( net, test_loader ):
    test_num  = 0
    test_loss = np.zeros(6,np.float32)
    test_acc  = 0
    for i, (inputs, truth_boxes, truth_labels, truth_instances, metas, indices) in enumerate(test_loader, 0):

        with torch.no_grad():
            inputs = Variable(inputs).cuda()
            net(inputs, truth_boxes,  truth_labels, truth_instances )
            loss = net.loss(inputs, truth_boxes,  truth_labels, truth_instances)

        # acc    = dice_loss(masks, labels) #todo

        batch_size = len(indices)
        test_acc  += 0 #batch_size*acc[0][0]
        test_loss += batch_size*np.array((
                           loss.cpu().data.numpy(),
                           net.rpn_cls_loss.cpu().data.numpy(),
                           net.rpn_reg_loss.cpu().data.numpy(),
                           net.rcnn_cls_loss.cpu().data.numpy(),
                           net.rcnn_reg_loss.cpu().data.numpy(),
                           net.mask_cls_loss.cpu().data.numpy(),
                         ))
        test_num  += batch_size

    assert(test_num == len(test_loader.sampler))
    test_acc  = test_acc/test_num
    test_loss = test_loss/test_num
    return test_loss, test_acc


#--------------------------------------------------------------
def main(resume, n_epoch, learn_rate):
    model_name = config['param']['model']
    c = config['train']
    batch_size = c.getint('n_batch')
    n_worker = c.getint('n_worker')
    n_ckpt_epoch = c.getint('n_ckpt_epoch')
    iter_accum = c.getint('iter_accum')
    cv_ratio = c.getfloat('cv_ratio')
    cv_seed = c.getint('cv_seed')
    data_src = json.loads(c.get('data_src'))
    data_major = json.loads(c.get('data_major'))
    data_sub = json.loads(c.get('data_sub'))

    out_dir  = TASK_OUTDIR
    # initial_checkpoint = INITIAL_CP_FILE
    # RESULTS_DIR + '/mask-rcnn-50-gray500-02/checkpoint/00014500_model.pth'
    ##

    '''
    # pretrain_file = PRETRAIN_FILE
    # None #RESULTS_DIR + '/mask-single-shot-dummy-1a/checkpoint/00028000_model.pth'
    skip = ['crop','mask']
    '''

    ## setup  -----------------
    os.makedirs(out_dir , exist_ok=True, mode=0o777)
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/train', exist_ok=True)
    # os.makedirs(out_dir +'/backup', exist_ok=True)
    # backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')

    ## net ----------------------
    log.write('** net setting **\n')
    cfg = Configuration()
    net = MaskRcnnNet(cfg).cuda()
    log.write('Classes map                 : %s\n'%(cfg.classes_map))
    log.write('num_classes                 : %s\n'%(cfg.num_classes))
    log.write('rpn single class            : %s\n'%(cfg.rpn_single_class))
    log.write('rpn using only P0 layer     : %s\n'%(cfg.rpn_p0_pool))
    log.write('Crop using only P0 layer    : %s\n'%(cfg.crop_one_layer))
    log.write('Increase feature resolution : %s\n'%(cfg.high_resolution))

    ## optimiser ----------------------------------
    # LR = None  #LR = StepLR([ (0, 0.01),  (200, 0.001),  (300, -1)])
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=learn_rate/iter_accum, momentum=0.9, weight_decay=0.0001)

    # resume checkpoint
    start_iter, start_epoch = 0, 0
    if resume:
        rate = get_learning_rate(optimizer)  #load all except learning rate
        start_iter, start_epoch = load_ckpt(out_dir, net, optimizer)
        '''
        if os.path.isfile(initial_checkpoint):
            log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
            net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
            #with open(out_dir +'/checkpoint/configuration.pkl','rb') as pickle_file:
            #    cfg = pickle.load(pickle_file)

            checkpoint  = torch.load(initial_checkpoint.replace('_model.pth','_optimizer.pth'))
            start_iter  = checkpoint['iter' ]
            start_epoch = checkpoint['epoch']

            rate = get_learning_rate(optimizer)  #load all except learning rate
            optimizer.load_state_dict(checkpoint['optimizer'])
            adjust_learning_rate(optimizer, rate)
        '''
    if start_epoch == 0:
        print('Grand new training ...')
    else:
        adjust_learning_rate(optimizer, rate)

    log.write('%s\n\n'%(type(net)))
    log.write('%s\n'%(net.version))
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    # Read Excel file
    # input_path = EXCEL_PATH
    # output_csv_path = COMMENT_CSV_PATH
    # sheet_name = Train_CSV_SHEET
    # csv_from_excel(input_path, output_csv_path, sheet_name)
    # runs the csv_from_excel function:
    comment_csv = pd.read_csv(COMMENT_CSV_PATH)
    df = comment_csv
    if 'all' not in data_src:
        df = df[df['source'].isin(data_src)]
    if 'all' not in data_major:
        df = df[df['major_category'].isin(data_major)]
    if 'all' not in data_sub:
        df = df[df['sub_category'].isin(data_sub)]
    comment_csv = df.reset_index()

    # comment_csv = comment_csv[comment_csv['category']=='Cloud'].reset_index()
    # split train/validation
    comment_csv = comment_csv.sample(frac = 1, random_state = cv_seed).reset_index(drop=True)
    split = int(np.floor(cv_ratio * comment_csv.shape[0]))
    comment_train_csv = comment_csv[split:].reset_index(drop=True)
    comment_valid_csv = comment_csv[:split]
    train_dataset = ScienceDataset(
                            comment_train_csv, COMMENT_CSV_PATH , mode='train',
                            # comment_csv , mode='train',
                            # 'train1_ids_gray2_500', mode='train',
                            #'debug1_ids_gray_only_10', mode='train',
                            #'disk0_ids_dummy_9', mode='train', #12
                            #'train1_ids_purple_only1_101', mode='train', #12
                            #'merge1_1', mode='train',
                            transform = augment.Compose()) # augment.train_augment
    train_loader  = DataLoader(
                        train_dataset,
                        sampler = RandomSampler(train_dataset),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = n_worker,
                        pin_memory  = True, # torch.cuda.is_available()
                        collate_fn  = train_collate)

    valid_dataset = valid_loader = None
    if len(comment_valid_csv) > 0:
        valid_dataset = ScienceDataset(
                                comment_valid_csv, COMMENT_CSV_PATH, mode='train',
                                # 'valid1_ids_gray2_43', mode='train',
                                #'debug1_ids_gray_only_10', mode='train',
                                #'disk0_ids_dummy_9', mode='train',
                                #'train1_ids_purple_only1_101', mode='train', #12
                                #'merge1_1', mode='train',
                                transform = augment.Compose()) # augment.valid_augment

        valid_loader  = DataLoader(
                            valid_dataset,
                            sampler     = SequentialSampler(valid_dataset),
                            batch_size  = batch_size,
                            drop_last   = False,
                            num_workers = n_worker,
                            pin_memory  = True, # torch.cuda.is_available()
                            collate_fn  = train_collate)

    log.write('\tWIDTH, HEIGHT = %d, %d\n'%(augment.WIDTH, augment.HEIGHT))
    log.write('\ttrain_dataset.split = %s\n'%(train_dataset.comment_path))
    log.write('\tlen(train_dataset)  = %d\n'%(len(train_dataset)))
    log.write('\tlen(train_loader)   = %d\n'%(len(train_loader)))
    if valid_dataset is not None:
        log.write('\tvalid_dataset.split = %s\n'%(valid_dataset.comment_path))
        log.write('\tlen(valid_dataset)  = %d\n'%(len(valid_dataset)))
        log.write('\tlen(valid_loader)   = %d\n'%(len(valid_loader)))
    else:
        log.write('\t!!! No valid dataset !!!\n')
    log.write('\tn_epoch = %d\n'%(n_epoch))
    log.write('\tbatch_size  = %d\n'%(batch_size))
    log.write('\titer_accum  = %d\n'%(iter_accum))
    log.write('\tbatch_size*iter_accum  = %d\n'%(batch_size*iter_accum))
    log.write('\tn_ckpt_epoch = %d\n'%(n_ckpt_epoch))
    log.write('\n')

    # decide log directory name
    txlog_dir = os.path.join(out_dir,
        'logs', '{}-{}'.format(model_name, augment.WIDTH),
        'ep_{},{}-lr_{}'.format(
            start_epoch,
            n_epoch + start_epoch,
            learn_rate,
        )
    )

    with SummaryWriter(txlog_dir) as txwriter:
        print('Training started...')
        run_train(  (train_loader, valid_loader), net, optimizer,
                    (start_iter, iter_accum), (start_epoch, n_epoch, n_ckpt_epoch),
                    (len(train_dataset), len(valid_dataset) if valid_dataset is not None else 0, batch_size),
                    log, txwriter)
        print('Training finished...')


def run_train(loaders, net, optimizer, iters, epochs, data_sizes, log, txwriter):
    train_loader, valid_loader = loaders
    start_iter, iter_accum = iters
    start_epoch, n_epoch, n_ckpt_epoch = epochs
    traindata_size, validdata_size, batch_size = data_sizes
    out_dir = TASK_OUTDIR

    if 0:
        debug_1()

    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write(' optimizer=%s\n' % str(optimizer) )
    log.write(' momentum=%f\n' % optimizer.param_groups[0]['momentum'])
    # log.write(' LR=%s\n\n'%str(LR) )

    log.write(' images_per_epoch = %d\n\n' % traindata_size)
    log.write(' rate    iter   epoch  num   | valid_loss               | train_loss               | batch_loss               |  time          \n')
    log.write('-------------------------------------------------------------------------------------------------------------------------------\n')

    train_loss, train_acc  = np.zeros(6,np.float32), 0.0
    valid_loss, valid_acc  = np.zeros(6,np.float32), 0.0
    batch_loss, batch_acc  = np.zeros(6,np.float32), 0.0
    rate = 0

    start = timer()
    j = 0
    i = 0

    def to_iter(epoch):
        return (epoch * traindata_size / (batch_size * iter_accum))

    num_iters = start_iter + to_iter(n_epoch)
    n_ckpt_iters = to_iter(n_ckpt_epoch)
    s, nckpt, n = [int(np.around(x)) for x in [start_iter, n_ckpt_iters, num_iters]]
    iter_save   = list(range((s + nckpt), n, nckpt)) + [n - 1]
    iter_smooth = to_iter(1)
    iter_valid  = to_iter(3)

    while i < num_iters:  # loop over the dataset multiple times
        sum_train_loss = np.zeros(6,np.float32)
        sum_train_acc  = 0.0
        sum = 0

        net.set_mode('train')
        optimizer.zero_grad()
        for inputs, truth_boxes, truth_labels, truth_instances, _, indices in train_loader:
            if all(len(b)==0 for b in truth_boxes): continue

            batch_size = len(indices)
            i = j / iter_accum + start_iter
            epoch = (i - start_iter) * batch_size * iter_accum / traindata_size + start_epoch
            num_products = epoch * traindata_size

            if i % iter_valid == 0 and validdata_size > 0:
                net.set_mode('valid')
                valid_loss, valid_acc = evaluate(net, valid_loader)
                net.set_mode('train')

                txwriter.add_scalar('CV/epoch_total_loss', valid_loss[0], int(epoch))
                txwriter.add_scalar('CV/epoch_rpn_cls_loss', valid_loss[1], int(epoch))
                txwriter.add_scalar('CV/epoch_rpn_reg_loss', valid_loss[2], int(epoch))
                txwriter.add_scalar('CV/epoch_rcnn_cls_loss', valid_loss[3], int(epoch))
                txwriter.add_scalar('CV/epoch_rcnn_reg_loss', valid_loss[4], int(epoch))
                txwriter.add_scalar('CV/epoch_mask_cls_loss', valid_loss[5], int(epoch))

                print('\r',end='',flush=True)
                log.write('%0.4f %5.1f k %6.1f %4.1f m | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %s\n' % (\
                         rate, i/1000, epoch, num_products/1000000,
                         valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5],#valid_acc,
                         train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5],#train_acc,
                         batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], batch_loss[4], batch_loss[5],#batch_acc,
                         time_to_str((timer() - start)/60)))
                time.sleep(0.01)

            if i in iter_save:
                save_ckpt(out_dir, net, optimizer, i, epoch)

            '''
            # learning rate schduler -------------
            if LR is not None:
                lr = LR.get_rate(i)
                if lr<0 : break
                adjust_learning_rate(optimizer, lr/iter_accum)
            '''
            rate = get_learning_rate(optimizer) * iter_accum

            # one iteration update  -------------
            inputs = Variable(inputs).cuda()
            net( inputs, truth_boxes, truth_labels, truth_instances )
            loss = net.loss( inputs, truth_boxes, truth_labels, truth_instances )

            # accumulated update
            loss.backward()
            if j % iter_accum == 0:
                #torch.nn.utils.clip_grad_norm(net.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()

            # print statistics  ------------
            batch_acc  = 0 #acc[0][0]
            batch_loss = np.array((
                           loss.cpu().data.numpy(),
                           net.rpn_cls_loss.cpu().data.numpy(),
                           net.rpn_reg_loss.cpu().data.numpy(),
                           net.rcnn_cls_loss.cpu().data.numpy(),
                           net.rcnn_reg_loss.cpu().data.numpy(),
                           net.mask_cls_loss.cpu().data.numpy(),
                         ))
            sum_train_loss += batch_loss
            sum_train_acc  += batch_acc
            sum += 1
            if i % iter_smooth == 0:
                train_loss = sum_train_loss/sum
                train_acc  = sum_train_acc /sum
                sum_train_loss = np.zeros(6,np.float32)
                sum_train_acc  = 0.
                sum = 0
                txwriter.add_scalar('training/epoch_total_loss', train_loss[0], int(epoch))
                txwriter.add_scalar('training/epoch_rpn_cls_loss', train_loss[1], int(epoch))
                txwriter.add_scalar('training/epoch_rpn_reg_loss', train_loss[2], int(epoch))
                txwriter.add_scalar('training/epoch_rcnn_cls_loss', train_loss[3], int(epoch))
                txwriter.add_scalar('training/epoch_rcnn_reg_loss', train_loss[4], int(epoch))
                txwriter.add_scalar('training/epoch_mask_cls_loss', train_loss[5], int(epoch))

            # log to summary
            txwriter.add_scalar('training/total_loss', batch_loss[0], int(i))
            txwriter.add_scalar('training/rpn_cls_loss', batch_loss[1], int(i))
            txwriter.add_scalar('training/rpn_reg_loss', batch_loss[2], int(i))
            txwriter.add_scalar('training/rcnn_cls_loss', batch_loss[3], int(i))
            txwriter.add_scalar('training/rcnn_reg_loss', batch_loss[4], int(i))
            txwriter.add_scalar('training/mask_cls_loss', batch_loss[5], int(i))

            print('\r%0.4f %5.1f k %6.1f %4.1f m | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %s  %d,%d,%s' % (\
                         rate, i/1000, epoch, num_products/1000000,
                         valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5],#valid_acc,
                         train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5],#train_acc,
                         batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], batch_loss[4], batch_loss[5],#batch_acc,
                         time_to_str((timer() - start)/60) ,i,j, ''), end='',flush=True)#str(inputs.size()))

            j = j + 1

            if 0: #if i % 10 ==0:
                debug_2()

        pass  #-- end of one data loader --
    pass #-- end of all iterations --

    if 1: #save last
        save_ckpt(out_dir, net, optimizer, i, epoch)

    log.write('\n')

    def debug_1():
        for inputs, truth_boxes, truth_labels, truth_instances, metas, indices in valid_loader:
            batch_size, C,H,W = inputs.size()
            print('batch_size=%d'%batch_size)

            images = inputs.cpu().numpy()
            for b in range(batch_size):
                image = (images[b].transpose((1,2,0))*255)
                image = np.clip(image.astype(np.float32)*2,0,255)

                contour_overlay = image.copy()
                box_overlay = image.copy()

                truth_box      = truth_boxes[b]
                truth_label    = truth_labels[b]
                truth_instance = truth_instances[b]
                for box,label,instance in zip(truth_box,truth_label,truth_instance):
                    print('label=%d'%label)

                    x0,y0,x1,y1 = box.astype(np.int32)
                    cv2.rectangle(box_overlay,(x0,y0),(x1,y1),(0,0,255),1)

                    mask = instance>0.5
                    contour = mask_to_inner_contour(mask)
                    contour_overlay[contour] = [0,255,0]


                image_show('contour_overlay',contour_overlay)
                image_show('box_overlay',box_overlay)
                cv2.waitKey(0)

    def debug_2():
        net.set_mode('test')
        with torch.no_grad():
            net( inputs, truth_boxes, truth_labels, truth_instances )

        batch_size,C,H,W = inputs.size()
        images = inputs.data.cpu().numpy()
        window          = net.rpn_window
        rpn_logits_flat = net.rpn_logits_flat.data.cpu().numpy()
        rpn_deltas_flat = net.rpn_deltas_flat.data.cpu().numpy()
        rpn_proposals   = net.rpn_proposals.data.cpu().numpy()

        rcnn_logits     = net.rcnn_logits.data.cpu().numpy()
        rcnn_deltas     = net.rcnn_deltas.data.cpu().numpy()
        rcnn_proposals  = net.rcnn_proposals.data.cpu().numpy()

        detections = net.detections.data.cpu().numpy()
        masks      = net.masks



        #print('train',batch_size)
        for b in range(batch_size): 

            image = (images[b].transpose((1,2,0))*255)
            image = image.astype(np.uint8)
            #image = np.clip(image.astype(np.float32)*2,0,255).astype(np.uint8)  #improve contrast

            truth_box      = truth_boxes[b]
            truth_label    = truth_labels[b]
            truth_instance = truth_instances[b]
            truth_mask     = instance_to_multi_mask(truth_instance)

            rpn_logit_flat = rpn_logits_flat[b]
            rpn_delta_flat = rpn_deltas_flat[b]
            rpn_prob_flat  = np_softmax(rpn_logit_flat)

            rpn_proposal = np.zeros((0,7),np.float32)
            if len(rpn_proposals)>0:
                index = np.where(rpn_proposals[:,0]==b)[0]
                rpn_proposal   = rpn_proposals[index]

            rcnn_proposal = np.zeros((0,7),np.float32)
            if len(rcnn_proposals)>0:
                index = np.where(rcnn_proposals[:,0]==b)[0]
                rcnn_logit     = rcnn_logits[index]
                rcnn_delta     = rcnn_deltas[index]
                rcnn_prob      = np_softmax(rcnn_logit)
                rcnn_proposal  = rcnn_proposals[index]

            mask = masks[b]


            #box = proposal[:,1:5]
            #mask = masks[b]

            ## draw --------------------------------------------------------------------------
            #contour_overlay = multi_mask_to_contour_overlay(truth_mask, image, [255,255,0] )
            #color_overlay   = multi_mask_to_color_overlay(mask)

            #all1 = draw_multi_rpn_prob(cfg, image, rpn_prob_flat)
            #all2 = draw_multi_rpn_delta(cfg, overlay_contour, rpn_prob_flat, rpn_delta_flat, window,[0,0,255])
            #all3 = draw_multi_rpn_proposal(cfg, image, proposal)
            #all4 = draw_truth_box(cfg, image, truth_box, truth_label)

            all5 = draw_multi_proposal_metric(cfg, image, rpn_proposal,  truth_box, truth_label,[0,255,255],[255,0,255],[255,255,0])
            all6 = draw_multi_proposal_metric(cfg, image, rcnn_proposal, truth_box, truth_label,[0,255,255],[255,0,255],[255,255,0])
            all7 = draw_mask_metric(cfg, image, mask, truth_box, truth_label, truth_instance)

            # image_show('color_overlay',color_overlay,1)
            # image_show('rpn_prob',all1,1)
            # image_show('rpn_prob',all1,1)
            # image_show('rpn_delta',all2,1)
            # image_show('rpn_proposal',all3,1)
            # image_show('truth_box',all4,1)
            # image_show('rpn_precision',all5,1)
            image_show('rpn_precision', all5,1)
            image_show('rcnn_precision',all6,1)
            image_show('mask_precision',all7,1)


            # summary = np.vstack([
            #     all5,
            #     np.hstack([
            #         all1,
            #         np.vstack( [all2, np.zeros((H,2*W,3),np.uint8)])
            #     ])
            # ])
            # draw_shadow_text(summary, 'iter=%08d'%i,  (5,3*HEIGHT-15),0.5, (255,255,255), 1)
            # image_show('summary',summary,1)

            name = train_dataset.ids[indices[b]].split('/')[-1]
            #cv2.imwrite(out_dir +'/train/%s.png'%name,summary)
            #cv2.imwrite(out_dir +'/train/%05d.png'%b,summary)

            cv2.imwrite(out_dir +'/train/%05d.rpn_precision.png'%b,  all5)
            cv2.imwrite(out_dir +'/train/%05d.rcnn_precision..png'%b,all6)
            cv2.waitKey(1)
            pass

        net.set_mode('train')


# main #################################################################
if __name__ == '__main__':
    c = config['train']
    log_name = c.get('log_name')
    learn_rate = c.getfloat('learn_rate')
    n_epoch = c.getint('n_epoch')
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.add_argument('--epoch', type=int, help='run number of epoch')
    parser.add_argument('--lr', type=float, dest='learn_rate', help='learning rate')
    parser.set_defaults(resume=True, epoch=n_epoch, learn_rate=learn_rate)
    args = parser.parse_args()

    print(' Task name: ', log_name)
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    main(args.resume, args.epoch, args.learn_rate)

    print('\nsucess!')

#  ffmpeg -f image2  -pattern_type glob -r 33 -i "iterations/*.png" -c:v libx264  iterations.mp4
#
#
