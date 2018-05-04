from common import *
from net.lib.box.process import*
from utility.draw import *




def make_empty_masks(cfg, mode, inputs):#<todo>
    masks = []
    batch_size,C,H,W = inputs.size()
    for b in range(batch_size):
        mask = np.zeros((H, W), np.float32)
        masks.append(mask)
    return masks





# def mask_nms( cfg, mode, inputs, proposals, mask_logits):
#
#     score_threshold = cfg.mask_test_score_threshold
#     mask_threshold  = cfg.mask_test_mask_threshold
#
#     proposals   = proposals.cpu().data.numpy()
#     mask_logits = mask_logits.cpu().data.numpy()
#     mask_probs  = np_sigmoid(mask_logits)
#
#     masks = []
#     batch_size,C,H,W = inputs.size()
#     for b in range(batch_size):
#         mask  = np.zeros((H,W),np.float32)
#         index = np.where(proposals[:,0]==b)[0]
#
#         instance_id=1
#         if len(index) != 0:
#             for i in index:
#                 p = proposals[i]
#                 prob = p[5]
#                 #print(prob)
#                 if prob>score_threshold:
#                     x0,y0,x1,y1 = p[1:5].astype(np.int32)
#                     h, w = y1-y0+1, x1-x0+1
#                     label = int(p[6]) #<todo>
#                     crop = mask_probs[i, label]
#                     crop = cv2.resize(crop, (w,h), interpolation=cv2.INTER_LINEAR)
#                     crop = crop>mask_threshold
#
#                     mask[y0:y1+1,x0:x1+1] = crop*instance_id + (1-crop)*mask[y0:y1+1,x0:x1+1]
#                     instance_id = instance_id+1
#
#                 if 0: #<debug>
#
#                     images = inputs.data.cpu().numpy()
#                     image = (images[b].transpose((1,2,0))*255).astype(np.uint8)
#                     image = np.clip(image.astype(np.float32)*4,0,255)
#
#                     image_show('image',image,2)
#                     image_show('mask',mask/mask.max()*255,2)
#                     cv2.waitKey(1)
#
#             #<todo>
#             #non-max-suppression to remove overlapping segmentation
#
#         masks.append(mask)
#     return masks

def mask_nms_1( cfg, mode, inputs, proposals, mask_probs):
    
    nms_threshold = cfg.mask_test_nms_pre_score_threshold
    threshold = cfg.mask_test_mask_threshold


    mask_probs  = mask_probs.cpu().data.numpy()
    proposals = proposals.cpu().data.numpy()

    masks = []
    batch_size,C,H,W = inputs.size()
    # print ('batch_size', batch_size)
    for b in range(batch_size):
        # print ('b',b)
        mask  = np.zeros((H,W),np.float32)
        index = np.where(proposals[:,0]==b)[0]

        instance_id=1
        if len(index) != 0:
            # print ('proposals', proposals[:,5])
            for i in index:
                p = proposals[i]
                prob = p[5]
                what = p[0]

                if prob>nms_threshold:
                    x0,y0,x1,y1 = p[1:5].astype(np.int32)
                    h, w = y1-y0+1, x1-x0+1
                    label = int(p[6]) #<todo>
                    crop = mask_probs[i, label] # 16x16
                    crop = cv2.resize(crop, (w,h), interpolation=cv2.INTER_LINEAR)
                    crop = crop>threshold
                    # print ('crop', crop)
                    # print ('1-crop', (1-crop))
                    mask_one = np.zeros((H,W),np.float32)
                    # print ('shape', crop.shape)
                    mask_one[y0:y1+1,x0:x1+1] = crop
                    area_one = np.count_nonzero(mask_one)
                    # print (area_one)

                    mask_true = mask > 0
                    mask_overlap = mask_one * mask_true
                    area_overlap = np.count_nonzero(mask_overlap)
                    if area_one == 0: continue
                    if (area_overlap / area_one > 0.5): continue
                    # print ('area_one', area_one, np.unique(mask_one))
                    # print ('area_overlap', area_overlap, np.unique(mask_overlap))
                    crop_one = mask_one * (1-mask_overlap)
                    area_crop_one = np.count_nonzero(crop_one)
                    # print ('area_crop_one', area_crop_one, np.unique(crop_one))
                    # mask[y0:y1+1,x0:x1+1] = crop*instance_id + (1-crop)*mask[y0:y1+1,x0:x1+1]
                    mask = crop_one*instance_id + (1-crop_one)*mask
                    # print ('mask', mask[y0:y1+1,x0:x1+1])
                    instance_id = instance_id+1

                if 0: #<debug>

                    images = inputs.data.cpu().numpy()
                    image = (images[b].transpose((1,2,0))*255).astype(np.uint8)
                    image = np.clip(image.astype(np.float32)*4,0,255)

                    image_show('image',image,2)
                    image_show('mask',mask/mask.max()*255,2)
                    cv2.waitKey(0)

            #<todo>
            #non-max-suppression to remove overlapping segmentation

        masks.append(mask)
    return masks

def mask_nms( cfg, mode, inputs, proposals, mask_logits):
    #images = (inputs.data.cpu().numpy().transpose((0,2,3,1))*255).astype(np.uint8)

    overlap_threshold   = cfg.mask_test_nms_overlap_threshold
    pre_score_threshold = cfg.mask_test_nms_pre_score_threshold
    mask_threshold      = cfg.mask_test_mask_threshold

    proposals   = proposals.cpu().data.numpy()
    mask_logits = mask_logits.cpu().data.numpy()
    mask_probs  = np_sigmoid(mask_logits)
    # print ('proposals at the first', proposals)
    masks = []
    keeps = []
    category_labels = []
    batch_size,C,H,W = inputs.size()
    for b in range(batch_size):
        mask  = np.zeros((H,W),np.float32)
        # get image category
        label_list = proposals[:,6]
        unique, counts = np.unique(label_list, return_counts = True)
        label_dict = dict(zip(unique, counts))
        label_sorted = sorted(label_dict.items(), key = lambda item:item[1], reverse = True)
        # print ("label_sorted",label_sorted)
        label_category  = label_sorted[0][0]
#   
        # print ("label_sorted",label_sorted[0][1])
        index = np.where((proposals[:,0]==b) & (proposals[:,5]>pre_score_threshold) & (proposals[:,6]==int(label_category)))[0]
        keep = []
        if len(index) != 0:

            instance=[]
            box=[]
            for i in index:
                m = np.zeros((H,W),np.bool)

                x0,y0,x1,y1 = proposals[i,1:5].astype(np.int32)
                h, w  = y1-y0+1, x1-x0+1
                label = int(proposals[i,6])
                crop  = mask_probs[i, label]
                crop  = cv2.resize(crop, (w,h), interpolation=cv2.INTER_LINEAR)
                crop  = crop > mask_threshold
                m[y0:y1+1,x0:x1+1] = crop

                instance.append(m)
                box.append((x0,y0,x1,y1))

                #<debug>----------------------------------------------
                if 0:

                    images = inputs.data.cpu().numpy()
                    image = (images[b].transpose((1,2,0))*255).astype(np.uint8)
                    image = np.clip(image.astype(np.float32)*4,0,255)

                    image_show('image',image,2)
                    image_show('mask',mask/mask.max()*255,2)
                    cv2.waitKey(1)

                #<debug>----------------------------------------------
            instance = np.array(instance,np.bool)
            box      = np.array(box, np.float32)

            #compute overlap
            box_overlap = cython_box_overlap(box, box)

            L = len(index)
            instance_overlap = np.zeros((L,L),np.float32)
            for i in range(L):
                instance_overlap[i,i] = 1
                for j in range(i+1,L):
                    if box_overlap[i,j]<0.01: continue

                    x0 = int(min(box[i,0],box[j,0]))
                    y0 = int(min(box[i,1],box[j,1]))
                    x1 = int(max(box[i,2],box[j,2]))
                    y1 = int(max(box[i,3],box[j,3]))

                    intersection = (instance[i,y0:y1,x0:x1] & instance[j,y0:y1,x0:x1]).sum()
                    area = (instance[i,y0:y1,x0:x1] | instance[j,y0:y1,x0:x1]).sum()
                    instance_overlap[i,j] = intersection/(area + 1e-12)
                    instance_overlap[j,i] = instance_overlap[i,j]

            #non-max suppress
            # print ('index before', index)
            score_dict = {}
            for i, one_index in enumerate(index):
                score_dict[one_index] = proposals[i,5]
            # print ('score_dict  before', score_dict)
            score_dict = sorted(score_dict.items(), key = lambda item:item[1], reverse = True)
            # print ('score_dict after ', score_dict)
            # score = proposals[index,5]
            new_index = []
            for i in range(len(index)):
                
                index_temp = score_dict[i][0]
                new_index.append(index_temp)
                # print ('new_index', new_index)
            # print ('index', new_index)
            # index = list(np.argsort(-score))
            # print ('proposals', proposals)
            # print ('score', score)
            # print ('index', index)
            score = proposals[index,5]

            new_index = list(np.argsort(-score))
            ##  https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
            keep = []
            while len(new_index) > 0:
                i = new_index[0]
                keep.append(index[i])
                delete_index = list(np.where(instance_overlap[i] > overlap_threshold)[0])
                new_index =  [e for e in new_index if e not in delete_index]

                #<todo> : merge?

            for i,k in enumerate(keep):
                ins_index = index.tolist().index(k)
                mask[np.where(instance[ins_index])] = i+1
        
        category_labels.append(label_category)
        keeps.append(keep)
        masks.append(mask)
    return masks, keeps, category_labels, label_sorted

##-----------------------------------------------------------------------------  
#if __name__ == '__main__':
#    print( '%s: calling main function ... ' % os.path.basename(__file__))
#
#
#
# 
 
