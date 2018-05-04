import numpy as np

from scipy import ndimage as ndi
from skimage.morphology import label, watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops
import os
import sys
import numpy as np
import csv
import json
from pathlib import Path
import PIL
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from skimage import exposure
from skimage import morphology
import torch
from common import *


def save_pred2img(uid, pred, pred_file, show=False):
    cmap = plt.get_cmap('prism') # prism for high frequence color bands
    cmap.set_bad('w') # map background(0) as transparent/white
    pred = pred.astype(float)
    pred[pred == 0] = np.nan
    plt.imsave(pred_file, pred, cmap=cmap)
    img = PIL.Image.open(pred_file) # workaroud
    if show:
        print(uid[:5], ': (h, w) = ', pred.shape)
        # refer https://stackoverflow.com/questions/9295026/matplotlib-plots-removing-axis-legends-and-white-spaces
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img, aspect='equal')
        plt.show()
        plt.close()
    return img

def show(x, y, uid, iou=None, save_file=None):
    # x, input image
    # y, predict image (not a predict array that each element is a probability)
    w, h = x.size
    if save_file is None:
        _dpi = 100
        figsize = (24,6) if w < 620 else (25, 5) # why ? try....
    else:
        _dpi = 200
        figsize = (24,6)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=figsize, dpi=_dpi)

    if iou is not None:
        fig.suptitle('{0} / iou = [{1}] = {2}'.format(uid, ', '.join(list(map(str, iou))), np.mean(iou)), y=1)
    else:
        fig.suptitle('{0} / (w, h)={1}'.format(uid, x.size), y=1)
    ax1.set_title('Image')
    ax2.set_title('Predict')
    ax3.set_title('Overlay')
    ax1.imshow(x)
    ax2.imshow(y)
    ax3.imshow(x)
    ax3.imshow(y, alpha=0.3)
    plt.tight_layout()
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)
    plt.close()


def save_fullresolution(x, y, uid, save_file=None):
#    print("Saving as %s" % save_file)
    result = Image.blend(x.convert('RGB'), y.convert('RGB'), alpha=0.2)
    matplotlib.image.imsave(save_file, result)


def rle2png_fullresolutionWithContour(rle_file, pred_dir, pred_img_outdir):
    rle_dict = {}
    rle_file = Path(rle_file)
    pred_dir = Path(pred_dir)
    pred_img_outdir = Path(pred_img_outdir)

    with rle_file.open('r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader) # header = 'ImageId,EncodedPixels'
        for row in reader:
            if len(row) != 2:
                print('[ERROR] ', row)
                raise ValueError
            img_id, rle_str = row
            if rle_str == '0 0' or rle_str == '1 1':
                continue
            if img_id not in rle_dict:
                rle_dict[img_id] = []
            rle_dict[img_id].append(rle_str)

    if not pred_img_outdir.exists():
        pred_img_outdir.mkdir()

    csv_file = pred_img_outdir / 'IoU_results.csv'
    # csv_file = Path(csv_file)
    Flag_write_IoU = 0
    with csv_file.open('w', newline='') as f:
        writer = csv.writer(f)
        thresholds = np.round(np.arange(0.5, 1, 0.05), 2)
        writer.writerow(['image_id'] + list(map(str, thresholds)) + ['Average'])
        img_count = 0
        for img_id in rle_dict:
            img_file = pred_dir / img_id / 'images' / (img_id + '.png')
            if not img_file.exists():
                print('[WARNING] no this image: ', img_id)
                continue
            # print('[YES] have this image: ', img_id)
            img = PIL.Image.open(img_file)
            w, h = img.size
            pred = np.zeros((h, w), dtype=int)
            imgWithContour = exposure.rescale_intensity(np.asarray(img), out_range=(0, 255))

            maxValue = np.max(imgWithContour)
            for i, rle_str in enumerate(rle_dict[img_id], 1):
                mask = rle_decode(rle_str, (h, w))
                mask[mask == 1] = i
                pred = np.maximum(pred, mask)
                contour = np.logical_xor(mask, morphology.binary_erosion(mask))
                try:
                    # imgWithContour[contour > 0, 0] = maxValue
                    imgWithContour[contour > 0] = maxValue
                except IndexError:
                    print('[WARNING] ', img_id, ',', contour.shape, ',', imgWithContour.shape)
                    continue

            masks_dir = pred_dir / img_id / 'masks'
            if masks_dir.exists():
                Flag_write_IoU = 1
                truth_labels, _,_ = label_masks(masks_dir)
                iou = dsb_iou_metric2(pred, truth_labels)
                iou.append(np.mean(iou))
                iou = np.round(iou, 3)
                writer.writerow([img_id] + list(map(str, iou)))

            # save predict image
            pred_file = pred_img_outdir / (img_id + '.png')
            pred_img = save_pred2img(img_id, pred, str(pred_file))
            #show(img, pred_img, img_id, save_file=str(pred_file))
            img_count = img_count + 1
            #print('Processing %3d: %s' % (img_count,img_id))
            save_fullresolution(PIL.Image.fromarray(np.uint8(imgWithContour)), pred_img, img_id, save_file=str(pred_file))

    if Flag_write_IoU == 1:
        print("%s is saved."%csv_file)
    else:
        print("%s is not saved without masks folder."%csv_file)
        os.system("rm %s"%csv_file)

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



##-----------------------------------------------------------------------------------------------------
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn

def evaluate_IoU (y_pred, labels):
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
    IoU = intersection / union
    # Precision helper function
    # Loop over IoU thresholds
    prec = []
    score = 0
    # print("Number of true objects:", true_objects)
    # print("Number of predicted objects:", pred_objects)

    # print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, IoU)
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

def rle_encode(mask):
    pixels = mask.T.flatten()
    # We need to allow for cases where there is a '1' at either end of the sequence.
    # We do this by padding with a zero at each end
    pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
    pixel_padded[1:-1] = pixels
    pixels = pixel_padded
    # 1) pixels[1:], so index + 1.
    # 2) submmit RLE index starts from 1 not 0, so index + 1
    # 3) padding a zero at the head (actually, padding with a zero at each end), so index - 1
    # bcz 1) and 2), need to shift (1 + 1 - 1)=1 index
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 1
    rle[1::2] = rle[1::2] - rle[0::2]
    return rle

# Used only for testing.
# This is copied from https://www.kaggle.com/paulorzp/run-length-encode-and-decode. Thanks to Paulo Pinto.
def rle_decode(rle_str, mask_shape):
    # mask_shpae = (high, width)
    h, w = mask_shape
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1 # bcz submmit start index is 1 not 0, so must substract 1
    ends = starts + lengths
    mask = np.zeros(np.prod(mask_shape), dtype=int)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape((w, h)).T

def image_ids_in(root_dir, train_error_ids=None):
    ids = []
    for id in root_dir.iterdir():
        if id.is_file():
            continue
        id = id.stem
        if train_error_ids is not None and id in train_error_ids:
            print('Skipping ID due to bad training data:', id)
        else:
            ids.append(id)
    return ids

def check_mask_value(mask):
    unique, counts = np.unique(mask, return_counts=True)
    for c, cnt in zip(unique, counts):
        if c != 0 and c != 255:
            print('[ERROR] unreasonable mask: color=', c, 'count=', cnt)

# Evaluate the average nucleus size.
def evaluate_size(image, ratio):
    label_image = label(image)
    label_counts = len(np.unique(label_image))
    #Sort Area sizes:
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    total_area = 0
    #To avoild eval_count ==0
    if int(label_counts * ratio)==0:
        eval_count = 1
    else:
        eval_count = int(label_counts * ratio)
    average_area = np.array(areas[:eval_count]).mean()
    size_index = average_area ** 0.5
    return size_index

# Segment image with watershed algorithm.
def seg_ws(image, size_scale=0.55, ratio=0.5):
    #Calculate the average size of the image.
    size_index = evaluate_size(image, ratio)
    '''
    Add noise to fix min_distance bug:
    If multiple peaks in the specified region have identical intensities,
    the coordinates of all such pixels are returned.
    '''
    noise = np.random.randn(image.shape[0], image.shape[1]) * 0.1
    distance = ndi.distance_transform_edt(image)+noise
    #2*min_distance+1 is the minimum distance between two peaks.
    local_maxi = peak_local_max(distance, min_distance=(size_index*size_scale), exclude_border=False, indices=False, labels=image)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=image)
    return labels

# Evaluate the average nucleus size.
def evaluate_size(image, ratio):
    label_image = label(image)
    label_counts = len(np.unique(label_image))
    #Sort Area sizes:
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    total_area = 0
    #To avoild eval_count ==0
    if int(label_counts * ratio)==0:
        eval_count = 1
    else:
        eval_count = int(label_counts * ratio)
    average_area = np.array(areas[:eval_count]).mean()
    size_index = average_area ** 0.5
    return size_index

# Segment image with watershed algorithm.
def seg_ws(image, size_scale=0.55, ratio=0.5):
    #Calculate the average size of the image.
    size_index = evaluate_size(image, ratio)
    '''
    Add noise to fix min_distance bug:
    If multiple peaks in the specified region have identical intensities,
    the coordinates of all such pixels are returned.
    '''
    noise = np.random.randn(image.shape[0], image.shape[1]) * 0.1
    distance = ndi.distance_transform_edt(image)+noise
    #2*min_distance+1 is the minimum distance between two peaks.
    local_maxi = peak_local_max(distance, min_distance=(size_index*size_scale), exclude_border=False, indices=False, labels=image)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=image)
    return labels

def label_masks(masks_dir):
    def check_mask_value(mask):
        unique, counts = np.unique(mask, return_counts=True)
        for c, cnt in zip(unique, counts):
            if c != 0 and c != 255:
                print('[ERROR] unreasonable mask: color=', c, 'count=', cnt)

    masks = np.array([])
    h, w = 0, 0
    for i, maskfile in enumerate(masks_dir.glob('*.png'), 1):
        mask = PIL.Image.open(str(maskfile)).convert('L')
        w, h = mask.size
        mask = np.array(mask)
        check_mask_value(mask)
        mask[mask == 255] = i
        if len(masks) == 0:
            masks = np.zeros((h, w), dtype=int)
        masks = np.maximum(masks, mask)
    return masks, h, w

def label_pred(img, pred_threshold=0.5, is_watershed=True):
    if is_watershed:
        return seg_ws(img > pred_threshold)
    return label(img > pred_threshold)

def dsb_iou_metric1(y_pred_in, masks_dir):
    # y_pred_in is np.array directly from the model output
    labels, _, _ = label_masks(masks_dir)
    # resize y_pred_in to the original size
    y_pred = label_pred(y_pred_in)
    return dsb_iou_metric2(y_pred, labels)

def dsb_iou_metric2(y_pred, labels):
    # y_pred and labels are segmented

    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

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
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    for t in np.round(np.arange(0.5, 1, 0.05), 2):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        prec.append(p)

    return prec

def pad_image(img, pad_w, pad_h, mode='replicate'):
    if mode == 'constant':
        # padding color should honor each image background, default is black (0)
        bgcolor = 'black' if np.median(img) < 100 else 'white'
        img = ImageOps.expand(img, (0, 0, pad_w, pad_h), bgcolor)
        return img
    elif mode == 'replicate':
        # replicate each border pixel's color
        x = np.asarray(img)
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        x = cv2.copyMakeBorder(x, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        if isinstance(img, Image.Image):
            x = Image.fromarray(x)
        return x
    else:
        raise NotImplementedError()

# checkpoint handling
def ckpt_path(out_dir, iter=None, epoch=None):
    checkpoint_dir = os.path.join(out_dir, 'checkpoint')
    current_path = os.path.join(out_dir, 'checkpoint', 'current.json')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if epoch is None:
        if os.path.exists(current_path):
            with open(current_path) as infile:
                data = json.load(infile)
                iter = data['iter']
                epoch = data['epoch']
        else:
            return ''
    else:
        with open(current_path, 'w') as outfile:
            json.dump({
                'iter': iter,
                'epoch': epoch
            }, outfile)
    return os.path.join(checkpoint_dir, 'ckpt-{}.pkl'.format(int(np.around(iter))))

def save_ckpt(out_dir, net, optimizer, iter, epoch):
    ckpt = ckpt_path(out_dir, iter, epoch)
    '''
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, ckpt)
    '''
    torch.save({
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter'     : iter,
        'epoch'    : epoch,
    }, ckpt)
    # with open(out_dir + '/checkpoint/configuration.pkl', 'wb') as pickle_file:
    #    pickle.dump(net.cfg, pickle_file, pickle.HIGHEST_PROTOCOL)


def load_ckpt(out_dir, net, optimizer=None):
    #ckpt = ckpt_path(out_dir)
    ckpt = config['param'].get('model_used')
    iter, epoch = 0, 0
    if os.path.isfile(ckpt):
        print("Loading checkpoint '{}'".format(ckpt))
        if torch.cuda.is_available():
            # Load all tensors onto previous state
            checkpoint = torch.load(ckpt)
        else:
            # Load all tensors onto the CPU
            checkpoint = torch.load(ckpt, map_location=lambda storage, loc: storage)
        iter = checkpoint['iter']
        epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['model'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
    return iter, epoch

def filter_fiber(blobs):
    objects = [(obj.area, obj.eccentricity, obj.label) for obj in regionprops(blobs)]
    objects = sorted(objects, reverse=True) # sorted by area in descending order
    # filter out the largest one which is (1) 5 times larger than 2nd largest one (2) eccentricity > 0.95
    if len(objects) > 1 and objects[0][0] > 5 * objects[1][0] and objects[0][1] > 0.95:
        print('\nfilter suspecious fiber', objects[0])
        blobs = np.where(blobs==objects[0][2], 0, blobs)
    return blobs
