from common import config
import os
import random
import json
import cv2
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

# Ignore skimage convertion warnings
import warnings
warnings.filterwarnings("ignore")

from utility.helper import pad_image
from .reader import multi_mask_to_annotation
from .transform import random_shift_scale_rotate_transform2,\
                        random_crop_transform2,\
                        random_horizontal_flip_transform2,\
                        random_vertical_flip_transform2,\
                        random_rotate90_transform2,\
                        fix_crop_transform2,\
                        pad_to_factor

WIDTH = config['maskrcnn'].getint('width')
HEIGHT = WIDTH

def train_augment(image, multi_mask, meta, index, label):
    # print('train_augment1> image:', index, image.shape) # image: 996 (512, 512, 3)
    # print('train_augment1> multi_mask:', index, multi_mask.shape) # multi_mask: 996 (512, 512)
    image, multi_mask = random_shift_scale_rotate_transform2( image, multi_mask,
                        shift_limit=[0,0], scale_limit=[1/2,2],
                        rotate_limit=[-45,45], borderMode=cv2.BORDER_REFLECT_101, u=0.5) #borderMode=cv2.BORDER_CONSTANT

    # overlay = multi_mask_to_color_overlay(multi_mask,color='cool')
    # overlay1 = multi_mask_to_color_overlay(multi_mask1,color='cool')
    # image_show('overlay',overlay)
    # image_show('overlay1',overlay1)
    # cv2.waitKey(0)

    image, multi_mask = random_crop_transform2(image, multi_mask, WIDTH, HEIGHT, u=0.5)
    image, multi_mask = random_horizontal_flip_transform2(image, multi_mask, 0.5)
    image, multi_mask = random_vertical_flip_transform2(image, multi_mask, 0.5)
    image, multi_mask = random_rotate90_transform2(image, multi_mask, 0.5)
    ##image,  multi_mask = fix_crop_transform2(image, multi_mask, -1,-1,WIDTH, HEIGHT)

    #---------------------------------------
    # print('train_augment2> image:', index, image.shape, type(image)) # image: 996 (256, 256, 3)
    # print('train_augment2> multi_mask:', index, multi_mask.shape, type(multi_mask)) # multi_mask: 996 (256, 256)
    input = torch.from_numpy(image.transpose((2,0,1))).float().div(255)
    # print('train_augment3> input:', index, input.shape) # input: 996 torch.Size([3, 256, 256])
    box, label, instance  = multi_mask_to_annotation(multi_mask, label)

    return input, box, label, instance, meta, index


def valid_augment(image, multi_mask, meta, index, label):

    image,  multi_mask = fix_crop_transform2(image, multi_mask, -1,-1,WIDTH, HEIGHT)

    #---------------------------------------
    input = torch.from_numpy(image.transpose((2,0,1))).float().div(255)
    box, label, instance  = multi_mask_to_annotation(multi_mask, label)

    return input, box, label, instance, meta, index

#-----------------------------------------------------------------------------------
def submit_augment(image, index, id):
    pad_image = pad_to_factor(image, factor=16)
    input = torch.from_numpy(pad_image.transpose((2,0,1))).float().div(255)
    return input, image, index, id


class Compose():
    def __init__(self, augment=True, padding=False, resizing=False, tensor=True):
        c = config['param']
        model_name = c.get('model')
        self.gcd_depth = c.getint('gcd_depth')
        width = config[model_name].getint('width')
        self.size = (width, width)

        c = config['pre']
        self.mean = json.loads(c.get('mean'))
        self.std = json.loads(c.get('std'))
        # self.label_binary = c.getboolean('label_to_binary')
        self.color_invert = c.getboolean('color_invert')
        self.color_jitter = c.getboolean('color_jitter')
        self.elastic_distortion = c.getboolean('elastic_distortion')
        self.color_equalize = c.getboolean('color_equalize')
        self.tensor = tensor
        self.augment = augment
        self.padding = padding
        self.resizing = resizing
        self.min_scale = c.getfloat('min_scale')
        self.max_scale = c.getfloat('max_scale')

    def __call__(self, image, multi_mask, meta, index, cat_id):
        # print('Compose1> ', index, image.shape, multi_mask.shape, cat_id) #Compose1>  996 (512, 512, 3) (512, 512) 3
        image = Image.fromarray(image)
        image_size = image.size
        label_gt = Image.fromarray(multi_mask)

        if self.augment:
            if self.color_equalize and random.random() > 0.5:
                image = clahe(image)

            # perform RandomResize() or just enlarge for image size < model input size
            if random.random() > 0.5:
                new_size = int(random.uniform(self.min_scale, self.max_scale) * np.min(image.size))
            else:
                new_size = int(np.min(image.size))
            if new_size < np.max(self.size): # make it viable for cropping
                new_size = int(np.max(self.size))
            image = tx.resize(image, new_size)
            # label_gt use NEAREST instead of BILINEAR (default) to avoid polluting instance labels after augmentation
            label_gt = tx.resize(label_gt, new_size, interpolation=Image.NEAREST)

            # perform RandomCrop()
            i, j, h, w = transforms.RandomCrop.get_params(image, self.size)
            image, label_gt = [tx.crop(x, i, j, h, w) for x in (image, label_gt)]

            # Note: RandomResizedCrop() is popularly used to train the Inception networks, but might not the best choice for segmentation?
            # # perform RandomResizedCrop()
            # i, j, h, w = transforms.RandomResizedCrop.get_params(
            #     image,
            #     scale=(0.5, 1.0)
            #     ratio=(3. / 4., 4. / 3.)
            # )
            # # label_gt use NEAREST instead of BILINEAR (default) to avoid polluting instance labels after augmentation
            # image, label, label_c, label_m = [tx.resized_crop(x, i, j, h, w, self.size) for x in (image, label, label_c, label_m)]
            # label_gt = tx.resized_crop(label_gt, i, j, h, w, self.size, interpolation=Image.NEAREST)

            # perform Elastic Distortion
            if self.elastic_distortion and random.random() > 0.75:
                indices = ElasticDistortion.get_params(image)
                image = ElasticDistortion.transform(image, indices)
                label_gt = ElasticDistortion.transform(label_gt, indices, spline_order=0) # spline_order=0 to avoid polluting instance labels

            # perform RandomHorizontalFlip()
            if random.random() > 0.5:
                image, label_gt = [tx.hflip(x) for x in (image, label_gt)]

            # perform RandomVerticalFlip()
            if random.random() > 0.5:
                image, label_gt = [tx.vflip(x) for x in (image, label_gt)]

            # perform Random Rotation (0, 90, 180, and 270 degrees)
            random_degree = random.randint(0, 3) * 90
            image, label_gt = [tx.rotate(x, random_degree) for x in (image, label_gt)]

            # perform random color invert, assuming 3 channels (rgb) images
            if self.color_invert and random.random() > 0.5:
                image = ImageOps.invert(image)

            # perform ColorJitter()
            if self.color_jitter and random.random() > 0.5:
                color = transforms.ColorJitter.get_params(0.5, 0.5, 0.5, 0.25)
                image = color(image)

        elif self.padding: # add border padding
            w, h = image_size
            gcd = self.gcd_depth
            pad_w = pad_h = 0
            if 0 != (w % gcd):
                pad_w = gcd - (w % gcd)
            if 0 != (h % gcd):
                pad_h = gcd - (h % gcd)
            image = pad_image(image, pad_w, pad_h)
            label_gt = ImageOps.expand(label_gt, (0, 0, pad_w, pad_h))

        else: # resize down image
            image = tx.resize(image, self.size)
            label_gt = tx.resize(label_gt, self.size, interpolation=Image.NEAREST)

        '''
        # Due to resize algorithm may introduce anti-alias edge, aka. non binary value,
        # thereafter map every pixel back to 0 and 255
        if self.label_binary:
            label = label.point(lambda p, threhold=100: 255 if p > threhold else 0)
        '''

        # _img = np.array(image), for print message
        multi_mask = np.array(label_gt)
        # print('compose2> image:', index, _img.shape) # image: 996 (256, 256, 3)
        # print('compose2> multi_mask:', index, multi_mask.shape) # multi_mask: 996 (256, 256)
        mask_box, mask_cat, mask_inst = multi_mask_to_annotation(multi_mask, cat_id)

        # perform ToTensor()
        if self.tensor:
            # image = torch.from_numpy(image.transpose((2,0,1))).float().div(255)
            image = tx.to_tensor(image)
            # print('compose3> image:', index, image.shape) # input: 996 torch.Size([3, 256, 256])
            # perform Normalize()
            image = tx.normalize(image, self.mean, self.std)

        '''
        # prepare a shadow copy of composed data to avoid screwup cached data
        x = sample.copy()
        x['image'], x['label'], x['label_gt'], x['mask_box'], x['mask_type'], x['mask_inst'] = \
                image, label, label_gt, mask_box, mask_type, mask_inst
        '''

        return image, mask_box, mask_cat, mask_inst, meta, index

    def denorm(self, tensor):
        tensor = tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

    def pil(self, tensor):
        return tx.to_pil_image(tensor)

    '''
    def show(self, sample):
        image, label_gt = \
                sample['image'], sample['label_gt']
        for x in (image, label_gt):
            if x.dim == 4:  # only dislay first sample
                x = x[0]
            if x.shape[0] > 1: # channel > 1
                x = self.denorm(x)
            x = self.pil(x)
            x.show()
    '''

def compose_mask(masks, pil=False):
    result = np.zeros_like(masks[0], dtype=np.int32)
    for i, m in enumerate(masks):
        mask = np.array(m) if pil else m.copy()
        mask[mask > 0] = i + 1 # zero for background, starting from 1
        result = np.maximum(result, mask) # overlay mask one by one via np.maximum, to handle overlapped labels if any
    if pil:
        result = Image.fromarray(result)
    return result

def decompose_mask(mask):
    num = mask.max()
    result = []
    for i in range(1, num+1):
        m = mask.copy()
        m[m != i] = 0
        m[m == i] = 255
        result.append(m)
    return result

# Note: the algorithm MUST guarantee (interior + contour = instance mask) & (interior within contour)
# def get_contour_interior(uid, mask):
#     contour = filters.scharr(mask)
#     scharr_threshold = np.amax(abs(contour)) / 2.
#     if uid in bright_field_list:
#         scharr_threshold = 0. # nuclei are much smaller than others in bright_field slice
#     contour = (np.abs(contour) > scharr_threshold).astype(np.uint8)*255
#     interior = (mask - contour > 0).astype(np.uint8)*255
#     return contour, interior
def get_contour_interior(uid, mask):
    # Note: find_boundaries() only have 1-pixel wide contour,
    #       use "dilation - twice erosion" to have 2-pixel wide contour
    # contour = find_boundaries(mask, connectivity=1, mode='inner')
    boundaries = dilation(mask) != erosion(erosion(mask))
    foreground_image = (mask != 0)
    boundaries &= foreground_image
    contour = (boundaries > 0).astype(np.uint8)*255
    interior = (mask - contour > 0).astype(np.uint8)*255
    return contour, interior

def get_instances_contour_interior(uid, instances_mask):
    result_c = np.zeros_like(instances_mask, dtype=np.uint8)
    result_i = np.zeros_like(instances_mask, dtype=np.uint8)
    weight = np.ones_like(instances_mask, dtype=np.float32)
    masks = decompose_mask(instances_mask)
    for m in masks:
        contour, interior = get_contour_interior(uid, m)
        result_c = np.maximum(result_c, contour)
        result_i = np.maximum(result_i, interior)
        # magic number 50 make weight distributed to [1, 5) roughly
        weight *= (1 + gaussian_filter(contour, sigma=1) / 50)
    return result_c, result_i, weight

def clahe(img):
    x = np.asarray(img, dtype=np.uint8)
    x = equalize_adapthist(x)
    x = img_as_ubyte(x)
    return Image.fromarray(x)


class ElasticDistortion():
    """Elastic deformation of image as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    def __init__(self):
        pass

    @staticmethod
    def get_params(img, alpha=2000, sigma=30):
        w, h = img.size
        dx = gaussian_filter((np.random.rand(*(h, w)) * 2 - 1),
                            sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((np.random.rand(*(h, w)) * 2 - 1),
                            sigma, mode="constant", cval=0) * alpha
        x, y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
        return indices

    @staticmethod
    def transform(img, indices, spline_order=1, mode='nearest'):
        x = np.asarray(img)
        if x.ndim == 2:
            x = np.expand_dims(x, -1)
        shape = x.shape[:2]
        result = np.empty_like(x)
        for i in range(x.shape[2]):
            result[:, :, i] = map_coordinates(
                x[:, :, i], indices, order=spline_order, mode=mode).reshape(shape)
        if result.shape[-1] == 1:
            result = np.squeeze(result)
        return Image.fromarray(result, mode=img.mode)

    def __call__(self, img, spline_order=1, mode='nearest'):
        """
        Args:
            img (PIL Image): Image to be transformed.
        Returns:
            PIL Image: Randomly distorted image.
        """
        indices = self.get_params(img)
        return self.transform(img, indices)

if __name__ == '__main__':
    compose = Compose(augment=True)
    print('Nothing happened')
