from common import *
import configparser
import json

#
# proposal i,x0,y0,x1,y1,score, label, (scale_level)
# roi      i,x0,y0,x1,y1
# box        x0,y0,x1,y1



class Configuration(object):

    def __init__(self):
        super(Configuration, self).__init__()
        self.version='configuration version \'mask-rcnn-resnet50-fpn-kaggle\''

        # net
        self.classes_map = json.loads(config['maskrcnn'].get('classes_map'))
        self.num_classes = int(np.fromiter(self.classes_map.values(), dtype=int).max()) + 1 #include background class

        self.rpn_single_class       = False
        self.rpn_p0_pool            = False
        self.crop_one_layer         = False
        self.high_resolution        = True

        #multi-rpn
        self.rpn_base_sizes         = [ 8, 16, 32, 64, 128 ] #diameter
        if self.high_resolution == True: 
            self.rpn_scales         = [ 1,  2,  4,  8, 16]
            self.rpn_stride         = 2
            self.layer0_stride      = 1
        else:
            self.rpn_scales         = [ 2,  4,  8,  16, 32]
            self.rpn_stride         = 1
            self.layer0_stride      = 2
        aspect = lambda s,x: (s*1/x**0.5,s*x**0.5)
        # self.rpn_base_apsect_ratios = [
        #     [(1,1) ],
        #     [(1,1),                    aspect(2**0.33,2), aspect(2**0.33,0.5),],
        #     [(1,1), aspect(2**0.66,1), aspect(2**0.33,2), aspect(2**0.33,0.5),aspect(2**0.33,3), aspect(2**0.33,0.33),  ],
        #     [(1,1), aspect(2**0.66,1), aspect(2**0.33,2), aspect(2**0.33,0.5),],
        # ]
        self.rpn_base_apsect_ratios = [
            [(1,1) ],
            [(1,1), aspect(2**0.5,2), aspect(2**0.5,0.5),],
            [(1,1), aspect(2**0.5,2), aspect(2**0.5,0.5),],
            # [(1,1), aspect(2**0.5,2), aspect(2**0.5,0.5),],
            # [(1,1), aspect(2**0.5,2), aspect(2**0.5,0.5),(2,2),(3,3)],
            [(1,1), aspect(2**0.5,2), aspect(2**0.5,0.5),],
            [(1,1), aspect(2**0.5,2), aspect(2**0.5,0.5),],
            # [(1,1), aspect(2**0.5,2), aspect(2**0.5,0.5),(2,2)],
        
        ]


        self.rpn_train_bg_thresh_high = 0.5
        self.rpn_train_fg_thresh_low  = 0.5

        self.rpn_train_nms_pre_score_threshold = 0.7
        self.rpn_train_nms_overlap_threshold   = 0.8  #higher for more proposals for mask training
        self.rpn_train_nms_min_size = 5

        self.rpn_test_nms_pre_score_threshold = 0.8
        self.rpn_test_nms_overlap_threshold   = 0.5
        self.rpn_test_nms_min_size = 5


        #rcnn
        if self.high_resolution == True: 
            self.rcnn_crop_size     = 7
        else:    
            self.rcnn_crop_size     = 14
        self.rcnn_train_batch_size  = 64 #per image
        self.rcnn_train_fg_fraction = 0.5
        self.rcnn_train_fg_thresh_low  = 0.5
        self.rcnn_train_bg_thresh_high = 0.5
        self.rcnn_train_bg_thresh_low  = 0.0

        self.rcnn_train_nms_pre_score_threshold = 0.05
        self.rcnn_train_nms_overlap_threshold   = 0.8  # high for more proposals for mask
        self.rcnn_train_nms_min_size = 5

        self.rcnn_test_nms_pre_score_threshold = 0.3
        self.rcnn_test_nms_overlap_threshold   = 0.5
        self.rcnn_test_nms_min_size = 5

        #mask
        self.mask_crop_size            = 14
        self.mask_train_batch_size     = 64 #per image
        self.mask_size                 = 28 #per image
        self.mask_train_min_size       = 5
        self.mask_train_fg_thresh_low  = self.rpn_train_fg_thresh_low

        self.mask_test_nms_pre_score_threshold = 0.4  #self.rpn_test_nms_pre_score_threshold
        self.mask_test_nms_overlap_threshold = 0.1
        self.mask_test_mask_threshold  = 0.5



    #-------------------------------------------------------------------------------------------------------
    def __repr__(self):
        d = self.__dict__.copy()
        str=''
        for k, v in d.items():
            str +=   '%32s = %s\n' % (k,v)

        return str


    def save(self, file):
        d = self.__dict__.copy()
        cfg = configparser.ConfigParser()
        cfg['all'] = d
        with open(file, 'w') as f:
            cfg.write(f)


    def load(self, file):
        # cfg = configparser.ConfigParser()
        # cfg.read(file)
        #
        # d = cfg['all']
        # self.num_classes     = eval(d['num_classes'])
        # self.multi_num_heads = eval(d['multi_num_heads'])

        raise NotImplementedError
