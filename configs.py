#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
from default_configs import DefaultConfigs

class configs(DefaultConfigs):

    def __init__(self):

        self.gpu = '4'
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu

        #########################
        #         I/O           #
        #########################

        # one out of [2, 3]. dimension the model operates in.
        self.dim = 3

        # one out of ['mrcnn', 'seg_mrcnn','retina_net', 'retina_unet', 'detection_unet', 'ufrcnn', 'detection_unet'].
        self.model = 'mrcnn'

        DefaultConfigs.__init__(self, self.model, self.dim)

        # path to preprocessed data.
        self.pp_name = 'abus_npy'
        self.input_df_name = 'info_df.pickle'
        self.input_id_name = 'fold_ids.pickle'
        self.pid_pth = 'fold_ids.txt'
        self.pp_data_path = '/shenlab/lab_stor6/yuezhou/ABUSdata/{}/'.format(self.pp_name)
        self.pp_test_data_path = self.pp_data_path #change if test_data in separate folder.

        #########################
        #      Data Loader      #
        #########################

        # data aug in training
        self.data_aug_training = False
        # select modalities from preprocessed data
        self.channels = [0]
        self.n_channels = len(self.channels)

        # patch_size to be used for training. pre_crop_size is the patch_size before data augmentation.
        #self.pre_crop_size_2D = [300, 300]
        #self.patch_size_2D = [288, 288]
        if self.data_aug_training == True:
            self.pre_crop_size_3D = [ 72,144,144]
        else:
            self.pre_crop_size_3D = [64,128,128]
        self.patch_size_3D = [64,128,128]#[128, 128, 64]
        self.patch_size = self.patch_size_3D#self.patch_size_2D if self.dim == 2 else self.patch_size_3D
        self.pre_crop_size = self.pre_crop_size_3D#self.pre_crop_size_2D if self.dim == 2 else self.pre_crop_size_3D

        # ratio of free sampled batch elements before class balancing is triggered
        # (>0 to include "empty"/background patches.)
        self.batch_sample_slack = 0.5#0.2

        #########################
        #      Architecture      #
        #########################

        self.backbone_path = 'models/backbone_vnet.py'
        self.multi_scale_det = True 
        self.start_filts = 48 if self.dim == 2 else 18
        if 'vnet' in self.backbone_path:
            if self.multi_scale_det == False:
                self.end_filts = [32,64,128,256,256] 
            else:
                self.end_filts = [36,36,36,36,36] #self.start_filts * 4 if self.dim == 2 else self.start_filts * 2
        if 'fpn' in self.backbone_path:
            self.end_filts = [36,36,36,36,36] #self.start_filts * 4 if self.dim == 2 else self.start_filts * 2
        self.res_architecture = 'resnet50' # 'resnet101' , 'resnet50'
        self.norm = None # one of None, 'instance_norm', 'batch_norm'
        self.weight_decay = 0

        # one of 'xavier_uniform', 'xavier_normal', or 'kaiming_normal', None (=default = 'kaiming_uniform')
        self.weight_init = None

        #########################
        #  Schedule / Selection #
        #########################
        self.debug = 0 
        if self.debug == 1:
            self.num_epochs = 2 
            self.num_train_batches = 2#2 if self.dim == 2 else 2 
            self.batch_size = 2#20 if self.dim == 2 else 2 
            self.n_workers = 1
        else:
            self.num_epochs = 250 
            self.num_train_batches = 200 if self.dim == 2 else 200 
            self.batch_size = 20 if self.dim == 2 else 2 
            self.n_workers = 16

        self.do_validation = True
        # decide whether to validate on entire patient volumes (like testing) or sampled patches (like training)
        # the former is morge accurate, while the latter is faster (depending on volume size)
        self.val_mode = 'val_sampling' # this is a bug only sampling one of 'val_sampling' , 'val_patient'
        if self.debug == 1:
            self.num_val_batches = 2
        else:
            self.num_val_batches = None 

        #########################
        # loss  #
        #########################
        # in ['BCE','roiDice','mapDice']
        self.mask_loss_flag = 'BCE'
        # in ['seg-only','mrcnn-only','frcnn-only','mrcnn-seg','frcnn-seg','mrcnn-seg-fusion']
        self.loss_flag = 'mrcnn-seg-fusion'
        #########################
        # fusion method  #
        #########################
        #fusion in prob or feature
        self.fusion_prob_feature = 'prob'
        # in ['cat-only','add-only','weight-cat','weight-add']
        self.fusion_method = 'add-only'
        # in ['before','after']
        self.fusion_feature_method = 'after'
        #fusion conoral number
        self.fusion_conv_num = 'less'#'no' or 'more'

        #########################
        #   Testing / Plotting  #
        #########################
        # show detection box score
        self.show_det_source_th = 0.1
        # patch stride during testing
        self.testing_patch_stride = [64,128,128]
        # set the top-n-epochs to be saved for temporal averaging in testing.
        self.save_n_models = 5 
        self.test_n_epochs = 5
        self.testing_epoch_num = 0
        # test the best epoch or the last epoch 
        self.test_last_epoch = False 
        # show image
        if self.debug == 1:
            self.show_train_images = 1 
            self.show_val_images = 1
        else:
            self.show_train_images = 5 
            self.show_val_images = 5

        #select detected box score
        #self.source_th = 0.1

        # set a minimum epoch number for saving in case of instabilities in the first phase of training.
        self.min_save_thresh = 0 if self.dim == 2 else 0

        self.report_score_level = ['patient', 'rois']  # choose list from 'patient', 'rois'
        self.class_dict = {1:'mass'}#{1: 'benign', 2: 'malignant'}  # 0 is background.
        self.patient_class_of_interest = 1  # patient metrics are only plotted for one class.
        self.ap_match_ious = [0.1]  # list of ious to be evaluated for ap-scoring.

        #['val_dice_seg','val_dice_mask','val_dice_fusion']#criteria to average over for saving epochs.
        if 'seg' in self.loss_flag and 'seg-only' not in self.loss_flag:
            self.model_selection_criteria = ['val_dice_fusion']#criteria to average over for saving epochs.
        if 'seg-only' in self.loss_flag:
            self.model_selection_criteria = ['val_deice_seg']
        if 'seg' not in self.loss_flag:
            self.model_selection_criteria = ['val_precision','val_recall']
        self.min_det_thresh = 0.1  # minimum confidence value to select predictions for evaluation.

        #########################
        #   Data Augmentation   #
        #########################

        self.da_kwargs={
        'do_elastic_deform': True,
        'alpha':(0., 1500.),
        'sigma':(30., 50.),
        'do_rotation':True,
        'angle_x': (0., 2 * np.pi),
        'angle_y': (0., 0),
        'angle_z': (0., 0),
        'do_scale': True,
        'scale':(0.8, 1.1),
        'random_crop':False,
        'rand_crop_dist':  (self.patch_size[0] / 2. - 3, self.patch_size[1] / 2. - 3),
        'border_mode_data': 'constant',
        'border_cval_data': 0,
        'order_data': 1
        }

        if self.dim == 3:
            self.da_kwargs['do_elastic_deform'] = False
            self.da_kwargs['angle_x'] = (0, 0.0)
            self.da_kwargs['angle_y'] = (0, 0.0) #must be 0!!
            self.da_kwargs['angle_z'] = (0., 2 * np.pi)


        #########################
        #   Add model specifics #
        #########################

        {'detection_unet': self.add_det_unet_configs,
         'mrcnn': self.add_mrcnn_configs,
         'ufrcnn': self.add_mrcnn_configs,
         'retina_net': self.add_mrcnn_configs,
         'retina_unet': self.add_mrcnn_configs,
        }[self.model]()


    def add_det_unet_configs(self):

        self.learning_rate = [1e-4] * self.num_epochs

        # aggregation from pixel perdiction to object scores (connected component). One of ['max', 'median']
        self.aggregation_operation = 'max'

        # max number of roi candidates to identify per batch element and class.
        self.n_roi_candidates = 10 if self.dim == 2 else 30

        # loss mode: either weighted cross entropy ('wce'), batch-wise dice loss ('dice), or the sum of both ('dice_wce')
        self.seg_loss_mode = 'dice_wce'

        # if <1, false positive predictions in foreground are penalized less.
        self.fp_dice_weight = 1 if self.dim == 2 else 1

        self.wce_weights = [1, 1, 1]
        self.detection_min_confidence = self.min_det_thresh

        # if 'True', loss distinguishes all classes, else only foreground vs. background (class agnostic).
        self.class_specific_seg_flag = True
        #self.class_specific_seg_flag = False 
        self.num_seg_classes = 1 if self.class_specific_seg_flag else 2
        self.head_classes = self.num_seg_classes

    def add_mrcnn_configs(self):
        # learning rate is a list with one entry per epoch.
        self.decrease_lr = 100 
        self.initial_learning_rate = 1e-4#[1e-4] * self.decrease_lr + [1e-5] * (self.num_epochs - self.decrease_lr)
        self.learning_rate_ratio = 0.1

        # disable the re-sampling of mask proposals to original size for speed-up.
        # since evaluation is detection-driven (box-matching) and not instance segmentation-driven (iou-matching),
        # mask-outputs are optional.
        self.return_masks_in_val = True
        self.return_masks_in_train = True
        self.return_masks_in_test = True 

        # set number of proposal boxes to plot after each epoch.
        self.n_plot_rpn_props = 5 if self.dim == 2 else 30

        # number of classes for head networks: n_foreground_classes + 1 (background)
        self.head_classes = 2 

        # seg_classes hier refers to the first stage classifier (RPN)
        self.num_seg_classes = 1  # foreground vs. background

        # feature map strides per pyramid level are inferred from architecture.
        if 'fpn' in self.backbone_path:
            self.backbone_strides = {'xy': [4, 8, 16, 32], 'z': [4 ,8 ,16 ,32]}
            self.rpn_anchor_scales = {'xy': [[8], [16], [32], [64]], 'z': [[2], [4], [8], [16]]}
        if 'vnet' in self.backbone_path:
            self.backbone_strides = {'xy':[1,2,4,8,16],'z':[1,2,4,8,16]}
            self.rpn_anchor_scales = {'xy': [[2],[4],[8],[16],[32]], 'z': [[2],[4],[8],[16],[32]]}
        # anchor scales are chosen according to expected object sizes in data set. Default uses only one anchor scale
        # per pyramid level. (outer list are pyramid levels (corresponding to BACKBONE_STRIDES), inner list are scales per level.)

        # choose which pyramid levels to extract features from: P2: 0, P3: 1, P4: 2, P5: 3.
        # for vnet [0,1,2,3,4]
        if self.multi_scale_det == False:
            self.pyramid_levels = [4]
        else:
            self.pyramid_levels = [0,1,2,3,4]

        # number of feature maps in rpn. typically lowered in 3D to save gpu-memory.
        self.n_rpn_features = 512 if self.dim == 2 else 128

        # anchor ratios and strides per position in feature maps.
        self.rpn_anchor_ratios = [0.5, 1, 2]
        self.rpn_anchor_stride = 1

        # Threshold for first stage (RPN) non-maximum suppression (NMS):  LOWER == HARDER SELECTION
        self.rpn_nms_threshold = 0.7 if self.dim == 2 else 0.7

        # loss sampling settings.
        self.rpn_train_anchors_per_image = 6  #per batch element
        self.train_rois_per_image = 6 #per batch element
        self.roi_positive_ratio = 0.5
        self.anchor_matching_iou = 0.7

        # factor of top-k candidates to draw from  per negative sample (stochastic-hard-example-mining).
        # poolsize to draw top-k candidates from will be shem_poolsize * n_negative_samples.
        self.shem_poolsize = 10

        self.pool_size = (3,7,7)#(7, 7) if self.dim == 2 else (7, 7, 3)
        self.mask_pool_size = (5,14,14)#(14, 14) if self.dim == 2 else (14, 14, 5)
        self.mask_shape = (10,28,28)#(28, 28) if self.dim == 2 else (28, 28, 10)

        self.rpn_bbox_std_dev = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
        self.bbox_std_dev = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
        self.window = np.array([0, 0, self.patch_size[0], self.patch_size[1], 0, self.patch_size_3D[2]])
        self.scale = np.array([self.patch_size[0], self.patch_size[1], self.patch_size[0], self.patch_size[1],
                               self.patch_size_3D[2], self.patch_size_3D[2]])
        if self.dim == 2:
            self.rpn_bbox_std_dev = self.rpn_bbox_std_dev[:4]
            self.bbox_std_dev = self.bbox_std_dev[:4]
            self.window = self.window[:4]
            self.scale = self.scale[:4]

        # pre-selection in proposal-layer (stage 1) for NMS-speedup. applied per batch element.
        self.pre_nms_limit = 3000 if self.dim == 2 else 6000

        # n_proposals to be selected after NMS per batch element. too high numbers blow up memory if "detect_while_training" is True,
        # since proposals of the entire batch are forwarded through second stage in as one "batch".
        self.roi_chunk_size = 2500 if self.dim == 2 else 600
        self.post_nms_rois_training = 75#500 if self.dim == 2 else 75
        self.post_nms_rois_inference = 500

        # Final selection of detections (refine_detections)
        self.model_max_instances_per_batch_element = 10 if self.dim == 2 else 30  # per batch element and class. used in refinedetection after num3D
        self.detection_nms_threshold = 1e-5  # needs to be > 0, otherwise all predictions are one cluster.
        self.model_min_confidence = 0.1#select detection boxes

        if self.dim == 2:
            self.backbone_shapes = np.array(
                [[int(np.ceil(self.patch_size[0] / stride)),
                  int(np.ceil(self.patch_size[1] / stride))]
                 for stride in self.backbone_strides['xy']])
        else:
            self.backbone_shapes = np.array(
                [[int(np.ceil(self.patch_size[0] / stride)),
                  int(np.ceil(self.patch_size[1] / stride)),
                  int(np.ceil(self.patch_size[2] / stride_z))]
                 for stride, stride_z in zip(self.backbone_strides['xy'], self.backbone_strides['z']
                                             )])

        if self.model == 'ufrcnn':
            self.operate_stride1 = True
            self.class_specific_seg_flag = False#true 
            self.num_seg_classes = 3 if self.class_specific_seg_flag else 2
            self.frcnn_mode = True

        if self.model == 'retina_net' or self.model == 'retina_unet' or self.model == 'prob_detector':
            # implement extra anchor-scales according to retina-net publication.
            self.rpn_anchor_scales['xy'] = [[ii[0], ii[0] * (2 ** (1 / 3)), ii[0] * (2 ** (2 / 3))] for ii in
                                            self.rpn_anchor_scales['xy']]
            self.rpn_anchor_scales['z'] = [[ii[0], ii[0] * (2 ** (1 / 3)), ii[0] * (2 ** (2 / 3))] for ii in
                                           self.rpn_anchor_scales['z']]
            self.n_anchors_per_pos = len(self.rpn_anchor_ratios) * 3

            self.n_rpn_features = 256 if self.dim == 2 else 64

            # pre-selection of detections for NMS-speedup. per entire batch.
            self.pre_nms_limit = 10000 if self.dim == 2 else 50000

            # anchor matching iou is lower than in Mask R-CNN according to https://arxiv.org/abs/1708.02002
            self.anchor_matching_iou = 0.5

            # if 'True', seg loss distinguishes all classes, else only foreground vs. background (class agnostic).
            self.num_seg_classes = 3 if self.class_specific_seg_flag else 2

            if self.model == 'retina_unet':
                self.operate_stride1 = True
