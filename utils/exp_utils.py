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

import numpy as np
import logging
import subprocess
import os
import torch
from collections import OrderedDict
import plotting
import sys
import importlib.util
import pandas as pd
import pickle

def convert_seg_to_bounding_box_coordinates(data_dict, dim, get_rois_from_seg_flag=False, class_specific_seg_flag=False):

    '''
    This function generates bounding box annotations from given pixel-wise annotations.
    :param data_dict: Input data dictionary as returned by the batch generator.
    :param dim: Dimension in which the model operates (2 or 3).
    :param get_rois_from_seg: Flag specifying one of the following scenarios:
    1. A label map with individual ROIs identified by increasing label values, accompanied by a vector containing
    in each position the class target for the lesion with the corresponding label (set flag to False)
    2. A binary label map. There is only one foreground class and single lesions are not identified.
    All lesions have the same class target (foreground). In this case the Dataloader runs a Connected Component
    Labelling algorithm to create processable lesion - class target pairs on the fly (set flag to True).
    :param class_specific_seg_flag: if True, returns the pixelwise-annotations in class specific manner,
    e.g. a multi-class label map. If False, returns a binary annotation map (only foreground vs. background).
    :return: data_dict: same as input, with additional keys:
    - 'bb_target': bounding box coordinates (b, n_boxes, (y1, x1, y2, x2, (z1), (z2)))
    - 'roi_labels': corresponding class labels for each box (b, n_boxes, class_label)
    - 'roi_masks': corresponding binary segmentation mask for each lesion (box). Only used in Mask RCNN. (b, n_boxes, y, x, (z))
    - 'seg': now label map (see class_specific_seg_flag)
    '''

    bb_target = []
    roi_masks = []
    roi_labels = []
    out_seg = np.copy(data_dict['seg'])
    for b in range(data_dict['seg'].shape[0]):

        p_coords_list = []
        p_roi_masks_list = []
        p_roi_labels_list = []

        if np.sum(data_dict['seg'][b]!=0) > 0:
            if get_rois_from_seg_flag:
                clusters, n_cands = lb(data_dict['seg'][b])
                data_dict['class_target'][b] = [data_dict['class_target'][b]] * n_cands
            else:
                n_cands = int(np.max(data_dict['seg'][b]))
                clusters = data_dict['seg'][b]

            rois = np.array([(clusters == ii) * 1 for ii in range(1, n_cands + 1)])  # separate clusters and concat
            for rix, r in enumerate(rois):
                if np.sum(r !=0) > 0: #check if the lesion survived data augmentation
                    seg_ixs = np.argwhere(r != 0)
                    coord_list = [np.min(seg_ixs[:, 1])-1, np.min(seg_ixs[:, 2])-1, np.max(seg_ixs[:, 1])+1,
                                     np.max(seg_ixs[:, 2])+1]
                    if dim == 3:

                        coord_list.extend([np.min(seg_ixs[:, 3])-1, np.max(seg_ixs[:, 3])+1])

                    p_coords_list.append(coord_list)
                    p_roi_masks_list.append(r)
                    # add background class = 0. rix is a patient wide index of lesions. since 'class_target' is
                    # also patient wide, this assignment is not dependent on patch occurrances.
                    p_roi_labels_list.append(data_dict['class_target'][b][rix] + 1)

                if class_specific_seg_flag:
                    out_seg[b][data_dict['seg'][b] == rix + 1] = data_dict['class_target'][b][rix] + 1

            if not class_specific_seg_flag:
                out_seg[b][data_dict['seg'][b] > 0] = 1

            bb_target.append(np.array(p_coords_list))
            roi_masks.append(np.array(p_roi_masks_list).astype('uint8'))
            roi_labels.append(np.array(p_roi_labels_list))


        else:
            bb_target.append([])
            roi_masks.append(np.zeros_like(data_dict['seg'][b])[None])
            roi_labels.append(np.array([-1]))

    if get_rois_from_seg_flag:
        data_dict.pop('class_target', None)

    #data_dict['bb_target'] = np.array(bb_target)
    #data_dict['roi_masks'] = np.array(roi_masks)
    #data_dict['class_target'] = np.array(roi_labels)
    #data_dict['seg'] = out_seg

    return np.array(bb_target) 


def learning_rate_decreasing(cf,epoch,lr_now,mode='step'):
    if mode == 'step':
        if epoch % cf.decrease_lr == 0:
            lr_next = lr_now * cf.learning_rate_ratio
        else:
            lr_next = lr_now
    return lr_next

    

def get_logger(exp_dir):
    """
    creates logger instance. writing out info to file and to terminal.
    :param exp_dir: experiment directory, where exec.log file is stored.
    :return: logger instance.
    """

    logger = logging.getLogger('medicaldetectiontoolkit')
    logger.setLevel(logging.DEBUG)
    log_file = exp_dir + '/exec.log'
    hdlr = logging.FileHandler(log_file)
    logger.addHandler(hdlr)
    logger.addHandler(ColorHandler())
    logger.propagate = False
    return logger



#def prep_exp(dataset_path, exp_path, server_env, use_stored_settings=True, is_training=True):
def prep_exp(exp_path, is_training=True):
    """
    I/O handling, creating of experiment folder structure. Also creates a snapshot of configs/model scripts and copies them to the exp_dir.
    This way the exp_dir contains all info needed to conduct an experiment, independent to changes in actual source code. Thus, training/inference of this experiment can be started at anytime. Therefore, the model script is copied back to the source code dir as tmp_model (tmp_backbone).
    Provides robust structure for cloud deployment.
    :param dataset_path: path to source code for specific data set. (e.g. medicaldetectiontoolkit/lidc_exp)
    :param exp_path: path to experiment directory.
    :param server_env: boolean flag. pass to configs script for cloud deployment.
    :param use_stored_settings: boolean flag. When starting training: If True, starts training from snapshot in existing experiment directory, else creates experiment directory on the fly using configs/model scripts from source code.
    :param is_training: boolean flag. distinguishes train vs. inference mode.
    :return:
    """

    if is_training:

        # the first process of an experiment creates the directories and copies the config to exp_path.
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)
            #os.mkdir(os.path.join(exp_path, 'plots'))
            #subprocess.call('cp {} {}'.format(os.path.join(dataset_path, 'configs.py'), os.path.join(exp_path, 'configs.py')), shell=True)
            #subprocess.call('cp {} {}'.format( 'configs.py', os.path.join(exp_path, 'configs.py')), shell=True)
            #subprocess.call('cp {} {}'.format('default_configs.py', os.path.join(exp_path, 'default_configs.py')), shell=True)

        #else:
            # run training with source code info and copy snapshot of model to exp_dir for later testing (overwrite scripts if exp_dir already exists.)
        cf_file = import_module('cf','configs.py')
        cf = cf_file.configs()
        cf.exp_dir = exp_path
        subprocess.call('cp {} {}'.format(cf.model_path, os.path.join(exp_path, 'model.py')), shell=True)
        subprocess.call('cp {} {}'.format(cf.backbone_path, os.path.join(exp_path, cf.backbone_path.split('/')[-1])), shell=True)
        subprocess.call('cp {} {}'.format('default_configs.py', os.path.join(exp_path, 'default_configs.py')), shell=True)
        subprocess.call('cp {} {}'.format('configs.py', os.path.join(exp_path, 'configs.py')), shell=True)

    else:
        # for testing, copy the snapshot model scripts from exp_dir back to the source_dir as tmp_model / tmp_backbone.
        cf_file = import_module('cf', os.path.join(exp_path, 'configs.py'))
        cf = cf_file.configs()
        tmp_model_path = os.path.join(exp_path, 'model.py')
        bp = os.listdir(exp_path)
        for bpp in bp:
            if 'backbone' in bpp:
                tmp_backbone_path = os.path.join(exp_path,bpp)
        cf.model_path = tmp_model_path
        cf.backbone_path = tmp_backbone_path
        cf.exp_dir = exp_path
        cf.test_dir = os.path.join(cf.exp_dir, 'test/')
        cf.plot_dir = os.path.join(cf.exp_dir, 'plots/')

        if not os.path.exists(cf.test_dir):
            os.makedirs(cf.test_dir)
        if not os.path.exists(cf.plot_dir):
            os.makedirs(cf.plot_dir)

    cf.experiment_name = exp_path.split("/")[-1]
    cf.created_fold_id_pickle = False

    return cf



def import_module(name, path):
    """
    correct way of importing a module dynamically in python 3.
    :param name: name given to module instance.
    :param path: path to module.
    :return: module: returned module instance.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def save_models(cf,net,optimizer,epoch,recall):
    state = {
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    # save checkpoint of current epoch.
    save_dir = os.path.join(cf.fold_dir, 'checkpoint'.format(epoch))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save(state, os.path.join(save_dir, 'params_{}_{}.pth'.format(epoch,recall)))


class ModelSelector:
    '''
    saves a checkpoint after each epoch as 'last_state' (can be loaded to continue interrupted training).
    saves the top-k (k=cf.save_n_models) ranked epochs. In inference, predictions of multiple epochs can be ensembled to improve performance.
    '''

    def __init__(self, cf, logger):
        self.cf = cf
        self.saved_epochs = [-1] * cf.save_n_models#5
        self.logger = logger

    def run_model_selection(self, net, optimizer, monitor_metrics, epoch):

        # take the mean over all selection criteria in each epoch
        non_nan_scores = np.mean(np.array([[0 if ii is None else ii for ii in monitor_metrics['val'][sc]] for sc in self.cf.model_selection_criteria]), 0)
        epochs_scores = [ii for ii in non_nan_scores[1:]]
        # ranking of epochs according to model_selection_criterion
        epoch_ranking = np.argsort(epochs_scores)[::-1] + 1 #epochs start at 1
        # if set in configs, epochs < min_save_thresh are discarded from saving process.
        epoch_ranking = epoch_ranking[epoch_ranking >= self.cf.min_save_thresh]#0
        # check if current epoch is among the top-k epchs.
        if epoch in epoch_ranking[:self.cf.save_n_models]:

            save_dir = os.path.join(self.cf.fold_dir, '{}_best_checkpoint'.format(epoch))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            torch.save(net.state_dict(), os.path.join(save_dir, 'params.pth'))
            with open(os.path.join(save_dir, 'monitor_metrics.pickle'), 'wb') as handle:
                pickle.dump(monitor_metrics, handle)
            # save epoch_ranking to keep info for inference.
            np.save(os.path.join(self.cf.fold_dir, 'epoch_ranking'), epoch_ranking[:self.cf.save_n_models])
            np.save(os.path.join(save_dir, 'epoch_ranking'), epoch_ranking[:self.cf.save_n_models])

            self.logger.info(
                "saving current epoch {} at rank {}".format(epoch, np.argwhere(epoch_ranking == epoch)))
            # delete params of the epoch that just fell out of the top-k epochs.
            for se in [int(ii.split('_')[0]) for ii in os.listdir(self.cf.fold_dir) if 'best_checkpoint' in ii]:
                if se in epoch_ranking[self.cf.save_n_models:]:
                    subprocess.call('rm -rf {}'.format(os.path.join(self.cf.fold_dir, '{}_best_checkpoint'.format(se))), shell=True)
                    self.logger.info('deleting epoch {} at rank {}'.format(se, np.argwhere(epoch_ranking == se)))

        state = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint of current epoch.
        save_dir = os.path.join(self.cf.fold_dir, 'last_checkpoint'.format(epoch))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(state, os.path.join(save_dir, 'params.pth'))
        np.save(os.path.join(save_dir, 'epoch_ranking'), epoch_ranking[:self.cf.save_n_models])
        with open(os.path.join(save_dir, 'monitor_metrics.pickle'), 'wb') as handle:
            pickle.dump(monitor_metrics, handle)



def load_checkpoint(checkpoint_path, net, optimizer):

    checkpoint_params = torch.load(os.path.join(checkpoint_path, 'params.pth'))
    net.load_state_dict(checkpoint_params['state_dict'])
    optimizer.load_state_dict(checkpoint_params['optimizer'])
    #with open(os.path.join(checkpoint_path, 'monitor_metrics.pickle'), 'rb') as handle:
    #    monitor_metrics = pickle.load(handle)
    starting_epoch = checkpoint_params['epoch'] + 1
    #return starting_epoch, monitor_metrics
    return starting_epoch



def prepare_monitoring(cf):
    """
    creates dictionaries, where train/val metrics are stored.
    """
    metrics = {}
    # first entry for loss dict accounts for epoch starting at 1.
    metrics['train'] = OrderedDict()
    metrics['val'] = OrderedDict()
    #metric_classes = []
    #print('in prepare_monitoring')
    #if 'rois' in cf.report_score_level:#['patients,rois']
    #    metric_classes.extend([v for k, v in cf.class_dict.items()])#{'1':benign,'2':malignant}
    #if 'patient' in cf.report_score_level:
    #    metric_classes.extend(['patient'])
    ##print('metric_classes',metric_classes)
    #for cl in metric_classes:#benign malignant patient
    #    metrics['train'][cl + '_ap'] = [None]
    #    metrics['val'][cl + '_ap'] = [None]
    #    if cl == 'patient':
    #        metrics['train'][cl + '_auc'] = [None]
    #        metrics['val'][cl + '_auc'] = [None]
    metrics['train']['train_recall'] = [None]
    metrics['train']['train_percision'] = [None]
    metrics['val']['val_recall'] = [None]
    metrics['val']['val_precision'] = [None]
    metrics['val']['val_dice_seg'] = [None]
    metrics['val']['val_dice_mask'] = [None]
    metrics['val']['val_dice_fusion'] = [None]
    metrics['train']['monitor_values'] = [[] for _ in range(cf.num_epochs + 1)]
    metrics['val']['monitor_values'] = [[] for _ in range(cf.num_epochs + 1)]

    # generate isntance of monitor plot class.
    #TrainingPlot = plotting.TrainingPlot_2Panel(cf)

    #return metrics, TrainingPlot
    return metrics


def create_csv_output(results_list, cf, logger):
    """
    Write out test set predictions to .csv file. output format is one line per prediction:
    PatientID | PredictionID | [y1 x1 y2 x2 (z1) (z2)] | score | pred_classID
    Note, that prediction coordinates correspond to images as loaded for training/testing and need to be adapted when
    plotted over raw data (before preprocessing/resampling).
    :param results_list: [[patient_results, patient_id], [patient_results, patient_id], ...]
    """

    logger.info('creating csv output file at {}'.format(os.path.join(cf.exp_dir, 'results.csv')))
    predictions_df = pd.DataFrame(columns = ['patientID', 'predictionID', 'coords', 'score', 'pred_classID'])
    for r in results_list:

        pid = r[1]

        #optionally load resampling info from preprocessing to match output predictions with raw data.
        #with open(os.path.join(cf.exp_dir, 'test_resampling_info', pid), 'rb') as handle:
        #    resampling_info = pickle.load(handle)

        for bix, box in enumerate(r[0][0]):
            assert box['box_type'] == 'det', box['box_type']
            coords = box['box_coords']
            score = box['box_score']
            pred_class_id = box['box_pred_class_id']
            out_coords = []
            if score >= cf.min_det_thresh:
                out_coords.append(coords[0]) #* resampling_info['scale'][0])
                out_coords.append(coords[1]) #* resampling_info['scale'][1])
                out_coords.append(coords[2]) #* resampling_info['scale'][0])
                out_coords.append(coords[3]) #* resampling_info['scale'][1])
                if len(coords) > 4:
                    out_coords.append(coords[4]) #* resampling_info['scale'][2] + resampling_info['z_crop'])
                    out_coords.append(coords[5]) #* resampling_info['scale'][2] + resampling_info['z_crop'])

                predictions_df.loc[len(predictions_df)] = [pid, bix, out_coords, score, pred_class_id]
    try:
        fold = cf.fold
    except:
        fold = 'hold_out'
    predictions_df.to_csv(os.path.join(cf.exp_dir, 'results_{}.csv'.format(fold)), index=False)



class _AnsiColorizer(object):
    """
    A colorizer is an object that loosely wraps around a stream, allowing
    callers to write text to the stream in a particular color.

    Colorizer classes must implement C{supported()} and C{write(text, color)}.
    """
    _colors = dict(black=30, red=31, green=32, yellow=33,
                   blue=34, magenta=35, cyan=36, white=37, default=39)

    def __init__(self, stream):
        self.stream = stream

    @classmethod
    def supported(cls, stream=sys.stdout):
        """
        A class method that returns True if the current platform supports
        coloring terminal output using this method. Returns False otherwise.
        """
        if not stream.isatty():
            return False  # auto color only on TTYs
        try:
            import curses
        except ImportError:
            return False
        else:
            try:
                try:
                    return curses.tigetnum("colors") > 2
                except curses.error:
                    curses.setupterm()
                    return curses.tigetnum("colors") > 2
            except:
                raise
                # guess false in case of error
                return False

    def write(self, text, color):
        """
        Write the given text to the stream in the given color.

        @param text: Text to be written to the stream.

        @param color: A string label for a color. e.g. 'red', 'white'.
        """
        color = self._colors[color]
        self.stream.write('\x1b[%sm%s\x1b[0m' % (color, text))



class ColorHandler(logging.StreamHandler):


    def __init__(self, stream=sys.stdout):
        super(ColorHandler, self).__init__(_AnsiColorizer(stream))

    def emit(self, record):
        msg_colors = {
            logging.DEBUG: "green",
            logging.INFO: "default",
            logging.WARNING: "red",
            logging.ERROR: "red"
        }
        color = msg_colors.get(record.levelno, "blue")
        self.stream.write(record.msg + "\n", color)

