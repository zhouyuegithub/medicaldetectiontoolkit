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

"""execution script."""

import argparse
import os
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import utils.exp_utils as utils
from evaluator import Evaluator 
from predictor import Predictor
from plotting import plot_batch_prediction, save_monitor_valuse,save_test_image
from tensorboardX import SummaryWriter
from utils.exp_utils import save_models
import utils.model_utils as mutils
import pickle
from models.mrcnn import compute_vnet_seg_loss

def train(logger):
    """
    perform the training routine for a given fold. saves plots and selected parameters to the experiment dir
    specified in the configs.
    """
    logger.info('performing training in {}D over fold {} on experiment {} with model {}'.format(
        cf.dim, cf.fold, cf.exp_dir, cf.model))
    
    writer = SummaryWriter(os.path.join(cf.exp_dir,'tensorboard'))

    net = model.net(cf, logger).cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=cf.learning_rate[0], weight_decay=cf.weight_decay)

    model_selector = utils.ModelSelector(cf, logger)
    train_evaluator = Evaluator(cf, logger, mode='train')
    val_evaluator = Evaluator(cf, logger, mode=cf.val_mode)#val_sampling

    starting_epoch = 1

    # prepare monitoring
    if cf.resume_to_checkpoint:#default: False
        lastepochpth = cf.resume_to_checkpoint + 'last_checkpoint/'
        best_epoch = np.load(lastepochpth + 'epoch_ranking.npy')[0] 
        df = open(lastepochpth+'monitor_metrics.pickle','rb')
        monitor_metrics = pickle.load(df)
        df.close()
        starting_epoch = utils.load_checkpoint(lastepochpth, net, optimizer)
        logger.info('resumed to checkpoint {} at epoch {}'.format(cf.resume_to_checkpoint, starting_epoch))
        num_batch = starting_epoch * cf.num_train_batches+1
        num_val = starting_epoch * cf.num_val_batches+1
    else:
        monitor_metrics = utils.prepare_monitoring(cf)
        num_batch = 0#for show loss
        num_val = 0
    logger.info('loading dataset and initializing batch generators...')
    batch_gen = data_loader.get_train_generators(cf, logger)
    best_train_recall,best_val_recall = 0,0
    for epoch in range(starting_epoch, cf.num_epochs + 1):

        logger.info('starting training epoch {}'.format(epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = cf.learning_rate[epoch - 1]

        start_time = time.time()

        net.train()
        train_results_list = []#this batch
        train_results_list_seg = []

        for bix in range(cf.num_train_batches):#200
            num_batch += 1
            batch = next(batch_gen['train'])#data,seg,pid,class_target,bb_target,roi_masks,roi_labels
            for ii,i in enumerate(batch['roi_labels']):
                if i[0] > 0:
                    batch['roi_labels'][ii] = [1]
                else:
                    batch['roi_labels'][ii] = [-1]

            tic_fw = time.time()
            results_dict = net.train_forward(batch)
            tic_bw = time.time()

            optimizer.zero_grad()
            results_dict['torch_loss'].backward()#total loss
            optimizer.step()
            
            if (num_batch) % cf.show_train_images == 0:
                fig = plot_batch_prediction(batch, results_dict, cf,'train')
                writer.add_figure('/Train/results',fig,num_batch)
                fig.clear()
            print('model',cf.exp_dir.split('/')[-2])
            logger.info('tr. batch {0}/{1} (ep. {2}) fw {3:.3f}s / bw {4:.3f}s / total {5:.3f}s || '
                        .format(bix + 1, cf.num_train_batches, epoch, tic_bw - tic_fw,
                                time.time() - tic_bw, time.time() - tic_fw) + results_dict['logger_string'])

            writer.add_scalar('Train/total_loss',results_dict['torch_loss'].item(),num_batch)
            writer.add_scalar('Train/rpn_class_loss',results_dict['monitor_losses']['rpn_class_loss'],num_batch)
            writer.add_scalar('Train/rpn_bbox_loss',results_dict['monitor_losses']['rpn_bbox_loss'],num_batch)
            writer.add_scalar('Train/mrcnn_class_loss',results_dict['monitor_losses']['mrcnn_class_loss'],num_batch)
            writer.add_scalar('Train/mrcnn_bbox_loss',results_dict['monitor_losses']['mrcnn_bbox_loss'],num_batch)
            writer.add_scalar('Train/mrcnn_mask_loss',results_dict['monitor_losses']['mrcnn_mask_loss'],num_batch)
            writer.add_scalar('Train/seg_dice_loss',results_dict['monitor_losses']['seg_loss_dice'],num_batch)
            writer.add_scalar('Train/fusion_dice_loss',results_dict['monitor_losses']['fusion_loss_dice'],num_batch)

            train_results_list.append([results_dict['boxes'], batch['pid']])#just gt and det
            monitor_metrics['train']['monitor_values'][epoch].append(results_dict['monitor_losses'])

        count_train = train_evaluator.evaluate_predictions(train_results_list,epoch,cf,flag = 'train')
        precision = count_train[0]/ (count_train[0]+count_train[2]+0.01)
        recall = count_train[0]/ (count_train[3])
        print('tp_patient {}, tp_roi {}, fp_roi {}, total_num {}'.format(count_train[0],count_train[1],count_train[2],count_train[3]))
        print('precision:{}, recall:{}'.format(precision,recall))
        monitor_metrics['train']['train_recall'].append(recall)
        monitor_metrics['train']['train_percision'].append(precision)
        writer.add_scalar('Train/train_precision',precision,epoch)
        writer.add_scalar('Train/train_recall',recall,epoch)
        train_time = time.time() - start_time

        logger.info('starting validation in mode {}.'.format(cf.val_mode))
        with torch.no_grad():
            net.eval()
            if cf.do_validation:
                val_results_list = []
                val_predictor = Predictor(cf, net, logger, mode='val')
                dice_val_seg,dice_val_mask,dice_val_fusion = [], [], []
                for _ in range(batch_gen['n_val']):#50
                    num_val += 1
                    batch = next(batch_gen[cf.val_mode])
                    print('eval',batch['pid'])
                    for ii,i in enumerate(batch['roi_labels']):
                        if i[0] > 0:
                            batch['roi_labels'][ii] = [1]
                        else:
                            batch['roi_labels'][ii] = [-1]
                    if cf.val_mode == 'val_patient':
                        results_dict = val_predictor.predict_patient(batch)#result of one patient
                    elif cf.val_mode == 'val_sampling':
                        results_dict = net.train_forward(batch, is_validation=True)
                    if (num_val) % cf.show_val_images == 0:
                        fig = plot_batch_prediction(batch, results_dict, cf, cf.val_mode)
                        writer.add_figure('Val/results',fig,num_val)
                        fig.clear()

                    # compute dice for vnet
                    this_batch_seg_label = torch.FloatTensor(mutils.get_one_hot_encoding(batch['seg'], cf.num_seg_classes+1)).cuda()
                    if cf.fusion_feature_method == 'after':
                        this_batch_dice_seg = mutils.dice_val(results_dict['seg_logits'],this_batch_seg_label)
                    else:
                        this_batch_dice_seg = mutils.dice_val(F.softmax(results_dict['seg_logits'],dim=1),this_batch_seg_label)
                    dice_val_seg.append(this_batch_dice_seg)
                    # compute dice for mask 
                    #mask_map = torch.from_numpy(results_dict['seg_preds']).cuda()
                    if cf.fusion_feature_method == 'after':
                        this_batch_dice_mask = mutils.dice_val(results_dict['seg_preds'],this_batch_seg_label)
                    else:
                        this_batch_dice_mask = mutils.dice_val(F.softmax(results_dict['seg_preds'], dim=1),this_batch_seg_label)
                    dice_val_mask.append(this_batch_dice_mask)
                    # compute dice for fusion
                    if cf.fusion_feature_method == 'after':
                        this_batch_dice_fusion = mutils.dice_val(results_dict['fusion_map'],this_batch_seg_label)
                    else:
                        this_batch_dice_fusion = mutils.dice_val(F.softmax(results_dict['fusion_map'], dim=1),this_batch_seg_label)
                    dice_val_fusion.append(this_batch_dice_fusion)

                    val_results_list.append([results_dict['boxes'], batch['pid']])
                    monitor_metrics['val']['monitor_values'][epoch].append(results_dict['monitor_values'])

                count_val = val_evaluator.evaluate_predictions(val_results_list,epoch,cf,flag = 'val')
                precision = count_val[0]/ (count_val[0]+count_val[2]+0.01)
                recall = count_val[0]/ (count_val[3])
                print('tp_patient {}, tp_roi {}, fp_roi {}, total_num {}'.format(count_val[0],count_val[1],count_val[2],count_val[3]))
                print('precision:{}, recall:{}'.format(precision,recall))
                val_dice_seg = sum(dice_val_seg)/float(len(dice_val_seg))
                val_dice_mask = sum(dice_val_mask)/float(len(dice_val_mask))
                val_dice_fusion = sum(dice_val_fusion)/float(len(dice_val_fusion))
                monitor_metrics['val']['val_recall'].append(recall)
                monitor_metrics['val']['val_precision'].append(precision) 
                monitor_metrics['val']['val_dice_seg'].append(val_dice_seg)
                monitor_metrics['val']['val_dice_mask'].append(val_dice_mask)
                monitor_metrics['val']['val_dice_fusion'].append(val_dice_fusion)

                writer.add_scalar('Val/val_precision',precision,epoch)
                writer.add_scalar('Val/val_recall',recall,epoch)
                writer.add_scalar('Val/val_dice_seg',val_dice_seg,epoch)
                writer.add_scalar('Val/val_dice_mask',val_dice_mask,epoch)
                writer.add_scalar('Val/val_dice_fusion',val_dice_fusion,epoch)
                model_selector.run_model_selection(net, optimizer, monitor_metrics, epoch)

            # update monitoring and prediction plots
            #TrainingPlot.update_and_save(monitor_metrics, epoch)
            epoch_time = time.time() - start_time
            logger.info('trained epoch {}: took {} sec. ({} train / {} val)'.format(
                epoch, epoch_time, train_time, epoch_time-train_time))
    writer.close()


def test(logger):
    """
    perform testing for a given fold (or hold out set). save stats in evaluator.
    """
    logger.info('starting testing model of fold {} in exp {}'.format(cf.fold, cf.exp_dir))
    net = model.net(cf, logger).cuda()

    test_predictor = Predictor(cf, net, logger, mode='test')
    test_evaluator = Evaluator(cf, logger, mode='test')

    batch_gen = data_loader.get_test_generator(cf, logger)

    test_results_list,test_results_list_mask,test_results_list_seg ,test_results_list_fusion,testing_epoch= test_predictor.predict_test_set(batch_gen,cf, return_results=True)

    count = test_evaluator.evaluate_predictions(test_results_list,testing_epoch,cf,pth = cf.test_dir,flag = 'test')
    print('tp_patient {}, tp_roi {}, fp_roi {}'.format(count[0],count[1],count[2]))
    save_test_image(test_results_list,test_results_list_mask,test_results_list_seg,test_results_list_fusion,testing_epoch,cf,cf.plot_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,  default='train',
                        help='one out of: train / test / train_test / analysis / create_exp')
    parser.add_argument('-f','--folds', nargs='+', type=int, default=[1],
                        help='None runs over all folds in CV. otherwise specify list of folds.')
    parser.add_argument('--exp_dir', type=str, default='/data/yuezhou/mrcnn/debug/',
                        help='path to experiment dir. will be created if non existent.')
    parser.add_argument('--resume_to_checkpoint', type=str, default=None,
                        help='if resuming to checkpoint, the desired fold still needs to be parsed via --folds.')
    parser.add_argument('--exp_source', type=str, default='experiments/abus_exp/',
                        help='specifies, from which source experiment to load configs and data_loader.')

    args = parser.parse_args()
    folds = args.folds
    resume_to_checkpoint = args.resume_to_checkpoint

    if args.mode == 'train':  

        cf = utils.prep_exp(args.exp_dir, is_training=True)

        cf.resume_to_checkpoint = resume_to_checkpoint#default:None

        model = utils.import_module('model', cf.model_path)
        data_loader = utils.import_module('dl', os.path.join(args.exp_source, 'data_loader.py'))

        for fold in folds:
            cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(fold))#path to save results
            cf.fold = fold
            if not os.path.exists(cf.fold_dir):
                os.mkdir(cf.fold_dir)
            logger = utils.get_logger(cf.fold_dir)#loginfo for this fold

            train(logger)

            cf.resume_to_checkpoint = None
            if args.mode == 'train_test':
                test(logger)

            for hdlr in logger.handlers:
                hdlr.close()
            logger.handlers = []

    elif args.mode == 'test':

        cf = utils.prep_exp(args.exp_dir, is_training=False)

        model = utils.import_module('model', cf.model_path)
        data_loader = utils.import_module('dl', os.path.join(args.exp_source, 'data_loader.py'))

        for fold in folds:
            cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(fold))
            logger = utils.get_logger(cf.fold_dir)
            cf.fold = fold
            test(logger)

            for hdlr in logger.handlers:
                hdlr.close()
            logger.handlers = []

