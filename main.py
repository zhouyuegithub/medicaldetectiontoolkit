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

import utils.exp_utils as utils
from evaluator import Evaluator
from predictor import Predictor
from plotting import plot_batch_prediction
from tensorboardX import SummaryWriter


def train(logger):
    """
    perform the training routine for a given fold. saves plots and selected parameters to the experiment dir
    specified in the configs.
    """
    logger.info('performing training in {}D over fold {} on experiment {} with model {}'.format(
        cf.dim, cf.fold, cf.exp_dir, cf.model))
    
    writer = SummaryWriter(os.path.join(cf.exp_dir,'tensorboard'))

    net = model.net(cf, logger).cuda()
    print('finish initial network')
    optimizer = torch.optim.Adam(net.parameters(), lr=cf.learning_rate[0], weight_decay=cf.weight_decay)
    #print('finish initial optimizer')
    model_selector = utils.ModelSelector(cf, logger)
    train_evaluator = Evaluator(cf, logger, mode='train')
    val_evaluator = Evaluator(cf, logger, mode=cf.val_mode)#val_sampling

    starting_epoch = 1

    # prepare monitoring
    monitor_metrics, TrainingPlot = utils.prepare_monitoring(cf)
    #print('finish initial metrics')
    if cf.resume_to_checkpoint:#default: False
        starting_epoch, monitor_metrics = utils.load_checkpoint(cf.resume_to_checkpoint, net, optimizer)
        logger.info('resumed to checkpoint {} at epoch {}'.format(cf.resume_to_checkpoint, starting_epoch))

    logger.info('loading dataset and initializing batch generators...')
    batch_gen = data_loader.get_train_generators(cf, logger)
    print('finish initial data_loader')
    #for k in batch_gen.keys():
    #    print('k in batch_gen are {}'.format(k))
    num_batch = 0
    for epoch in range(starting_epoch, cf.num_epochs + 1):

        logger.info('starting training epoch {}'.format(epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = cf.learning_rate[epoch - 1]

        start_time = time.time()

        net.train()
        train_results_list = []#this batch

        #print('net.train()')
        for bix in range(cf.num_train_batches):#200
            num_batch += 1
            batch = next(batch_gen['train'])
            print('*'*50+'load batch !')
            print('loading patient',batch['pid'])
            #print('bbox',batch['bb_target'][0])#y1,x1,y2,x2,z1,z2
            #print('data',batch['data'][0].shape)
            #for k in batch.items():
            #    print('k',k)
            tic_fw = time.time()
            results_dict = net.train_forward(batch)
            tic_bw = time.time()
            optimizer.zero_grad()
            results_dict['torch_loss'].backward()#total loss
            optimizer.step()
            logger.info('tr. batch {0}/{1} (ep. {2}) fw {3:.3f}s / bw {4:.3f}s / total {5:.3f}s || '
                        .format(bix + 1, cf.num_train_batches, epoch, tic_bw - tic_fw,
                                time.time() - tic_bw, time.time() - tic_fw) + results_dict['logger_string'])
            train_results_list.append([results_dict['boxes'], batch['pid']])
            monitor_metrics['train']['monitor_values'][epoch].append(results_dict['monitor_values'])
            #print('num_batch',num_batch)
            #results_dict['monitor_values'] = {'loss': loss.item(), 'mrcnn_class_loss': mrcnn_class_loss.item(), 'mrcnn_bbox_loss':mrcnn_bbox_loss.item(), 'mrcnn_mask_loss':mrcnn_mask_loss.item(),'rpn_class_loss':batch_rpn_class_loss.item(),'rpn_bbox_loss':batch_rpn_bbox_loss.item()}
            writer.add_scalar('Train/total_loss',results_dict['torch_loss'].item(),num_batch)
            writer.add_scalar('Train/rpn_class_loss',results_dict['monitor_losses']['rpn_class_loss'],num_batch)
            writer.add_scalar('Train/rpn_bbox_loss',results_dict['monitor_losses']['rpn_bbox_loss'],num_batch)
            writer.add_scalar('Train/mrcnn_class_loss',results_dict['monitor_losses']['mrcnn_class_loss'],num_batch)
            writer.add_scalar('Train/mrcnn_bbox_loss',results_dict['monitor_losses']['mrcnn_bbox_loss'],num_batch)
            writer.add_scalar('Train/mrcnn_mask_loss',results_dict['monitor_losses']['mrcnn_mask_loss'],num_batch)

        _, monitor_metrics['train'] = train_evaluator.evaluate_predictions(train_results_list, monitor_metrics['train'])
        train_time = time.time() - start_time

        logger.info('starting validation in mode {}.'.format(cf.val_mode))
        with torch.no_grad():
            net.eval()
            if cf.do_validation:
                val_results_list = []
                val_predictor = Predictor(cf, net, logger, mode='val')
                for _ in range(batch_gen['n_val']):#50
                    batch = next(batch_gen[cf.val_mode])
                    if cf.val_mode == 'val_patient':
                        results_dict = val_predictor.predict_patient(batch)
                    elif cf.val_mode == 'val_sampling':
                        results_dict = net.train_forward(batch, is_validation=True)
                    val_results_list.append([results_dict['boxes'], batch['pid']])
                    monitor_metrics['val']['monitor_values'][epoch].append(results_dict['monitor_values'])

                _, monitor_metrics['val'] = val_evaluator.evaluate_predictions(val_results_list, monitor_metrics['val'])
                model_selector.run_model_selection(net, optimizer, monitor_metrics, epoch)

            # update monitoring and prediction plots
            TrainingPlot.update_and_save(monitor_metrics, epoch)
            epoch_time = time.time() - start_time
            logger.info('trained epoch {}: took {} sec. ({} train / {} val)'.format(
                epoch, epoch_time, train_time, epoch_time-train_time))
            batch = next(batch_gen['val_sampling'])
            results_dict = net.train_forward(batch, is_validation=True)
            logger.info('plotting predictions from validation sampling.')
            plot_batch_prediction(batch, results_dict, cf)
    writer.close()


def test(logger):
    """
    perform testing for a given fold (or hold out set). save stats in evaluator.
    """
    logger.info('starting testing model of fold {} in exp {}'.format(cf.fold, cf.exp_dir))
    net = model.net(cf, logger).cuda()
    print('initial net for testing')
    test_predictor = Predictor(cf, net, logger, mode='test')
    test_evaluator = Evaluator(cf, logger, mode='test')
    batch_gen = data_loader.get_test_generator(cf, logger)
    test_results_list = test_predictor.predict_test_set(batch_gen, return_results=True)
    test_evaluator.evaluate_predictions(test_results_list)
    test_evaluator.score_test_df()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str,  default='train_test',
                        help='one out of: train / test / train_test / analysis / create_exp')
    parser.add_argument('-f','--folds', nargs='+', type=int, default=None,
                        help='None runs over all folds in CV. otherwise specify list of folds.')
    parser.add_argument('--exp_dir', type=str, default='./experiments/abus_exp/debug/',
                        help='path to experiment dir. will be created if non existent.')
    parser.add_argument('--server_env', default=False, action='store_true',
                        help='change IO settings to deploy models on a cluster.')
    parser.add_argument('--data_dest', type=str, default=None, help="path to final data folder if different from config.")
    parser.add_argument('--use_stored_settings', default=False, action='store_true',
                        help='load configs from existing exp_dir instead of source dir. always done for testing, '
                             'but can be set to true to do the same for training. useful in job scheduler environment, '
                             'where source code might change before the job actually runs.')
    parser.add_argument('--resume_to_checkpoint', type=str, default=None,
                        help='if resuming to checkpoint, the desired fold still needs to be parsed via --folds.')
    parser.add_argument('--exp_source', type=str, default='experiments/abus_exp/',
                        help='specifies, from which source experiment to load configs and data_loader.')
    parser.add_argument('-d', '--dev', default=False, action='store_true', help="development mode: shorten everything")

    args = parser.parse_args()
    folds = args.folds
    resume_to_checkpoint = args.resume_to_checkpoint

    if args.mode == 'train' or args.mode == 'train_test':

        cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, args.use_stored_settings)
        if args.dev: # default: false
            folds = [0,1]
            cf.batch_size, cf.num_epochs, cf.min_save_thresh, cf.save_n_models = 3 if cf.dim==2 else 1, 1, 0, 1
            cf.num_train_batches, cf.num_val_batches, cf.max_val_patients = 5, 1, 1
            cf.test_n_epochs =  cf.save_n_models
            cf.max_test_patients = 1

        cf.data_dest = args.data_dest
        ### import model and dataloader
        model = utils.import_module('model', cf.model_path)
        data_loader = utils.import_module('dl', os.path.join(args.exp_source, 'data_loader.py'))
        if folds is None:
            folds = range(cf.n_cv_splits)

        for fold in folds:
            cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(fold))#path to save results
            cf.fold = fold
            cf.resume_to_checkpoint = resume_to_checkpoint#default:None
            if not os.path.exists(cf.fold_dir):
                os.mkdir(cf.fold_dir)
            logger = utils.get_logger(cf.fold_dir)#loginfo for this fold
            train(logger)
            print('*'*100+'finish train all epoches')
            cf.resume_to_checkpoint = None
            if args.mode == 'train_test':
                test(logger)

            for hdlr in logger.handlers:
                hdlr.close()
            logger.handlers = []

    elif args.mode == 'test':

        cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, is_training=False, use_stored_settings=True)
        if args.dev:
            folds = [0,1]
            cf.test_n_epochs =  1; cf.max_test_patients = 1

        cf.data_dest = args.data_dest
        model = utils.import_module('model', cf.model_path)
        data_loader = utils.import_module('dl', os.path.join(args.exp_source, 'data_loader.py'))
        if folds is None:
            folds = range(cf.n_cv_splits)

        for fold in folds:
            cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(fold))
            logger = utils.get_logger(cf.fold_dir)
            cf.fold = fold
            test(logger)

            for hdlr in logger.handlers:
                hdlr.close()
            logger.handlers = []

    # load raw predictions saved by predictor during testing, run aggregation algorithms and evaluation.
    elif args.mode == 'analysis':
        cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, is_training=False, use_stored_settings=True)
        logger = utils.get_logger(cf.exp_dir)

        if cf.hold_out_test_set:
            cf.folds = args.folds
            predictor = Predictor(cf, net=None, logger=logger, mode='analysis')
            results_list = predictor.load_saved_predictions(apply_wbc=True)
            utils.create_csv_output(results_list, cf, logger)

        else:
            if folds is None:
                folds = range(cf.n_cv_splits)
            for fold in folds:
                cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(fold))
                cf.fold = fold
                predictor = Predictor(cf, net=None, logger=logger, mode='analysis')
                results_list = predictor.load_saved_predictions(apply_wbc=True)
                logger.info('starting evaluation...')
                evaluator = Evaluator(cf, logger, mode='test')
                evaluator.evaluate_predictions(results_list)
                evaluator.score_test_df()

    # create experiment folder and copy scripts without starting job.
    # useful for cloud deployment where configs might change before job actually runs.
    elif args.mode == 'create_exp':
        cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, use_stored_settings=True)
        logger = utils.get_logger(cf.exp_dir)
        logger.info('created experiment directory at {}'.format(args.exp_dir))

    else:
        raise RuntimeError('mode specified in args is not implemented...')
