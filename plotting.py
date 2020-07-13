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
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from copy import deepcopy
import torch
import torch.nn.functional as F
from utils.model_utils import dice_val
from utils.model_utils import get_one_hot_encoding 
def save_seg_result(cf,epoch,pid,seg_map,mask_map,fusion_map):
    if cf.test_last_epoch == False:
        pth = cf.plot_dir + '3D_result_epoch{}/'.format(epoch) 
    else:
        pth = cf.plot_dir + '3D_result_lastepoch{}/'.format(epoch)
    if not os.path.exists(pth):
        os.mkdir(pth)
    seg_map = np.squeeze(seg_map).astype(np.uint8)
    mask_map = np.squeeze(mask_map).astype(np.uint8)
    fusion_map = np.squeeze(fusion_map).astype(np.uint8)

    seg_map_pth = pth + '{}_epoch{}_segmap.nii.gz'.format(pid,epoch)
    mask_map_pth = pth + '{}_epoch{}_maskmap.nii.gz'.format(pid,epoch)
    fusion_map_pth = pth + '{}_epoch{}_fusionmap.nii.gz'.format(pid,epoch)

    seg_map = sitk.GetImageFromArray(seg_map)
    sitk.WriteImage(seg_map,seg_map_pth)
    mask_map = sitk.GetImageFromArray(mask_map)
    sitk.WriteImage(mask_map,mask_map_pth)
    fusion_map = sitk.GetImageFromArray(fusion_map)
    sitk.WriteImage(fusion_map,fusion_map_pth)
def savedice_csv(cf,epoch,pidlist,seg_dice,mask_dice,fusion_dice):
    if cf.test_last_epoch == True:
        pth = cf.test_dir + 'dice_lastepoch{}.csv'.format(epoch)
    else:
        pth = cf.test_dir + 'dice_epoch{}.csv'.format(epoch)
    print('saving csv to',pth)
    f = open(pth,'w+')
    f.write('%s,%s,%s,%s\n'%('patient','maskdice','segdice','fusiondice'))
    for ii,pid in enumerate(pidlist):
        print('pid',pid)
        f.write('%s,%.2f,%.2f,%.2f\n'%(pid,(mask_dice[ii]),(seg_dice[ii]),(fusion_dice[ii])))
        f.flush()
    maskdice = sum(mask_dice)/float(len(mask_dice))
    segdice = sum(seg_dice)/float(len(seg_dice))
    fusiondice = sum(fusion_dice)/float(len(fusion_dice))
    f.write('%s,%.2f,%.2f,%.2f\n'%('average',(maskdice),(segdice),(fusiondice)))
    f.flush()
    f.close()
def save_test_image(results_list,results_list_mask,results_list_seg,results_list_fusion, epoch,cf,pth,mode = 'test'):
    print('in save_test_image')
    if cf.test_last_epoch == False:
        pth = pth + 'epoch_{}/'.format(epoch)
    else:
        pth = pth + 'lastepoch_{}/'.format(epoch)
    if not os.path.exists(pth):
        os.mkdir(pth)
    mask_dice,seg_dice,fusion_dice,pidlist =[], [],[],[]
    for ii,box_pid in enumerate(results_list):
        pid = box_pid[1]
        pidlist.append(pid)
        boxes = box_pid[0][0]

        img = np.load(cf.pp_test_data_path + pid + '_img.npy')
        img = np.transpose(img,axes = (1,2,0))[np.newaxis]
        data = np.transpose(img, axes=(3, 0, 1, 2))#128,1,64,128
        seg = np.load(cf.pp_test_data_path + pid + '_rois.npy')
        seg = np.transpose(seg,axes = (1,2,0))[np.newaxis]
        this_batch_seg_label = np.expand_dims(seg,axis=0)#seg[np.newaxis,:,:,:,:]
        this_batch_seg_label = get_one_hot_encoding(this_batch_seg_label, cf.num_seg_classes+1)
        seg = np.transpose(seg, axes=(3, 0, 1, 2))#128,1,64,128

        mask_map = np.squeeze(results_list_mask[ii][0])
        mask_map = np.transpose(mask_map,axes = (0,1,2))[np.newaxis]
        mask_map_ = np.expand_dims(mask_map,axis=0)
        this_batch_dice_mask = dice_val(torch.from_numpy(mask_map_),torch.from_numpy(this_batch_seg_label))
        mask_map = np.transpose(mask_map, axes=(3, 0, 1, 2))#128,1,64,128
        mask_map[mask_map>0.5] = 1
        mask_map[mask_map<1] = 0

        seg_map = np.squeeze(results_list_seg[ii][0])
        seg_map = np.transpose(seg_map,axes = (0,1,2))[np.newaxis]
        seg_map_ = np.expand_dims(seg_map,axis=0)
        this_batch_dice_seg = dice_val(torch.from_numpy(seg_map_),torch.from_numpy(this_batch_seg_label))
        seg_map = np.transpose(seg_map, axes=(3, 0, 1, 2))#128,1,64,128
        seg_map[seg_map>0.5] = 1
        seg_map[seg_map<1] = 0

        fusion_map = np.squeeze(results_list_fusion[ii][0])
        fusion_map = np.transpose(fusion_map,axes = (0,1,2))[np.newaxis]
        fusion_map_ = np.expand_dims(fusion_map,axis=0)
        this_batch_dice_fusion = dice_val(torch.from_numpy(fusion_map_),torch.from_numpy(this_batch_seg_label))
        fusion_map = np.transpose(fusion_map, axes=(3, 0, 1, 2))#128,1,64,128
        fusion_map[fusion_map>0.5] = 1
        fusion_map[fusion_map<1] = 0

        save_seg_result(cf,epoch,pid,seg_map,mask_map,fusion_map)

        mask_dice.append(this_batch_dice_mask)
        seg_dice.append(this_batch_dice_seg)
        fusion_dice.append(this_batch_dice_fusion)

        gt_boxes = [box['box_coords'] for box in boxes if box['box_type'] == 'gt']
        slice_num = 5 
        if len(gt_boxes) > 0:
            center = int((gt_boxes[0][5]-gt_boxes[0][4])/2+gt_boxes[0][4])
            z_cuts = [np.max((center - slice_num, 0)), np.min((center + slice_num, data.shape[0]))]#max len = 10
        else:
            z_cuts = [data.shape[0]//2 - slice_num, int(data.shape[0]//2 + np.min([slice_num, data.shape[0]//2]))]
        roi_results = [[] for _ in range(data.shape[0])] 
        for box in boxes:#box is a list
            b = box['box_coords']
            # dismiss negative anchor slices.
            slices = np.round(np.unique(np.clip(np.arange(b[4], b[5] + 1), 0, data.shape[0]-1)))
            for s in slices:
                roi_results[int(s)].append(box)
                roi_results[int(s)][-1]['box_coords'] = b[:4]#change 3d box to 2d
        roi_results = roi_results[z_cuts[0]: z_cuts[1]]#extract slices to show
        data = data[z_cuts[0]: z_cuts[1]]
        seg = seg[z_cuts[0]:z_cuts[1]]
        seg_map = seg_map[z_cuts[0]:z_cuts[1]]
        mask_map = mask_map[z_cuts[0]:z_cuts[1]]
        fusion_map = fusion_map[z_cuts[0]:z_cuts[1]]
        pids = [pid] * data.shape[0]

        kwargs={'linewidth':0.2,
                'alpha':1,
                }
        show_arrays = np.concatenate([data,data,data,data], axis=1).astype(float)#10,2,79,219
        approx_figshape = (4*show_arrays.shape[0], show_arrays.shape[1])
        fig = plt.figure(figsize=approx_figshape)
        gs = gridspec.GridSpec(show_arrays.shape[1] + 1, show_arrays.shape[0])
        gs.update(wspace=0.1, hspace=0.1)
        for b in range(show_arrays.shape[0]):#10(0...9)
            for m in range(show_arrays.shape[1]):#4(0,1,2,3)
                ax = plt.subplot(gs[m, b])
                ax.axis('off')
                arr = show_arrays[b, m]#get image to be shown
                cmap = 'gray'
                vmin = None
                vmax = None

                if m == 1:
                    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.contour(np.squeeze(mask_map[b][0:1,:,:]),colors = 'yellow',linewidth=1,alpha=1)
                if m == 2:
                    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.contour(np.squeeze(seg_map[b][0:1,:,:]),colors = 'lime',linewidth=1,alpha=1)
                if m == 3:
                    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.contour(np.squeeze(fusion_map[b][0:1,:,:]),colors = 'orange',linewidth=1,alpha=1)
                if m == 0:
                    plt.title('{}'.format(pids[b][:10]), fontsize=8)
                    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.contour(np.squeeze(seg[b][0:1,:,:]),colors = 'red',linewidth=1,alpha=1)
                    plot_text = False 
                    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
                    for box in roi_results[b]:
                        coords = box['box_coords']
                        #print('coords',coords)
                        #print('type',box['box_type'])
                        if box['box_type'] == 'det':
                            #print('score',box['box_score'])
                            if box['box_score'] > 0.1:# and box['box_score'] > cf.source_th:#detected box
                                plot_text = True
                                #score = np.max(box['box_score'])
                                score = box['box_score']
                                score_text = '{:.2f}'.format(score*100)#'{}|{:.0f}'.format(box['box_pred_class_id'], score*100)
                                score_font_size = 7 
                                text_color = 'w'
                                text_x = coords[1] #+ 10*(box['box_pred_class_id'] -1) #avoid overlap of scores in plot.
                                text_y = coords[2] + 10
                            #else:#background and small score don't show
                            #    continue
                        color_var = 'box_type'#'extra_usage' if 'extra_usage' in list(box.keys()) else 'box_type'
                        color = cf.box_color_palette[box[color_var]]
                        ax.plot([coords[1], coords[3]], [coords[0], coords[0]], color=color, linewidth=1, alpha=1) # up
                        ax.plot([coords[1], coords[3]], [coords[2], coords[2]], color=color, linewidth=1, alpha=1) # down
                        ax.plot([coords[1], coords[1]], [coords[0], coords[2]], color=color, linewidth=1, alpha=1) # left
                        ax.plot([coords[3], coords[3]], [coords[0], coords[2]], color=color, linewidth=1, alpha=1) # right
                        if plot_text:
                            ax.text(text_x, text_y, score_text, fontsize=score_font_size, color=text_color)
        if cf.test_last_epoch == False:
            outfile = pth+'result_{}_{}_{}.png'.format(mode,pid,epoch)
        else:
            outfile = pth+'result_{}_{}_lastepoch_{}.png'.format(mode,pid,epoch)
        print('outfile',outfile)
        try:
            plt.savefig(outfile)
        except:
            raise Warning('failed to save plot.')

    savedice_csv(cf,epoch,pidlist,seg_dice,mask_dice,fusion_dice)
def save_monitor_valuse(cf,test_df,epoch,flag = 'val'):
    pth = cf.exp_dir
    filename = flag+'_{}'.format(epoch)+'.csv'
    print('pth',pth+filename)
    test_df.to_csv(pth+filename)

def plot_batch_prediction(batch, results_dict, cf, mode,outfile= None):
    """
    plot the input images, ground truth annotations, and output predictions of a batch. If 3D batch, plots a 2D projection
    of one randomly sampled element (patient) in the batch. Since plotting all slices of patient volume blows up costs of
    time and space, only a section containing a randomly sampled ground truth annotation is plotted.
    :param batch: dict with keys: 'data' (input image), 'seg' (pixelwise annotations), 'pid'
    :param results_dict: list over batch element. Each element is a list of boxes (prediction and ground truth),
    where every box is a dictionary containing box_coords, box_score and box_type.
    """
    #print('in ploting image')
    data = batch['data']
    pids = batch['pid']
    segs = batch['seg']
    # for 3D, repeat pid over batch elements.
    if len(set(pids)) == 1:
        pids = [pids] * data.shape[0]

    if mode == 'val_patient':
        mask_map = results_dict['seg_preds']#.cpu().detach().numpy()
        seg_map = results_dict['seg_logits']#.cpu().detach().numpy()
        fusion_map = results_dict['fusion_map']#.cpu().detach().numpy()
    else:
    #mask_map = torch.tensor(results_dict['seg_preds']).cuda()
        if cf.fusion_feature_method == 'after':
            mask_map = results_dict['seg_preds'][:,1:2,:,:,:].cpu().detach().numpy()
        else:
            mask_map = F.softmax(results_dict['seg_preds'], dim=1)[:,1:2,:,:,:].cpu().detach().numpy()# N,2,64,128,128

        if cf.fusion_feature_method == 'after':
            seg_map = results_dict['seg_logits'][:,1:2,:,:,:].cpu().detach().numpy()
        else:
            seg_map = F.softmax(results_dict['seg_logits'], dim=1)[:,1:2,:,:,:].cpu().detach().numpy()

        fusion_map = results_dict['fusion_map'][:,1:2,:,:,:].cpu().detach().numpy()
        we_layer_seg = results_dict['we_layer'][:,1:2,:,:,:].cpu().detach().numpy()
        we_layer_mask = results_dict['we_layer'][:,3:4,:,:,:].cpu().detach().numpy()

    roi_results = deepcopy(results_dict['boxes'])#len == batch size
    # Randomly sampled one patient of batch and project data into 2D slices for plotting.
    if cf.dim == 3:
        patient_ix = np.random.choice(data.shape[0])
        data = np.transpose(data[patient_ix], axes=(3, 0, 1, 2))#128,1,64,128
        # select interesting foreground section to plot.
        gt_boxes = [box['box_coords'] for box in roi_results[patient_ix] if box['box_type'] == 'gt']
        if len(gt_boxes) > 0:
            center = int((gt_boxes[0][5]-gt_boxes[0][4])/2+gt_boxes[0][4])
            z_cuts = [np.max((center - 5, 0)), np.min((center + 5, data.shape[0]))]#max len = 10
        else:
            z_cuts = [data.shape[0]//2 - 5, int(data.shape[0]//2 + np.min([5, data.shape[0]//2]))]
        p_roi_results = roi_results[patient_ix]
        roi_results = [[] for _ in range(data.shape[0])]#len = 128

        # iterate over cubes and spread across slices.
        for box in p_roi_results:#box is a list
            b = box['box_coords']
            # dismiss negative anchor slices.
            slices = np.round(np.unique(np.clip(np.arange(b[4], b[5] + 1), 0, data.shape[0]-1)))
            for s in slices:
                roi_results[int(s)].append(box)
                roi_results[int(s)][-1]['box_coords'] = b[:4]#change 3d box to 2d
        roi_results = roi_results[z_cuts[0]: z_cuts[1]]#extract slices to show
        data = data[z_cuts[0]: z_cuts[1]]
        segs = np.transpose(segs[patient_ix], axes=(3, 0, 1, 2))[z_cuts[0]: z_cuts[1]]#gt
        mask_map = np.transpose(mask_map[patient_ix], axes=(3, 0, 1, 2))[z_cuts[0]: z_cuts[1]]#pred seg
        seg_map = np.transpose(seg_map[patient_ix], axes=(3, 0, 1, 2))[z_cuts[0]: z_cuts[1]]#pred seg
        fusion_map = np.transpose(fusion_map[patient_ix], axes=(3, 0, 1, 2))[z_cuts[0]: z_cuts[1]]#pred seg
        we_layer_seg = np.transpose(we_layer_seg[patient_ix],axes=(3,0,1,2))[z_cuts[0]:z_cuts[1]]
        we_layer_mask = np.transpose(we_layer_mask[patient_ix],axes=(3,0,1,2))[z_cuts[0]:z_cuts[1]]
        pids = [pids[patient_ix]] * data.shape[0]

    try:
        # all dimensions except for the 'channel-dimension' are required to match
        for i in [0, 2, 3]:
            assert data.shape[i] == segs.shape[i] == mask_map.shape[i]
    except:
        raise Warning('Shapes of arrays to plot not in agreement!'
                      'Shapes {} vs. {} vs {}'.format(data.shape, segs.shape, mask_map.shape))

    show_arrays = np.concatenate([data[:,0][:,None], segs, mask_map, seg_map, fusion_map,we_layer_mask,we_layer_seg], axis=1).astype(float)
    approx_figshape = (4 * show_arrays.shape[0], 4 * show_arrays.shape[1])
    fig = plt.figure(figsize=approx_figshape)
    gs = gridspec.GridSpec(show_arrays.shape[1] + 1, show_arrays.shape[0])
    gs.update(wspace=0.1, hspace=0.1)

    for b in range(show_arrays.shape[0]):
        for m in range(show_arrays.shape[1]):

            ax = plt.subplot(gs[m, b])
            ax.axis('off')
            arr = show_arrays[b, m]#get image to be shown

            if m == 0:
                cmap = 'gray'
                vmin = None
                vmax = None
            else:
                cmap = 'jet' 
                vmin = 0
                vmax = 1#cf.num_seg_classes - 1

            ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            if m == 0: 
                plot_text = False
                plt.title('{}'.format(pids[b][:10]), fontsize=8)
                for box in roi_results[b]:
                    coords = box['box_coords']
                    if box['box_type'] == 'det':
                        # dont plot background preds or low confidence boxes.
                        if  box['box_score'] > cf.show_det_source_th:#detected box
                            plot_text = True
                            score = box['box_score']
                            score_text = '{:.2f}'.format(score*100)
                            score_font_size = 7
                            text_color = 'w'
                            text_x = coords[1] #+ 10*(box['box_pred_class_id'] -1) #avoid overlap of scores in plot.
                            text_y = coords[2] + 5
                    color_var = 'box_type'#'extra_usage' if 'extra_usage' in list(box.keys()) else 'box_type'
                    color = cf.box_color_palette[box[color_var]]
                    ax.plot([coords[1], coords[3]], [coords[0], coords[0]], color=color, linewidth=1, alpha=1) # up
                    ax.plot([coords[1], coords[3]], [coords[2], coords[2]], color=color, linewidth=1, alpha=1) # down
                    ax.plot([coords[1], coords[1]], [coords[0], coords[2]], color=color, linewidth=1, alpha=1) # left
                    ax.plot([coords[3], coords[3]], [coords[0], coords[2]], color=color, linewidth=1, alpha=1) # right
                    if plot_text:
                        ax.text(text_x, text_y, score_text, fontsize=score_font_size, color=text_color)
    return fig

class TrainingPlot_2Panel():


    def __init__(self, cf):
        self.file_name = cf.plot_dir + '/monitor_{}'.format(cf.fold)
        #print('file_name monitor',self.file_name)
        self.exp_name = cf.fold_dir
        self.do_validation = cf.do_validation
        self.separate_values_dict = cf.assign_values_to_extra_figure#{}
        self.figure_list = []
        for n in range(cf.n_monitoring_figures):#1
            self.figure_list.append(plt.figure(figsize=(10, 6)))
            self.figure_list[-1].ax1 = plt.subplot(111)
            self.figure_list[-1].ax1.set_xlabel('epochs')
            self.figure_list[-1].ax1.set_ylabel('loss / metrics')
            self.figure_list[-1].ax1.set_xlim(0, cf.num_epochs)
            self.figure_list[-1].ax1.grid()

        self.figure_list[0].ax1.set_ylim(0, 1.5)
        self.color_palette = ['b', 'c', 'r', 'purple', 'm', 'y', 'k', 'tab:gray']

    def update_and_save(self, metrics, epoch):

        for figure_ix in range(len(self.figure_list)):
            fig = self.figure_list[figure_ix]
            detection_monitoring_plot(fig.ax1, metrics, self.exp_name, self.color_palette, epoch, figure_ix,
                                      self.separate_values_dict,
                                      self.do_validation)
            fig.savefig(self.file_name + '_{}'.format(figure_ix))


def detection_monitoring_plot(ax1, metrics, exp_name, color_palette, epoch, figure_ix, separate_values_dict, do_validation):

    monitor_values_keys = metrics['train']['monitor_values'][1][0].keys()
    separate_values = [v for fig_ix in separate_values_dict.values() for v in fig_ix]
    if figure_ix == 0:
        plot_keys = [ii for ii in monitor_values_keys if ii not in separate_values]
        plot_keys += [k for k in metrics['train'].keys() if k != 'monitor_values']
    else:
        plot_keys = separate_values_dict[figure_ix]


    x = np.arange(1, epoch + 1)

    for kix, pk in enumerate(plot_keys):
        if pk in metrics['train'].keys():
            y_train = metrics['train'][pk][1:]
            if do_validation:
                y_val = metrics['val'][pk][1:]
        else:
            y_train = [np.mean([er[pk] for er in metrics['train']['monitor_values'][e]]) for e in x]
            if do_validation:
                y_val = [np.mean([er[pk] for er in metrics['val']['monitor_values'][e]]) for e in x]

        ax1.plot(x, y_train, label='train_{}'.format(pk), linestyle='--', color=color_palette[kix])
        if do_validation:
            ax1.plot(x, y_val, label='val_{}'.format(pk), linestyle='-', color=color_palette[kix])

    if epoch == 1:
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.set_title(exp_name)


def plot_prediction_hist(label_list, pred_list, type_list, outfile):
    """
    plot histogram of predictions for a specific class.
    :param label_list: list of 1s and 0s specifying whether prediction is a true positive match (1) or a false positive (0).
    False negatives (missed ground truth objects) are artificially added predictions with score 0 and label 1.
    :param pred_list: list of prediction-scores.
    :param type_list: list of prediction-types for stastic-info in title.
    """
    #print('in plot_prediction_hist')
    #print('label_list',label_list)
    #print('pred_list',pred_list)
    #print('type_list',type_list)
    #print('outfile',outfile)
    preds = np.array(pred_list)
    labels = np.array(label_list)
    title = outfile.split('/')[-1] + ' count:{}'.format(len(label_list))
    plt.figure()
    plt.yscale('log')
    if 0 in labels:
        plt.hist(preds[labels == 0], alpha=0.3, color='g', range=(0, 1), bins=50, label='false pos.')
    if 1 in labels:
        plt.hist(preds[labels == 1], alpha=0.3, color='b', range=(0, 1), bins=50, label='true pos. (false neg. @ score=0)')

    if type_list is not None:
        fp_count = type_list.count('det_fp')
        fn_count = type_list.count('det_fn')
        tp_count = type_list.count('det_tp')
        pos_count = fn_count + tp_count
        title += ' tp:{} fp:{} fn:{} pos:{}'. format(tp_count, fp_count, fn_count, pos_count)

    plt.legend()
    plt.title(title)
    plt.xlabel('confidence score')
    plt.ylabel('log n')
    plt.savefig(outfile)
    plt.close()


def plot_stat_curves(stats, outfile):
    print('in plot_stat_curves')
    print('outfile',outfile)
    for c in ['roc', 'prc']:
        plt.figure()
        for s in stats:
            if s[c] is not None:
                plt.plot(s[c][0], s[c][1], label=s['name'] + '_' + c)
        plt.title(outfile.split('/')[-1] + '_' + c)
        plt.legend(loc=3 if c == 'prc' else 4)
        plt.xlabel('precision' if c == 'prc' else '1-spec.')
        plt.ylabel('recall')
        plt.savefig(outfile + '_' + c)
        plt.close()
