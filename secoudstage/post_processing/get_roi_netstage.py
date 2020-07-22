import SimpleITK as sitk
import numpy as np
import os
import csv
import cc3d
import pickle
import pandas as pd

def aggregate_meta_info(exp_dir):

    files = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if 'meta_info' in f]
    df = pd.DataFrame(columns=['pid', 'class_target', 'spacing'])
    for f in files:
        with open(f, 'rb') as handle:
            df.loc[len(df)] = pickle.load(handle)

    df.to_pickle(os.path.join(exp_dir, 'info_df.pickle'))
    print ("aggregated meta info to df with length", len(df))
def getbbox(data):
    y,x,z = [],[],[]
    thisbox = []
    for xx in range(data.shape[1]):
        if data[:,xx,:].max() > 0:
            x.append(xx)
    for yy in range(data.shape[0]):
        if data[yy,:,:].max() > 0:
            y.append(yy)
    for zz in range(data.shape[2]):
        if data[:,:,zz].max() > 0:
            z.append(zz)
    x0 = x[0]
    y0 = y[0]
    z0 = z[0]
    x1 = x[-1]
    y1 = y[-1]
    z1 = z[-1]
    cy = int(float(y1-y0)/2 + y0)
    cx = int(float(x1-x0)/2 + x0)
    cz = int(float(z1-z0)/2 + z0)
    return [cy, cx, cz] 
def preocess_box(boxlist):
    processed_boxlist = []
    for ii,b in enumerate(boxlist):
        bsp = b.split(' ')
        thisbox = []
        for bb in bsp:
            if '[' in bb and len(bb) != 1:
                thisbox.append(int(bb[1:]))
            if '[' not in bb  and len(bb) >0 and ']' not in bb:
                thisbox.append(int(bb))
            if ']' in bb:
                thisbox.append(int(bb[:-1]))
        processed_boxlist.append(thisbox)
    return processed_boxlist 
def readcsv(pth):
    with open(pth,'r') as f:
        reader = csv.reader(f)
        det_score = [row[1] for row in reader]
    det_score_str = det_score[1:]
    det_score = []
    for s in det_score_str:
        det_score.append(float(s))
    with open(pth,'r') as f:
        reader = csv.reader(f)
        det_bbox = [row[2] for row in reader]
    det_bbox = det_bbox[1:]
    det_bbox = preocess_box(det_bbox)
    with open(pth,'r') as f:
        reader = csv.reader(f)
        det_pid = [row[-4] for row in reader]
    det_pid = det_pid[1:]
    return det_bbox,det_score,det_pid
def readcsv_new(pth):
    with open(pth,'r') as f:
        reader = csv.reader(f)
        det_score = [row[2] for row in reader]
    det_score_str = det_score[1:]
    det_score = []
    for s in det_score_str:
        det_score.append(float(s))
    with open(pth,'r') as f:
        reader = csv.reader(f)
        det_bbox = [row[1] for row in reader]
    det_bbox = det_bbox[1:]
    det_bbox = preocess_box(det_bbox)
    with open(pth,'r') as f:
        reader = csv.reader(f)
        det_pid = [row[0] for row in reader]
    det_pid = det_pid[1:]
    return det_bbox,det_score,det_pid

if __name__ == '__main__':
    connectivity = 6 # only 26, 18, and 6 are allowed
    patch_size = [64,128,128]
    det_result_pth = '/data/yuezhou/newdata/0713_mrcnn_seg_fusprob_opt_newdata/test_next_val/test_epoch_115_score_0.5.csv'
    seg_result_pth = '/data/yuezhou/newdata/0713_mrcnn_seg_fusprob_opt_newdata/plots_next_val/3D_result_epoch115/'
    ori_data_pth = '/shenlab/lab_stor6/yuezhou/ABUSdata/abus_npy/'
    crop_pth = '/data/yuezhou/newdata/0713_mrcnn_seg_fusprob_opt_newdata/roi_val/'

    fusion_result_pth_list, pid_list = [], []
    for files in os.listdir(seg_result_pth):
        if 'fusion' in files:
            fusion_result_pth_list.append(files)
            pid = files[:9]
            if '_' == pid[-1]:
                pid = files[:8]
            pid_list.append(pid)
    print(len(pid_list))
    det_bbox,det_score,det_pid = readcsv_new(det_result_pth)
    centers = []
    pids = []
    for ii,pp in enumerate(pid_list):
        print('processing patient:',pp)
        fusion_result_pth = seg_result_pth + fusion_result_pth_list[ii]
        fusion_map = sitk.ReadImage(fusion_result_pth)
        fusion_map = sitk.GetArrayFromImage(fusion_map)
        fusion_map = np.transpose(fusion_map,(1,2,0)) 
        fusion_map_connection = cc3d.connected_components(fusion_map, connectivity=connectivity)
        fusion_map_regions_num = fusion_map_connection.max()
        print('fusion_map_regions_num',fusion_map_regions_num)
        if fusion_map_regions_num>0:
            for i in range(fusion_map_regions_num):
                fusion_map_this = fusion_map_connection == i+1
                fusion_map_this = fusion_map_this.astype(np.uint8)
                this_center = getbbox(fusion_map_this)
                centers.append(this_center)
                pids.append(pp)
        #else:
        #    for i,p in enumerate(det_pid):
        #        if p == pp:
        #            det_bbox_this = det_bbox[i]
        #            cy = int(float(det_bbox_this[2]-det_bbox_this[0])/2 + det_bbox_this[0])
        #            cx = int(float(det_bbox_this[3]-det_bbox_this[1])/2 + det_bbox_this[1])
        #            cz = int(float(det_bbox_this[5]-det_bbox_this[4])/2 + det_bbox_this[4])
        #            this_center = [cy,cx,cz]
        #            centers.append(this_center)
        #            pids.append(pp)
    print('total center num',len(centers))
    #print('detcenter',len(pids))
    print('selected patient num',len(list(set(pids))))
    has_fg = []# = 0
    has_bg = []
    real_coords = []
    read_pids = []
    saved = 0
    for ii,pp in enumerate(pids):
        img_pth = ori_data_pth + pp + '_img.npy'
        label_pth = ori_data_pth + pp + '_rois.npy'
        img = np.load(img_pth)
        img = np.transpose(img,(1,2,0))
        label = np.load(label_pth)
        label = np.transpose(label,(1,2,0))
        label[label>0] = 1
        label[label<1] = 0
        img_pad = np.pad(img,[(patch_size[0],patch_size[0]),(patch_size[1],patch_size[1]),(patch_size[2],patch_size[2])],mode = 'constant',constant_values=0)
        label_pad = np.pad(label,[(patch_size[0],patch_size[0]),(patch_size[1],patch_size[1]),(patch_size[2],patch_size[2])],mode = 'constant',constant_values=0)
        this_center = centers[ii]
        #print('pid',pp+'_'+str(ii+1))
        this_center = centers[ii]
        #print('this_center',this_center)
        #print('img_pad',img_pad.shape)
        beginy = int(patch_size[0] + this_center[0] - 32)
        beginx = int(patch_size[1] + this_center[1] - 64)
        beginz = int(patch_size[2] + this_center[2] - 64)
        #print('beginy:{},beginx:{},beginz:{}'.format(beginy,beginx,beginz))
        croped_img = img_pad[beginy:beginy+64,beginx:beginx+128,beginz:beginz+128]
        croped_label = label_pad[beginy:beginy+64,beginx:beginx+128,beginz:beginz+128]

        if sum(sum(sum(croped_label))) > 0:
            has_fg.append(pp)
        else:
            has_bg.append(pp)
        croped_img_pth = crop_pth + '{}_{}_img.npy'.format(pp,ii+1)
        croped_label_pth = crop_pth + '{}_{}_rois.npy'.format(pp,ii+1)
        np.save(croped_img_pth,croped_img)
        np.save(croped_label_pth,croped_label)
        temp_pikle_pth = crop_pth + 'meta_info_{}_{}.pickle'.format(pp,ii+1)
        with open(temp_pikle_pth,'wb') as handle:
            meta_info_dict = {'pid': pp+'_'+str(ii+1), 'class_target': [1], 'spacing': [0.5,0.5,0.5]}#, 'fg_slices': fg_slices}
            pickle.dump(meta_info_dict, handle)
        ### for test crop right or not
        #pad_img_pth = crop_pth + '{}_{}_img_pad_{}_{}_{}.nii.gz'.format(pp,ii+1,beginy,beginx,beginz)
        #pad_label_pth = crop_pth + '{}_{}_rois_pad_{}_{}_{}.nii.gz'.format(pp,ii+1,beginy,beginx,beginz)
        #img = sitk.GetImageFromArray(img_pad)
        #label = sitk.GetImageFromArray(label_pad)
        #sitk.WriteImage(img,pad_img_pth)
        #sitk.WriteImage(label,pad_label_pth)

        #croped_img_pth = crop_pth + '{}_{}_img.nii.gz'.format(pp,ii+1)
        #croped_label_pth = crop_pth + '{}_{}_rois.nii.gz'.format(pp,ii+1)
        #img = sitk.GetImageFromArray(croped_img)
        #label = sitk.GetImageFromArray(croped_label)
        #sitk.WriteImage(img,croped_img_pth)
        #sitk.WriteImage(label,croped_label_pth)
    aggregate_meta_info(crop_pth)
    print('total has_fg',len(has_fg))
    print('total has_bg',len(has_bg))
    print('pids',len(pids))

    print('has_fg',len(list(set(has_fg))))
    print('has_bg',len(list(set(has_bg))))
    print('pid',len(list(set(pids))))

