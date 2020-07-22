import os
import csv
import cc3d
import numpy as np
import SimpleITK as sitk
def getbbox(data):
    y,x,z = [],[],[]
    thisbox = []
    for xx in range(data.shape[0]):
        if data[xx,:,:].max() > 0:
            x.append(xx)
    for yy in range(data.shape[1]):
        if data[:,yy,:].max() > 0:
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
    thisbox.append(y0)
    thisbox.append(x0)
    thisbox.append(y1)
    thisbox.append(x1)
    thisbox.append(z0)
    thisbox.append(z1)
    return thisbox


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
        if len(thisbox) != 6:
            print('*'*100)
            print((thisbox))
        processed_boxlist.append(thisbox)
    return processed_boxlist 
def readcsv(pth):
    with open(pth,'r') as f:
        reader = csv.reader(f)
        predscore = [row[1] for row in reader]
    predscore = predscore[1:]
    with open(pth,'r') as f:
        reader = csv.reader(f)
        predbox = [row[2] for row in reader]
    predbox = predbox[1:]
    with open(pth,'r') as f:
        reader = csv.reader(f)
        prediou = [row[3] for row in reader]
    prediou = prediou[1:]
    with open(pth,'r') as f:
        reader = csv.reader(f)
        gtbox = [row[4] for row in reader]
    gtbox = gtbox[1:]
    with open(pth,'r') as f:
        reader = csv.reader(f)
        gtlabel = [row[5] for row in reader]
    gtlabel = gtlabel[1:]
    with open(pth,'r') as f:
        reader = csv.reader(f)
        patientlist = [row[-4] for row in reader]
    patientlist = patientlist[1:]
    with open(pth,'r') as f:
        reader = csv.reader(f)
        dettype = [row[-3] for row in reader]
    dettype = dettype[1:]
    return predscore, predbox, gtbox, gtlabel,patientlist,prediou,dettype

if __name__ == '__main__':
    epoch_num = 148 
    iou_th = 0.2
    score_th = 0.5
    model_pth = '/data/yuezhou/newdata/0717_mrcnn/'
    gt_pth = '/shenlab/lab_stor6/yuezhou/ABUSdata/abus_npy/'
    fusionmap_pth = model_pth + 'plots/3D_result_epoch{}/'.format(epoch_num)
    det_pth = model_pth + 'test/test_epoch_{}.csv'.format(epoch_num)
    files = os.listdir(fusionmap_pth)
    patient_list,fusionmap_pth_list = [],[]
    for ff in files:
        if 'fusion' in ff:
            tempff = ff[:9]
            if tempff[-1] == '_':
                patient_list.append(ff[:8])
            else:
                patient_list.append(ff[:9])
            fusionmap_pth_list.append(fusionmap_pth + ff)
    print('patient_list',patient_list)
    tp_roi,fp_roi = 0,0
    tp_patient = []
    total_roi = 0
    connectivity = 6 # only 26, 18, and 6 are allowed
    predbox_seg,predpatient_seg = [],[]
    for ii,ff in enumerate(fusionmap_pth_list):
        data = sitk.ReadImage(ff)
        seg = sitk.GetArrayFromImage(data)
        this_gt_pth = gt_pth + patient_list[ii] + '_rois.npy'
        gt = np.load(this_gt_pth)
        
        segout = cc3d.connected_components(seg, connectivity=connectivity)
        this_p_seg_num = segout.max()
        print('this patient has seg region',segout.max())

        for i in range(this_p_seg_num):
            total_roi += 1
            seg_each = segout == i+1
            seg_each = seg_each.astype(np.uint8)
            this_pred_bbox_seg = getbbox(seg_each)
            predbox_seg.append(this_pred_bbox_seg)
            predpatient_seg.append(patient_list[ii])
            ## evaluate seg as det
            n = seg_each * gt 
            intersation = sum(sum(sum(n)))
            u = sum(sum(sum(seg_each))) + sum(sum(sum(gt))) - intersation
            iou = float(intersation) / u
            if iou < iou_th: # this is a fp
                fp_roi += 1
            else:
                tp_roi += 1
                tp_patient.append(patient_list[ii])
    tp_patient = list(set(tp_patient))
    smooth = 0.001
    recall = float(len(tp_patient)) / (len(patient_list)+smooth) 
    precision = float(len(tp_patient)) / (len(tp_patient) + fp_roi + smooth)
    fp_per_volume = float(fp_roi)/len(patient_list)
    print('recall:{},precision:{},fp_per_volume:{}'.format(recall,precision,fp_per_volume))
    ## evaluate det combine seg
    predscore_det, predbox_det, gtbox_det, gtlabel_det,patientlist_det,prediou_det,dettype = readcsv(det_pth)
    preocessed_box = preocess_box(predbox_det) 
    selected_score,selected_box,selected_gtbox,selected_gtlabel,selected_patient,selected_iou,selected_type = [], [],[],[],[],[],[]
    # process each pred box
    for ii,bb in enumerate(preocessed_box):
        pat = patientlist_det[ii]
        segpth =fusionmap_pth + pat + '_epoch{}_fusionmap.nii.gz'.format(epoch_num)
        segmap = sitk.ReadImage(segpth)
        segmap = sitk.GetArrayFromImage(segmap)
        detmap = np.zeros(segmap.shape)
        detmap[bb[1]:bb[3],bb[0]:bb[2],bb[4]:bb[5]] = 1
        u = sum(sum(sum(segmap * detmap)))
        n = sum(sum(sum(segmap))) + sum(sum(sum(detmap)))-u
        iou = float(u)/n
        ## select det has overlap with seg and score > th or no segment keep all dets 
        #if iou > 0.0 and float(predscore_det[ii]) > 0.3 or sum(sum(sum(segmap))) == 0:
        if float(predscore_det[ii]) > score_th:
            selected_score.append(predscore_det[ii])
            selected_box.append(predbox_det[ii])
            selected_gtbox.append(gtbox_det[ii])
            selected_gtlabel.append(gtlabel_det[ii])
            selected_patient.append(patientlist_det[ii])
            selected_iou.append(prediou_det[ii])
            selected_type.append(dettype[ii])
    ## save selected det box in a new file
    newpth = det_pth[:-4] + '_score_0.5.csv'
    print('newpth',newpth)
    f = open(newpth,'w+')
    f.write('%s,%s,%s,%s,%s\n'%('pid','det','score','type','gtlabel'))
    for ii,pp in enumerate(selected_patient):
        f.write('%s,%s,%s,%s,%s\n'%(pp,selected_box[ii],selected_score[ii],selected_type[ii],selected_gtlabel[ii]))
    f.close()
    #for ii,pp in enumerate(predpatient_seg):
        #f.write('%s,%s,%s,%s,%s\n'%(pp,[predbox_seg[ii]],0.9,'det_tp',1))

