import os
import numpy as np
import csv
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

colorlist = ['red','navy','orange','green','blue','pink','black']
fig_width = 3.487#3.487/1.618
fig_height = 3.487#/1.618
params = {
  'backend': 'ps',
  #'text.latex.preamble': ['\usepackage{gensymb}'],
  'axes.labelsize': 10, # fontsize for x and y labels (was 10)
  'axes.titlesize': 8,
  'font.size':       10, # was 10
  'legend.fontsize': 7, # was 10
  'xtick.labelsize': 7,
  'ytick.labelsize': 7,
#   'text.usetex': Tre,
  'figure.figsize': [fig_width,fig_height],
  'font.family': 'serif'
}
matplotlib.rcParams.update(params)

def drawROC(gtlist,problistnew):
    gtnp = np.array(gtlist)
    probnp = np.array(problistnew)
    fpr,tpr,th = metrics.roc_curve(gtnp,probnp,pos_label = 1)
    auc = metrics.roc_auc_score(gtnp,probnp)
    return fpr,tpr,auc

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
    #print('predscore',len(predscore))
    #print('predbox',len(predbox))
    #print('gtbox',len(gtbox))
    return predscore, predbox, gtbox, gtlabel,patientlist,prediou
def preocess_box(boxlist,patient_list):
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
            print(patient_list[ii])
        processed_boxlist.append(thisbox)
    return processed_boxlist 
def preocess_score_label(predscore,gtlabel):
    processed_scorelist = []
    for s in predscore:
        processed_scorelist.append(float(s))

    preocess_gt = []
    for l in gtlabel:
        preocess_gt.append(int(l))

    return processed_scorelist, preocess_gt
def selecte_result(predbox_list,gtbox_list,predscore_list,predgt_list,patient_list,iou,th):
    selected_boxlist,selected_gtlist,selected_scorelist, selected_gtlabel,selecte_patient = [], [], [],[],[]
    selecte_iou = []
    for ii, s in enumerate(predscore_list):
        if s > th:
            selected_scorelist.append(s)
            selected_boxlist.append(predbox_list[ii])
            selected_gtlist.append(gtbox_list[ii])
            selected_gtlabel.append(predgt_list[ii])
            selecte_patient.append(patient_list[ii])
            selecte_iou.append(iou[ii])
    return selected_boxlist, selected_gtlist, selected_scorelist, selected_gtlabel, selecte_patient,selecte_iou
def preocess_iou(iou_list):
    iou = []
    for i in iou_list:
        iou.append(float(i[1:-1]))
    return iou

def count(selecte_patient,box,gt,th):
    tp,fp,tn,fn_roi = 0,0,0,0
    detected_pat = []
    for ii,bb in enumerate(box):
        p = selecte_patient[ii]
        #print('bb',bb)
        pred = np.zeros((79,290,318))
        gt_map = np.zeros((79,290,318))
        pred[bb[0]:bb[1],bb[2]:bb[3],bb[4]:bb[5]] = 1
        gt_bb = gt[ii]
        #print('gt',gt_bb)
        gt_map[gt_bb[0]:gt_bb[1],gt_bb[2]:gt_bb[3],gt_bb[4]:gt_bb[5]] = 1
        u = sum(sum(sum(np.multiply(gt_map,pred))))#sum(np.multiply(gt_map,pred))
        n = sum(sum(sum(pred+gt_map))) - u
        iou = u/float(n)
        #print('iou',iou)
        
        if iou > th:
            tp += 1
            detected_pat.append(p)
        if iou < th:
            fp += 1
        if pred.max() == 0:
            fn_roi += 1
    return tp,fp,tn,fn_roi,detected_pat
def saveFROC(xlabel,ylabel,auc,maxfp,name,th,lastmodel):
    ax = plt.subplot(111)
    for ii,n in enumerate(name):
        #plt.plot(xlabel[ii],ylabel[ii],color=colorlist[ii], label=n+'(AUC:'+str(round(auc[ii],3))+')', linewidth=1)
        plt.plot(xlabel[ii],ylabel[ii],color=colorlist[ii], label=n, linewidth=1)
    #plt.plot([0,1],[0,1],color = 'black',linewidth = 0.5,linestyle = '--')
    plt.ylim(0,1.2)
    plt.xlim(0,np.array(maxfp).max()) 
    ymajorLocator = MultipleLocator(0.2)
    ymajorFormatter = FormatStrFormatter('%1.1f')
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_major_formatter(ymajorFormatter)

    xmajorLocator = MultipleLocator(0.5)
    xmajorFormatter = FormatStrFormatter('%1.1f')
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_major_formatter(xmajorFormatter)
    plt.title('FROC')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive per volume')
    plt.legend(loc='lower right')
    plt.grid(linewidth = 0.5,linestyle = '--')
    plt.tight_layout()
    if lastmodel == True:
        #name = '/shenlab/lab_stor6/yuezhou/ABUSdata/mrcnn/froc/lastmodel/{}_roc_{}.pdf'.format('lastmodel',th)
        #name = '/data/yuezhou/mrcnn-fusion-seg/0709_mfs_add_as_roiDice/plots/{}_roc_{}.pdf'.format('lastmodel',th)
        name = '/data/yuezhou/models/baseline/compare/lastepoch_roc_{}_new.pdf'.format(th)
    else:
        name = '/data/yuezhou/temp_det_result/roc_{}.pdf'.format(th)
    plt.savefig(name)


if __name__ == '__main__':
    th = 0.7
    iou_th = 0.3
    lastmodel = False 
    result_pths = '/data/yuezhou/temp_det_result/'
    models = os.listdir(result_pths)
    pths,mm =[], []
    maxfp = []
    for m in models:
        #if 'rcnn' in m:
        print(m)
        mm.append(m)
        pth_ = result_pths+m+'/test/'
        f = os.listdir(pth_)
        for ff in f:
            if lastmodel == True and 'last' in ff and 'dice' not in ff:
                pths.append(pth_+ff)
            if lastmodel == False and 'last' not in ff and 'test' in ff:
                pths.append(pth_+ff)
    #pths = []
    #pth = '/data/yuezhou/mrcnn-fusion-seg/0709_mfs_add_as_roiDice/test/test_epoch_35.csv'
    #pths.append(pth)
    print(pths)
    xlabel,ylabel,name,auclist =[], [],[],[]
    for ii, result_pth in enumerate(pths):
        print('*'*50)
        print('processing model',mm[ii])
        predscore, predbox, gtbox, gtlabel,patient_list,iou_list = readcsv(result_pth)
        predbox_list = preocess_box(predbox,patient_list)
        gtbox_list = preocess_box(gtbox,patient_list)
        predscore_list, predgt_list = preocess_score_label(predscore,gtlabel)
        iou_list = preocess_iou(iou_list)

        fpr,tpr,auc = drawROC(predgt_list,predscore_list)
        #print('fpr{},tpr{},auc{}'.format(fpr,tpr,auc))

        selected_boxlist, selected_gtlist, selected_scorelist, selected_gtlabel,selecte_patient,selecte_iou = selecte_result(predbox_list,gtbox_list,predscore_list,predgt_list,patient_list,iou_list,th) 
        p = []
        [p.append(i) for i in selecte_patient if not  i  in p]
        selecte_patient_num = len(p)

        #print('afterselect patient',(p))
        #print('selecte_patient',selecte_patient_num)
        #print('afterselect bbox',len(selected_boxlist))
        tp,fp,fn,tn,detected_pat = count(selecte_patient,selected_boxlist,selected_gtlist,iou_th)
        p = []
        [p.append(i) for i in detected_pat if not  i  in p]
        tp_patinet = len(p)
        fn_patient = 34 - tp_patinet
        fp_roi = fp
        tp_roi = tp
        print('tp_patinet:{},fn_patient:{}'.format(tp_patinet,fn_patient))
        print('tp_roi:{},fp_roi:{}'.format(tp_roi,fp_roi))
        recall = tp_patinet/float(tp_patinet+fn_patient+0.001)
        percision = tp_patinet/float(tp_patinet+fp_roi+0.001)
        fp_per_volume = fp_roi/float(34)
        print('recall:{},percision:{},fp per volume:{}'.format(recall,percision,fp_per_volume))

        fpr,tpr,auc = drawROC(selected_gtlabel,selected_scorelist)
        FP_num = fpr*fp/float(34)
        xlabel.append(FP_num) 
        ylabel.append(tpr)
        auclist.append(auc)
        name.append(mm[ii])
        #name.append(result_pth.split('/')[-3])
        maxfp.append(np.array(FP_num).max())

    saveFROC(xlabel,ylabel,auclist,maxfp,name,th,lastmodel)
