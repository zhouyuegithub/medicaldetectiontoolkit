import os 
import pickle
import numpy as np
import pandas as pd

def aggregate_meta_info(exp_dir):

    files = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if 'meta_info' in f]
    df = pd.DataFrame(columns=['pid', 'class_target', 'spacing'])
    for f in files:
        with open(f, 'rb') as handle:
            df.loc[len(df)] = pickle.load(handle)

    df.to_pickle(os.path.join(exp_dir, 'info_df.pickle'))
    print ("aggregated meta info to df with length", len(df))

if __name__ == '__main__':
    train_pth = '/data/yuezhou/newdata/0713_mrcnn_seg_fusprob_opt_newdata/roi_train/info_df.pickle'
    val_pth = '/data/yuezhou/newdata/0713_mrcnn_seg_fusprob_opt_newdata/roi_val/info_df.pickle'
    test_pth = '/data/yuezhou/newdata/0713_mrcnn_seg_fusprob_opt_newdata/roi_test/info_df.pickle'
    foldinfo_pth = '/data/yuezhou/newdata/0713_mrcnn_seg_fusprob_opt_newdata/roi_gt/fold_ids.txt'
    train_pid = []
    val_pid = []
    test_pid = []
    f = open(train_pth,'rb')
    data_train = pickle.load(f)
    for p in data_train['pid']:
        train_pid.append(p)
    f = open(val_pth,'rb')
    data_val = pickle.load(f)
    for p in data_val['pid']:
        val_pid.append(p)
    f = open(test_pth,'rb')
    data_test = pickle.load(f)
    for p in data_test['pid']:
        test_pid.append(p)
    print('train_pid',len(train_pid))
    print('val_pid',len(val_pid))
    print('test_pid',len(test_pid))
    f = open(foldinfo_pth,'w+')
    for n in train_pid:
        f.write('train:'+n)
        f.write('\n')
    for n in val_pid:
        f.write('val:'+n)
        f.write('\n')
    for n in test_pid:
        f.write('test:'+n)
        f.write('\n')
    f.close()
    info_pth = '/data/yuezhou/newdata/0713_mrcnn_seg_fusprob_opt_newdata/roi_gt/'
    aggregate_meta_info(info_pth)
