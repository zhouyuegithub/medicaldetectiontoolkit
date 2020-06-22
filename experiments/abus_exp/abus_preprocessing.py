import os
import tqdm
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool
import pandas as pd
import numpy.testing as npt
from skimage.transform import resize
import subprocess
import pickle
import csv

import configs
cf = configs.configs()

def readcsv(pth):
    with open(pth,'r') as f:
        reader = csv.reader(f)
        name = [row[0] for row in reader]
    name = name[1:]
    with open(pth,'r') as f:
        reader = csv.reader(f)
        classes = [row[1] for row in reader]
    classes = classes[1:]
    return name,classes


def aggregate_meta_info(exp_dir):

    files = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if 'meta_info' in f]
    df = pd.DataFrame(columns=['pid', 'class_target', 'spacing', 'fg_slices'])
    for f in files:
        with open(f, 'rb') as handle:
            df.loc[len(df)] = pickle.load(handle)

    df.to_pickle(os.path.join(exp_dir, 'info_df.pickle'))
    print ("aggregated meta info to df with length", len(df))

if __name__ == "__main__":
    img_path = '/shenlab/lab_stor6/yuezhou/ABUSdata/newresize/image/'
    label_path = '/shenlab/lab_stor6/yuezhou/ABUSdata/newresize/label/'
    save_path = '/shenlab/lab_stor6/yuezhou/ABUSdata/abus_npy/'
    img_path = '/shenlab/lab_stor6/yuezhou/ABUSdata/roi/image/'
    label_path = '/shenlab/lab_stor6/yuezhou/ABUSdata/roi/label/'
    save_path = '/shenlab/lab_stor6/yuezhou/ABUSdata/abus_roi_npy/'
    #csvpth = '/shenlab/lab_stor6/yuezhou/ABUSdata/data_info/class_info.csv'
    #names, classes = readcsv(csvpth) 
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    filenames = [filename for filename in os.listdir(img_path)]
    for filename in tqdm.tqdm(filenames):
        pid = filename.split('.')[0]
        #idx = names.index(pid)
        #this_class = classes[idx]
        img = sitk.ReadImage(os.path.join(img_path, filename))
        img_array = sitk.GetArrayFromImage(img)
        label = sitk.ReadImage(os.path.join(label_path, filename))
        label_array = sitk.GetArrayFromImage(label)
        label_array[label_array != 0] = 1
        fg_slices = [ii for ii in np.unique(np.argwhere(label_array != 0)[:, 0])]

        np.save(os.path.join(save_path, '{}_rois.npy'.format(pid)), label_array)
        np.save(os.path.join(save_path, '{}_img.npy'.format(pid)), img_array)
        with open(os.path.join(save_path, 'meta_info_{}.pickle'.format(pid)), 'wb') as handle:
            #meta_info_dict = {'pid': pid, 'class_target': [int(this_class)+1], 'spacing': img.GetSpacing(), 'fg_slices': fg_slices}
            meta_info_dict = {'pid': pid, 'class_target': [1], 'spacing': img.GetSpacing(), 'fg_slices': fg_slices}
            pickle.dump(meta_info_dict, handle)

    aggregate_meta_info(save_path)
