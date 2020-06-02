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

import configs
cf = configs.configs()

def aggregate_meta_info(exp_dir):

    files = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if 'meta_info' in f]
    df = pd.DataFrame(columns=['pid', 'class_target', 'spacing', 'fg_slices'])
    for f in files:
        with open(f, 'rb') as handle:
            df.loc[len(df)] = pickle.load(handle)

    df.to_pickle(os.path.join(exp_dir, 'info_df.pickle'))
    print ("aggregated meta info to df with length", len(df))

if __name__ == "__main__":
    img_path = './image/'
    label_path = './label/'
    save_path = './abus_data'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    filenames = [filename for filename in os.listdir(img_path)]
    for filename in tqdm.tqdm(filenames):
        pid = filename.split('.')[0]
        img = sitk.ReadImage(os.path.join(img_path, filename))
        img_array = sitk.GetArrayFromImage(img)
        label = sitk.ReadImage(os.path.join(label_path, filename))
        label_array = sitk.GetArrayFromImage(label)
        label_array[label_array != 0] = 1
        fg_slices = [ii for ii in np.unique(np.argwhere(label_array != 0)[:, 0])]

        np.save(os.path.join(save_path, '{}_rois.npy'.format(pid)), label_array)
        np.save(os.path.join(save_path, '{}_img.npy'.format(pid)), img_array)
        with open(os.path.join(save_path, 'meta_info_{}.pickle'.format(pid)), 'wb') as handle:
            meta_info_dict = {'pid': pid, 'class_target': [1], 'spacing': img.GetSpacing(), 'fg_slices': fg_slices}
            pickle.dump(meta_info_dict, handle)

    aggregate_meta_info(save_path)
