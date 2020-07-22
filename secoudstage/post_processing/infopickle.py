import os
import pickle
import pandas as pd

exp_dir = '/data/yuezhou/newdata/0713_mrcnn_seg_fusprob_opt_newdata/roi_train/'
files = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if 'meta_info' in f]
df = pd.DataFrame(columns=['pid', 'class_target', 'spacing'])
for f in files:
    with open(f, 'rb') as handle:
        df.loc[len(df)] = pickle.load(handle)

df.to_pickle(os.path.join(exp_dir, 'info_df.pickle'))
print ("aggregated meta info to df with length", len(df))
