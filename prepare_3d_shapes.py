"""Converts 3dshapes dataset from hdf5 to npz, and reorders factors"""

import h5py
import numpy as np
from dl_utils.label_funcs import compress_labels


with h5py.File('3dshapes.h5') as orig_dset:
    orig_labels = orig_dset['labels'][()]

new_labels = np.empty_like(orig_labels)
for feature_num in range(orig_labels.shape[1]):
    feature_values = orig_labels[:,feature_num]
    feature_values -= feature_values.min()
    feature_values /= feature_values.max()
    new_feature_labels = np.round(feature_values * len(set(feature_values))).astype(int)
    new_labels[:,feature_num] = compress_labels(new_feature_labels)[0] # Also returns trans_dict and changed

new_labels = new_labels[:,np.array([0,1,3,5,4,2])] # Make order similar to what I use for dsprites
with h5py.File('3dshapes.h5') as orig_dset:
    X = orig_dset['images'][()]

new_labels = new_labels.astype(int)
np.savez('3dshapes.npz',imgs=X,latents_classes=new_labels)

med_mask = np.random.choice(len(X),size=10000,replace=False)
np.savez('3dshapes_medium.npz',imgs=X[med_mask],latents_classes=new_labels[med_mask])

small_mask = np.random.choice(len(X),size=1000,replace=False)
np.savez('3dshapes_small.npz',imgs=X[small_mask],latents_classes=new_labels[small_mask])
