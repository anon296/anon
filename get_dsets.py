from dl_utils.torch_misc import CifarLikeDataset
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision.transforms import Compose, ToTensor
from os.path import join


def get_dsprite(test_level,zs_combo,zs_combo_vals,dirpath='.'):
    if test_level == 2:
        archive = np.load(join(dirpath,'dsprites_small.npz'))
    elif test_level == 1:
        archive = np.load(join(dirpath,'dsprites_medium.npz'))
    else:
        archive = np.load(join(dirpath,'dsprites.npz'))
    imgs = archive['imgs']
    class_labels = archive['latents_classes']
    train_set_imgs, test_set_imgs, train_set_labels, test_set_labels = maybe_zs_combo_split(
            imgs, class_labels, zs_combo, zs_combo_vals)
    to_float = lambda t: t.float()
    transform = Compose([ToTensor(),to_float])
    train_set = CifarLikeDataset(train_set_imgs,train_set_labels,transform)
    test_set = CifarLikeDataset(test_set_imgs,test_set_labels,transform)
    return train_set, test_set

def get_maybe_zs_dset(dset_name,test_level,zs_combo,zs_combo_vals,dirpath='datasets'):
    if test_level == 2:
        archive = np.load(join(dirpath,f'{dset_name}_small.npz'))
    elif test_level == 1:
        archive = np.load(join(dirpath,f'{dset_name}_medium.npz'))
    else:
        archive = np.load(join(dirpath,f'{dset_name}.npz'))
    imgs = archive['imgs']
    class_labels = archive['latents_classes']
    train_set_imgs, test_set_imgs, train_set_labels, test_set_labels = maybe_zs_combo_split(
            imgs, class_labels, zs_combo, zs_combo_vals)
    to_float = lambda t: t.float()
    transform = Compose([ToTensor(),to_float])
    train_set = CifarLikeDataset(train_set_imgs,train_set_labels,transform)
    test_set = CifarLikeDataset(test_set_imgs,test_set_labels,transform)
    return train_set, test_set

def get_celebA(test_level,zs_combo,zs_combo_vals,dirpath='.'):
    if test_level == 2:
        archive = np.load(join(dirpath,'dsprites_small.npz'))
    elif test_level == 1:
        archive = np.load(join(dirpath,'dsprites_medium.npz'))
    else:
        archive = np.load(join(dirpath,'dsprites.npz'))
    imgs = archive['imgs']
    class_labels = archive['latents_classes']
    class_labels = class_labels[:,np.array([4,3,1,2,0])] # Put low-level features first
    to_float = lambda t: t.float()
    transform = Compose([ToTensor(),to_float])
    train_set_imgs, train_set_labels, test_set_imgs, test_set_labels = maybe_zs_combo_split(
            imgs, class_labels, zs_combo, zs_combo_vals)
    train_set = CifarLikeDataset(train_set_imgs,train_set_labels,transform)
    test_set = CifarLikeDataset(test_set_imgs,test_set_labels,transform)
    return train_set, test_set

def maybe_zs_combo_split(X,y,zs_combo,zs_combo_vals,test_size=0.2):
    if zs_combo == [-1,-1]:
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size)
    else:
        m1 = y[:,zs_combo[0]] == zs_combo_vals[0]
        m2 = y[:,zs_combo[1]] == zs_combo_vals[1]
        test_set_mask = np.logical_and(m1,m2)
        X_train, y_train = X[~test_set_mask], y[~test_set_mask]
        X_test, y_test = X[test_set_mask], y[test_set_mask]
    return X_train, X_test, y_train, y_test
