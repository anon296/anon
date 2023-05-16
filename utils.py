import numpy as np
from os.path import join
import torch
import torch.nn as nn
from scipy import stats


class ShapePrinter(nn.Module):
    def __init__(self,do_print):
        super().__init__()
        self.do_print = do_print

    def forward(self,inp):
        if self.do_print:
            print(inp.shape)
        return inp

def make_zs_str(zs_combo,zs_combo_vals,is_test,normal_testset_number=0):
    assert len(zs_combo) == 2
    assert len(zs_combo_vals) == 2
    assert all([x>=0 for x in zs_combo_vals])

    if zs_combo == [-1,-1]:
        zs_str = 'normal_testset'
    else:
        zs_str = f'{zs_combo[0]}{zs_combo_vals[0]}-{zs_combo[1]}{zs_combo_vals[1]}'
    if is_test:
        zs_str += 'test'
    elif zs_combo == [-1,-1]:
        if not isinstance(normal_testset_number,str):
            assert isinstance(normal_testset_number,int)
            normal_testset_number = str(normal_testset_number)
        zs_str += normal_testset_number

    return zs_str

def enc_block(nin,nout,ksize,stride,padding=0,do_print=False,is_pool=False,**kwargs):
    block = [nn.Conv2d(nin,nout,ksize,stride,**kwargs),
                ShapePrinter(do_print), nn.BatchNorm2d(nout), nn.ReLU()]
    if is_pool:
        block = block[:1] + [nn.MaxPool2d(2)] + block[1:]
    return nn.Sequential(*block)

def dec_block(nin,nout,ksize,stride,is_last=False,do_print=False,**kwargs):
    if is_last:
        block = [nn.ConvTranspose2d(nin,nout,ksize,stride,**kwargs),
                    ShapePrinter(do_print), nn.BatchNorm2d(nout), nn.ReLU()]
    else:
        block = [nn.ConvTranspose2d(nin,nout,ksize,stride,**kwargs), ShapePrinter(do_print), nn.Sigmoid()]
    return nn.Sequential(*block)

def conv_block(nin,nout,ksize,is_pool,do_print):
    mods = [nn.Conv2d(nin,nout,ksize),nn.ReLU(),nn.BatchNorm2d(nout,track_running_stats=False),ShapePrinter(do_print)]
    if is_pool:
        mods.append(nn.MaxPool2d(2))
    return nn.Sequential(*mods)

def load_latents_and_gts(dset,vae,zs_str,test):
    exps_dir = 'experiments_tests' if test else 'experiments'
    latents_dir = join(exps_dir,f'{dset}_{vae}_{zs_str}')
    if vae in ['pvae','pvae-semisup']:
        train_data = np.load(join(latents_dir,f'{dset}_{vae}_train_data_{zs_str}.npz'))
        test_data = np.load(join(latents_dir,f'{dset}_{vae}_test_data_{zs_str}.npz'))
        c_latents_train, z_latents_train = train_data['latents_c'],train_data['latents_z']
        c_gts_train, z_gts_train = train_data['gts'][:,4],np.delete(train_data['gts'],4,axis=1)
        c_latents_test, z_latents_test = test_data['latents_c'],test_data['latents_z']
        c_gts_test, z_gts_test = test_data['gts'][:,4],np.delete(test_data['gts'],4,axis=1)
        # Need for the linear/SN only
        latents_train, gts_train = np.concatenate((z_latents_train,c_latents_train),axis=1), np.concatenate((z_gts_train,np.expand_dims(c_gts_train,1)),axis=1)
        latents_test, gts_test = np.concatenate((z_latents_test,c_latents_test),axis=1), np.concatenate((z_gts_test,np.expand_dims(c_gts_test,1)),axis=1)
    else:
        train_data = np.load(join(latents_dir,f'{dset}_{vae}_train_data_{zs_str}.npz'))
        test_data = np.load(join(latents_dir,f'{dset}_{vae}_test_data_{zs_str}.npz'))
        latents_train, gts_train = train_data['latents'], train_data['gts']
        latents_test, gts_test = test_data['latents'], test_data['gts']
    if vae == 'weakde':
        print(f'latents shape is {latents_train.shape} {latents_test.shape}')
    return latents_train, gts_train, latents_test, gts_test

def normalize(x):
    return (x-x.min()) / (x.max() - x.min())

def load_trans_dict(trans_dict_fpath):
    return dict([p for p in np.load(trans_dict_fpath)])

def convert_reg_targets_to_classif_targets(y):
    classif_targets = np.empty_like(y)
    for factor in range(y.shape[1]):
        reg_fts = y[:,factor]
        classif_targets[:,factor] = sum([i*(reg_fts==v) for i,v in enumerate(set(reg_fts))])
    int_classif_targets = classif_targets.astype(int)
    assert np.allclose(classif_targets,int_classif_targets)
    return int_classif_targets

def np_ent(p):
    counts = np.bincount(p)
    return stats.entropy(counts)
