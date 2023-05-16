import numpy as np
import sys
import json
from utils import make_zs_str, load_latents_and_gts
import os
from os.path import join
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import torch
import torch.nn.functional as F
import argparse
from scipy.optimize import linear_sum_assignment
#from entropy_estimators import continuous
from dl_utils.misc import check_dir
from dl_utils.torch_misc import CifarLikeDataset
from dl_utils.tensor_funcs import numpyify
from dl_utils.label_funcs import get_num_labels, get_trans_dict, get_trans_dict_from_cost_mat
from utils import normalize, load_trans_dict, np_ent


def cond_ent_for_alignment(x,y):
    num_classes = get_num_labels(y)
    dividers_idx = np.arange(0,len(x),len(x)/num_classes).astype(int)
    bin_dividers = np.sort(x)[dividers_idx]
    bin_vals = sum([x<bd for bd in bin_dividers])
    total_ent = 0
    for bv in np.unique(bin_vals):
        bv_mask = bin_vals==bv
        gts_for_this_val = y[bv_mask]
        new_ent = np_ent(gts_for_this_val)
        total_ent += new_ent*bv_mask.sum()/len(x)
    return total_ent

def combo_acc(cs1,cs2):
    return np.logical_and(cs1,cs2).mean()

def round_maybe_list(x,round_factor=100):
    if isinstance(x,float):
        return round(round_factor*x,2)
    elif isinstance(x,list):
        return [round(round_factor*item,2) for item in x]
    else: return x

class PredictorAligner():
    def __init__(self,dset_name,vae_name,latents_name,test,max_epochs,verbose=False):
        self.test = test
        self.latents_name = latents_name
        self.dset_name = dset_name
        self.vae_name = vae_name
        self.max_epochs = max_epochs
        #self.zs_combo = (2,-1) if vae_name=='pvae' else (2,4)
        self.zs_combo = (2,4)
        self.results = {'train': {}, 'test': {}}
        self.verbose = verbose
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_nt = latents_name.startswith('normal_testset')
        check_dir('mi_mats')
        check_dir('trans_dicts')

    def set_cost_mat(self,latents_,gts):
        if self.test:
            self.cost_mat = np.random.rand(self.nz,self.n_factors)
        latents = normalize(latents_)
        assert latents.ndim == 2
        assert gts.ndim == 2
        self.nz = latents.shape[1]
        cost_mat_ = np.empty((self.nz,self.n_factors)) # tall thin matrix normally
        for i in range(self.nz):
            if self.verbose:
                print(i)
            factor_latents = latents[:,i]
            for j in range(self.n_factors):
                if self.verbose:
                    print('\t' + str(j))
                factor_gts = gts[:,j]
                cost_mat_[i,j] = cond_ent_for_alignment(factor_latents,factor_gts)

        raw_ents = np.array([np_ent(self.gts[:,j]) for j in range(self.n_factors)])
        self.cost_mat = np.transpose(cost_mat_ - raw_ents) # transpose added because bug
        if self.verbose:
            print(self.cost_mat)
        np.save(f'mi_mats/{self.dset_name}_mi_mat_{self.latents_name}.npy',-self.cost_mat)
        if not self.cost_mat.max() <= 0.03: # -MI should be negative but error in computing entropy
            breakpoint()

    def set_maybe_loaded_trans_dict(self,is_load=False,is_save=False):
        possible_fpath = join('trans_dicts',f'{self.dset_name}_{self.vae_name}_trans_dict_{self.latents_name}.npy')
        if is_load and os.path.isfile(possible_fpath):
            self.trans_dict = load_trans_dict(possible_fpath)
        else:
            self.set_cost_mat(self.latents,self.gts)
            self.trans_dict = get_trans_dict_from_cost_mat(self.cost_mat)
        if is_save:
            np.save(possible_fpath,np.array([[k,v] for k,v in self.trans_dict.items()]))

    def set_data(self,latents,gts,latents_test,gts_test):
        assert latents.ndim == 2
        assert gts.ndim == 2
        assert latents_test.ndim == 2
        assert gts_test.ndim == 2
        assert latents.shape[1] == latents_test.shape[1]
        assert gts.shape[1] == gts_test.shape[1]
        assert latents.shape[0] == gts.shape[0]
        assert latents_test.shape[0] == gts_test.shape[0]
        self.latents, self.gts = latents, gts
        self.latents_test, self.gts_test = latents_test, gts_test
        self.n_factors = self.gts.shape[1]
        self.nz = self.latents.shape[1]

    def save_and_print_results(self,save_fpath):
        if self.test:
            save_fpath += '.test'
        if os.path.isfile(save_fpath):
            print(f'WARNING: accs file already exists at {save_fpath}')
            save_fpath += '.safe'
            print(f'saving to {save_fpath} instead')
        self.results = {k:{k2:round_maybe_list(v2) for k2,v2 in v.items()}
                            for k,v in self.results.items()}
        for k,v in self.results.items():
            print('\n'+k.upper()+':')
            for k2,v2 in v.items():
                print(f'{k2}: {v2}')

        with open(save_fpath,'w') as f:
            json.dump(self.results,f)

    def train_classif(self,x,y,x_test,y_test,is_mlp=True):
        ngts = max(y.max(), y_test.max()) + 1
        if x.ndim == 1: x=np.expand_dims(x,1)
        if is_mlp:
            fc = nn.Sequential(nn.Linear(x.shape[1],256,device=self.device),nn.ReLU(),nn.Linear(256,ngts,device=self.device))
        else:
            fc = nn.Linear(x.shape[1],ngts,device=self.device)
        opt = Adam(fc.parameters(),lr=1e-2,weight_decay=0e-2)
        scheduler = lr_scheduler.ExponentialLR(opt,gamma=0.65)

        assert not ((x_test == 'none') ^ (y_test=='none'))
        if x_test == 'none':
            dset = CifarLikeDataset(x,y)
            len_train_set = int(len(x)*.8)
            lengths = [len_train_set,len(x)-len_train_set]
            train_set, test_set = random_split(dset,lengths)
            test_gt = y[test_set.indices]
        else:
            if x_test.ndim == 1: x_test=np.expand_dims(x_test,1)
            train_set = CifarLikeDataset(x,y)
            test_set = CifarLikeDataset(x_test,y_test)
            test_gt = y_test
        train_loader = DataLoader(train_set,shuffle=True,batch_size=4096)
        test_loader = DataLoader(test_set,shuffle=False,batch_size=4096)
        train_losses = []
        tol = 0
        best_acc = 0
        best_corrects = np.zeros(len(y_test)).astype(bool)
        for i in range(self.max_epochs):
            if i > 0 and (i&(i-1) == 0): # power of 2
                scheduler.step()
            train_preds_list = []
            train_gts_list = []
            for batch_idx,(xb,yb) in enumerate(train_loader):
                preds = fc(xb.to(self.device))
                #if not (yb>=0).all():
                    #breakpoint()
                #if not yb.max() < preds.shape[1]:
                    #breakpoint()
                loss = F.cross_entropy(preds,yb.to(self.device))
                loss.backward(); opt.step()
                for p in fc.parameters():
                    p.grad = None
                train_losses.append(loss.item())
                train_preds_list.append(preds.argmax(axis=1))
                train_gts_list.append(yb)

            train_preds_array = numpyify(torch.cat(train_preds_list))
            train_gt = numpyify(torch.cat(train_gts_list))
            train_corrects = (train_preds_array==train_gt)
            train_acc = train_corrects.mean()
            preds_list = []
            with torch.no_grad():
                for xb,yb in test_loader:
                    preds = fc(xb.to(self.device))
                    preds_list.append(preds.argmax(axis=1))
                preds_array = numpyify(torch.cat(preds_list))
                corrects = (preds_array==test_gt)
            acc = corrects.mean()
            if self.verbose:
                print(f'Epoch: {i}\ttrain acc: {train_acc:.3f}\ttest acc:{acc:.3f}')
            if train_acc > .99:
                if i==0:
                    best_corrects=corrects
                break
            if train_acc>best_acc:
                tol = 0
                best_acc = train_acc
                best_corrects = corrects
            else:
                tol += 1
                if tol == 4 and self.verbose:
                    print('breaking at', i)
                    break
            if self.test: break
        return train_corrects, best_corrects

    def simple_accs(self,x,y,xt,yt):
        num_classes = get_num_labels(y)
        bin_dividers = np.sort(x)[np.arange(0,len(x),len(x)/num_classes).astype(int)]
        bin_dividers[0] = min(bin_dividers[0],min(xt))
        bin_vals = sum([x<bd for bd in bin_dividers]) # sum is over K np arrays where K is num classes, produces labels for the dset
        bin_vals_test = sum([xt<bd for bd in bin_dividers])
        trans_dict = get_trans_dict(bin_vals,y,subsample_size=30000)
        train_corrects = np.array([trans_dict[z] for z in bin_vals]) == y
        test_corrects = np.array([trans_dict[z] for z in bin_vals_test]) == yt
        return train_corrects, test_corrects
        train_correctsa = (bin_vals==y)
        train_correctsb = ((num_classes-1-bin_vals)==y)
        use_reverse_encoding = train_correctsb.mean() > train_correctsa.mean()
        if use_reverse_encoding:
            test_corrects = ((num_classes-1-bin_vals_test)==yt)
            return train_correctsb, test_corrects
        else:
            test_corrects = (bin_vals_test==yt)
            return train_correctsa, test_corrects

    def accs_from_alignment(self,is_single_neurons,is_mlp):
        if is_single_neurons and not hasattr(self,'trans_dict'):
            self.set_maybe_loaded_trans_dict()
        all_train_corrects = []
        all_test_corrects = []
        for factor in range(self.gts.shape[1]):
            factor_gts = self.gts[:,factor]
            factor_gts_test = self.gts_test[:,factor]
            if is_single_neurons:
                corresponding_latent = self.trans_dict[factor]
                if self.verbose:
                    print(f'predicting gt {factor} with neuron {corresponding_latent}')
                factor_latents = self.latents[:,corresponding_latent]
                factor_latents_test = self.latents_test[:,corresponding_latent]
                train_corrects,test_corrects = self.simple_accs(factor_latents,factor_gts,factor_latents_test,factor_gts_test)
            else:
                train_corrects,test_corrects = self.train_classif(self.latents,factor_gts,self.latents_test,factor_gts_test,is_mlp=is_mlp)
            all_train_corrects.append(train_corrects)
            all_test_corrects.append(test_corrects)
        return all_train_corrects, all_test_corrects

    def zs_combo_accs_from_corrects(self,all_cs):
        zs_acc = combo_acc(all_cs[self.zs_combo[0]],all_cs[self.zs_combo[1]])
        return [c.mean() for c in all_cs] + [zs_acc]

    def standarize(x):
        x = x-x.mean(axis=0)
        x /= (x.std(axis=0)+1e-8)
        return x

    def progressive_knockout(self,start_at):
        self.results['train']['full_knockouts'] = {k:{} for k in range(self.n_factors)}
        self.results['test']['full_knockouts'] = {k:{} for k in range(self.n_factors)}
        if not hasattr(self,'trans_dict'):
            self.set_maybe_loaded_trans_dict()
        for m in range(self.n_factors):
            y = self.gts[:,m]
            y_test = self.gts_test[:,m]
            X = np.copy(self.latents)
            X_test = np.copy(self.latents_test)
            cost_mat = np.transpose(self.cost_mat)
            tmp_trans_dict = self.trans_dict
            for num_excluded in range(self.nz):
                if num_excluded >= start_at:
                    print(f'predicting factor {m} with {num_excluded} neurons knocked out')
                    tr_corrects, ts_corrects = self.train_classif(X,y,X_test,y_test,is_mlp=True)
                    self.results['train']['full_knockouts'][m][num_excluded] = tr_corrects.mean()
                    self.results['test']['full_knockouts'][m][num_excluded] = tr_corrects.mean()
                tmp_trans_dict = get_trans_dict_from_cost_mat(cost_mat)
                latent_to_exclude = tmp_trans_dict[m]
                X = np.delete(X,latent_to_exclude,1)
                X_test = np.delete(X_test,latent_to_exclude,1)
                cost_mat = np.delete(cost_mat,latent_to_exclude,1)

    def exclusive_mlp(self,test_factor,exclude_factor,num_to_exclude):
        first_latent_to_exclude = self.trans_dict[exclude_factor]
        X = np.delete(self.latents,first_latent_to_exclude,1)
        X_test = np.delete(self.latents_test,first_latent_to_exclude,1)
        if num_to_exclude == 2:
            cost_mat2 = np.delete(self.cost_mat,first_latent_to_exclude,1)
            trans_dict2 = get_trans_dict_from_cost_mat(cost_mat2)
            second_latent_to_exclude = trans_dict2[exclude_factor]
            X = np.delete(X,second_latent_to_exclude,1)
            X_test = np.delete(X_test,second_latent_to_exclude,1)

        y = self.gts[:,test_factor]
        y_test = self.gts_test[:,test_factor]
        return self.train_classif(X,y,X_test,y_test,is_mlp=True)

    def set_exclusive_mlp_results(self,num_to_exclude):
        if not hasattr(self,'trans_dict'):
            self.set_maybe_loaded_trans_dict()
        # cs = corrects, re = reflexive exclude
        train_cs1, test_cs1 = self.exclusive_mlp(self.zs_combo[0],self.zs_combo[1],num_to_exclude)
        train_cs2, test_cs2 = self.exclusive_mlp(self.zs_combo[1],self.zs_combo[0],num_to_exclude)
        zs_train_acc = combo_acc(train_cs1, train_cs2)
        zs_test_acc = combo_acc(test_cs1, test_cs2)
        self.results['train'][f'e{num_to_exclude}f1'] = train_cs1.mean()
        self.results['train'][f'e{num_to_exclude}f2'] = train_cs2.mean()
        self.results['train'][f'e{num_to_exclude}zs'] = zs_train_acc
        self.results['test'][f'e{num_to_exclude}f1'] = test_cs1.mean()
        self.results['test'][f'e{num_to_exclude}f2'] = test_cs2.mean()
        self.results['test'][f'e{num_to_exclude}zs'] = zs_test_acc

    def set_reflexive_exclude_mlp_results(self,num_to_exclude):
        results_name = f're{num_to_exclude}'
        self.results['train'][results_name] = []
        self.results['test'][results_name] = []
        if self.is_nt:
            for fn in range(self.n_factors):
                re_train_cs,re_test_cs = self.exclusive_mlp(fn,fn,num_to_exclude)
                self.results['train'][results_name].append(re_train_cs.mean())
                self.results['test'][results_name].append(re_test_cs.mean())

    def predict_unsupervised_vae(self,latents,gts,latents_test,gts_test):
        self.set_data(latents,gts,latents_test,gts_test)
        sn_train_cs, sn_test_cs = self.accs_from_alignment(is_single_neurons=True,is_mlp='none')
        self.results['train']['single-neuron'] = self.zs_combo_accs_from_corrects(sn_train_cs)
        self.results['test']['single-neuron'] = self.zs_combo_accs_from_corrects(sn_test_cs)
        self.set_exclusive_mlp_results(num_to_exclude=1)
        self.set_exclusive_mlp_results(num_to_exclude=2)
        if self.is_nt:
            self.set_reflexive_exclude_mlp_results(num_to_exclude=1)
            self.set_reflexive_exclude_mlp_results(num_to_exclude=2)
        self.set_full_accs_from_classif_func('none')

    def predict_pvae(self,c_latents_train,c_gts_train,c_latents_test,c_gts_test,
                     z_latents_train,z_gts_train,z_latents_test,z_gts_test):

        full_latents_train = np.c_[z_latents_train,c_latents_train]
        full_gts_train = np.c_[z_gts_train,c_gts_train]
        full_latents_test = np.c_[z_latents_test,c_latents_test]
        full_gts_test = np.c_[z_gts_test,c_gts_test]

        print('COMPUTING SINGLE ACCS')
        # Subset neuron (equiv. of single-neuron) accs
        self.set_data(z_latents_train,z_gts_train,z_latents_test,z_gts_test)
        unsuper_z_train_cs, unsuper_z_test_cs = self.accs_from_alignment(is_single_neurons=True,is_mlp='none')
        z_train_cs, z_test_cs = self.accs_from_alignment(is_single_neurons=False,is_mlp='none') # use in excl. below
        class_from_c_train_cs,class_from_c_test_cs = self.train_classif(
                    c_latents_train,c_gts_train,c_latents_test,c_gts_test,is_mlp=True)
        unsuper_zs_train_acc = combo_acc(class_from_c_train_cs,unsuper_z_train_cs[2])
        unsuper_zs_test_acc = combo_acc(class_from_c_test_cs,unsuper_z_test_cs[2])
        self.results['train']['subset-neurons'] = [a.mean() for a in unsuper_z_train_cs] + [class_from_c_train_cs.mean(),unsuper_zs_train_acc]
        self.results['test']['subset-neurons'] = [a.mean() for a in unsuper_z_test_cs] + [class_from_c_test_cs.mean(),unsuper_zs_test_acc]

        if self.is_nt:
            print('COMPUTING REFLEXIVE EXCLUDE ACCS')
            # Equiv of refl. excl. accs
            self.set_data(full_latents_train,z_gts_train,full_latents_test,z_gts_test)
            self.set_maybe_loaded_trans_dict()
            self.set_reflexive_exclude_mlp_results(num_to_exclude=1)
            self.set_reflexive_exclude_mlp_results(num_to_exclude=2)
            class_from_z_train_cs,class_from_z_test_cs = self.train_classif(
                        z_latents_train,c_gts_train,z_latents_test,c_gts_test,is_mlp=True)
            self.results['train'][f're1'].append(class_from_z_train_cs.mean())
            self.results['test'][f're1'].append(class_from_z_test_cs.mean())

        print('COMPUTING EXCLUDE ACCS')
        # Equiv of excl. accs
        excl_zs_train_acc = combo_acc(class_from_c_train_cs,z_train_cs[2])
        excl_zs_test_acc = combo_acc(class_from_c_test_cs,z_test_cs[2])
        self.results['train']['e1'] = [a.mean() for a in z_train_cs] + [class_from_c_train_cs.mean(),excl_zs_train_acc]
        self.results['test']['e1'] = [a.mean() for a in z_test_cs] + [class_from_c_test_cs.mean(),excl_zs_test_acc]

        # Full MLP and linear accs
        print('COMPUTING FULL ACCS')
        self.set_data(full_latents_train,full_gts_train,full_latents_test,full_gts_test)
        self.set_full_accs_from_classif_func('none')

        self.results['test']['NB'] = 'Class predictions are always last in the list, right before zs'

    def default_full_classif_func_(self,is_mlp): # pvae used not use this, may not again in future
        train_cs, test_cs = self.accs_from_alignment(is_single_neurons=False,is_mlp=is_mlp)
        train_accs = self.zs_combo_accs_from_corrects(train_cs)
        test_accs = self.zs_combo_accs_from_corrects(test_cs)
        return train_accs, test_accs

    def set_full_accs_from_classif_func(self,classif_func):
        if classif_func == 'none':
            classif_func = self.default_full_classif_func_

        mlp_train_accs, mlp_test_accs = classif_func(is_mlp=True)
        linear_train_accs, linear_test_accs = classif_func(is_mlp=False)
        self.results['train']['mlp'] = mlp_train_accs
        self.results['test']['mlp'] = mlp_test_accs
        self.results['train']['linear'] = linear_train_accs
        self.results['test']['linear'] = linear_test_accs

    def predict_supervised_vae(self,latents,gts,latents_test,gts_test):
        self.set_data(latents,gts,latents_test,gts_test)
        self.set_full_accs_from_classif_func('none')

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dset',type=str,default='dsprites')
    parser.add_argument('--max_epochs',type=int,default=75)
    parser.add_argument('--is_save_trans_dict',action='store_true')
    parser.add_argument('--test','-t',action='store_true')
    parser.add_argument('--is_load_trans_dict',action='store_true')
    parser.add_argument('--is_mlp',action='store_true')
    parser.add_argument('--is_exclusive_mlp',action='store_true')
    parser.add_argument('--is_single_neurons','-sn',action='store_true')
    parser.add_argument('--linear_only',action='store_true')
    parser.add_argument('--single_neurons_only',action='store_true')
    parser.add_argument('--excl1_only',action='store_true')
    parser.add_argument('--full_knockouts',action='store_true')
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--fix_after_costmat_bug',action='store_true')
    parser.add_argument('--num_to_exclude',type=int,default=2)
    #parser.add_argument('--latents',type=str,default='full')
    parser.add_argument('--vae',type=str,choices=['betaH','btcvae','factor','pvae','pvae-semisup','weakde','lei'],default='betaH')
    parser.add_argument('--data_dir',type=str,default='experiments')
    parser.add_argument('--zs_combo',type=int,nargs='+',default=(2,4))
    parser.add_argument('-nt','--normal_testset_number',type=str,default='0')
    parser.add_argument('--zs_combo_vals',type=int,nargs='+',default=(0,0))
    parser.add_argument('--gpu',type=str,default='0')
    ARGS = parser.parse_args()

    assert not (ARGS.linear_only and ARGS.excl1_only)
    os.environ['CUDA_VISIBLE_DEVICES'] = ARGS.gpu
    zs_str = make_zs_str(ARGS.zs_combo,ARGS.zs_combo_vals,ARGS.test,ARGS.normal_testset_number)
    exps_dir = 'experiments_tests' if ARGS.test else 'experiments'
    latents_dir = join(exps_dir,f'{ARGS.dset}_{ARGS.vae}_{zs_str}')
    print(zs_str)
    print(exps_dir)
    print(latents_dir)

    factor_aligner = PredictorAligner(vae_name=ARGS.vae,latents_name=zs_str,dset_name=ARGS.dset,test=ARGS.test,max_epochs=ARGS.max_epochs,verbose=ARGS.verbose)
    #if ARGS.vae == 'pvae':
    #    train_data = np.load(join(latents_dir,f'{ARGS.dset}_pvae_train_data_{zs_str}.npz'))
    #    test_data = np.load(join(latents_dir,f'{ARGS.dset}_pvae_test_data_{zs_str}.npz'))
    #    c_latents_train, z_latents_train = train_data['latents_c'],train_data['latents_z']
    #    c_gts_train, z_gts_train = train_data['gts'][:,4],np.delete(train_data['gts'],4,axis=1)
    #    c_latents_test, z_latents_test = test_data['latents_c'],test_data['latents_z']
    #    c_gts_test, z_gts_test = test_data['gts'][:,4],np.delete(test_data['gts'],4,axis=1)
    #    # Need for the linear/SN only
    #    latents_train, gts_train = np.concatenate((z_latents_train,c_latents_train),axis=1), np.concatenate((z_gts_train,np.expand_dims(c_gts_train,1)),axis=1)
    #    latents_test, gts_test = np.concatenate((z_latents_test,c_latents_test),axis=1), np.concatenate((z_gts_test,np.expand_dims(c_gts_test,1)),axis=1)
    #else:
    #    train_data = np.load(join(latents_dir,f'{ARGS.dset}_{ARGS.vae}_train_data_{zs_str}.npz'))
    #    test_data = np.load(join(latents_dir,f'{ARGS.dset}_{ARGS.vae}_test_data_{zs_str}.npz'))
    #    latents_train, gts_train = train_data['latents'], train_data['gts']
    #    latents_test, gts_test = test_data['latents'], test_data['gts']
    latents_train, gts_train, latents_test, gts_test = load_latents_and_gts(ARGS.dset,ARGS.vae,zs_str,ARGS.test)

    # if specifying a zs_combo, check that this combo is indeed what defines the test gts
    if ARGS.vae not in ['pvae','pvae-semisup']:
        possible_zs_suffix = zs_str.split('_')[-1]
        if all([possible_zs_suffix[i] in '0123456789' for i in [0,1,3,4]]) and possible_zs_suffix[2] == '-':
            print('checking that the zs matches')
            assert (gts_test[:,int(possible_zs_suffix[0])] == int(possible_zs_suffix[1])).all()
            assert (gts_test[:,int(possible_zs_suffix[3])] == int(possible_zs_suffix[4])).all()
    if ARGS.linear_only:
        factor_aligner.set_data(latents_train,gts_train,latents_test,gts_test)
        print('running linear only')
        train_cs, test_cs = factor_aligner.accs_from_alignment(is_single_neurons=False,is_mlp=False)
        train_accs = factor_aligner.zs_combo_accs_from_corrects(train_cs)
        test_accs = factor_aligner.zs_combo_accs_from_corrects(test_cs)
        with open(join(latents_dir,'extra_linear_accs.txt'),'w') as f:
            f.write('Train:')
            f.write(' '.join([str(a) for a in train_accs]))
            f.write('\nTest:')
            f.write(' '.join([str(a) for a in test_accs]))
        print('Train:', round_maybe_list(train_accs))
        print('Test:', round_maybe_list(test_accs))
        sys.exit()
    elif ARGS.single_neurons_only:
        factor_aligner.set_data(latents_train,gts_train,latents_test,gts_test)
        print('running single neurons only')
        train_cs, test_cs = factor_aligner.accs_from_alignment(is_single_neurons=True,is_mlp=False)
        train_accs = factor_aligner.zs_combo_accs_from_corrects(train_cs)
        test_accs = factor_aligner.zs_combo_accs_from_corrects(test_cs)
        with open(join(latents_dir,'extra_single_neuron_accs.txt'),'w') as f:
            f.write('Train:')
            f.write(' '.join([str(a) for a in train_accs]))
            f.write('\nTest:')
            f.write(' '.join([str(a) for a in test_accs]))
        print('Train:', round_maybe_list(train_accs))
        print('Test:', round_maybe_list(test_accs))
        sys.exit()
    elif ARGS.excl1_only:
        print('running excl1 only')
        factor_aligner.set_data(latents_train,gts_train,latents_test,gts_test)
        with open(join(latents_dir,'accs.txt')) as f:
            factor_aligner.results = json.load(f)
        factor_aligner.set_exclusive_mlp_results(num_to_exclude=1)
    elif ARGS.fix_after_costmat_bug:
        print('running NK1, SNC and exclNCFF')
        factor_aligner.set_data(latents_train,gts_train,latents_test,gts_test)
        with open(join(latents_dir,'accs.txt')) as f:
            x = json.load(f)
            factor_aligner.results = {'train':{k:round_maybe_list(v,0.01) for k,v in x['train'].items()},'test':{k:round_maybe_list(v,0.01) for k,v in x['test'].items()}}
        factor_aligner.set_exclusive_mlp_results(num_to_exclude=1)
        sn_train_cs, sn_test_cs = factor_aligner.accs_from_alignment(is_single_neurons=True,is_mlp='none')
        factor_aligner.results['train']['single-neuron'] = factor_aligner.zs_combo_accs_from_corrects(sn_train_cs)
        factor_aligner.results['test']['single-neuron'] = factor_aligner.zs_combo_accs_from_corrects(sn_test_cs)
        if factor_aligner.is_nt:
            factor_aligner.set_reflexive_exclude_mlp_results(num_to_exclude=1)
    elif ARGS.full_knockouts:
        factor_aligner.set_data(latents_train,gts_train,latents_test,gts_test)
        factor_aligner.progressive_knockout(start_at=3)
        results_fpath = join(exps_dir,f'{ARGS.dset}_{ARGS.vae}_{zs_str}','full_knockouts.txt')
        factor_aligner.save_and_print_results(results_fpath)
        sys.exit()

    #elif ARGS.vae == 'pvae':
        #factor_aligner.predict_pvae(
                    #c_latents_train,c_gts_train,c_latents_test,c_gts_test,
                    #z_latents_train,z_gts_train,z_latents_test,z_gts_test
            #)
    else:
        factor_aligner.predict_unsupervised_vae(latents_train,gts_train,latents_test,gts_test)

    results_fpath = join(latents_dir,'redone_results.json')
    factor_aligner.save_and_print_results(results_fpath)
