import numpy as np
import os
import torch
from align_and_predict import PredictorAligner
from disentangling_vae_fork.run import run_bvae_like_model
from weak_disentanglement.train import run_weak_de
from parted_vae_fork.run import run_pvae
from dear.train import run_dear
import argparse
from os.path import join
from dl_utils.misc import set_experiment_dir, asMinutes
from utils import make_zs_str
from time import time


ALL_N_FACTORS_DICT = {
    'dsprites': [32,32,6,40,3],
    '3dshapes': [10,10,8,15,4,10],
    'mpi3d': [40,40,2,6,6,3,3],
}

def main(args):
    start_time = time()
    n_factors = ALL_N_FACTORS_DICT[args.dataset]
    for f, val in zip(args.zs_combo, args.zs_combo_vals):
        assert val < n_factors[f], "ZS-combo val is out of range"

    exps_dir = 'experiments_tests' if args.test else 'experiments'
    zs_str = make_zs_str(args.zs_combo,args.zs_combo_vals,args.test,args.normal_testset_number)
    results_dir = join(exps_dir,f'{args.dataset}_{args.vae}_{zs_str}{args.name_suffix}')
    set_experiment_dir(results_dir,args.overwrite)

    train_save_fpath = join(results_dir,f'{args.dataset}_{args.vae}_train_data_{zs_str}.npz')
    test_save_fpath = join(results_dir,f'{args.dataset}_{args.vae}_test_data_{zs_str}.npz')
    print('results_dir:',results_dir)
    print('train_save_fpath:',train_save_fpath)
    print('test_save_fpath:',test_save_fpath)

    pa = PredictorAligner(vae_name=ARGS.vae,latents_name=zs_str,
                          dset_name=args.dataset,test=args.test,
                          max_epochs=args.max_predict_epochs,verbose=args.verbose)

    summary_fpath = join(results_dir,'summary.txt')
    accs_fpath = join(results_dir,'accs.txt')

    if args.vae in ['pvae-semisup','pvae-unsup']:
        latents_c_train, latents_z_train, gts_train, latents_c_test, latents_z_test, gts_test = run_pvae(ARGS)

        np.savez(train_save_fpath,latents_c=latents_c_train,latents_z=latents_z_train,gts=gts_train)
        np.savez(test_save_fpath,latents_c=latents_c_test,latents_z=latents_z_test,gts=gts_test)

        latents_train = np.concatenate([latents_c_train,latents_z_train],axis=1)
        latents_test = np.concatenate([latents_c_test,latents_z_test],axis=1)
        #gts_c_train, gts_z_train = gts_train[:,4],np.delete(gts_train,4,axis=1)
        #gts_c_test, gts_z_test = gts_test[:,4],np.delete(gts_test,4,axis=1)

        #pa.predict_pvae(
                    #latents_c_train,gts_c_train,latents_c_test,gts_c_test,
                    #latents_z_train,gts_z_train,latents_z_test,gts_z_test
            #)
        #pa.save_and_print_results(accs_fpath)

    else:
        if args.vae == 'weakde':
            latents_train, gts_train, latents_test, gts_test = run_weak_de(ARGS)
        elif args.vae == 'dear':
            latents_train, gts_train, latents_test, gts_test = run_dear(ARGS)
        else:
            latents_train, gts_train, latents_test, gts_test = run_bvae_like_model(ARGS)

        np.savez(train_save_fpath,latents=latents_train,gts=gts_train)
        np.savez(test_save_fpath,latents=latents_test,gts=gts_test)

    pa.predict_unsupervised_vae(latents_train,gts_train,latents_test,gts_test)
    pa.save_and_print_results(accs_fpath)

    total_time = asMinutes(time() - start_time)
    with open(summary_fpath,'w') as f:
        f.write(f'\nTime: {total_time}\n')
        f.write('\nARGS:\n')
        for k,v in args.__dict__.items():
            f.write(f'{k}: {v}\n')
        if ARGS.vae == 'weakde':
            f.write(f'nz: {latents_train.shape[-1]}\n')

    print(f'Time: {total_time}')


if __name__ == '__main__':
    DATASETS = ['dsprites','3dshapes','mpi3d']

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--gpu',type=str,default='0')
    parser.add_argument('--max_predict_epochs',type=int,default=75)
    parser.add_argument('--name_suffix',type=str,default='')
    parser.add_argument('--no_progress_bar',action='store_true')
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--reload_from',type=str,default='none')
    parser.add_argument('--short_test','-tt',action='store_true')
    parser.add_argument('--test','-t',action='store_true')
    parser.add_argument('--vae',type=str,default='betaH')
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--no_pvae_supervision',action='store_true')
    parser.add_argument('--zs_combo',type=int,nargs='+',default=(2,4))
    parser.add_argument('--zs_combo_vals',type=int,nargs='+',default=(0,0))
    parser.add_argument('-d', '--dataset',type=str,choices=DATASETS,default='dsprites')
    parser.add_argument('-e','--epochs',type=int,default=100)
    parser.add_argument('-nt','--normal_testset_number',type=str,default='0')
    ARGS = parser.parse_args()

    if ARGS.short_test:
        ARGS.test_level = 2
    elif ARGS.short_test:
        ARGS.test_level = 1
    else:
        ARGS.test_level = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = ARGS.gpu
    if ARGS.short_test: ARGS.test=True
    if ARGS.test:
        ARGS.overwrite=True
        ARGS.epochs=2
        ARGS.max_predict_epochs=2
    else:
        torch.backends.cudnn.benchmark = True

    main(ARGS)
