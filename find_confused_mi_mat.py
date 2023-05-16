import numpy as np
from align_and_predict import PredictorAligner
from utils import make_zs_str, load_latents_and_gts
from os.path import join


TEST = False

#for dset in ['dsprites','3dshapes','mpi3d']:
for dset in ['dsprites']:
    for vae in ['betaH','btcvae','factor','pvae','pvae-semisup','weakde']:
        for nt in range(5):
            zs_str = make_zs_str([-1,-1],[0,0],TEST,nt)
            factor_aligner = PredictorAligner(vae_name=vae,latents_name=zs_str,dset_name=dset,max_epochs=1,test=TEST)
            exps_dir = 'experiments_tests' if TEST else 'experiments'
            latents_dir = join(exps_dir,f'{dset}_{vae}_{zs_str}')
            try:
                latents_train, gts_train, latents_test, gts_test = load_latents_and_gts(dset,vae,zs_str,TEST)
            except FileNotFoundError as e:
                print(e)
                continue
            factor_aligner.set_data(latents_train,gts_train,latents_test,gts_test)
            factor_aligner.set_cost_mat(latents_train,gts_train)
            matches = factor_aligner.cost_mat.argmin(axis=1)
            print(f'{dset}-{vae}-{zs_str}: {matches}')
            if len(set(matches)) < len(matches):
                outfpath = f'confused_mi_mats/{dset}_{vae}_normal_testset{nt}_confused_mi_mat.npy'
                print('found one, saving to',outfpath)
                np.save(outfpath,factor_aligner.cost_mat)
            else:
                print('this one is alright')
