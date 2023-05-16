import json
from os.path import join
from os import listdir
from utils import load_latents_and_gts
from dci import _compute_dci
from mig import _compute_mig
from sap_score import _compute_sap
from irs import scalable_disentanglement_score
from med import _compute_med
import numpy as np


TEST = False

for dset in ['dsprites','3dshapes','mpi3d']:
#for dset in ['3dshapes']:
    print(dset)
    for vae in ['betaH','btcvae','factor','pvae','pvae-semisup','weakde']:
    #for vae in ['weakde']:
        print(f'\t{vae}')
        for nt in range(5):
            zs_str = f'normal_testset{nt}'
            print(f'\t\t{nt}')
            try:
                mutr, ytr, muts, yts = load_latents_and_gts(dset,vae,zs_str,TEST)
            except FileNotFoundError as e:
                print("couldn't find latents files:",e)
                continue
            #mig_results = _compute_mig(muts.transpose(),yts.transpose())
            #print(f'MIG: {mig_results}')
            #sap_results = _compute_sap(mutr.transpose(),ytr.transpose(),muts.transpose(),yts.transpose(),continuous_factors=None)
            #print(f'SAP: {sap_results}')
            #irs_results = scalable_disentanglement_score(yts,muts)
            #print(f'IRS: {irs_results}')
            #dci_results = _compute_dci(mutr.transpose(),ytr.transpose(),muts.transpose(),yts.transpose())
            med_results = _compute_med(mutr.transpose(),ytr.transpose(),muts.transpose(),yts.transpose(),topk=2)
            #print(f'DCI: {dci_results}')
            print('med2:',med_results)
            exps_dir = 'experiments_tests' if TEST else 'experiments'
            #outfpath = join(exps_dir,f'{dset}_{vae}_{zs_str}','other_metrics.txt')
            outfpath = join(exps_dir,f'{dset}_{vae}_{zs_str}','med.txt')
            #results_dict = {'DCI':dci_results,'MIG':sap_results,'SAP':sap_results,'IRS':irs_results,'med':med_results}
            results_dict = {'med':med_results}
            with open(outfpath,'w') as f:
                json.dump(results_dict,f)

