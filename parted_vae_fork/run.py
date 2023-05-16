import itertools
import torch
from torch import optim
from dl_utils.tensor_funcs import numpyify
from dl_utils.misc import set_experiment_dir
import numpy as np
from .partedvae.models import VAE
from .partedvae.training import Trainer
from .utils.dataloaders import get_maybe_zs_dataloader, get_celeba_dataloader
#from .utils.load_pvae_model import load_pvae_model
#from viz.visualize import Visualizer
from .utils.metrics import dis_by_fact_metric
import argparse
import os
from os.path import join


def run_pvae(args):
    if args.dataset == 'dsprites':
        z = 5
        c = 3
    elif args.dataset == '3dshapes':
        z = 6
        c = 4
    elif args.dataset == 'mpi3d':
        z = 7
        c = 6
    #disc_priors = [[0.33, 0.33, 0.34]]
    disc_priors = [(np.arange(c)/c).tolist()]
    img_size = (1 if args.dataset=='dsprites' else 3, 64, 64)
    latent_spec = {
        'z': z,
        'c': [c],
        'single_u': 1,
    }
    z_capacity = [0., 30., 300000, 50.]
    u_capacity = [0., 5., 300000, 50.]
    g_c, g_h = 100., 10.
    g_bc = 10.
    bc_threshold = 0.1
    test_level = 2 if args.test else 0

    recon_type = 'bce'

    trainloader, testloader, warmuploader = get_maybe_zs_dataloader(test_level, args.dataset, args.zs_combo, args.zs_combo_vals, batch_size=args.batch_size, fraction=1)

    if args.vae =='pvae-unsup':
        warmuploader = None
    else:
        assert args.vae == 'pvae-semisup'

    model = VAE(img_size=img_size, latent_spec=latent_spec, c_priors=disc_priors)

    optimizer_warm_up = optim.Adam(itertools.chain(*[
        model.img_to_features.parameters(),
        model.features_to_hidden.parameters(),
        model.h_to_c_logit_fc.parameters()
    ]), lr=5e-4)
    optimizer_model = optim.Adam(model.parameters(), lr=5e-4)
    optimizers = [optimizer_warm_up, optimizer_model]

    trainer = Trainer(model, optimizers, args.test, dataset=args.dataset, recon_type=recon_type,
                      z_capacity=z_capacity, u_capacity=u_capacity, c_gamma=g_c, entropy_gamma=g_h,
                      bc_gamma=g_bc, bc_threshold=bc_threshold)
    trainer.train(trainloader, warmuploader=warmuploader, epochs=args.epochs, run_after_epoch=None,
                      run_after_epoch_args=[])

    def generate_latents_dset(dloader,fragment_name):
        c_latents_list = []
        z_latents_list = []
        gts_list = []
        for xb,yb in dloader:
            latent_dict = model.encode(xb.cuda())
            new_c_latent = latent_dict['log_c']
            new_z_latent = latent_dict['z'][0]
            c_latents_list.append(numpyify(new_c_latent))
            z_latents_list.append(numpyify(new_z_latent))
            gts_list.append(numpyify(yb))
        c_latents_array = np.concatenate(c_latents_list)
        z_latents_array = np.concatenate(z_latents_list)
        gts_array = np.concatenate(gts_list)
        print(len(c_latents_array))
        return c_latents_array, z_latents_array, gts_array

    c_latents_train, z_latents_train, gts_train = generate_latents_dset(trainloader,'train')
    c_latents_test, z_latents_test, gts_test = generate_latents_dset(testloader,'test')
    return c_latents_train, z_latents_train, gts_train, c_latents_test, z_latents_test, gts_test

def save(trainer, z_capacity, u_capacities, latent_spec, epochs, lr_warm_up, lr_model, dset, recon_type):
    torch.save(trainer.model.state_dict(), 'model.pt')
    with open('specs.json', 'w') as f:
        f.write('''{
        "z_capacity": %s,
        "u_capacity": %s,
        "latent_spec": %s,
        "epochs": %d,
        "lr_warm_up": %f,
        "lr_model": %f,
        "dataset": "%s",
        "recon_type": "%s"
        }''' % (str(z_capacity), str(u_capacities), str(latent_spec).replace("'", '"'), epochs,
                lr_warm_up, lr_model, dset, recon_type))
