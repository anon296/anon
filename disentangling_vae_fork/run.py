import argparse
import math
from dl_utils.tensor_funcs import numpyify
import numpy as np
import logging
import os

from torch import optim
import torch

from .disvae import init_specific_model, Trainer#, Evaluator
from .disvae.utils.modelIO import save_model, load_model#, load_metadata
from .disvae.models.losses import LOSSES, get_loss_f
from .disvae.models.vae import MODELS
from .disvae_utils.datasets import get_dataloaders, DATASETS
from .disvae_utils.helpers import (create_safe_directory, get_device, set_seed, get_n_param,
                           get_config_section, update_namespace_, FormatterNoDuplicate)


CONFIG_FILE = "disentangling_vae_fork/hyperparam.ini"
RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom', "debug", "best_celeba", "best_dsprites"]
EXPERIMENTS = ADDITIONAL_EXP + ["{}_{}".format(loss, data)
                                for loss in LOSSES
                                for data in DATASETS]


def silly_way_to_store_defaults():
    default_config = get_config_section([CONFIG_FILE], "Custom")

    description = "PyTorch implementation and evaluation of disentangled Variational AutoEncoders and metrics."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)
    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('--name', type=str,default='dummy_name',
                         help="Name of the model for storing and loading purposes.")
    general.add_argument('-L', '--log-level', help="Logging levels.",
                         default=default_config['log_level'], choices=LOG_LEVELS)
    general.add_argument('--no-progress-bar', action='store_true',
                         default=default_config['no_progress_bar'],
                         help='Disables progress bar.')
    general.add_argument('--no-cuda', action='store_true',
                         default=default_config['no_cuda'],
                         help='Disables CUDA training, even when have one.')
    general.add_argument('-s', '--seed', type=int, default=default_config['seed'],
                         help='Random seed. Can be `None` for stochastic behavior.')

    # Learning options
    training = parser.add_argument_group('Training specific options')
    training.add_argument('--checkpoint-every',
                          type=int, default=default_config['checkpoint_every'],
                          help='Save a checkpoint of the trained model every n epoch.')
    training.add_argument('-x', '--experiment',
                          default=default_config['experiment'], choices=EXPERIMENTS,
                          help='Predefined experiments to run. If not `custom` this will overwrite some other arguments.')
    training.add_argument('-e', '--epochs', type=int,
                          default=default_config['epochs'],
                          help='Maximum number of epochs to run for.')
    training.add_argument('-b', '--batch-size', type=int,
                          default=default_config['batch_size'],
                          help='Batch size for training.')
    training.add_argument('--lr', type=float, default=default_config['lr'],
                          help='Learning rate.')

    # Model Options
    model = parser.add_argument_group('Model specfic options')
    model.add_argument('-m', '--model-type',
                       default=default_config['model'], choices=MODELS,
                       help='Type of encoder and decoder to use.')
    model.add_argument('-z', '--latent-dim', type=int,
                       default=default_config['latent_dim'],
                       help='Dimension of the latent variable.')
    model.add_argument('-l', '--loss',
                       default=default_config['loss'], choices=LOSSES,
                       help="Type of VAE loss function to use.")
    model.add_argument('-a', '--reg-anneal', type=float,
                       default=default_config['reg_anneal'],
                       help="Number of annealing steps where gradually adding the regularisation. What is annealed is specific to each loss.")

    # Loss Specific Options
    betaH = parser.add_argument_group('BetaH specific parameters')
    betaH.add_argument('--betaH-B', type=float,
                       default=default_config['betaH_B'],
                       help="Weight of the KL (beta in the paper).")

    betaB = parser.add_argument_group('BetaB specific parameters')
    betaB.add_argument('--betaB-initC', type=float,
                       default=default_config['betaB_initC'],
                       help="Starting annealed capacity.")
    betaB.add_argument('--betaB-finC', type=float,
                       default=default_config['betaB_finC'],
                       help="Final annealed capacity.")
    betaB.add_argument('--betaB-G', type=float,
                       default=default_config['betaB_G'],
                       help="Weight of the KL divergence term (gamma in the paper).")

    factor = parser.add_argument_group('factor VAE specific parameters')
    factor.add_argument('--factor-G', type=float,
                        default=default_config['factor_G'],
                        help="Weight of the TC term (gamma in the paper).")
    factor.add_argument('--lr-disc', type=float,
                        default=default_config['lr_disc'],
                        help='Learning rate of the discriminator.')

    btcvae = parser.add_argument_group('beta-tcvae specific parameters')
    btcvae.add_argument('--btcvae-A', type=float,
                        default=default_config['btcvae_A'],
                        help="Weight of the MI term (alpha in the paper).")
    btcvae.add_argument('--btcvae-G', type=float,
                        default=default_config['btcvae_G'],
                        help="Weight of the dim-wise KL term (gamma in the paper).")
    btcvae.add_argument('--btcvae-B', type=float,
                        default=default_config['btcvae_B'],
                        help="Weight of the TC term (beta in the paper).")

    # Evaluation options
    evaluation = parser.add_argument_group('Evaluation specific options')
    evaluation.add_argument('--is-eval-only', action='store_true',
                            default=default_config['is_eval_only'],
                            help='Whether to only evaluate using precomputed model `name`.')
    evaluation.add_argument('--is-metrics', action='store_true',
                            default=default_config['is_metrics'],
                            help="Whether to compute the disentangled metrcics. Currently only possible with `dsprites` as it is the only dataset with known true factors of variations.")
    evaluation.add_argument('--no-test', action='store_true',
                            default=default_config['no_test'],
                            help="Whether not to compute the test losses.`")
    evaluation.add_argument('--eval-batchsize', type=int,
                            default=default_config['eval_batchsize'],
                            help='Batch size for evaluation.')

    args = parser.parse_args("")
    if args.experiment != 'custom':
        if args.experiment not in ADDITIONAL_EXP:
            # update all common sections first
            model, dataset = args.experiment.split("_")
            common_data = get_config_section([CONFIG_FILE], "Common_{}".format(dataset))
            update_namespace_(args, common_data)
            common_model = get_config_section([CONFIG_FILE], "Common_{}".format(model))
            update_namespace_(args, common_model)

        try:
            experiments_config = get_config_section([CONFIG_FILE], args.experiment)
            update_namespace_(args, experiments_config)
        except KeyError as e:
            if args.experiment in ADDITIONAL_EXP:
                raise e  # only reraise if didn't use common section

    return args


def run_bvae_like_model(real_cl_args):
    args = silly_way_to_store_defaults()
    for k,v in real_cl_args.__dict__.items():
        #exec(f"args.{k} = {v}")
        setattr(args,k,v)
    args.loss = real_cl_args.vae

    if args.dataset == 'dsprites':
        args.rec_dist = 'bernoulli'
        args.img_size = (1,64,64)
    else:
        args.rec_dist = 'gaussian'
        args.img_size = (3,64,64)
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())
    stream = logging.StreamHandler()
    stream.setLevel(args.log_level.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    set_seed(args.seed)
    device = get_device(is_gpu=not args.no_cuda)
    exp_dir = os.path.join(RES_DIR, args.name)
    logger.info("Root directory for saving and loading experiments: {}".format(exp_dir))

    #set_experiment_dir(args.name,args.overwrite)
    mu_list = []

    create_safe_directory(exp_dir, logger=logger)

    if args.loss == "factor":
        logger.info("FactorVae needs 2 batches per iteration. To replicate this behavior while being consistent, we double the batch size and the the number of epochs.")
        args.batch_size *= 2
        args.epochs *= 2

    # PREPARES DATA
    train_loader = get_dataloaders(args.dataset,
                                   img_size=args.img_size,
                                   batch_size=args.batch_size,
                                   zs_combo=args.zs_combo,
                                   zs_combo_vals=args.zs_combo_vals,
                                   logger=logger,
                                   is_test=args.test)
    logger.info("Train {} with {} samples".format(args.dataset, len(train_loader.dataset)))

    # PREPARES MODEL
    #args.img_size = get_img_size(args.dataset)  # stores for metadata
    #args.img_size = train_loader.dataset.img_size
    if args.reload_from == 'none':
        model = init_specific_model(args.model_type, args.img_size, args.latent_dim)
    else:
        model = load_model(args.reload_from,is_gpu=True)
    logger.info('Num parameters in model: {}'.format(get_n_param(model)))

    # TRAINS
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model = model.to(device)  # make sure trainer and viz on same device
    loss_f = get_loss_f(args.loss,
                        n_data=len(train_loader.dataset),
                        device=device,
                        **vars(args))
    trainer = Trainer(model, optimizer, loss_f,
                      is_test=args.test,
                      device=device,
                      logger=logger,
                      save_dir=exp_dir,
                      is_progress_bar=not args.no_progress_bar,
                      gif_visualizer=None)
    trainer(train_loader,
            epochs=args.epochs,
            checkpoint_every=args.checkpoint_every,)

    save_model(trainer.model, exp_dir, metadata=vars(args))

    test_loader = get_dataloaders(args.dataset,
                                  batch_size=args.eval_batchsize,
                                   img_size=args.img_size,
                                  shuffle=False,
                                  zs_combo='none',
                                  zs_combo_vals=args.zs_combo_vals,
                                  logger=logger,
                                  is_test=args.test)
    mu_list = []
    with torch.no_grad():
        for xb,yb in test_loader:
            mu,logvar = model.encoder(xb.to(device))
            assert mu.shape[1] == 10
            mu_list.append(numpyify(mu))
            if args.short_test: break

    mus = np.concatenate(mu_list,axis=0)
    gts = test_loader.dataset.class_labels

    test_set_mask = train_loader.dataset.test_set_mask
    if args.test: mus = np.repeat(mus,math.ceil(len(gts)/len(mus)),axis=0)[:len(gts)]
    train_latents = mus[~test_set_mask]
    test_latents = mus[test_set_mask]
    train_gts = gts[~test_set_mask]
    test_gts = gts[test_set_mask]

    return train_latents, train_gts, test_latents, test_gts

if __name__ == '__main__':
    pass
