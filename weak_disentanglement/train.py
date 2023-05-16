import math
import numpy as np
import torch
from dl_utils.tensor_funcs import numpyify
from torch import nn

from .config import make_config
from .model import MLP, Conv64Encoder, Conv64Decoder, SemiSupervisedAbstractionAutoencoder
from .loss import GeneratorLossCompute, DiscriminatorLossCompute, ReconstructionLossCompute
from .prior import Uniform, AdaptiveRelationalPrior
from get_dsets import get_maybe_zs_dset



def set_trainable(trainables):
    for model, trainable in trainables.items():
        if trainable:
            for p in model.parameters():
                p.requires_grad = True
        else:
            for p in model.parameters():
                p.requires_grad = False

def train_model(args, cfg, tr_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("- Init model...")
    nc = 1 if args.dataset == 'dsprites' else 3
    model = SemiSupervisedAbstractionAutoencoder(
        encoder=Conv64Encoder(nc=nc, nf=cfg.model.nf_dim, nz=cfg.model.z_dim),
        decoder=Conv64Decoder(nc=nc, nf=cfg.model.nf_dim, nz=cfg.model.z_dim),
        discriminator=MLP(
            3,
            (cfg.model.z_dim + cfg.data.n_classes, 1024, 1),
            activation=torch.tanh, out_activation=lambda x: x
        ),
        prior=Uniform(cfg, z_dim=cfg.model.z_dim, low=-1., high=1.)
    )
    model.to(device)

    reconstruction_criterion = nn.BCEWithLogitsLoss().to(device)
    gan_gen_criterion = nn.BCEWithLogitsLoss().to(device)
    gan_disc_criterion = nn.BCEWithLogitsLoss().to(device)
    #prior_criterion = RelationalLoss(model.prior).to(device)

    dec_optim = torch.optim.Adam(model.decoder.parameters(), lr=cfg.train.lr.dec)
    enc_optim = torch.optim.Adam(model.encoder.parameters(), lr=cfg.train.lr.enc)
    disc_optim = torch.optim.Adam(model.discriminator.parameters(), lr=cfg.train.lr.disc)
    #prior_optim = torch.optim.Adam(model.prior.parameters(), lr=cfg.train.lr.prior)

    losses = {
        "reconstruction": ReconstructionLossCompute(cfg, reconstruction_criterion, enc_optim, dec_optim, train=True),
        "gan_gen": GeneratorLossCompute(cfg, gan_gen_criterion, enc_optim, train=cfg.loss.adversarial_train),
        "gan_disc": DiscriminatorLossCompute(cfg, gan_disc_criterion, disc_optim, train=cfg.loss.adversarial_train),
        #"prior": RelationalLossCompute(cfg, prior_criterion, prior_optim, train=False)
    }

    model.prior = AdaptiveRelationalPrior(
        cfg,
        cfg.model.z_dim,
        cfg.data.n_classes
    ).to(device)
    model.prior.init_adaptive_prior(tr_loader, model, args.dataset)
    best_l = np.inf
    best_epoch_loss = np.inf
    patience = 0
    epoch_patience = 0
    for epoch in range(args.epochs):
        if epoch == 1:
            enc_optim.param_groups[0]['lr'] = 1e-4
            dec_optim.param_groups[0]['lr'] = 1e-4
            disc_optim.param_groups[0]['lr'] = 1e-4

        y = torch.zeros((args.batch_size, cfg.data.n_classes), dtype=torch.long).to(device)
        epoch_loss = 0
        for it, (x, _) in enumerate(tr_loader):
            x = x.view(-1, nc, 64, 64).to(device)

            set_trainable({model.prior: False,model.encoder: True,
                model.decoder: True,model.discriminator: False})
            x_rec, _ = model(x=x, phase="ae")
            l=losses["reconstruction"](x_rec, x)
            epoch_loss += l
            if l < best_l:
                best_l = l
                patience = 0
            else:
                patience += 1

            if patience == 10000:
                enc_optim.param_groups[0]['lr'] *= 0.9
                dec_optim.param_groups[0]['lr'] *= 0.9
                disc_optim.param_groups[0]['lr'] *= 0.9
                print('patienced out, now', enc_optim.param_groups[0]['lr'])
                patience = 0

        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            epoch_patience = 0
        else:
            epoch_patience += 1
            print(f'\nEpoch patience: {epoch_patience}\n')
        if epoch_patience == 7:
            break

            #entities_ids = list(range(27)) if args.dataset=='dsprites' else list(range(120))
            #if cfg.train.phase != "warmup":
                #set_trainable({model.prior: True,model.encoder: False,
                #    model.decoder: False, model.discriminator: False})
                #if model.training: # training batch
                #    batch = new_batch_of_relations(
                #        cfg, args.dataset, cfg.data.prior_batch_size, model.prior,
                #        entities_ids
                #    ).to(device)
                #else: # validation batch
                #    batch = new_batch_of_relations(
                #        cfg, args.dataset, cfg.data.prior_batch_size, model.prior,
                #        entities_ids
                #    ).to(device)

                #z_true_rel1 = batch.rel1_target_sample
                #z_true_rel2 = batch.rel2_target_sample
                #z_true_rel3 = batch.rel3_target_sample
                #z_true_rel4 = batch.rel4_target_sample
                #z_true_rel5 = batch.rel5_target_sample
                #z_rel1, z_rel2, z_rel3, z_rel4, z_rel5 = model(relational_batch=batch, phase="prior")
                #losses["prior"](
                #    z_true_rel1, z_true_rel2, z_true_rel3, z_true_rel4, z_true_rel5,
                #    z_rel1, z_rel2, z_rel3, z_rel4, z_rel5
                #)

            set_trainable({model.prior: False,model.encoder: True,
                model.decoder: False,model.discriminator: False})
            z_gen_score_g = model(x=x, y=y, phase="gan_gen")
            losses["gan_gen"](z_gen_score_g)

            set_trainable({model.prior: False,model.encoder: False,
                model.decoder: False,model.discriminator: True})
            z_gen_score, z_prior_score, z_gen, z_prior = model(x=x, y=y, phase="gan_disc")
            losses["gan_disc"](z_gen_score, z_prior_score, z_gen, z_prior)
            if args.test: break
        print("- epoch:", epoch + 1, "/", args.epochs,f'loss: {epoch_loss:.3f}')

    return model

def generate_latents(encoder,dloader,test):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mu_list = []
    with torch.no_grad():
        for xb,yb in dloader:
            mu = encoder(xb.to(device))
            #assert mu.shape[1] == cfg.model.z_dim
            mu_list.append(numpyify(mu))
            if test: break
    mus = np.concatenate(mu_list,axis=0)
    N = len(dloader.dataset)
    if test: mus = np.repeat(mus,math.ceil(N/len(mus)),axis=0)[:N]
    return mus

def run_weak_de(args):
    cfg = make_config()
    test_level = 2 if args.test else 0
    tr_set, ts_set = get_maybe_zs_dset(args.dataset,test_level,args.zs_combo,args.zs_combo_vals)

    tr_loader = torch.utils.data.DataLoader(tr_set,batch_size=args.batch_size,
        shuffle=True, pin_memory=True, drop_last=True)

    ts_loader = torch.utils.data.DataLoader(ts_set,batch_size=args.batch_size,
        shuffle=False, pin_memory=True, drop_last=False)

    model = train_model(args, cfg, tr_loader)
    new_tr_loader = torch.utils.data.DataLoader(tr_set,batch_size=args.batch_size,
        shuffle=False, pin_memory=True, drop_last=False)
    train_latents = generate_latents(model.encoder,new_tr_loader,args.short_test)
    test_latents = generate_latents(model.encoder,ts_loader,args.short_test)
    train_gts = tr_set.targets
    test_gts = ts_set.targets
    print('finished training and generating latents')
    return train_latents,train_gts,test_latents,test_gts
