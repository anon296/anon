import numpy as np
from time import time
import math
import torch
from torch import nn, optim
from torch.nn import functional as F

from .bgm import BGM, BigJointDiscriminator
from dl_utils.tensor_funcs import numpyify
#from sagan import *
from .config import get_config, extend_args
#from causal_model import *
#import utils
from get_dsets import get_maybe_zs_dset


latent_dim = 20

def run_dear(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extend_args(args)
    #if 'pendulum' in args.dataset: still not sure what this is but
    #    label_idx = range(4)  guessing it's the number of label types
    #elif args.labels == 'smile':
    #    label_idx = [31, 20, 19, 21, 23, 13]
    #elif args.labels == 'age':
    #    label_idx = [39, 20, 28, 18, 13, 3]
    #else:
    #    raise NotImplementedError("Not supported structure.")

    num_label = 5 if args.dataset=='dsprites' else 6 if args.dataset=='3dshapes' else 7
    nc = 1 if args.dataset=='dsprites' else 3
    label_idx = range(num_label)
    tr_set, ts_set = get_maybe_zs_dset(args.dataset, args.test_level, args.zs_combo, args.zs_combo_vals)
    tr_loader = torch.utils.data.DataLoader(tr_set,batch_size=args.batch_size,
        shuffle=True, pin_memory=True, drop_last=True)

    print('Build models...')
    model = BGM(latent_dim, args.g_conv_dim, nc,
                args.enc_dist, args.enc_arch, args.enc_fc_size, args.enc_noise_dim, args.dec_dist,
                args.prior, num_label)
    discriminator = BigJointDiscriminator(latent_dim, args.d_conv_dim, args.image_size,
                                          args.dis_fc_size)
    if args.dataset == 'dsprites':
        #model.encoder.encoder.conv1 = nn.Conv2d(1,64,7,2,3,device=device)
        #model.decoder.decoder.toRGB = nn.Conv2d(64,1,3,1,1,device=device)
        discriminator.discriminator.fromRGB = nn.Conv2d(1,32,1,1)

    A_optimizer = None
    prior_optimizer = None
    if 'scm' in args.prior:
        enc_param = model.encoder.parameters()
        dec_param = list(model.decoder.parameters())
        prior_param = list(model.prior.parameters())
        A_optimizer = optim.Adam(prior_param[0:1], lr=1e-3)
        prior_optimizer = optim.Adam(prior_param[1:], lr=5e-5, betas=(0, 0.999))
    else:
        enc_param = model.encoder.parameters()
        dec_param = model.decoder.parameters()
    encoder_optimizer = optim.Adam(enc_param, lr=5e-5, betas=(0, 0.999))
    decoder_optimizer = optim.Adam(dec_param, lr=5e-5, betas=(0, 0.999))
    D_optimizer = optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0, 0.999))

    # Load model from checkpoint
    #if args.resume:
    #    ckpt_dir = args.ckpt_dir if args.ckpt_dir != '' else save_dir + args.model_type + str(
    #        args.start_epoch - 1) + '.sav'
    #    checkpoint = torch.load(ckpt_dir)
    #    model.load_state_dict(checkpoint['model'])
    #    discriminator.load_state_dict(checkpoint['discriminator'])
    #    del checkpoint

    model = nn.DataParallel(model.to(device))
    discriminator = nn.DataParallel(discriminator.to(device))

    patience = 0
    best_epoch_loss = np.inf
    print('Start training...')
    for epoch in range(args.epochs):
        #if epoch == 1:
        #    encoder_optimizer.param_groups[0]['lr'] = 1e-4
        #    decoder_optimizer.param_groups[0]['lr'] = 1e-4
        #    D_optimizer.param_groups[0]['lr'] = 1e-4
        epoch_loss = train(args, model, discriminator, encoder_optimizer, decoder_optimizer, D_optimizer, tr_loader, label_idx,prior_optimizer, A_optimizer)
        print(f"- epoch, {epoch+1}, /, {args.epochs} loss: {epoch_loss:.4f}")
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            patience = 0
        else:
            patience+=1
        if patience == 2:
            encoder_optimizer.param_groups[0]['lr'] *= 0.9
            decoder_optimizer.param_groups[0]['lr'] *= 0.9
            D_optimizer.param_groups[0]['lr'] *= 0.9
            print('patienced out, now', encoder_optimizer.param_groups[0]['lr'])
            patience = 0
            if encoder_optimizer.param_groups[0]['lr'] < 5e-5:
                break


    new_tr_loader = torch.utils.data.DataLoader(tr_set,batch_size=args.batch_size,
        shuffle=True, pin_memory=True, drop_last=False)

    new_ts_loader = torch.utils.data.DataLoader(ts_set,batch_size=args.batch_size,
        shuffle=True, pin_memory=True, drop_last=False)

    latents_train = generate_latents(model.module.encoder,new_tr_loader,args.test)
    latents_test = generate_latents(model.module.encoder,new_ts_loader,args.test)
    gts_train = tr_set.targets
    gts_test = ts_set.targets
    return latents_train, gts_train, latents_test, gts_test


def train(args, model, discriminator, encoder_optimizer, decoder_optimizer, D_optimizer,
              train_loader, label_idx,
              prior_optimizer, A_optimizer):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    discriminator.train()

    epoch_loss = 0
    batch_start_time = time()
    running_disc_weight = 1
    running_dec_weight = 1
    running_enc_weight = 1
    for batch_idx, (x, label) in enumerate(train_loader):
        x = x.to(device)
        # supervision flag
        sup_flag = label[:, 0] != -1
        if sup_flag.sum() > 0:
            label = label[sup_flag, :][:, label_idx].float()
        if 'pendulum' in args.dataset:
            # Normalize labels to 0,1
            scale = get_scale()
            label = (label - scale[0]) / (scale[1] - scale[0])
        num_labels = len(label_idx)
        label = label.to(device)

        # ================== TRAIN DISCRIMINATOR ================== #
        discriminator.zero_grad()

        # Sample z from prior p_z
        z = torch.randn(x.size(0), latent_dim, device=x.device)

        # Get inferred latent z = E(x) and generated image x = G(z)
        z_fake, x_fake, z_fake_mean = model(x, z)

        # Compute D loss, high score means real which means from encoder
        if (running_disc_weight/running_dec_weight > 0.2) or \
            (running_disc_weight/running_enc_weight > 0.2):
            encoder_score = discriminator(x, z_fake.detach())
            decoder_score = discriminator(x_fake.detach(), z.detach())
            nll_encoder = F.softplus(-encoder_score).mean()
            nll_decoder = F.softplus(decoder_score).mean()

            loss_d_ = nll_encoder + nll_decoder
            #loss_d_ = (1+running_disc_weight) * (nll_encoder/(1+running_enc_weight) + nll_decoder/(1+running_enc_weight))
            loss_d = running_disc_weight*torch.clamp(loss_d_,max=1)
            running_disc_weight = (3*running_disc_weight+prob_from_nll(loss_d_).item())/4
            loss_d.backward()
            D_optimizer.step()
        else:
            running_disc_weight = (3*running_disc_weight+0.2)/4
            loss_d_ = -1

        #z = torch.randn(x.size(0), latent_dim, device=x.device)
        #z_fake, x_fake, z_fake_mean = model(x, z) # I think we can get away with removing this

        # ================== TRAIN ENCODER ================== #
        if batch_idx%10 == 0: # train with labels 10% of the time
            label_z = z_fake_mean[sup_flag, :num_labels]
            sup_loss = F.mse_loss(label_z, label)
        if running_enc_weight/running_disc_weight > 0.4:
            model.zero_grad()
            # WITH THE GENERATIVE LOSS
            encoder_score = discriminator(x, z_fake)
            #loss_encoder = encoder_score.mean()
            gan_loss_encoder = F.softplus(encoder_score)
            yy = gan_loss_encoder.max()
            gan_loss_encoder = torch.clamp(gan_loss_encoder,max=1).nanmean()*(~gan_loss_encoder.isnan()).float().mean()
            if gan_loss_encoder.isnan().any():
                breakpoint()

            # WITH THE SUPERVISED LOSS
            loss_encoder = sup_loss + running_enc_weight*gan_loss_encoder
            running_enc_weight = (3*running_enc_weight+prob_from_nll(gan_loss_encoder).item())/4

            if 'scm' in args.prior:
                prior_optimizer.step()
            epoch_loss += loss_encoder.item()
            if round(gan_loss_encoder.item(),4) == 0.6321:
                breakpoint()
        else:
            gan_loss_encoder = -1
            running_enc_weight = (3*running_enc_weight+prob_from_nll(nll_encoder).item())/4
            if batch_idx%10==0:
                loss_encoder = sup_loss
                loss_encoder.backward()
                encoder_optimizer.step()


        # ================== TRAIN GENERATOR ================== #
        if running_dec_weight/running_disc_weight > 0.2:
            model.zero_grad()

            decoder_score = discriminator(x_fake, z)
            # with scaling clipping for stabilization
            #r_decoder = torch.exp(decoder_score.detach())
            #s_decoder = r_decoder.clamp(0.5, 2)
            #loss_decoder = -(s_decoder * decoder_score).mean()
            loss_decoder_ = F.softplus(-decoder_score).mean()
            loss_decoder = running_dec_weight*torch.clamp(loss_decoder_,max=1)
            loss_decoder.backward()
            decoder_optimizer.step()
            if 'scm' in args.prior:
                model.module.prior.set_zero_grad()
                A_optimizer.step()
                prior_optimizer.step()
            running_dec_weight = (3*running_dec_weight+prob_from_nll(loss_decoder_).item())/4
        else:
            running_enc_weight = (3*running_enc_weight+prob_from_nll(nll_decoder).item())/4
            loss_decoder_ = -1


        if batch_idx%10 == 0:
            s =f'Batch: {batch_idx}\tsup: {sup_loss:.4f}\t enc: {gan_loss_encoder:.4f} {running_enc_weight:.4f}\tdec: {loss_decoder_:.4f} {running_dec_weight:.4f}\tdiscr:{loss_d_:.4f} {running_disc_weight:.4f}'
            if yy.max() > 10:
                s += f' max encoder loss: {yy}'
            #batch_loss=epoch_loss/(batch_idx+1)
            print(s)
    return epoch_loss

def prob_from_nll(x):
    return 1 - (-x).exp()

def generate_latents(encoder,dloader,test):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mu_list = []
    with torch.no_grad():
        for xb,yb in dloader:
            mu,logvar = encoder(xb.to(device))
            #assert mu.shape[1] == cfg.model.z_dim
            mu_list.append(numpyify(mu))
            if test: break
    mus = np.concatenate(mu_list,axis=0)
    N = len(dloader.dataset)
    if test: mus = np.repeat(mus,math.ceil(N/len(mus)),axis=0)[:N]
    return mus

def get_scale():
    '''return max and min of training data'''
    scale = torch.Tensor([[0.0000, 48.0000, 2.0000, 2.0178], [40.5000, 88.5000, 14.8639, 14.4211]])
    return scale

def get_stats():
    '''return mean and std of training data'''
    mm = torch.Tensor([20.2500, 68.2500, 6.9928, 8.7982])
    ss = torch.Tensor([11.8357, 11.8357, 2.8422, 2.1776])
    return mm, ss
