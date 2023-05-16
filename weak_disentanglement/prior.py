import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform as Unif
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.one_hot_categorical import OneHotCategorical
import numpy as np
from .model import MLP



class AdaptiveRelationalPrior(nn.Module):
    def __init__(self, cfg, z_dim, n_classes):
        super(AdaptiveRelationalPrior, self).__init__()
        self.name = "adaptive_graph"
        self.dimensions = z_dim
        self.n_classes = n_classes
        self.categorical = OneHotCategorical(probs=torch.tensor([1/n_classes for _ in range(n_classes)]))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.label_fn = label_fn
        self.cfg = cfg
        #gaussians, locs, scale_trils = self._init_mog_flower_prior(cfg)
        #gaussians, locs = self._init_mog_line_prior(cfg)
        gaussians = self._init_mog_random_prior(cfg)
        self.gaussians = gaussians

        self.relation_layer = MLP(
            n_layers=cfg.model.n_relational_layers,
            dimensions=(cfg.model.z_dim*2, 1024, cfg.model.z_dim),
            activation=torch.tanh,
            out_activation=lambda x: x
        )

    # flower non isotropic mog
    def _init_mog_flower_prior(self, cfg):
        gaussians = []
        locs = nn.ParameterList()
        scale_trils = nn.ParameterList()

        radial_noise = 0.1
        tangent_noise = 0.01

        n = cfg.data.n_classes
        d = cfg.model.z_dim

        for l in range(self.n_classes):

            # flower prior
            z_mean = torch.tensor([1 * np.cos((l*2*np.pi) / n), 1 * np.sin((l*2*np.pi) / n)] + [ 0 for _ in range(d-2) ], dtype=torch.float)
            v1 = [np.cos((l*2*np.pi) / n), np.sin((l*2*np.pi) / n)]
            v2 = [-np.sin((l*2*np.pi) / n), np.cos((l*2*np.pi) / n)]
            a1 = radial_noise   # radial axis (center-to-outer)
            a2 = tangent_noise # tangent axis (along the circle)
            M = np.eye(d)
            S = np.eye(d)
            np.fill_diagonal(S, [a1] + [ a2 for _ in range(d-1) ])
            M[0:2, 0:2] = np.vstack((v1,v2)).T
            z_cov = torch.tensor(np.dot(np.dot(M, S), np.linalg.inv(M)), dtype=torch.float)

            gaussian = MultivariateNormal(loc=z_mean, covariance_matrix=z_cov)

            gaussian.loc = nn.Parameter(gaussian.loc.to(self.device), requires_grad=False)
            gaussian._unbroadcasted_scale_tril = nn.Parameter(gaussian._unbroadcasted_scale_tril.to(self.device), requires_grad=False)

            gaussians.append(gaussian)
            locs.append(gaussian.loc)
            scale_trils.append(gaussian._unbroadcasted_scale_tril)


        return gaussians, locs, scale_trils

    # adaptive mog prior
    def _init_mog_adaptive_prior(self, cfg, tr_set, model, dset):
        xs = []
        ys = []
        nc = 1 if dset=='dsprites' else 3
        for i, (x, y) in enumerate(tr_set):
            if i >= cfg.prior.supervision_amount:
                break

            if dset == 'dsprites':
                y = (y[:,:2]*torch.tensor([27/32,9/32])).round().sum(axis=1)+y[:,-1]
            elif dset == '3dshapes':
                y = (y[:,:2]*torch.tensor([36/10,12/10])).round().sum(axis=1)+y[:,-2]
            elif dset == 'mpi3d':
                y = (y[:,:2]*torch.tensor([45/40,15/40])).round().sum(axis=1)+y[:,-3]
            else:
                print(888)
            xs.append(x.reshape((x.shape[0], nc, 64, 64)).cpu())
            ys.append(y.reshape(y.shape[0], 1).cpu())

        x = torch.cat(xs, dim=0)
        y = torch.cat(ys, dim=0)

        gaussians = []
        n_classes = 27 if dset=='dsprites' else 36 if dset=='3dshapes' else 45
        for i in range(n_classes):
            if i not in y:
                z_mean = np.zeros(cfg.model.z_dim)
                z_cov = np.eye(cfg.model.z_dim)
            else:
                x_i = x[y[:, 0] == i, :]

                with torch.no_grad():
                    x_i = torch.Tensor(x_i).to(self.device)
                    z_i = model.encoder(x_i).cpu().numpy()

                z_mean = np.mean(z_i, axis=0)
                z_cov  = np.cov(z_i, rowvar=False) * cfg.prior.gm_cov


            gaussians.append(
                MultivariateNormal(
                    torch.Tensor(z_mean).to(self.device),
                    torch.Tensor(z_cov+1e-4*np.eye(cfg.model.z_dim)).to(self.device)
                )
            )

        return gaussians

    # flower isotropic gauss
    def _init_mog_flower_isotropic_prior(self, cfg):
        gaussians = []
        locs = nn.ParameterList()

        n = cfg.data.n_classes
        d = cfg.model.z_dim

        for l in range(self.n_classes):
            z_mean = torch.tensor([1 * np.cos((l*2*np.pi) / n), 1 * np.sin((l*2*np.pi) / n)] + [ 0 for _ in range(d-2) ], dtype=torch.float)
            z_cov = torch.eye(d).to(self.device) * cfg.prior.gm_cov
            gaussian = MultivariateNormal(loc=z_mean, covariance_matrix=z_cov)
            gaussian.loc = nn.Parameter(gaussian.loc.to(self.device), requires_grad=True)
            gaussians.append(gaussian)
            locs.append(gaussian.loc)

        return gaussians #, locs

    # flower isotropic prior
    def _init_mog_line_prior(self, cfg):
        gaussians = []
        locs = nn.ParameterList()

        n = cfg.data.n_classes
        d = cfg.model.z_dim

        for l in range(self.n_classes):
            z_mean = torch.tensor([l, 0] + [ 0 for _ in range(d-2) ], dtype=torch.float)
            z_cov = torch.eye(d).to(self.device) * cfg.prior.gm_cov
            gaussian = MultivariateNormal(loc=z_mean, covariance_matrix=z_cov)
            gaussian.loc = nn.Parameter(gaussian.loc.to(self.device), requires_grad=True)
            gaussians.append(gaussian)
            locs.append(gaussian.loc)

        return gaussians #, locs

    # flower isotropic prior
    def _init_mog_random_prior(self, cfg):
        gaussians = []
        locs = nn.ParameterList()

        n = cfg.data.n_classes
        d = cfg.model.z_dim

        for l in range(self.n_classes):
            z_mean = (torch.rand((d,)) * 2.) - 1.
            z_cov = torch.eye(d).to(self.device) * cfg.prior.gm_cov
            gaussian = MultivariateNormal(loc=z_mean, covariance_matrix=z_cov)
            gaussian.loc = nn.Parameter(gaussian.loc.to(self.device), requires_grad=True)
            gaussians.append(gaussian)
            locs.append(gaussian.loc)

        return gaussians #, locs

        def forward(self, batch):
            rel1_inputs = torch.cat([batch.rel1_sample, torch.tensor([1, 0, 0, 0, 0],device=self.device)], dim=-1)
            rel2_inputs = torch.cat([batch.rel2_sample, torch.tensor([0, 1, 0, 0, 0],device=self.device)], dim=-1)
            rel3_inputs = torch.cat([batch.rel3_sample, torch.tensor([0, 0, 1, 0, 0],device=self.device)], dim=-1)
            rel4_input = torch.cat([batch.rel4_sample, torch.tensor([0, 0, 0, 1, 0],device=self.device)], dim=-1)
            rel5_inputs = torch.cat([batch.rel5_sample, torch.tensor([0, 0, 0, 0, 1],device=self.device)], dim=-1)


            z_rel1_pred = self.relation_layer(batch.rel1_inputs)
            z_rel2_pred = self.relation_layer(batch.rel2_inputs)
            z_rel3_pred = self.relation_layer(batch.rel3_inputs)
            z_rel4_pred = self.relation_layer(batch.rel4_input)
            z_rel5_pred = self.relation_layer(batch.rel5_inputs)

            return z_rel1_pred, z_rel2_pred, z_rel3_pred, z_rel4_pred, z_rel5_pred

    def sample(self, y):
        z_sample = torch.empty((y.shape[0], self.dimensions))

        y_sample = y

        for i, y_sample_i in enumerate(y_sample):
            z_sample[i, :] = self.gaussians[y_sample_i.argmax()].sample()

        z_sample = z_sample.to(self.device)
        #y_sample = y_sample.to(self.device)

        return z_sample

    def classify(self, z, threshold=None):
        log_threshold = torch.log(torch.Tensor([threshold]).to(self.device))

        log_probs = torch.stack([ self.gaussians[i].log_prob(z) for i in range(self.cfg.data.n_entities)], dim=-1).to(self.device)

        max_val, max_i = log_probs.max(dim=-1)
        max_i[max_val < log_threshold ] = -1

        return max_i

    def init_adaptive_prior(self, tr_set, model, dset):
        gaussians = self._init_mog_adaptive_prior(self.cfg, tr_set, model, dset)
        self.gaussians = gaussians

class Uniform(nn.Module): # isotropic uniform dist
    def __init__(self, cfg, z_dim, low, high):
        super(Uniform, self).__init__()
        self.cfg = cfg
        self.name = "uniform"
        # not actually needeed, just so that the prior optim doesn't get an empty list of parameters
        self.low = nn.Parameter(torch.tensor(low))
        self.high = nn.Parameter(torch.tensor(high))

        self.p = Unif(torch.tensor([low]*z_dim), torch.tensor([high]*z_dim))

        self.categorical = OneHotCategorical(probs=torch.tensor([1/self.cfg.data.n_classes for _ in range(self.cfg.data.n_classes)]))

    def forward(self):
        #y_sample = self.categorical.sample((self.cfg.data.batch_size,)).to(self.device)
        sample = self.p.sample((self.cfg.data.batch_size,)).to(self.device)
        return sample

    def sample(self, y):
        #y_sample = self.categorical.sample((self.cfg.data.batch_size,)).to(self.device)
        sample = self.p.sample((y.shape[0],)).to(self.device)
        return  sample
