import torch.nn as nn
import torch
import torch.nn.functional as F



class SemiSupervisedAbstractionAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, discriminator, prior):
        super(SemiSupervisedAbstractionAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.prior = prior

    def forward_ae(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z

    def forward_prior(self, relational_batch):
        z_left, z_right, z_up, z_down, z_shape = self.prior(relational_batch)
        return z_left, z_right, z_up, z_down, z_shape

    def forward_gan_disc(self, x, y):
        z_prior = self.prior.sample(y)
        z_gen = self.encoder(x)

        latents_prior = torch.cat([y, z_prior], dim=-1)
        latents_gen   = torch.cat([y, z_gen], dim=-1)

        prior_score = self.discriminator(latents_prior)
        gen_score = self.discriminator(latents_gen)

        return gen_score, prior_score, z_gen, z_prior

    def forward_gan_gen(self, x, y):
        z_gen = self.encoder(x)

        latents_gen = torch.cat([y, z_gen], dim=-1)
        gen_score = self.discriminator(latents_gen)
        return gen_score

    def forward(self, x=None, y=None, relational_batch=None, phase=None):
        if phase == "ae":
            return self.forward_ae(x)
        elif phase == "prior":
            return self.forward_prior(relational_batch)
        elif phase == "gan_gen":
            return self.forward_gan_gen(x, y)
        elif phase == "gan_disc":
            return self.forward_gan_disc(x, y)
        else:
            raise NotImplementedError

class Conv64Decoder(nn.Module):
    # nz = z_dim, ngf = size of feature maps, nc = number of channels
    def __init__(self, nz, nf, nc):
        super(Conv64Decoder, self).__init__()
        ngf = nf
        self.ngf =ngf
        self.nz = nz

        #self.tconv1 = nn.ConvTranspose2d( nz, ngf * 4, 4, 1, 0, bias=False)
        self.tconv2 = nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False)
        self.tconv3 = nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False)
        self.tconv4 = nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False)
        self.tconv5 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)

        #self.batchnorm1 = nn.BatchNorm2d(ngf)
        self.batchnorm2 = nn.BatchNorm2d(ngf)
        self.batchnorm3 = nn.BatchNorm2d(ngf)
        self.batchnorm4 = nn.BatchNorm2d(ngf)

        self.linear1 = nn.Linear(nz, 1024)
        self.linear2 = nn.Linear(1024, self.ngf * 4 * 4) # changed from 6 to 4

    def forward(self, x):
        #print("FWD decoder")
        #print("X:", x.shape)
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.1)
        #print("X:", x.shape)

        x = self.linear2(x)
        x = F.leaky_relu(x, 0.1)
        #print("X:", x.shape)

        #x = self.tconv1(x)
        #x = self.batchnorm1(x)
        #x = nn.LeakyReLU(0.2)(x)

        x = x.view((-1, self.ngf, 4, 4)) # changed from 6 to 4
        #print("X:", x.shape)

        x = self.tconv2(x)
        x = self.batchnorm2(x)
        x = F.leaky_relu(x, 0.1)
        #print("X:", x.shape)

        x = self.tconv3(x)
        x = self.batchnorm3(x)
        x = F.leaky_relu(x, 0.1)
        #print("X:", x.shape)

        x = self.tconv4(x)
        x = self.batchnorm4(x)
        x = F.leaky_relu(x, 0.1)
        #print("X:", x.shape)

        x = self.tconv5(x)
        #print("X:", x.shape)
        return x

class Conv64Encoder(nn.Module):
    # ndf = size of feature maps, nc = number of channels
    def __init__(self, nc, nf, nz):
        super(Conv64Encoder, self).__init__()
        #self.ngpu = ngpu
        self.nz = nz
        ndf = nf
        self.ndf = ndf

        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False)
        #self.conv5 = nn.Conv2d(ndf, nz, 4, 1, 0, bias=False)

        self.batchnorm1 = nn.BatchNorm2d(ndf)
        self.batchnorm2 = nn.BatchNorm2d(ndf)
        self.batchnorm3 = nn.BatchNorm2d(ndf)
        self.batchnorm4 = nn.BatchNorm2d(ndf)

        self.linear1 = nn.Linear(self.ndf * 4 * 4, 1024) # changed from 6 to 4
        self.linear2 = nn.Linear(1024, nz)

    def forward(self, x):
        #print("Encoder FWD pass")
        #print("X:", x.shape)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.leaky_relu(x, 0.1)
        #print("X:", x.shape)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.leaky_relu(x, 0.1)
        #print("X:", x.shape)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.leaky_relu(x, 0.1)
        #print("X:", x.shape)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = nn.LeakyReLU(0.2)(x)
        #print("X:", x.shape)

        #x = self.conv5(x)
        #x = x.view((-1, self.ndf*4*5*5))
        #x = self.linear(x)
        #print("X:", x.shape)

        x = x.view((-1, self.ndf * 4 * 4)) # changed from 6 to 4
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.1)
        #print("X:", x.shape)

        x = self.linear2(x)
        #print("X:", x.shape)

        x = x.view((-1, self.nz))
        #print("X:", x.shape)

        return x

class MLP(nn.Module):
    def __init__(self, n_layers, dimensions, activation, out_activation, stochastic=None):
        super(MLP, self).__init__()
        self.stochastic = stochastic
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        input_dim = dimensions[0]
        hidden_dim = dimensions[1]
        output_dim = dimensions[2]

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])

        self.output = nn.Linear(hidden_dim, output_dim)
        # self.output_std = nn.Linear(hidden_dim, output_dim)

        self.activation = activation
        self.out_activation = out_activation

    def forward(self, x):
        h = self.input_layer(x)
        h = self.activation(h)

        if self.stochastic is not None:
            epsilon = self.stochastic.sample(h.shape).to(self.device)
        else:
            epsilon = torch.zeros_like(h).to(self.device)

        h = h + epsilon

        for hidden_layer in self.hidden_layers:
            h = hidden_layer(h)
            h = self.activation(h)
        # print("h.shape", h.shape)

        out = self.output(h)
        out = self.out_activation(out)
        return out
