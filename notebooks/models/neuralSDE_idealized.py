# import -----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import math


from typing import Sequence, Optional, Union

import torch
import tqdm
from torch import nn
from torch import optim
from torch.distributions import Normal

from collections import namedtuple

import torchsde

#for plotting
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import scipy as scp
import scipy.stats as ss
from scipy.optimize import minimize
from scipy import sparse
from scipy.sparse.linalg import spsolve
from mpl_toolkits import mplot3d
from matplotlib import cm
import scipy.special as scsp
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator


from sys import exit


#data-------------------------------------------------------------------------------------

np.random.seed(seed=42)
N = 24  # time steps
paths = 10  # number of paths
T = 24
T_vec, dt = np.linspace(0, T, N, retstep=True)
kappa = 3
theta = 0.5
sigma = 0.5
std_asy = np.sqrt(sigma**2 / (2 * kappa))  
X0 = 0.
X = np.zeros((N, paths))
X[0, :] = X0
W = ss.norm.rvs(loc=0, scale=1, size=(N - 1, paths))
std_dt = np.sqrt(sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))
for t in range(0, N - 1):
    X[t + 1, :] = theta + np.exp(-kappa * dt) * (X[t, :] - theta) + std_dt * W[t, :]


# parameters for neuralSDE ---------------------------------------------
ts_len = N
batch_size=paths
hidden_size=8
lr_init=1e-2
t0=0.
t1=float(ts_len)
lr_gamma=0.997
kl_anneal_iters=100
noise_std=5.

num_samples=batch_size
num_iters = 1000



# torch tensors -------------------------------------------
xs = torch.empty((int(t1), batch_size,1), dtype=torch.float32)
ts = torch.empty(int(t1), dtype=torch.float32)
x0 = torch.empty((batch_size, 1), dtype=torch.float32)

xs[:,:,0] = torch.tensor(X)

for i in range(ts_len):
    ts[i] = float(i)

# model ---------------------------------------------------
device = torch.device('cpu')


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val



class LatentSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal" 

    def __init__(self, hidden_size,sigma,kappa, theta): 
        super(LatentSDE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_size),
            #nn.Softplus(),
            #nn.Linear(hidden_size, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, 2),
        )

        self.qz0_net = nn.Linear(2, 2)
        self.register_buffer("kappa", torch.tensor([[kappa]]))
        self.register_buffer("theta", torch.tensor([[theta]]))
        self.register_buffer("sigma", torch.tensor([[sigma]]))


        self.f_net = nn.Sequential(
            nn.Linear(3, hidden_size),
            #nn.Softplus(),
            #nn.Linear(hidden_size, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, 1),
        )

    
        #Decoder: 
        self.projector =nn.Sequential(
            nn.Linear(1, hidden_size),
            #nn.Softplus(),
            #nn.Linear(hidden_size, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, 1),
        )
        self.pz0_mean = nn.Parameter(torch.zeros(1, 1))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, 1))

        self._ctx = None

    def contextualize(self, ctx):
        self._ctx = ctx  

    def f(self, t, y): # (posterior) Drift 
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        return self.f_net(torch.cat((y, ctx[i]), dim=1))
    

    def h(self, t, y): # (prior) Drift
        return self.kappa * (self.theta - y)

    def g(self, t, y):  # Diagonal diffusion. 
        return self.sigma.repeat(y.size(0), 1)
    

    def forward(self, xs, ts, noise_std):
        ctx = self.encoder(torch.flip(xs, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx))

        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)

        zs, log_ratio = torchsde.sdeint(self, z0, ts, dt=1, logqp=True, method='milstein', rtol = 1e-3, atol = 1e-3)

        _xs = self.projector(zs)
        xs_dist = Normal(loc=_xs, scale=noise_std)
        log_pxs = xs_dist.log_prob(xs).sum(dim=(0, 2)).mean(dim=0)

        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        logqp_path = log_ratio.sum(dim=0).mean(dim=0)
        return log_pxs, logqp0 + logqp_path

    #We can use these parameters to sample new similar points from the latent space
    @torch.no_grad()
    def sample(self, batch_size, ts, bm=None):
        eps = torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
        z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
        zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h'}, dt=1, bm=bm)
        # Most of the times in ML, we don't sample the observation noise for visualization purposes.
        _xs = self.projector(zs)
        return _xs



latent_sde = LatentSDE(
        hidden_size=hidden_size,
        sigma = sigma, kappa = kappa, theta = theta,
    ).to(device)

optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer)
kl_scheduler = LinearScheduler(iters=kl_anneal_iters)


# Fix the same Brownian motion for visualization.
bm_vis = torchsde.BrownianInterval(
        t0=t0, t1=t1, size=(batch_size, 1,), device=device, levy_area_approximation="space-time")


for global_step in tqdm.tqdm(range(1, num_iters + 1)):
    latent_sde.zero_grad()
    log_pxs, log_ratio = latent_sde(xs, ts, noise_std)
    loss = -log_pxs + log_ratio * kl_scheduler.val
    #print(loss)
    loss.backward()
    optimizer.step()
    scheduler.step()
    kl_scheduler.step()


xs_l = latent_sde.sample(batch_size=xs.size(1), ts=ts, bm=bm_vis)

fig = plt.figure(figsize=(20, 9))
input_data = xs.cpu().numpy()
output_data = xs_l.cpu().numpy()
tt = ts.numpy()
for i in range(batch_size-1):
    plt.plot(tt,input_data[:,i,0], label = 'data',color='blue')
    plt.plot(tt,output_data[:,i,0], label = 'model',color='green')
plt.plot(tt,input_data[:,batch_size-1,0], label = 'data',color='blue')
plt.plot(tt,output_data[:,batch_size-1,0], label = 'model',color='green')
plt.xlabel('t')
plt.ylabel('x')
plt.legend()
plt.show()
