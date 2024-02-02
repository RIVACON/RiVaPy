# import -----------------------------------------------
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

import xarray as xr

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


import numpy as np
import scipy as scp
import scipy.stats as ss
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from mpl_toolkits import mplot3d
from matplotlib import cm
import scipy.special as scsp
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator

from sys import exit


# parameters ---------------------------------------------
data_size = 1
batch_size=1#512#1024
ts_len = 100
latent_size=1
context_size=1
hidden_size=2#128
lr_init=1e-3
t0=0.
t1=float(ts_len)
lr_gamma=0.997
kl_anneal_iters=1000
pause_every=1
noise_std=.1
adjoint=False
train_dir='./'
method="milstein"

num_samples=batch_size



# data ---------------------------------------------------
spot_data = pd.read_pickle('spot.pickle')
demand_data = pd.read_pickle('demand.pickle')
wind_solar_data = pd.read_pickle('wind_solar.pickle')
spot_data = spot_data.dropna()
demand_data = demand_data.dropna()
wind_solar_data = wind_solar_data.dropna()

spot = np.array(spot_data['Spot'])
demand = np.array(demand_data['Demand'])
wind = np.array(wind_solar_data['Wind'])


spot_r1y = np.resize(spot[0:ts_len*batch_size],(ts_len,batch_size))
demand_r1y = np.resize(demand[0:ts_len*batch_size],(ts_len,batch_size))
wind_r1y = np.resize(wind[0:ts_len*batch_size],(ts_len,batch_size))

spot_mean = np.mean(spot_r1y[0,:])
demand_mean = np.mean(demand_r1y[0,:])
wind_mean = np.mean(wind_r1y[0,:])
spot_std = np.std(spot_r1y[0,:])
demand_std = np.std(demand_r1y[0,:])
wind_std = np.std(wind_r1y[0,:])

spot_r1y = (spot_r1y - spot_mean) / spot_std
demand_r1y = (demand_r1y - demand_mean) / demand_std
wind_r1y = (wind_r1y - wind_mean) / wind_std








# torch tensors -------------------------------------------
xs = torch.empty((int(t1), batch_size, data_size), dtype=torch.float32)
ts = torch.empty(int(t1), dtype=torch.float32)
x0 = torch.empty((batch_size, latent_size), dtype=torch.float32)

#xs[:,:,0] = torch.tensor(spot_r1y)
#xs[:,:,1] = torch.tensor(demand_r1y)
#xs[:,:,2] = torch.tensor(wind_r1y)

xs[:,0,0] = torch.tensor(X[0,:])
#xs[:,1,0] = torch.tensor(X[1,:])
#xs[:,0,1] = torch.tensor(X[2,:])
#xs[:,1,1] = torch.tensor(X[3,:])
#xs[:,0,2] = torch.tensor(X[4,:])
#xs[:,1,2] = torch.tensor(X[5,:])

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


#Encoder 
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.lin1 =nn.Linear(input_size, hidden_size) 
        self.softplus = nn.Softplus()
        self.lin2 = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        out = self.lin1(inp)
        out = self.softplus(out)
        out = self.lin2(out)
        return out


class LatentSDE(nn.Module):
    # stochastic ito integral
    sde_type = "ito"
    # diffusion is element wise 
    noise_type = "diagonal" 

    def __init__(self, data_size, latent_size, context_size, hidden_size): # decoder_hiddensiz
        super(LatentSDE, self).__init__()
        # Encoder.
        self.encoder = Encoder(input_size=data_size, hidden_size=hidden_size, output_size=context_size)
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)

        # Decoder.
        # corresponding (posterior) drift
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        # corresponding to prior drift to calculate KL divergence (section 5 in Scalable Gradients for SDEs by Li et al., 2020) to reduce overfitting
        self.h_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        # Diffusion: This needs to be an element-wise function for the SDE to satisfy diagonal noise.
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.Softplus(),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid()
                )
                for _ in range(latent_size)
            ]
        )
        #self.projector = nn.Linear(latent_size, data_size)
        #Decoder: 
        self.projector =nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, data_size),
        )
        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        self._ctx = None

    def contextualize(self, ctx):
        self._ctx = ctx  

    def f(self, t, y): # (posterior) Drift 
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        return self.f_net(torch.cat((y, ctx[i]), dim=1))

    def h(self, t, y): # (prior) Drift
        return self.h_net(y)

    def g(self, t, y):  # Diagonal diffusion. 
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def forward(self, xs, ts, noise_std, adjoint=False, method="midpoint"):
        ctx = self.encoder(torch.flip(xs, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx))

        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)

        if adjoint:
            # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
            adjoint_params = (
                    (ctx,) +
                    tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(self.h_net.parameters())
            )
            zs, log_ratio = torchsde.sdeint_adjoint(
                self, z0, ts, adjoint_params=adjoint_params, dt=1, logqp=True, method=method)
        else:
            zs, log_ratio = torchsde.sdeint(self, z0, ts, dt=1, logqp=True, method=method)

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
        data_size=data_size, 
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
    ).to(device)

optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer)#ExponentialLR(optimizer=optimizer, gamma=lr_gamma)
kl_scheduler = LinearScheduler(iters=kl_anneal_iters)

# Fix the same Brownian motion for visualization.
bm_vis = torchsde.BrownianInterval(
        t0=t0, t1=t1, size=(batch_size, latent_size,), device=device, levy_area_approximation="space-time")



num_iters = 1000
for global_step in tqdm.tqdm(range(1, num_iters + 1)):
    latent_sde.zero_grad()
    log_pxs, log_ratio = latent_sde(xs, ts, noise_std, adjoint, method)
    loss = -log_pxs + log_ratio * kl_scheduler.val
    #print(loss)
    loss.backward()
    optimizer.step()
    scheduler.step()
    kl_scheduler.step()



f_ten = torch.empty(int(t1), dtype=torch.float32)
h_ten = torch.empty(int(t1), dtype=torch.float32)
g_ten = torch.empty(int(t1), dtype=torch.float32)
for i in range(ts_len):
    f_ten[i] = latent_sde.f(t=ts[i],y=xs[i] )
    h_ten[i] = latent_sde.h(t=ts[i],y=xs[i] )
    g_ten[i] = latent_sde.g(t=ts[i],y=xs[i] )


xs_l = latent_sde.sample(batch_size=xs.size(1), ts=ts, bm=bm_vis)

fig = plt.figure(figsize=(20, 9))
#gs = gridspec.GridSpec(1, 2)
#ax00 = fig.add_subplot(gs[0, 0], projection='3d')
#ax01 = fig.add_subplot(gs[0, 1], projection='3d')

input_data = xs.cpu().numpy()
output_data = xs_l.cpu().numpy()
f_data = f_ten.detach().numpy()
h_data = h_ten.detach().numpy()
g_data = g_ten.detach().numpy()
tt = ts.numpy()
plt.plot(tt,input_data[:,0,0], label = 'data')
plt.plot(tt,output_data[:,0,0], label = 'model')
plt.plot(tt,f_data[:],label = 'f')
plt.plot(tt,h_data[:],label = 'h')
plt.plot(tt,g_data[:],label = 'g')
plt.legend()


# Left plot: data.
#z1, z2, z3 = np.split(xs.cpu().numpy(), indices_or_sections=3, axis=-1)
#[ax00.plot(z1[:, i, 0], z2[:, i, 0], z3[:, i, 0]) for i in range(num_samples)]
#ax00.scatter(z1[0, :num_samples, 0], z2[0, :num_samples, 0], z3[0, :num_samples, 0], marker='x')
#ax00.set_xlabel('$norm. price$', labelpad=0., fontsize=16)
#ax00.set_ylabel('$norm. demand$', labelpad=.5, fontsize=16)
#ax00.set_zlabel('$norm. wind$', labelpad=0., horizontalalignment='center', fontsize=16)
#ax00.set_title('Data', fontsize=20)
#xlim = ax00.get_xlim()
#ylim = ax00.get_ylim()
#zlim = ax00.get_zlim()

# Right plot: model.
#z1, z2, z3 = np.split(xs_l.cpu().numpy(), indices_or_sections=3, axis=-1)
#[ax01.plot(z1[:, i, 0], z2[:, i, 0], z3[:, i, 0]) for i in range(num_samples)]
#ax01.scatter(z1[0, :num_samples, 0], z2[0, :num_samples, 0], z3[0, :num_samples, 0], marker='x')
#ax01.set_xlabel('$norm. price$', labelpad=0., fontsize=16)
#ax01.set_ylabel('$norm. demand$', labelpad=.5, fontsize=16)
#ax01.set_zlabel('$norm. wind$', labelpad=0., horizontalalignment='center', fontsize=16)
#ax01.set_title('Samples', fontsize=20)
#ax01.set_xlim(xlim)
#ax01.set_ylim(ylim)
#ax01.set_zlim(zlim)

plt.show()
#plt.savefig('test.png')

    

