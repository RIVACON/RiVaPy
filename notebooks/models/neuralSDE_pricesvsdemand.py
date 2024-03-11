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



# data ---------------------------------------------------
df = pd.read_pickle('spot_demand_wind_solar.pickle')

sel = df[(df['Day_of_week'] ==3)]# & (df['isHoliday?'] == False)]

df = sel[['Hour_of_day','Spot','ResidualDemand']]

df = df[-6*24:]




# parameters ---------------------------------------------
data_size = 2 
ts_len = 24
batch_size=6#int(len(sel['Spot'])/(ts_len))#512#1024
latent_size=2
context_size=1
hidden_size=8
lr_init=5e-3#1e-2
wd_init = 1e-5
lr_gamma=0.997
t0=0.
t1=float(ts_len)
kl_anneal_iters=100
noise_std=.01
adjoint=False
method="milstein"

num_samples=ts_len*batch_size
num_iters = 30000



# torch tensors and data preparation -------------------------------------------
xs = torch.empty((int(t1), batch_size, data_size), dtype=torch.float32)
ts = torch.empty(int(t1), dtype=torch.float32)

spot_rz = np.resize(df['Spot'][0:ts_len*batch_size],(ts_len,batch_size))
resdemand_rz = np.resize(df['ResidualDemand'][0:ts_len*batch_size],(ts_len,batch_size))

spot_mean = np.mean(spot_rz[0,:])
spot_std = np.std(spot_rz[0,:])
spot_rz = (spot_rz - spot_mean) / spot_std

resdemand_mean = np.mean(resdemand_rz[0,:])
resdemand_std = np.std(resdemand_rz[0,:])
resdemand_rz = (resdemand_rz - resdemand_mean) / resdemand_std

xs[:,:,1] = torch.tensor(spot_rz)
xs[:,:,0] = torch.tensor(resdemand_rz)

#print(xs[:,:,0])

#plt.scatter(xs[:,:,0],xs[:,:,1])
#plt.show()



for i in range(ts_len):
    ts[i] = float(i)

#model-----------------------------------------------------------------------------------
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

    def __init__(self, hidden_size,context_size,latent_size,input_size):

        super(LatentSDE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, context_size),
        )

        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)


        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
        )

        self.h_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )

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
    
        #Decoder: 
        self.projector =nn.Sequential(
            #nn.Linear(latent_size, input_size)

            nn.Linear(latent_size, hidden_size),
            #nn.ReLU(),
            #nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
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
    
    def h(self, t, y):
        return self.h_net(y)
    
    def g(self, t, y):  # Diagonal diffusion.
            y = torch.split(y, split_size_or_sections=1, dim=1)
            out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
            return torch.cat(out, dim=1)
    
    

    def forward(self, xs, ts, noise_std):
        ctx = self.encoder(torch.flip(xs, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx))

        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean  + qz0_logstd.exp() * torch.randn_like(qz0_mean)

        zs, log_ratio = torchsde.sdeint(self, z0, ts, dt = 0.1, logqp=True, method='euler', rtol = 1e-3, atol = 1e-3, names={'drift': 'f'})

        _xs = self.projector(zs)
        xs_dist = Normal(loc=_xs, scale=noise_std)
        log_pxs = xs_dist.log_prob(xs).sum(dim=(0, 2)).mean(dim=0)

        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        logqp_path = log_ratio.sum(dim=0).mean(dim=0)
        return log_pxs, logqp0 + logqp_path, _xs, zs

    @torch.no_grad()
    def sample(self, batch_size, ts, bm=None):
        eps = torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
        z0 = self.pz0_mean.repeat(batch_size,1)   + self.pz0_logstd.exp() * eps  
        zs = torchsde.sdeint(self, z0, ts, dt = 0.1, names={'drift': 'h'}, bm = bm)
        _xs = self.projector(zs)
        return _xs, zs



latent_sde = LatentSDE(
        hidden_size=hidden_size,
        context_size = context_size, input_size = data_size, latent_size = 2
    ).to(device)



optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer) #ExponentialLR(optimizer=optimizer, gamma = lr_gamma)
kl_scheduler = LinearScheduler(iters=kl_anneal_iters)


# Fix the same Brownian motion for visualization.
bm_vis = torchsde.BrownianInterval(
        t0=ts[0], t1=ts[-1], size=(batch_size, latent_size,), device=device, levy_area_approximation="space-time")



#training and testing model--------------------------------------------------------------------------------


loss_trend = []
if True:
    for global_step in tqdm.tqdm(range(1, num_iters + 1)):
        latent_sde.zero_grad()
        log_pxs, log_ratio = latent_sde(xs, ts, noise_std)[:2]
        loss = -log_pxs   + log_ratio * kl_scheduler.val
        if (global_step)% (num_iters/100) == 0:
            loss_trend.append(loss.detach().numpy())
        loss.backward()
        print('Loss: ' + str(loss.item()) + '  log_pxs: '+ str(log_pxs.item()) + '  log_ratio, kl_scheduler: ' + str((log_ratio.item(),kl_scheduler.val)))
        optimizer.step()
        scheduler.step()
        kl_scheduler.step()

    torch.save(latent_sde.state_dict(), 'sdepricedemand.pth')
else:
    latent_sde.load_state_dict(torch.load('sdepricedemand.pth'))
    latent_sde.eval()

xs_l, zs_l = latent_sde.sample(batch_size=xs.size(1), ts=ts, bm=bm_vis)
_xs = latent_sde(xs, ts, noise_std)[2].cpu().detach().numpy()
_zs = latent_sde(xs, ts, noise_std)[3].cpu().detach().numpy()

input_data = xs.cpu().numpy()
output_data = xs_l.cpu().numpy()
print(input_data)
print(output_data.shape)
print(input_data[:,:,0])
c = np.array([i/ts_len*0.6 for i in range(ts_len)]).repeat(6)
plt.scatter(input_data[:,:,0]*resdemand_std + resdemand_mean,input_data[:,:,1]*spot_std + spot_mean, c = c, label = 'data', marker = 'x', edgecolors='black')
#plt.gray()
plt.scatter(output_data[:,:,0]*resdemand_std + resdemand_mean, output_data[:,:,1]*spot_std + spot_mean, c = c, label = 'model', edgecolors='black')
#plt.gray()
plt.title('Thursdays in 2024')
plt.xlabel('residual demand')
plt.ylabel('spot price')
plt.legend()
plt.show()