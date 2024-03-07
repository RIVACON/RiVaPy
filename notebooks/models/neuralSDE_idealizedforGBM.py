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


#data OU-------------------------------------------------------------------------------------

np.random.seed(seed=42)
N = 365  # time steps
paths = 200  # number of paths
T = 1.
T_vec, dt = np.linspace(0, T, N, retstep=True)
kappa = 3
theta = .5
sigma = 1.1
std_asy = np.sqrt(sigma**2 / (2 * kappa))  
X0 = 3.
X = np.zeros((N, paths))
X[0, :] = X0
W = ss.norm.rvs(loc=0, scale=1, size=(N - 1, paths))
std_dt = np.sqrt(sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))
for t in range(0, N - 1):
    X[t + 1, :] = theta + np.exp(-kappa * dt) * (X[t, :] - theta) + std_dt * W[t, :]




#data GBM--------------------------------------------------------------------------------------
    
#np.random.seed(seed=42)
##N = 365  # number of time steps
#paths = 200  # number of paths
#T = 1.
#T_vec, dt = np.linspace(0, T, N, retstep=True)
#mu = 0.1
#sigma = 0.2
#S0 = 1
#X0 = np.zeros((paths, 1))  # each path starts at zero
#W = ss.norm.rvs((mu - 0.5 * sigma**2) * dt, np.sqrt(dt) * sigma, (paths, N-1))
#X = np.concatenate((X0, W), axis=1).cumsum(1)
#S_T = np.exp(np.transpose(X))




#visualisation of tests-------------------------------------------------
figure, axis = plt.subplots(3, 1, figsize = (8,12), gridspec_kw={'height_ratios': [1, 2, 2]})
    
    

# parameters for neuralSDE ---------------------------------------------
num_iters = 5000
num_samples = 1
ts_len = N
batch_size=paths
hidden_size=64
context_size = 1
input_size = 1
latent_size = 1
lr_init=5e-2
lr_final = 1e-5
lr_gamma = math.pow(lr_final / lr_init, 1/num_iters)
noise_std=0.01
num_samples=batch_size
kl_anneal_iters=int(num_iters/5)    



# torch tensors -------------------------------------------
xs = torch.empty((N, batch_size,1), dtype=torch.float32)
ts = torch.tensor(T_vec, dtype=torch.float32)
x0 = torch.empty((batch_size, 1), dtype=torch.float32)

xs[:,:,0] = torch.tensor(X)


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

    def __init__(self, hidden_size,context_size,latent_size,input_size,sigma,kappa, theta):
    #def __init__(self, hidden_size, context_size, latent_size, input_size, sigma, mu): 
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

        self.register_buffer("sigma", torch.tensor([[sigma]]))
        self.register_buffer("kappa", torch.tensor([[kappa]]))
        self.register_buffer("theta", torch.tensor([[theta]]))

        #self.register_buffer("mu", torch.tensor([[mu]]))


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
    

    #def h(self, t, y): # (prior) Drift
    #    return self.kappa * (self.theta - y)
    #    #return self.mu * y

    
    #def g(self, t, y):  # Diagonal diffusion. 
    #    return self.sigma.repeat(y.size(0), 1)
    

    def forward(self, xs, ts, noise_std):
        ctx = self.encoder(torch.flip(xs, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx))

        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean               #+ qz0_logstd.exp() * torch.randn_like(qz0_mean)

        zs, log_ratio = torchsde.sdeint(self, z0, ts,  logqp=True, method='euler', rtol = 1e-3, atol = 1e-3, names={'drift': 'f'})

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
        z0 = self.pz0_mean.repeat(batch_size,1)             #+ self.pz0_logstd.exp() * eps  #torch.zeros((batch_size,1)) 
        zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h'}, bm = bm)
        _xs = self.projector(zs)
        return _xs, zs



latent_sde = LatentSDE(
        hidden_size=hidden_size,
        sigma = sigma, kappa = kappa, theta = theta,
        context_size = 2, input_size = 1, latent_size = 1

    ).to(device)

#latent_sde = LatentSDE(
#        hidden_size=hidden_size,
#        sigma = sigma, mu=mu,
#        context_size = 2, input_size = 1, latent_size = 1
#    ).to(device)

optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma = lr_gamma)
kl_scheduler = LinearScheduler(iters=kl_anneal_iters)


# Fix the same Brownian motion for visualization.
bm_vis = torchsde.BrownianInterval(
        t0=T_vec[0], t1=T, size=(batch_size, 1,), device=device, levy_area_approximation="space-time")



#training and testing model--------------------------------------------------------------------------------

tt = ts.numpy()
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

    torch.save(latent_sde.state_dict(), 'sde.pth')
else:
    latent_sde.load_state_dict(torch.load('sde.pth'))
    latent_sde.eval()

xs_l, zs_l = latent_sde.sample(batch_size=xs.size(1), ts=ts, bm=bm_vis)
_xs = latent_sde(xs, ts, noise_std)[2].cpu().detach().numpy()
_zs = latent_sde(xs, ts, noise_std)[3].cpu().detach().numpy()


#visualisation---------------------------------------------------------------------------------------------

input_data = xs.cpu().numpy()
latent_data = zs_l.cpu().numpy()
output_data = xs_l.cpu().numpy()
inputmean = np.array([np.mean(input_data[i]) for i in range(N)])
outputmean = np.array([np.mean(output_data[i]) for i in range(N)])
reconmean = np.array([np.mean(_xs[i]) for i in range(N)])
latentmean = np.array([np.mean(latent_data[i]) for i in range(N)])



print(np.mean(outputmean))
print(np.mean(inputmean))

for i in range(100):
    axis[1].plot(tt,input_data[:,i,0], color='blue', linewidth = 0.2)
    axis[1].plot(tt,_xs[:,i,0], color='orangered', linewidth = 0.2)
    axis[1].plot(tt,_zs[:,i,0], color='dimgray', linewidth = 0.2)
    #axis[2].plot(tt,latent_data[:,i,0], color='orangered', linewidth = 0.2)
    axis[2].plot(tt,output_data[:,i,0], color='darkgreen', linewidth = 0.2)
    axis[2].plot(tt,input_data[:,i,0], color = 'blue', linewidth = 0.1)
axis[1].plot(tt,input_data[:,batch_size-1,0],color='blue', linewidth = 0.2)
axis[1].plot(tt,_xs[:,batch_size-1,0], color='orangered', linewidth = 0.2)
axis[1].plot(tt,inputmean, color='blue', label = 'data', linewidth = 1)
axis[1].plot(tt,reconmean, color='orangered', label = 'reconstructed data', linewidth = 1)

axis[2].plot(tt,latent_data[:,batch_size-1,0], color='orangered', linewidth = 0.2)
axis[2].plot(tt,output_data[:,batch_size-1,0], color='darkgreen', linewidth = 0.2)
axis[2].plot(tt,inputmean, color='blue', label = 'data', linewidth = 1)
axis[2].plot(tt,outputmean, color='darkgreen', label = 'model', linewidth = 1)
axis[2].plot(tt,latentmean, color='orangered', label = 'latent', linewidth = 1)

axis[1].set_xlabel('t')
axis[1].set_ylabel('x')
axis[1].set_xlabel('t')
axis[1].set_ylabel('x')
axis[1].set_title('data')
axis[2].set_title('model')
axis[2].legend()
axis[1].legend()
        

num_values = [num_iters/100*i for i in range(len(loss_trend))]

axis[0].plot(num_values,loss_trend)
axis[0].set_xlabel('iteration')
axis[0].set_ylabel('loss')
#plt.savefig(fname = str(num_iters) + 'iters_' + str(lr_init) + 'lr_' 
#            + str(paths) + 'paths_' + str(N) + 'N' 
#            +'0.1dtt_0.01dts' '.jpeg', dpi = 500)
plt.show()