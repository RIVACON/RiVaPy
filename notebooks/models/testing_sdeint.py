import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
from torch import nn
from torch import optim
from torch.distributions import Normal
import scipy.stats as ss
from torchsde import sdeint

#OU-construction----------------------------------------

N = 365  # time steps
paths = 15  # number of paths
T = 1.0
T_vec, dt = np.linspace(0, T, N, retstep=True)
kappa = 3
theta = 0.5
sigma = 0.5
std_asy = np.sqrt(sigma**2 / (2 * kappa))  
X0 = 3.
X = np.zeros((N, paths))
X[0, :] = X0
W = ss.norm.rvs(loc=0, scale=1, size=(N - 1, paths))
std_dt = np.sqrt(sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))
for t in range(0, N - 1):
    X[t + 1, :] = theta + np.exp(-kappa * dt) * (X[t, :] - theta) + std_dt * W[t, :]

# parameters for neuralSDE ---------------------------------------------
ts_len = N
batch_size=paths
hidden_size=64
lr_init=1e-4
t0=0.
t1=float(ts_len)

#modeling------------------------------------------------
xs = torch.empty((int(t1), batch_size,1), dtype=torch.float32)
X0 = X0 * torch.ones((batch_size, 1), dtype=torch.float32)

xs[:,:,0] = torch.tensor(X)

   

class SDE(nn.Module):

    def __init__(self,kappa,theta,sigma):
        super().__init__()
        self.register_buffer("kappa", torch.tensor([[kappa]]))
        self.register_buffer("theta", torch.tensor([[theta]]))
        self.register_buffer("sigma", torch.tensor([[sigma]]))
        self.noise_type = "diagonal"
        self.sde_type = "ito"
    
    def f(self, t, y):
        return self.kappa * (self.theta - y)
    
    def g(self, t, y):
        return self.sigma.repeat(y.size(0), 1)


sde = SDE(kappa,theta, sigma)
ts = torch.linspace(0, T, N)
y0 = torch.zeros(batch_size, 1).fill_(0.1)  # (batch_size, d)
xs = xs.cpu().numpy()
with torch.no_grad():
    ys = sdeint(sde, X0, ts, method='srk')  # (T, batch_size, d) = (100, 3, 1).

plt.figure()
for i in range(batch_size):
    plt.plot(ts, ys[:, i].squeeze(), label=f'sample {i}', color = 'orangered',linewidth = 0.5)
    plt.plot(ts,xs[:,i].squeeze(), label=f'data {i}', color = 'blue',linewidth = 0.5)
plt.xlabel('$t$')
plt.ylabel('$y_t$')
plt.legend()
plt.show()
