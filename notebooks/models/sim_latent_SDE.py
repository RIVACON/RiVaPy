import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.distributions.multivariate_normal import MultivariateNormal
import tqdm


import matplotlib


import numpy as np
import scipy.stats as ss

import math
import yaml
import pickle
import pprint 
import os
import logging
import sys

import matplotlib
import matplotlib.pyplot as plt
device = torch.device('cpu') #'cuda' if torch.cuda.is_available() else 'cpu')
training = None


#params----------------------------------------------------------------------------------------
#dataset:
N = 365  # time steps
paths = 200  # number of paths
T = 1.
T_vec, dt = np.linspace(0, T, N, retstep=True)
kappa = 3
theta = .5
sigma = 1.1
std_asy = np.sqrt(sigma**2 / (2 * kappa))  
X0 = 3.

#model: 
num_iters = 6
kl_iters = int(num_iters/4)
batch_size = paths
in_dim = 1
hidden_dim = 32
n_layers = 4
latent_dim = 2
num_sample = 20
optimizer_name = 'Adam'
lr_scheduler = 'ExponentialLR'
depth = 4
act = 'nn.Softplus'
lr_mu = 0.001
lr_sigma = 0.001
lr_ae = 0.01
lr_gamma = 0.998
usetorchkl = False

#setup----------------------------------------------------------------------------------------
#data:
def OUdata(N,paths,X0):
    X = np.zeros((N, paths))
    X[0, :] = X0
    W = ss.norm.rvs(loc=0, scale=1, size=(N - 1, paths))
    std_dt = np.sqrt(sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))
    for t in range(0, N - 1):
        X[t + 1, :] = theta + np.exp(-kappa * dt) * (X[t, :] - theta) + std_dt * W[t, :]

    xs = torch.empty((batch_size, N, 1), dtype=torch.float32)
    ts = torch.tensor(T_vec, dtype=torch.float32)
    xs[:,:,0] = torch.tensor(X.T)

    print(torch.mean(xs))

    dataset = torch.utils.data.TensorDataset(xs)
    train_dataloader = torch.utils.data.DataLoader(dataset, num_workers = 0,  batch_size = 1, drop_last=True)
    return train_dataloader, ts

#scheduler for KL divergence:
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


#networks:
class StochConvAE(nn.Module):
    def __init__(self, in_channels: int, 
            hidden_dim:      int , 
            n_layers:   int , 
            latent_dim: int , 
            act = nn.LeakyReLU(), 
            loss = 'exact', 
            sig_x = 0.1, 
            sigma_type='diagonal',
            pz_var=100):

        '''
        This method initializes an autoencoder with parameters

        in_channels: number of input channels
        hidden_dim     : multiple to increase the hidden_dim of each conv layer
        n_layers  : number of levels of convolutions
        latent_dim: size of the latent dimension
        latent_dim_im_size: size of the downscaled image
        fs        : filter size
        act       : activation function
        pooling   : pooling layer
        loss      : either 'mc' or 'exact' for computing the KL divergence
        sigmoid   : last layer include sigmoid or not
        use_skip  : add a skip connection between the input and output layers
        add_det   : add deterministic component
        flow      : add normalizing flow (not yet implemented)
        sig_x     : variance on the decoder (how to penalize reconstruction loss)
        sig_det   : variance on deterministic part
        sigma_type: if sigma should be diagonal or full matrix
        pz_var    : variance on the prior z
        use_conv  : encoder conv or not
        '''

        super(StochConvAE,self).__init__()

        self.loss     = loss
        self.sigma_type = sigma_type
        self.sig_x   = sig_x

        if pz_var:
            self.pz = MultivariateNormal(torch.zeros(latent_dim,device='cpu'), torch.eye(latent_dim,device='cpu') * pz_var)

        self.latent_dim = latent_dim

        enc_modules = [nn.Linear(in_channels, hidden_dim), act]
        dec_modules = [nn.Linear(latent_dim, hidden_dim)]

        for i in range(1, n_layers):

            enc_modules += [nn.Linear(hidden_dim, hidden_dim),
                    #nn.BatchNorm1d(hidden_dim),
                    act]
            dec_modules += [nn.Linear(hidden_dim, hidden_dim),
                    #nn.BatchNorm1d(hidden_dim), 
                    act]

        dec_last_layer = [nn.Linear(hidden_dim, in_channels)]

        self.mu_linear = nn.Linear(hidden_dim, latent_dim, bias=True)
        self.sigma_linear = nn.Linear(hidden_dim, latent_dim, bias=True)

        self.encoder = nn.Sequential( * enc_modules)
        self.decoder = nn.Sequential( * dec_modules)
        self.decoder_last = nn.Sequential( * dec_last_layer)
        self.latent_dim = latent_dim




        for key, value in {'weight':'xavier', 'bias':'zeros'}.items():
            param = getattr(self.encoder, key, None)
            if param is not None:
                nn.init.xavier_normal_(param.data, gain=0.5)
            param = getattr(self.decoder, key, None)
            if param is not None:
                nn.init.xavier_normal_(param.data, gain=0.5)


    def get_increments(self, q_mu, q_sigma, dt=None):
        '''
        Gets increments given a mu and sigma coming from the encoder
        returns the increments, the sampled latent points z, and the full sigma q
        '''

        if dt:
            z = q_mu * dt + q_sigma * torch.randn_like(q_sigma).normal_(0, np.sqrt(dt))
        else:
            z = q_mu + q_sigma * torch.randn_like(q_sigma)
        inc = z[1:] - z[:-1]
        q_sigma_full = torch.diag_embed(q_sigma)

        return inc, z, q_sigma_full


    def get_next_z(self, z_init, ts, dt, mu, sigma):
        '''
        Method to compute z given mu and sigma functions, not encoder mu and sigma
        z_init : initial z to integrate from
        ts : time stamp
        dt : change in time
        mu : mu function
        sigma : sigma function
        returns z_n, one step integrated using the learned mu and sigma
        '''

        net_inputs = torch.cat( (ts.unsqueeze(1), z_init), dim=1)

        mu_hat, sigma_hat = sample_mu_sigma(mu, sigma, net_inputs)

        #z_n  = z_init + mu_hat * dt + sigma_hat * torch.randn_like(sigma_hat).normal_(0,np.sqrt(dt)) 

        lower_triangular = triangle_vec_to_lower(sigma_hat, self.latent_dim)
        sigma_hat_full   = torch.bmm(lower_triangular, lower_triangular.permute(0,2,1))
        epsilon          = torch.randn((mu_hat.shape[0], mu_hat.shape[1], 1)).normal_(0, np.sqrt(dt)).to(z_init.device) 
        z_n  = z_init + mu_hat * dt + torch.bmm(sigma_hat_full, epsilon).squeeze(2)

        return z_n
    

    def step(self, frames, ts, dt, mu, sigma,  plus_one=False):
        '''
        The main function for training, predicts the next step

        frames: input frames, tensor sized (batch_size, n_channels, w, w)
        NOTE the batch size is also acting as the time index, it is assumed to be in order
        ts: time step
        mu: mu function
        sigma: sigma function
        detach_ac: when training if detaching the autoencoder should occur
        plus_one : if predicting the next step should occur
        '''
        # Get the parameters for the latent distributions
        q_mu, q_sigma = self.encode(frames)

        # get a sample from our estimated distribution
        inc, z, q_sigma_full = self.get_increments(q_mu, q_sigma)


        # now we want to minimize the kl divergence with the
        # parameters of the SDE
        net_inputs = torch.cat((ts[:-1].unsqueeze(1), q_mu[:-1,:]), dim=1)      #why q_mu and not z?

        mu_hat, sigma_hat = sample_mu_sigma(mu, sigma, net_inputs)

        # helpers to go from cholesky vector to full matrix
        # if diagonal, keep it the same
        lower_triangular = triangle_vec_to_lower(sigma_hat, self.latent_dim)
        sigma_hat_full = torch.bmm(lower_triangular, lower_triangular.permute(0,2,1))       #huhhh

        # since we assume gaussian parameterization
        # difference betwen mu is a new gaussian
        q_mu_inc = q_mu[1:] - q_mu[:-1]
        
        sig_q       = q_sigma_full

        # likewise, sum of variances is also gaussian
        q_sig_inc = sig_q[1:] + sig_q[:-1]


        # minimize the KL between the distribution from the encoder and the latent SDE
        # specifically, the increments of our multivariate gaussian should match the mu_hat 
        kl_loss   = kl_div_exact(q_mu_inc, q_sig_inc, mu_hat, lower_triangular, dt)


        # add the prior 
        if self.pz:
            nan_problem = self.pz.log_prob(z).mean()
            kl_loss += nan_problem


        # pass to the decoder

        if plus_one:
            # We will use the mean as the current state
            # Then, we sample the next state according to the SDE
            z_step = self.get_next_z(q_mu, ts, dt, mu, sigma)
            decode_vec = z_step[:-1]
        else:
            decode_vec = z

        # after sampling the latent space reconstruct the image
        conditional_frame = frames[0].unsqueeze(0).repeat(q_mu.size(0)-1,1,1,1)
        frames_hat = self.decode(decode_vec,x=conditional_frame)

        # reconstruction loss
        if plus_one:
            l2_loss = 0.5 * F.mse_loss(frames_hat, frames[1:]) / self.sig_x ** 2
        else:
            l2_loss = 0.5 * F.mse_loss(frames_hat, frames) / self.sig_x ** 2


        return kl_loss, l2_loss, frames_hat, mu_hat, q_mu, sigma_hat_full, q_sigma_full, inc, z, nan_problem
    
    def sample_new_data(self, mu, sigma, ts, dt, num_samples):
        '''
        Samples new data from trained latent SDE
        '''
        samples = torch.empty((num_samples,len(ts),latent_dim))

        with torch.no_grad():

            for sample in samples:

                sample[0] = self.encode(torch.tensor([3.]))[0]     #torch.zeros((1,latent_dim))

                for i, time in enumerate(ts[:-1]):
                    net_inputs = torch.cat((torch.tensor([time]), sample[i]), dim = 0).unsqueeze(0)

                    mu_hat, sigma_hat = sample_mu_sigma(mu, sigma, net_inputs)

                    #z_n  = z_init + mu_hat * dt + sigma_hat * torch.randn_like(sigma_hat).normal_(0,np.sqrt(dt)) 

                    lower_triangular = triangle_vec_to_lower(sigma_hat, self.latent_dim)
                    sigma_hat_full   = torch.bmm(lower_triangular, lower_triangular.permute(0,2,1))
                    epsilon          = torch.randn((mu_hat.shape[0], mu_hat.shape[1], 1)).normal_(0, np.sqrt(dt)).to(sample[i-1].device) 
                    sample[i+1]  = sample[i] + mu_hat * dt + torch.bmm(sigma_hat_full, epsilon).squeeze(2)
                    
            return self.decode(samples)


    def encode(self, x):
        '''
        Takes in a frame (x)
        Outputs mu and sigma for the frame
        '''
        latent = self.encoder(x)
        mu = self.mu_linear(latent)
        sigma = self.sigma_linear(latent)

        return mu, sigma.exp()

    def decode(self, z, x=None):
        up_to_last = self.decoder(z)
        x = self.decoder_last(up_to_last)
        return x

    def forward(self, x, t):
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x, latent
    




class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size,
                 weight_init='xavier', bias_init='zeros', gain=0.5, **params):
        super(MLP, self).__init__()

        self.first_layer = ScaledLinear(input_size, hidden_size, bias=False, **params)
        self.hidden_layers = nn.ModuleList()
        for i in range(num_layers):
            self.hidden_layers.append(ScaledLinear(hidden_size, hidden_size, bias=False, **params))
        self.out= nn.Linear(hidden_size, out_size, bias=True)
        for key, value in {'weight':'xavier', 'bias':'zeros'}.items():
            param = getattr(self, key, None)
            if param is not None:
                nn.init.xavier_normal_(param.data, gain=0.5)

    def forward(self,*inputs):
        inputs = torch.cat(inputs,dim=1)
        out = self.first_layer(inputs)
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.out(out)
        return out


class ScaledLinear(nn.Module):
    def __init__(self, input_size, output_size, activation='nn.ReLU', activation_parameters = {}, bias=True):
        super(ScaledLinear, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias)
        if activation_parameters.get('output_size', False) is None:
            activation_parameters['output_size'] = output_size
        self.activation = eval(activation)(**activation_parameters)

    def forward(self,x):
        out = self.activation(self.linear(x))
        return out

def sample_mu_sigma(mu, sigma, net_inputs):
    '''
    Helper method to evaluate mu and sigma
    mu : matrix or nn.Module describing the drift
    sigma : matrix or nn.Module describing the diffusion
    net_inputs : tensor with size (t x dims) 
    '''
    hidden_dim = mu.first_layer.linear.weight.shape[1]
    if hidden_dim == net_inputs.shape[1]:
        mu_hat     = mu(net_inputs)
    else:
        mu_hat     = mu(net_inputs[:,1:])

    sigma_hat = 0 

    if sigma is not None:
            sigma_hat  = sigma(net_inputs)
            

    return mu_hat, sigma_hat.exp()


def kl_div_exact(q_mu, q_sigma, mu_hat, lower_triangular, dt):

    n_var = mu_hat.shape[1]
    mu_diff = (mu_hat - q_mu).unsqueeze(2)

    if not usetorchkl:

        trysomthingnew = False
        if trysomthingnew:

            jitter_value = 1e-6  # Small constant value
            n_var = mu_hat.shape[1]
            mu_diff = (mu_hat - q_mu).unsqueeze(2)

            B = torch.eye(n_var).unsqueeze(0).repeat(mu_hat.shape[0], 1, 1).to(mu_hat.device)
            q_sigma = torch.clamp(q_sigma, min=jitter_value)  # Clamping and exponentiation
            lower_triangular = torch.clamp(lower_triangular, min=jitter_value)  # Clamping and exponentiation
            
            # Adding jitter for numerical stability
            B += jitter_value * torch.eye(n_var, device=B.device)
            
            sigma_inv = torch.cholesky_solve(B, lower_triangular + jitter_value * torch.eye(n_var, device=lower_triangular.device))

            # Using torch.slogdet for stable log determinant calculation
            try:
                # Using torch.slogdet for stable log determinant calculation
                sign1, log_det_sig = torch.slogdet(lower_triangular)
                sign2, log_det_sig_q = torch.slogdet(q_sigma)
                
                # Check the sign to ensure the determinants are positive
                if not (torch.all(sign1 > 0) and torch.all(sign2 > 0)):
                    raise ValueError("Determinants should be positive")

                # ... (rest of your calculation for kl)
            
            except ValueError as e:
                print(f"Error: {e}")
                print("lower_triangular matrix with non-positive determinant:")
                print(lower_triangular)
                print("q_sigma matrix with non-positive determinant:")
                print(q_sigma)
                # Re-raise the exception if you need to stop execution
                raise

            #sign1, log_det_sig = torch.slogdet(lower_triangular)
            #sign2, log_det_sig_q = torch.slogdet(q_sigma)
            
            # Check the sign to ensure the determinants are positive
            #assert torch.all(sign1 > 0) and torch.all(sign2 > 0), "Determinants should be positive"
            
            trace = (torch.diagonal(sigma_inv, dim1=-2, dim2=-1) * torch.diagonal(q_sigma, dim1=-2, dim2=-1)).sum(1)
            intermediate = torch.bmm(sigma_inv, mu_diff)
            result = torch.bmm(mu_diff.transpose(1,2), intermediate)

            kl = 1/2 * (trace + result.squeeze() - n_var + log_det_sig - log_det_sig_q).mean()


        else:

            # we're using cholesky solve to invert the matrix
            # so we want to solve AX= B where B is the identity matrix
            # in this case, A is our cholesky factorization

            B = torch.eye(n_var).unsqueeze(0).repeat(mu_hat.shape[0],1,1).to(mu_hat.device)
            sigma_inv = torch.cholesky_solve(B, lower_triangular)

            det_sig = lower_triangular.det()
            det_sig_q = q_sigma.det()
                

            #kl = 1/2 * ( trace + torch.bmm( mu_diff.permute(0,2,1), torch.bmm(sigma_inv, mu_diff) ) * dt \
            #        + det_sig.log() - det_sig_q.log() ).mean()

            trace =  (torch.diagonal(sigma_inv, dim1=-2, dim2=-1) * torch.diagonal(q_sigma, dim1=-2, dim2=-1)).sum(1)
            intermediate = torch.bmm(sigma_inv, mu_diff)
            result = torch.bmm(mu_diff.transpose(1,2), intermediate)

            kl = 1/2 * (trace +  result.squeeze() - n_var + det_sig.log() - det_sig_q.log()).mean()

    else:
        kl_vec = torch.empty(len(q_mu))
        for i in range(len(q_mu)):
            q = MultivariateNormal(mu_diff[i], q_sigma[i])
            p = MultivariateNormal(mu_hat[i], lower_triangular[i])
            kl_i = F.kl_div(q,p)
            kl_vec[i] = kl_i

        kl = kl_vec.mean()

        print('JIPPYYYYY')
        exit()
    

    return kl

def triangle_vec_to_lower(vec,N):

    # helper method for converting vector to lower triangular matrix
    tri_inds = torch.tril_indices(N,N)

    lower = torch.zeros(vec.shape[0],N,N).to(vec.device)
    lower[:,tri_inds[0,:],tri_inds[1,:]] = vec

    return lower



#initializing:
def setup():

    train_dataloader, ts = OUdata(N=N,paths = paths, X0=X0)
    
    ae    = StochConvAE(in_channels = in_dim, hidden_dim=hidden_dim, n_layers=n_layers,
                        latent_dim=latent_dim).to(device)

    tri_inds = torch.tril_indices(latent_dim,latent_dim)
    upper_tri = torch.eye(latent_dim)[tri_inds[0,:], tri_inds[1,:]]

    mu    = MLP(latent_dim + 1,hidden_dim,depth,latent_dim,activation=act).to(device)
    sigma = MLP(latent_dim + 1,hidden_dim,depth,int((latent_dim+1)*latent_dim/2),activation=act).to(device)


    nn.init.zeros_(mu.out.weight.data)
    nn.init.ones_(mu.out.bias.data)
    nn.init.zeros_(sigma.out.weight.data)
    opt_params = [{'params': mu.parameters(),'lr': lr_mu},
            {'params': sigma.parameters(), 'lr': lr_sigma}, 
            {'params': ae.parameters(), 'lr': lr_ae}]
    optimizer = getattr(optim, optimizer_name)(opt_params)
    scheduler = getattr(optim.lr_scheduler, lr_scheduler)(optimizer, gamma = lr_gamma)
    kl_scheduler = LinearScheduler(kl_iters)

    initialized = {'ae' : ae, 
            'mu'    : mu, 
            'sigma' : sigma, 
            'dt'    : dt,
            'ts'    : ts,
            'train_data' : train_dataloader, 
            #'val_data'   : val_dataloader,
            #'test_data'  : test_dataloader,
            'optimizer'  : optimizer, 
            'scheduler'  : scheduler, 
            'n_epochs'   : num_iters,
            'kl_scheduler' : kl_scheduler
            }
    return initialized



#train-----------------------------------------------------------------------------------------
def train(ae: StochConvAE, 
        mu, 
        sigma, 
        dt, 
        ts,
        train_data, 
        #val_data, 
        optimizer, 
        scheduler, 
        n_epochs,
        kl_scheduler, 
        **kwargs):

    '''
    The main training routine:

    ae : neural network (torch.Module subclass) that represents our autoencoder
    mu : network or parameter that describes the latent drift
    sigma : network or parameter that describes the latent diffusion
    dt : time step
    train_data : dataloader with the training data
    val_data : dataloader with validation data
    optimizer : optimization algorithm torch.optim 
    scheduler : lr decay schedule
    n_epochs  : number of epochs to run
    data_params : parameters associated with the dataset

    returns statistics with respect to training
    '''


    #train_dataset = train_data.dataset.dataset
    inner_num = 1


    nan_problem = []
    losses_train = []
    x_recon = []


    # setup the stats dict
    stats = {'kl': np.Inf, 
            'l2' : np.Inf, 
            'l2_valid': np.Inf, 
            'kl_valid': np.Inf, 
            'mu_mse': 0, 
            'mu_mse_valid': 0, 
            'mu_rel': 0,
            'mu_rel_valid': 0,
            'sde_mse': 0, 
            'sde_mse_valid': 0,
            'sde_rel': 0,
            'sde_rel_valid': 0,
            'val_cond_met': False}

    for epoch in tqdm.tqdm(range(n_epochs)):

        loss_per_iteration = []
        nan_problem_it = []
        kl_loss_it = []
        l2_loss_it = []
        ae.train()
        mu.train()
        sigma.train()
        for idx, frames in enumerate(train_data):     #train_data

            ts     = ts.float().to(device)

            frames = frames[0][0]   
            for _ in range(inner_num):

                optimizer.zero_grad()

                kl_loss, l2_loss,\
                        frames_hat, mu_hat, q_mu, sigma_hat_full, q_sigma_full, inc, z, nan_problem1 = ae.step(frames, ts, dt, mu, sigma)


                kl_loss1, l2_loss1,\
                        _, _, _, _, _, _, _,nan_problem2 = ae.step(frames, ts, dt, mu, sigma, plus_one=True)

                #sigma.data = sigma / sigma.norm(2) * torch.ones(z.shape[1]).norm(2)
                loss = l2_loss + l2_loss1 + 20*sigma_hat_full.norm(1) + kl_loss  * kl_scheduler.val + kl_loss1 * kl_scheduler.val 

                loss_per_iteration.append(l2_loss.item() + l2_loss1.item())                  #(kl_loss.item(), l2_loss.item()))
                if training:
                    loss.backward()
                    optimizer.step()

                # And that's the end of the train routine
            kl_loss_it.append(np.array([kl_loss.detach(), kl_loss1.detach()]))
            l2_loss_it.append(np.array([l2_loss.detach(), l2_loss1.detach()]))
            #nan_problem_it.append(np.array([kl_loss.detach(), kl_loss1.detach()]))

            if kl_loss < stats['kl']:

                stats['kl'] = kl_loss.item()
                stats['mu'] = mu_hat.mean().item()

            if l2_loss < stats['l2']:
                stats['l2'] = l2_loss.item()


            plots_list = [(q_mu[1:]-q_mu[:-1]).detach().cpu().numpy(), mu_hat.detach().cpu().numpy()]
            plot_titles = ['q_mu', 'mu_hat'] 

            if epoch == num_iters-1 :
                x_recon.append(frames_hat.detach().numpy())

        #print(nan_problem_it)
        l2_losss_av = np.mean(l2_loss_it)
        kl_losss_av = np.mean(kl_loss_it)
        losss = np.mean(loss_per_iteration)
        losses_train.append(losss)
        print('Loss: ' + str(losss.item()) + '    kl loss: ' + str(kl_losss_av.item()) + '    l2 loss: ' + str(l2_losss_av) + '    LR sheduler: ' 
                + str(scheduler.get_last_lr()[0]))


        if scheduler and training:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(l2_loss)
            else:
                scheduler.step()

    return stats, x_recon



###################################################################################################################################

training = False
path = 'latentSDEpresent.pth'
num_paths_vis = 5
initialized = setup()

if training:

    stats , x_recon= train(**initialized)

    savedict = {'ae': initialized['ae'].state_dict(), 'mu' : initialized['mu'].state_dict(), 'sigma' : initialized['sigma'].state_dict()}
    torch.save(savedict, path)
else:
    checkpoint = torch.load(path)
    initialized['ae'].load_state_dict(checkpoint['ae'])
    initialized['mu'].load_state_dict(checkpoint['mu'])
    initialized['sigma'].load_state_dict(checkpoint['sigma'])
    num_iters = 1
    initialized['n_epochs'] = 1
    stast, x_recon = train(**initialized)


x_sam = initialized['ae'].sample_new_data(initialized['mu'], initialized['sigma'], initialized['ts'], initialized['dt'], num_sample)

for path in x_sam:

    plt.plot(T_vec, path, color = 'blue')

for i, path in enumerate(initialized['train_data']):
    if i <= num_paths_vis or i >= len(initialized['train_data']) - num_paths_vis:
        plt.plot(T_vec, path[0][0], linewidth = 0.3, color = 'black')


plt.show()


for i, path in enumerate(initialized['train_data']):
    if i <= num_paths_vis or i >= len(initialized['train_data']) - num_paths_vis:
        plt.plot(T_vec, path[0][0], linewidth = 0.7, color = 'black')

for i in range(num_paths_vis):
    plt.plot(T_vec, x_recon[i], color = 'blue', linewidth = 0.7)
    plt.plot(T_vec, x_recon[-i], color = 'orangered', linewidth = 0.7)

plt.savefig('latentSDE_for_OUnewcode.png', dpi=600)
plt.show()