import torch
import torch.nn as nn
import gpytorch
from torch import distributions as pyd
import math
from torch.distributions.utils import _standard_normal
import torch.nn.functional as F
import numpy as np
import re


__REDUCE__ = lambda b: 'mean' if b else 'none'


def orthogonal_init(m):
    """Orthogonal layer initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MLP(nn.Module):
    """Simple multi-layer perceptron (MLP) with ELU activation by default."""
    def __init__(self, input_dim, hidden_dim, output_dim, act_fn=nn.ELU()):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.act_fn = act_fn
        
        self.apply(orthogonal_init)
        
    def forward(self, x):
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))
        x = self.fc3(x)
        return x
    

class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    """
    Stochastic Variational Gaussian Process (SVGP) layer with Grid Interpolation.
    Input is a tensor of shape (batch_size, output_dim, 2).
    Output is a MultivariateNormal distribution (of dimension output_dim).
    """
    def __init__(self, output_dim, grid_bounds=[(-1., 1.), (-1., 1.)], grid_size=16):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size**2, batch_shape=torch.Size([output_dim])
        )

        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=grid_bounds,
                variational_distribution=variational_distribution,
            ), 
            num_tasks=output_dim,
        )
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=2,
                batch_shape=torch.Size([output_dim]),
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            ),
            batch_shape=torch.Size([output_dim])
        )
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([output_dim]))
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, act_fn=nn.ELU()):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act_fn = act_fn
        
        self.apply(orthogonal_init)

    def forward(self, x):
        x = self.act_fn(self.fc1(x))
        return self.fc2(x)
    

class SVDKL(gpytorch.Module):
    """Replace the last layer of a MLP with a SVGP (with SKI) layer."""
    def __init__(self, input_dim, hidden_dim, output_dim, act_fn=nn.ELU(), grid_bound=(-1., 1.)):
        super(SVDKL, self).__init__()
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim, 2, act_fn)

        grid_bounds = [grid_bound,] * 2
        self.gp_layer = GaussianProcessLayer(output_dim=output_dim, grid_bounds=grid_bounds)

        self.grid_bounds = grid_bounds
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(grid_bound[0], grid_bound[1])

        self.apply(orthogonal_init)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.scale_to_bounds(x)
        return self.gp_layer(x)
    

class QNet(nn.Module):
    """Q-network with LayerNorm"""
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.act_fn1 = nn.Tanh()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act_fn2 = nn.ELU()
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self.apply(orthogonal_init)
        self.fc3.weight.data.fill_(0)
        self.fc3.bias.data.fill_(0)
        
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        x = self.act_fn1(self.ln1(self.fc1(x)))
        x = self.act_fn2(self.fc2(x))
        x = self.fc3(x)
        return x
    

class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, action_lower_bound, action_upper_bound):
        super(PolicyNet, self).__init__()
        self.mlp = MLP(input_dim, hidden_dim, output_dim)
        self.action_lower_bound = torch.tensor(action_lower_bound).to(0)
        self.action_upper_bound = torch.tensor(action_upper_bound).to(0)
        self.action_lower_bound.requires_grad = False
        self.action_upper_bound.requires_grad = False
        
    def action_scale(self, act):
        act = torch.tanh(act)
        act = self.action_lower_bound + (self.action_upper_bound - self.action_lower_bound) * (act + 1) / 2
        return act

    def forward(self, x):
        return self.action_scale(self.mlp(x))
    

def linear_schedule(schdl, step):
    """
    Outputs values following a linear decay schedule.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
    raise NotImplementedError(schdl)


class TruncatedNormal(pyd.Normal):
    """
    Utility class implementing the truncated normal distribution.
    Adapted from https://github.com/nicklashansen/tdmpc/
    """
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)
    

def ema(m, m_target, tau):
    """
    Update slow-moving average of online network (target network) at rate tau.
    Adapted from https://github.com/nicklashansen/tdmpc/
    """
    with torch.no_grad():
        for p, p_target in zip(m.parameters(), m_target.parameters()):
            p_target.data.lerp_(p.data, tau)
            

def mse(pred, target, reduce=False):
    """
    Computes the MSE loss between predictions and targets.
    Adapted from https://github.com/nicklashansen/tdmpc/
    """
    return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))

def l1(pred, target, reduce=False):
    """
    Computes the L1-loss between predictions and targets.
    Adapted from https://github.com/nicklashansen/tdmpc/
    Remark. This is used to compute the priority loss for the value functions.
    Intuitively, it should encourage one of the Q network to be more accurate than the other.
    """
    return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))


class Episode(object):
    """
    Storage object for a single episode.
    Adapted from https://github.com/nicklashansen/tdmpc/
    """
    def __init__(self, cfg, init_obs):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float32 if cfg.modality == 'state' else torch.uint8
        self.obs = torch.empty((cfg.episode_length+1, *init_obs.shape), dtype=dtype, device=self.device)
        self.obs[0] = torch.tensor(init_obs, dtype=dtype, device=self.device)
        self.action = torch.empty((cfg.episode_length, cfg.action_dim), dtype=torch.float32, device=self.device)
        self.reward = torch.empty((cfg.episode_length,), dtype=torch.float32, device=self.device)
        self.cumulative_reward = 0
        self.done = False
        self._idx = 0
    
    def __len__(self):
        return self._idx

    @property
    def first(self):
        return len(self) == 0
    
    def __add__(self, transition):
        self.add(*transition)
        return self

    def add(self, obs, action, reward, done):
        self.obs[self._idx+1] = torch.tensor(obs, dtype=self.obs.dtype, device=self.obs.device)
        self.action[self._idx] = action
        self.reward[self._idx] = reward
        self.cumulative_reward += reward
        self.done = done or self._idx + 1 >= self.cfg.episode_length
        self._idx += 1
        

class ReplayBuffer():
    """
    Storage and sampling functionality for training TD-MPC / TOLD.
    The replay buffer is stored in GPU memory when training from state.
    Uses prioritized experience replay by default.
    Adapted from https://github.com/nicklashansen/tdmpc/
    """
    def __init__(self, cfg):
        self.cfg = cfg
        cfg.train_steps = int(eval(cfg.train_steps)) if isinstance(cfg.train_steps, str) else cfg.train_steps
        cfg.episode_length = int(eval(cfg.episode_length))
        cfg.obs_shape = [cfg.obs_dim,]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.capacity = min(cfg.train_steps, cfg.max_buffer_size)
        dtype = torch.float32 if cfg.modality == 'state' else torch.uint8
        obs_shape = (cfg.obs_dim,) if cfg.modality == 'state' else (3, *cfg.obs_shape[-2:])
        self._obs = torch.empty((self.capacity+1, *obs_shape), dtype=dtype, device=self.device)
        self._last_obs = torch.empty((self.capacity//cfg.episode_length, *cfg.obs_shape), dtype=dtype, device=self.device)
        self._action = torch.empty((self.capacity, cfg.action_dim), dtype=torch.float32, device=self.device)
        self._reward = torch.empty((self.capacity,), dtype=torch.float32, device=self.device)
        self._priorities = torch.ones((self.capacity,), dtype=torch.float32, device=self.device)
        self._eps = 1e-6
        self._full = False
        self.idx = 0

    def __add__(self, episode: Episode):
        self.add(episode)
        return self

    def add(self, episode: Episode):
        self._obs[self.idx:self.idx+self.cfg.episode_length] = episode.obs[:-1] if self.cfg.modality == 'state' else episode.obs[:-1, -3:]
        self._last_obs[self.idx//self.cfg.episode_length] = episode.obs[-1]
        self._action[self.idx:self.idx+self.cfg.episode_length] = episode.action
        self._reward[self.idx:self.idx+self.cfg.episode_length] = episode.reward
        if self._full:
            max_priority = self._priorities.max().to(self.device).item()
        else:
            max_priority = 1. if self.idx == 0 else self._priorities[:self.idx].max().to(self.device).item()
        mask = torch.arange(self.cfg.episode_length) >= self.cfg.episode_length-self.cfg.horizon
        new_priorities = torch.full((self.cfg.episode_length,), max_priority, device=self.device)
        new_priorities[mask] = 0
        self._priorities[self.idx:self.idx+self.cfg.episode_length] = new_priorities
        self.idx = (self.idx + self.cfg.episode_length) % self.capacity
        self._full = self._full or self.idx == 0

    def update_priorities(self, idxs, priorities):
        self._priorities[idxs] = priorities.to(self.device) + self._eps

    def _get_obs(self, arr, idxs):
        if self.cfg.modality == 'state':
            return arr[idxs]
        obs = torch.empty((self.cfg.batch_size, 3*self.cfg.frame_stack, *arr.shape[-2:]), dtype=arr.dtype, device=torch.device('cuda'))
        obs[:, -3:] = arr[idxs].cuda()
        _idxs = idxs.clone()
        mask = torch.ones_like(_idxs, dtype=torch.bool)
        for i in range(1, self.cfg.frame_stack):
            mask[_idxs % self.cfg.episode_length == 0] = False
            _idxs[mask] -= 1
            obs[:, -(i+1)*3:-i*3] = arr[_idxs].cuda()
        return obs.float()

    def sample(self):
        probs = (self._priorities if self._full else self._priorities[:self.idx]) ** self.cfg.per_alpha
        probs /= probs.sum()
        total = len(probs)
        idxs = torch.from_numpy(np.random.choice(total, self.cfg.batch_size, p=probs.cpu().numpy(), replace=not self._full)).to(self.device)
        weights = (total * probs[idxs]) ** (-self.cfg.per_beta)
        weights /= weights.max()

        obs = self._get_obs(self._obs, idxs)
        next_obs_shape = self._last_obs.shape[1:] if self.cfg.modality == 'state' else (3*self.cfg.frame_stack, *self._last_obs.shape[-2:])
        next_obs = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, *next_obs_shape), dtype=obs.dtype, device=obs.device)
        action = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, *self._action.shape[1:]), dtype=torch.float32, device=self.device)
        reward = torch.empty((self.cfg.horizon+1, self.cfg.batch_size), dtype=torch.float32, device=self.device)
        for t in range(self.cfg.horizon+1):
            _idxs = idxs + t
            next_obs[t] = self._get_obs(self._obs, _idxs+1)
            action[t] = self._action[_idxs]
            reward[t] = self._reward[_idxs]

        mask = (_idxs+1) % self.cfg.episode_length == 0
        next_obs[-1, mask] = self._last_obs[_idxs[mask]//self.cfg.episode_length].cuda().float()
        if not action.is_cuda:
            action, reward, idxs, weights = \
                action.cuda(), reward.cuda(), idxs.cuda(), weights.cuda()

        return obs, next_obs, action, reward.unsqueeze(2), idxs, weights
