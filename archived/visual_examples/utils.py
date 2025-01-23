import torch
import torch.nn as nn
import gpytorch
from torch import distributions as pyd
import math
from torch.distributions.utils import _standard_normal
import torch.nn.functional as F
import numpy as np
import re

import linear_operator
from linear_operator.utils.interpolation import left_interp
import warnings


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


def _get_out_shape(in_shape, layers):
	"""Utility function. Returns the output shape of a network for a given input shape."""
	x = torch.randn(*in_shape).unsqueeze(0)
	return (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x).squeeze(0).shape


class NormalizeImg(nn.Module):
	"""Normalizes pixel observations to [0,1) range."""
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.)


class Flatten(nn.Module):
	"""Flattens its input to a (batched) vector."""
	def __init__(self):
		super().__init__()
		
	def forward(self, x):
		return x.view(x.size(0), -1)


def Encoder(cfg):
	"""Returns a TOLD encoder."""
	if cfg.modality == 'pixels':
		C = int(6)
		layers = [NormalizeImg(),
                  nn.Conv2d(C, cfg.num_channels, 7, stride=2), nn.ReLU(),
				  nn.Conv2d(cfg.num_channels, cfg.num_channels, 5, stride=2), nn.ReLU(),
                  nn.Conv2d(cfg.num_channels, cfg.num_channels, 5, stride=2), nn.ReLU(),
				  nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU()]
		out_shape = _get_out_shape((C, cfg.img_size, cfg.img_size), layers)
		layers.extend([Flatten(), nn.Linear(np.prod(out_shape), cfg.latent_dim)])
	else:
		layers = [nn.Linear(cfg.obs_shape[0], cfg.enc_dim), nn.ELU(),
				  nn.Linear(cfg.enc_dim, cfg.latent_dim)]
	return nn.Sequential(*layers)


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
    

class SVSKILayer(gpytorch.models.ApproximateGP):
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
        self.gp_layer = SVSKILayer(output_dim=output_dim, grid_bounds=grid_bounds)

        self.grid_bounds = grid_bounds
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(grid_bound[0], grid_bound[1])

        self.apply(orthogonal_init)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.scale_to_bounds(x)
        return self.gp_layer(x)
    

class DKLv0(gpytorch.models.ExactGP):
    def __init__(self, input_dim, hidden_dim, output_dim, likelihood=None, grid_bound=(-1., 1.)):
        # Generate dummy data to initialize the model
        dummy_train_inputs = torch.zeros(input_dim, 10)
        dummy_train_targets = torch.zeros(output_dim, 10)
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([output_dim]))
        super(DKLv0, self).__init__(
            train_inputs=dummy_train_inputs, 
            train_targets=dummy_train_targets, 
            likelihood=likelihood
        )

        self.output_dim = output_dim
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim, 2 * output_dim)
        
        grid_bounds = [grid_bound,] * 2
        self.grid_bounds = grid_bounds
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(grid_bound[0], grid_bound[1])

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([output_dim]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=2,
                batch_shape=torch.Size([output_dim]),
            ),
            batch_shape=torch.Size([output_dim]),
        )

        self.apply(orthogonal_init)

        print("DKLv0 model initialized.")

    def forward(self, x):
        """
        Input shape: (input_dim, batch_shape)
        Output mean shape: (output_dim, batch_shape)
        """
        x = self.feature_extractor(x.t()).t()
        x = x.reshape(self.output_dim, 2, -1).permute(0, 2, 1)
        x = self.scale_to_bounds(x)
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    

class KISSGP(gpytorch.models.ExactGP):
    def __init__(self, output_dim, grid_bounds=[(-1., 1.), (-1., 1.)], grid_size=32, likelihood=None):
        super(KISSGP, self).__init__(
            train_inputs=torch.zeros(output_dim, 10, 2),
            train_targets=torch.zeros(output_dim, 10),
            likelihood=likelihood
        )

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([output_dim]))
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=2,
                    batch_shape=torch.Size([output_dim]),
                    # num_mixtures=4,
                ),
                batch_shape=torch.Size([output_dim]),
            ),
            num_dims=2,
            grid_bounds=grid_bounds,
            grid_size=grid_size,
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    


class DKLOVC(gpytorch.models.ExactGP):
    """
    DKL with OVC cache for constant-time update and inference.
    """
    def __init__(
            self, input_dim, hidden_dim, output_dim, likelihood=None,
            num_inducing_points=128, latent_gp_dim=3, error_tol=1e-5
        ):
        # Generate dummy data to initialize the model
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([output_dim]))
        super(DKLOVC, self).__init__(
            train_inputs=None, 
            train_targets=None, 
            likelihood=likelihood
        )

        self.latent_gp_dim = latent_gp_dim
        self.output_dim = output_dim
        
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim, output_dim * latent_gp_dim)
        # self.feature_extractor = lambda x : x.view(1, -1).repeat(4, 1).t()
        
        grid_bound = (-1., 1.) # output of feature extractor is in (-1, 1) (due to the Tanh activation)
        grid_bounds = [grid_bound,] * latent_gp_dim
        self.grid_bounds = grid_bounds
        # self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(grid_bound[0], grid_bound[1])
        self.scale_to_bounds = torch.nn.Tanh()

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([output_dim]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                # nu=2.5,
                ard_num_dims=latent_gp_dim,
                batch_shape=torch.Size([output_dim]),
                # num_mixtures=4,
            ),
            batch_shape=torch.Size([output_dim]),
        )

        self.num_inducing_points = num_inducing_points
        self.pseudo_inputs = None
        self.error_tol = error_tol # error tolerance for pivoted cholesky decomposition

        self.c = torch.nn.Parameter(
            data=torch.zeros(self.output_dim, self.num_inducing_points),
            requires_grad=False
        )
        self.C = torch.nn.Parameter(
            data=torch.zeros(self.output_dim, self.num_inducing_points, self.num_inducing_points),
            requires_grad=False
        )
        self.m_u = torch.nn.Parameter(
            data=torch.zeros(self.output_dim, self.num_inducing_points),
            requires_grad=False
        )
        self.S_u = torch.nn.Parameter(
            data=torch.zeros(self.output_dim, self.num_inducing_points, self.num_inducing_points),
            requires_grad=False
        )
        self.w = torch.nn.Parameter(
            data=torch.zeros(self.output_dim, self.num_inducing_points),
            requires_grad=False
        )

    @torch.no_grad()
    def clear_cache(self, train_inputs, train_targets):
        if not self.eval():
            raise RuntimeError("Model must be in eval mode to update cache.")

        self.c.fill_(0.0)
        self.C.fill_(0.0)
        self.m_u.fill_(0.0)
        self.S_u.fill_(0.0)
        self.w.fill_(0.0)

        with linear_operator.settings.cholesky_jitter(1e-6), linear_operator.settings.cholesky_max_tries(10):
            self.update_cache(train_inputs, train_targets)

    @torch.no_grad()
    def update_cache(self, x, y):
        if not self.eval():
            raise RuntimeError("Model must be in eval mode to update cache.")
        
        x = self.feature_extractor(x).t()
        x = x.reshape(self.output_dim, self.latent_gp_dim, -1).permute(0, 2, 1)
        y = (y - self.mean_module(x)).unsqueeze(-1)
        noise_inv = self.likelihood.noise ** -1
        pseudo_inputs = None

        # try:
        # select the inducing points
        L, pivots = self.covar_module(x).pivoted_cholesky(
            rank=self.num_inducing_points,
            error_tol=self.error_tol,
            return_pivots=True
        )
        rank = L.shape[-1]
        pivots = pivots[:,:rank] # (output_dim, rank)
        pivots = pivots.unsqueeze(-1).expand(self.output_dim, rank, self.latent_gp_dim)
        new_pseudo_inputs = x.gather(1, pivots)

        # OVC update
        K_uv = self.covar_module(new_pseudo_inputs, x)
        c = (K_uv.matmul(y).squeeze(-1).to_dense() * noise_inv)
        C = K_uv.matmul(K_uv.transpose(-1, -2)).to_dense() * noise_inv.unsqueeze(-1)

        # Compute m_u (do not compute S_u)
        K_uu = self.covar_module(new_pseudo_inputs)
        M = K_uu + C
        v = M.solve(c.unsqueeze(-1))
        m_u = K_uu.matmul(v).squeeze(-1).detach().to_dense()

        # Compute cache w
        w = K_uu.solve(m_u.unsqueeze(-1)).squeeze(-1).detach().to_dense()

        # Update pseudo inputs
        pseudo_inputs = new_pseudo_inputs

        # except:
        #     print("Failed to update cache. Skippping update.")
        #     return
        
        # Update cache
        if pseudo_inputs is not None:
            self.c = torch.nn.Parameter(c, requires_grad=False)
            self.C = torch.nn.Parameter(C, requires_grad=False)
            self.m_u = torch.nn.Parameter(m_u, requires_grad=False)
            self.w = torch.nn.Parameter(w, requires_grad=False)
            self.pseudo_inputs = pseudo_inputs


    @torch.no_grad()
    def mean_inference(self, x):
        x = self.feature_extractor(x).t()
        x = x.reshape(self.output_dim, self.latent_gp_dim, -1).permute(0, 2, 1)

        mean = self.mean_module(x)
        covar = self.covar_module(x, self.pseudo_inputs)

        return mean + covar.matmul(self.w.unsqueeze(-1)).squeeze(-1)

    def forward(self, x):
        """
        Input shape: (input_dim, batch_shape)
        Output mean shape: (output_dim, batch_shape)
        """
        x = self.feature_extractor(x).t()
        x = x.reshape(self.output_dim, self.latent_gp_dim, -1).permute(0, 2, 1)
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class DKL(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, likelihood=None, 
                 grid_bound=(-1., 1.), ski_dim=2, grid_size=16):
        super(DKL, self).__init__()
        
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([output_dim]))

        self.output_dim, self.ski_dim = output_dim, ski_dim
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim, ski_dim * output_dim)
        
        grid_bounds = [grid_bound,] * ski_dim
        self.grid_bounds = grid_bounds
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(grid_bound[0], grid_bound[1])

        self.gp_layer = KISSGP(
            output_dim=output_dim, 
            grid_bounds=grid_bounds, 
            grid_size=grid_size,
            likelihood=likelihood
        )

        self.apply(orthogonal_init)

    def forward(self, x):
        """
        Input shape: (input_dim, batch_shape)
        Output mean shape: (output_dim, batch_shape)
        """
        x = self.feature_extractor(x.t()).t()
        x = x.reshape(self.output_dim, self.ski_dim, -1).permute(0, 2, 1)
        x = self.scale_to_bounds(x)
        return self.gp_layer(x)
    
    def mean_inference(self, x):
        x = self.feature_extractor(x.t()).t()
        x = x.reshape(self.output_dim, self.ski_dim, -1).permute(0, 2, 1)
        x = self.scale_to_bounds(x)

        # Compute mean
        test_mean = self.gp_layer.mean_module(x)

        # Fetch WISKI mean cache, do inference
        if self.gp_layer.prediction_strategy:
            mean_cache = self.gp_layer.prediction_strategy.fantasy_mean_cache
            test_interp_indices, test_interp_values = self.gp_layer.covar_module._compute_grid(x)

            return test_mean + left_interp(test_interp_indices, test_interp_values, mean_cache).squeeze(-1)
        else:
            print("Did not find prediction strategy, use the default prediction.")
            return self.gp_layer(x).mean

        
    

def construct_M0(k_ZZ, noises):
    """
    Compute M_0 = (\sigma^2 K_ZZ^{-1} + I)^{-1} = K_ZZ (K_ZZ + \sigma^2 I)^{-1}.
    This is form the simple identity that (I+A^{-1})^{-1} = A(A+I)^{-1}.
    TODO: Try to solve with conjugate gradient
    """
    I = torch.zeros_like(k_ZZ) + torch.eye(k_ZZ.size(-1), device=k_ZZ.device)
    hat_k_ZZ = k_ZZ + noises**2 * I
    inv_hat_k_ZZ = torch.cholesky_inverse(torch.linalg.cholesky(hat_k_ZZ))
    return k_ZZ @ inv_hat_k_ZZ


def get_w_x(idx, value, num_inducing):
    """
    Obtain the sparse interpolation vector w_x (batched) given indices and values.
    """
    if idx.dim() == 2:
        """
        Input: idx (output_dim, num_interp), value (output_dim, num_interp)
        Output: a torch.sparse_ooc_tensor w_x (output_dim, num_inducing) 
        """
        output_dim = idx.size(0)
        batch_idx = torch.arange(output_dim, device=idx.device).unsqueeze(-1).expand_as(idx)
        idx = torch.cat([batch_idx.unsqueeze(-1), idx.unsqueeze(-1)], dim=-1).reshape(-1, 2).t()
        value = value.reshape(-1)
        return torch.sparse_coo_tensor(idx, value, (output_dim, num_inducing))
    elif idx.dim() == 3:
        """
        Input: idx (output_dim, batch_size, num_interp), value (output_dim, batch_size, num_interp)
        Output: a dense tensor w_x (output_dim, batch_size, num_inducing)
        This outputs a dense tensor, which is not memory efficient.
        TODO: Implement a memory efficient version outputing a sparse tensor.
        """
        flat_idx = idx.reshape(-1, idx.size(-1))
        flat_value = value.reshape(-1)
        w_x = get_w_x(flat_idx, flat_value, num_inducing)
        w_x = w_x.to_dense().reshape(idx.size(0), idx.size(1), num_inducing)
        return w_x
    else:
        raise ValueError("Invalid input shape for idx. Should be either 2D or 3D.")
    

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.capacity = min(cfg.train_steps, cfg.max_buffer_size)
        dtype = torch.float32 if cfg.modality == 'state' else torch.uint8
        obs_shape = (cfg.obs_dim,) if cfg.modality == 'state' else (6, *cfg.obs_shape[-2:])
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
        self._obs[self.idx:self.idx+self.cfg.episode_length] = episode.obs[:-1] if self.cfg.modality == 'state' else episode.obs[:-1]
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
        else:
            return arr[idxs].cuda().float()
        obs = torch.empty((self.cfg.batch_size, 6, *arr.shape[-2:]), dtype=arr.dtype, device=torch.device('cuda'))
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
        next_obs_shape = self._last_obs.shape[1:] if self.cfg.modality == 'state' else (6, *self._last_obs.shape[-2:])
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
