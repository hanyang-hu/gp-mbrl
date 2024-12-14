import utils
from copy import deepcopy
import torch
import gpytorch
import numpy as np
import warnings


class Agent():
    """
    Base class for online planning agents (encoder-free for toy examples).
    The planning and loss comptation methods are adapted from https://github.com/nicklashansen/tdmpc/
    """
    def __init__(self, cfg):
        self.cfg = cfg
        obs_dim = cfg.obs_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.std = utils.linear_schedule(cfg.std_schedule, 0)

        # Dynamics model (Default: MLP)
        self.dynamics_model = utils.MLP(
            obs_dim + cfg.action_dim, cfg.mlp_dim, obs_dim
        ).to(self.device)

        # (Double) Q function
        self.q1_net = utils.QNet(obs_dim, cfg.action_dim, cfg.mlp_dim).to(self.device)
        self.q2_net = utils.QNet(obs_dim, cfg.action_dim, cfg.mlp_dim).to(self.device)
        self.q1_target = deepcopy(self.q1_net)
        self.q2_target = deepcopy(self.q2_net)

        # Policy network
        self.pi_net = utils.PolicyNet(obs_dim, cfg.mlp_dim, cfg.action_dim, cfg.action_lower_bound, cfg.action_upper_bound).to(self.device)
        self.pi_optim = torch.optim.Adam(self.pi_net.parameters(), lr=cfg.lr)

        # Reward network
        self.rew_fn = utils.MLP(obs_dim + cfg.action_dim, cfg.mlp_dim, 1).to(self.device)
        self.rew_ctr = 0.0

        train_steps = int(eval(cfg.train_steps)) + int(eval(cfg.episode_length))
        self.memory = {
            "obs": torch.zeros((train_steps, cfg.obs_dim)).to(self.device),
            "act": torch.zeros((train_steps, cfg.action_dim)).to(self.device),
            "next_obs": torch.zeros((train_steps, cfg.obs_dim)).to(self.device),
            "rew": torch.zeros((train_steps, 1)).to(self.device)
        }
        self.current_step = 0

        # Optimizer
        self.optim = torch.optim.Adam(
            [
                {'params': self.dynamics_model.parameters()},
                {'params': self.q1_net.parameters()},
                {'params': self.q2_net.parameters()},
                {'params': self.rew_fn.parameters()}
            ],
            lr=self.cfg.lr
        )

        # Turn to eval
        self.dynamics_model.eval()
        self.q1_net.eval()
        self.q2_net.eval()
        self.q1_target.eval()
        self.q2_target.eval()
        self.pi_net.eval()
        self.rew_fn.eval()

    def save(self, path):
        torch.save(
            {
                'dynamics_model': self.dynamics_model.state_dict(),
                'q1_net': self.q1_net.state_dict(),
                'q2_net': self.q2_net.state_dict(),
                'pi_net': self.pi_net.state_dict(),
                'rew_fn': self.rew_fn.state_dict()
            }, 
            path
        )

    def Q(self, z, a):
        return torch.min(self.q1_net(z, a), self.q2_net(z, a))
    
    def Q_target(self, z, a):
        return torch.min(self.q1_target(z, a), self.q2_target(z, a))
    
    def pi(self, z, std=0):
        mu = self.pi_net(z)
        if std > 0:
            std = torch.ones_like(mu) * std
            return utils.TruncatedNormal(mu, std).sample(clip=0.3)
        return mu
    
    @torch.no_grad()
    def estimate_value(self, z, actions, horizon):
        G, discount = 0, 1
        for t in range(horizon):
            z, reward = self.next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount
        G += discount * self.Q(z, self.pi(z, self.cfg.min_std)) # terminal value
        return G

    @torch.no_grad()
    def plan(self, obs, eval_mode=False, step=None, t0=True):
        """
        Plan next action using TD-MPC inference without latent.
        obs: raw input observation.
        eval_mode: uniform sampling and action noise is disabled during evaluation.
        step: current time step. determines e.g. planning horizon.
        t0: whether current step is the first step of an episode.

        Remark. Must run with no_grad to avoid memory accumulation.
        """
        # Seed steps (return random actions for exploration)
        if step < self.cfg.seed_steps and not eval_mode:
            return torch.empty(self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)

        # Sample policy trajectories (no need to predict reward)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        horizon = int(min(self.cfg.horizon, utils.linear_schedule(self.cfg.horizon_schedule, step)))
        num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
        if num_pi_trajs > 0:
            pi_actions = torch.empty(horizon, num_pi_trajs, self.cfg.action_dim, device=self.device)
            z = obs.repeat(num_pi_trajs, 1)
            for t in range(horizon):
                pi_actions[t] = self.pi(z, self.cfg.min_std)
                z, _ = self.next(z, pi_actions[t])

        # Initialize state and parameters
        z = obs.repeat(self.cfg.num_samples+num_pi_trajs, 1)
        mean = torch.zeros(horizon, self.cfg.action_dim, device=self.device)
        std = 2*torch.ones(horizon, self.cfg.action_dim, device=self.device)
        if not t0 and hasattr(self, '_prev_mean'):
            mean[:-1] = self._prev_mean[1:]

        # Iterate CEM
        for _ in range(self.cfg.iterations):
            actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                torch.randn(horizon, self.cfg.num_samples, self.cfg.action_dim, device=std.device), -1, 1)
            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)

            # Compute elite actions
            value = self.estimate_value(z, actions, horizon).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature*(elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
            _std = _std.clamp_(self.std, 2)
            mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std

        # Outputs
        score = score.squeeze(1).detach().cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        mean, std = actions[0], _std[0]
        a = mean
        if not eval_mode:
            a += std * torch.randn(self.cfg.action_dim, device=std.device)

        return a
    
    def correction(self, obs, act, rew, next_obs, step):
        # Update memory
        self.memory["obs"][step] = torch.tensor(obs).to(self.device)
        self.memory["act"][step] = act.detach().to(self.device)
        self.memory["rew"][step] = torch.tensor(rew).to(self.device)
        self.memory["next_obs"][step] = torch.tensor(next_obs).to(self.device)
        self.current_step = step

        # Reward centering
        self.rew_ctr += self.cfg.rew_beta * (rew - self.rew_ctr)

    def next(self, z, a):
        x = torch.cat([z, a], dim=-1)
        return z.clone().detach() + self.dynamics_model(x), self.rew_fn(x)
    
    def track_q_grad(self, enable=True):
        for m in [self.q1_net, self.q2_net]:
            for p in m.parameters():
                p.requires_grad_(enable)
    
    def update_pi(self, zs):
        """Update policy using a sequence of latent states."""
        self.pi_optim.zero_grad(set_to_none=True)

        self.track_q_grad(False)

        # Loss is a weighted sum of Q-values
        pi_loss = 0.0
        for t, z in enumerate(zs):
            a = self.pi(z, self.cfg.min_std)
            Q = self.Q(z, a)
            pi_loss += -Q.mean() * (self.cfg.rho ** t)

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pi_net.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)

        self.pi_optim.step()

        self.track_q_grad(True)

        return pi_loss.item()

    def to_train(self):
        self.dynamics_model.train()
        self.q1_net.train()
        self.q2_net.train()
        self.pi_net.train()
        self.rew_fn.train()

    def to_eval(self):
        self.dynamics_model.eval()
        self.q1_net.eval()
        self.q2_net.eval()
        self.pi_net.eval()
        self.rew_fn.eval()

    def ema(self, tau):
        utils.ema(self.q1_net, self.q1_target, tau)
        utils.ema(self.q2_net, self.q2_target, tau)

    @torch.no_grad()
    def _td_target(self, next_obs, reward):
        """Compute the TD-target from a reward and the observation at the following time step."""
        next_z = next_obs
        td_target = reward + self.cfg.discount * self.Q_target(next_z, self.pi(next_z, self.cfg.min_std))
        return td_target

    def update(self, replay_buffer, step):
        """Main update function. Corresponds to one iteration of the TOLD model learning."""
        raise NotImplementedError
    

class TDMPC(Agent):
    """
    TD-MPC without latent, adapted from https://github.com/nicklashansen/tdmpc/
    """
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def update(self, replay_buffer, step):
        with torch.autograd.set_detect_anomaly(False):
            obs, next_obses, action, reward, idxs, weights = replay_buffer.sample()

            # Reset action to a leaf node
            action = action.detach()

            # Centering reward signals
            if self.cfg.reward_centering:
                centered_reward = reward - self.rew_ctr
            else:
                centered_reward = reward

            self.optim.zero_grad(set_to_none=True)
            self.std = utils.linear_schedule(self.cfg.std_schedule, step)
            
            # Turn to train
            self.to_train()

            # Do not encode representation
            z = [None] * (self.cfg.horizon + 1)
            z[0] = obs.clone()

            batch_size = obs.size(0)
            consistency_loss, reward_loss, value_loss, priority_loss = (torch.zeros(batch_size, device=obs.device),) * 4
            
            for t in range(self.cfg.horizon):
                # Predictions
                Q1, Q2 = self.q1_net(z[t], action[t]), self.q2_net(z[t], action[t])
                z[t+1], reward_pred = self.next(z[t], action[t]) # z[t+1] is the prediction

                next_obs = next_obses[t]
                next_z = next_obs # no encoding (supposedly from target encoder)
                td_target = self._td_target(next_obs, centered_reward[t]).detach()

                # Losses
                rho = (self.cfg.rho ** t)
                consistency_loss += rho * torch.mean(utils.mse(z[t+1], next_z), dim=1)
                reward_loss += rho * utils.mse(reward_pred, centered_reward[t]).squeeze(1)
                value_loss += rho * (utils.mse(Q1, td_target) + utils.mse(Q2, td_target)).squeeze(1)
                priority_loss += rho * (utils.l1(Q1, td_target) + utils.l1(Q2, td_target)).squeeze(1)

            # Optimize model
            total_loss = self.cfg.consistency_coef * consistency_loss.clamp(max=1e4) + \
                        self.cfg.reward_coef * reward_loss.clamp(max=1e4) + \
                        self.cfg.value_coef * value_loss.clamp(max=1e4)

            weighted_loss = (total_loss * weights).mean()
            weighted_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon)) 
            weighted_loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.dynamics_model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
            torch.nn.utils.clip_grad_norm_(self.q1_net.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
            torch.nn.utils.clip_grad_norm_(self.q2_net.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
            torch.nn.utils.clip_grad_norm_(self.rew_fn.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)

            self.optim.step()
            
            replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())

            # Update policy + target network
            pi_loss = self.update_pi([z_t.detach() for z_t in z])
            if step % self.cfg.update_freq == 0:
                self.ema(self.cfg.tau)

            # Turn to eval
            self.to_eval()

            return {
                'consistency_loss': float(consistency_loss.mean().item()),
                'reward_loss': float(reward_loss.mean().item()),
                'value_loss': float(value_loss.mean().item()),
                'pi_loss': pi_loss,
                'total_loss': float(total_loss.mean().item()),
                'weighted_loss': float(weighted_loss.mean().item())
            }
        

class MOPOCv0(TDMPC):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Deep Dynamics Kernel
        self.dynamics_likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([cfg.obs_dim])).to(self.device)
        self.dynamics_gp = utils.DKLv0(
            cfg.obs_dim + cfg.action_dim, cfg.mlp_dim, cfg.obs_dim,
            likelihood=self.dynamics_likelihood,
        ).to(self.device)
        self.dynamics_gp_mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.dynamics_gp.likelihood, self.dynamics_gp)

        # Deep Reward Kernel
        self.rew_likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([1])).to(self.device)
        self.rew_gp = utils.DKLv0(
            cfg.obs_dim + cfg.action_dim, cfg.mlp_dim, 1,
            likelihood=self.rew_likelihood,
        ).to(self.device)
        self.rew_gp_mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.rew_gp.likelihood, self.rew_gp)

        self.gp_optim = torch.optim.Adam(
            [
                {'params': self.dynamics_gp.feature_extractor.parameters(), 'weight_decay': 1e-4},
                {'params': self.dynamics_gp.mean_module.parameters()},
                {'params': self.dynamics_gp.covar_module.parameters(), 'lr': cfg.lr},
                {'params': self.dynamics_likelihood.parameters(), 'lr': cfg.lr},
                {'params': self.rew_gp.feature_extractor.parameters(), 'weight_decay': 1e-4},
                {'params': self.rew_gp.mean_module.parameters()},
                {'params': self.rew_gp.covar_module.parameters(), 'lr': cfg.lr},
                {'params': self.rew_likelihood.parameters(), 'lr': cfg.lr},
            ],
            lr=self.cfg.lr
        )

        # Turn to eval
        self.dynamics_gp.eval()
        self.dynamics_likelihood.eval()
        self.rew_gp.eval()
        self.rew_likelihood.eval()

        self.cache = False

    def to_eval(self):
        super().to_eval()

        self.dynamics_gp.eval()
        self.dynamics_likelihood.eval()
        self.rew_gp.eval()
        self.rew_likelihood.eval()

    def to_train(self):
        super().to_train()

        self.dynamics_gp.train()
        self.dynamics_likelihood.train()
        self.rew_gp.train()
        self.rew_likelihood.train()

    def update_gp_prior(self):
        self.to_train()

        # Compute GP loss over memory
        inputs = torch.cat([self.memory["obs"][:self.current_step+1], self.memory["act"][:self.current_step+1]], dim=-1)
        dm_targets = self.memory["next_obs"][:self.current_step+1] - self.memory["obs"][:self.current_step+1]
        rew_targets = self.memory["rew"][:self.current_step+1]

        # Subsample data to maintain computational efficiency (TODO: Use InfoRS instead of random sampling)
        subsample_size = min(inputs.size(0), self.cfg.gp_subsample_size)
        idxs = torch.randperm(inputs.size(0))[:subsample_size]
        inputs, dm_targets, rew_targets = inputs[idxs], dm_targets[idxs], rew_targets[idxs]
        
        inputs, dm_targets, rew_targets = inputs.t(), dm_targets.t(), rew_targets.t()

        self.dynamics_gp.set_train_data(inputs, dm_targets, strict=False)
        self.rew_gp.set_train_data(inputs, rew_targets, strict=False)

        for _ in range(self.cfg.gp_update_per_iter):
            self.gp_optim.zero_grad(set_to_none=True)

            dm_gp_loss = -self.dynamics_gp_mll(self.dynamics_gp(inputs), dm_targets).mean()
            rew_gp_loss = -self.rew_gp_mll(self.rew_gp(inputs), rew_targets).mean()

            # Optimize GP
            gp_loss = dm_gp_loss + rew_gp_loss
            gp_loss.backward()
            self.gp_optim.step()

        self.to_eval()

        return gp_loss.item()
    
    @torch.no_grad()
    def construct_L(self):
        self.cache = True

        # Fetch data from memory
        subsample_size = min(self.current_step + 1, self.cfg.mem_subsample_size)
        indices = torch.randperm(self.current_step + 1)[:subsample_size]

        obs = self.memory["obs"][indices]
        act = self.memory["act"][indices]
        next_obs = self.memory["next_obs"][indices]
        rew = self.memory["rew"][indices]

        input = torch.cat([obs, act], dim=-1)

        # Construct dynamics cache
        gp_dm_input = self.dynamics_gp.feature_extractor(input)
        gp_dm_input = gp_dm_input.reshape(self.cfg.obs_dim, 2, -1).permute(0, 2, 1)
        gp_dm_input = self.dynamics_gp.scale_to_bounds(gp_dm_input)
        self.gp_dm_base = gp_dm_input.clone()
        dm_K = self.dynamics_gp.covar_module(gp_dm_input).evaluate()
        noises = (self.dynamics_gp.likelihood.noise).unsqueeze(-1)
        batch_eyes = torch.eye(subsample_size, device=self.device).unsqueeze(0).expand(self.cfg.obs_dim, -1, -1)
        dm_K_hat = dm_K + noises * batch_eyes
        self.dm_L = torch.linalg.cholesky(dm_K_hat)
        v = (next_obs - obs - self.dynamics_model(input)).t().unsqueeze(-1)
        dm_K_hat_inv = torch.cholesky_inverse(self.dm_L)
        self.dm_cache = torch.bmm(dm_K_hat_inv, v).squeeze(-1) # (obs_dim, batch_size)

        # Construct reward cache
        gp_rew_input = self.rew_gp.feature_extractor(input)
        gp_rew_input = gp_rew_input.reshape(1, 2, -1).permute(0, 2, 1)
        gp_rew_input = self.rew_gp.scale_to_bounds(gp_rew_input)
        self.gp_rew_base = gp_rew_input.clone()
        rew_K = self.rew_gp.covar_module(gp_rew_input).evaluate()
        noises = (self.rew_gp.likelihood.noise).unsqueeze(-1)
        batch_eyes = torch.eye(subsample_size, device=self.device).unsqueeze(0).expand(1, -1, -1)
        rew_K_hat = rew_K + noises * batch_eyes
        self.rew_L = torch.linalg.cholesky(rew_K_hat)
        v = (rew - self.rew_fn(input)).t().unsqueeze(-1)
        rew_K_hat_inv = torch.cholesky_inverse(self.rew_L)
        self.rew_cache = torch.bmm(rew_K_hat_inv, v).squeeze(-1) # (1, batch_size)
        
    
    @torch.no_grad()
    def update_dm_cache(self, input, target):
        """
        TODO: Leverage the sparsity to speedup computation
        TODO: (Important) Fix exploding entries.
        """
        gp_input = self.dynamics_gp.feature_extractor(input)
        gp_input = gp_input.reshape(self.cfg.obs_dim, 2, -1).permute(0, 2, 1)
        gp_input = self.dynamics_gp.scale_to_bounds(gp_input)

        idx, value = self.dynamics_gp.covar_module._compute_grid(gp_input) # idx, value are of the shape (output_dim, 1, K) where K is a constant integer
        idx, value = idx.squeeze(1), value.squeeze(1)
        v = target - self.dynamics_model(input) # the correction (residual) term
        w_t = utils.get_w_x(idx, value, self.grid_size**2).to_dense().unsqueeze(-1) 
        Mw_t = torch.bmm(self.dm_M, w_t) 
        self.dm_M = self.dm_M + 1 / (1 + torch.bmm(w_t.permute(0, 2, 1), Mw_t)) * torch.bmm(Mw_t, Mw_t.permute(0, 2, 1)) 
        self.dm_Wv = self.dm_Wv + w_t * v.reshape(-1, 1, 1)
        self.dm_MWv = torch.bmm(self.dm_M, self.dm_Wv)

        print("dot product: ", 1 + torch.bmm(w_t.permute(0, 2, 1), Mw_t))
        print("w_t: ", w_t.norm())
        print("Mw_t: ", Mw_t.norm())
        print("M: ", self.dm_M.norm())
        if torch.isnan(self.dm_MWv).any():
            exit()

    @torch.no_grad()
    def update_rew_cache(self, input, rew):
        """
        TODO: Leverage the sparsity to speedup computation
        """
        gp_input = self.rew_gp.feature_extractor(input)
        gp_input = gp_input.reshape(1, 2, -1).permute(0, 2, 1)
        gp_input = self.rew_gp.scale_to_bounds(gp_input)

        idx, value = self.rew_gp.covar_module._compute_grid(gp_input)
        idx, value = idx.squeeze(1), value.squeeze(1)
        v = rew - self.rew_fn(input)
        w_t = utils.get_w_x(idx, value, self.grid_size**2).to_dense().unsqueeze(-1)
        Mw_t = torch.bmm(self.rew_M, w_t)
        self.rew_M = self.rew_M +  1 / (1 + torch.bmm(w_t.permute(0, 2, 1), Mw_t)) * torch.bmm(Mw_t, Mw_t.permute(0, 2, 1))
        self.rew_Wv = self.rew_Wv + w_t * v
        self.rew_MWv = torch.bmm(self.rew_M, self.rew_Wv)

    @torch.no_grad()
    def correction(self, obs, act, rew, next_obs, step):
        """
        Update cache matrix M_t and W^T v and their product for both dynamics and reward.
        """
        super().correction(obs, act, rew, next_obs, step)

        if True or not self.cache:
            return

        obs = torch.tensor(obs).to(self.device)
        act = act.detach().to(self.device)
        next_obs = torch.tensor(next_obs).to(self.device)
        rew = torch.tensor(rew).to(self.device)

        input = torch.cat([obs, act], dim=-1)

        self.update_dm_cache(input, next_obs)
        self.update_rew_cache(input, rew)        

    @torch.no_grad()
    def corrected_next(self, z, a):
        """
        TODO: Leverage the sparsity to speedup computation
        """
        x = torch.cat([z, a], dim=-1)

        # original predictions
        z_pred = z.clone().detach() + self.dynamics_model(x) 
        r_pred = self.rew_fn(x)

        if not self.cache:
            return z_pred, r_pred

        # dynamics correction
        gp_dm_input = self.dynamics_gp.feature_extractor(x)
        gp_dm_input = gp_dm_input.reshape(self.cfg.obs_dim, 2, -1).permute(0, 2, 1)
        gp_dm_input = self.dynamics_gp.scale_to_bounds(gp_dm_input)

        test_train_covar = self.dynamics_gp.covar_module(gp_dm_input, self.gp_dm_base).evaluate()
        z_corr = torch.bmm(test_train_covar, self.dm_cache.unsqueeze(-1)).squeeze(-1).t()

        # reward correction
        gp_rew_input = self.rew_gp.feature_extractor(x)
        gp_rew_input = gp_rew_input.reshape(1, 2, -1).permute(0, 2, 1)
        gp_rew_input = self.rew_gp.scale_to_bounds(gp_rew_input)

        test_train_covar = self.rew_gp.covar_module(gp_rew_input, self.gp_rew_base).evaluate()
        r_corr = torch.bmm(test_train_covar, self.rew_cache.unsqueeze(-1)).squeeze(-1).t()

        return z_pred + z_corr, r_pred + r_corr

    @torch.no_grad()
    def estimate_value(self, z, actions, horizon):
        """
        Use the corrected_next method to estimate the value.
        """
        G, discount = 0, 1
        for t in range(horizon):
            z, reward = self.corrected_next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount
        G += discount * self.Q(z, self.pi(z, self.cfg.min_std)) # terminal value
        return G


class MOPOC(TDMPC):
    """
    Build on top of TD-MPC with online correction based on Gaussian Processes.
    Update Gaussian Processes when updating the target networks. (i.e. follow cfg.update_freq)

    TODO: (Important) Have numerical issues, pending to fix cache update methods.
    """
    def __init__(self, cfg, grid_size=20):
        super().__init__(cfg)

        self.grid_size = grid_size

        # Deep Dynamics Kernel
        self.dynamics_likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([cfg.obs_dim])).to(self.device)
        self.dynamics_gp = utils.DKL(
            cfg.obs_dim + cfg.action_dim, cfg.mlp_dim, cfg.obs_dim,
            likelihood=self.dynamics_likelihood,
            grid_size=self.grid_size
        ).to(self.device)
        self.dynamics_gp_mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.dynamics_gp.likelihood, self.dynamics_gp)

        # Deep Reward Kernel
        self.rew_likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([1])).to(self.device)
        self.rew_gp = utils.DKL(
            cfg.obs_dim + cfg.action_dim, cfg.mlp_dim, 1,
            likelihood=self.rew_likelihood,
            grid_size=self.grid_size
        ).to(self.device)
        self.rew_gp_mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.rew_gp.likelihood, self.rew_gp)

        self.gp_optim = torch.optim.Adam(
            [
                {'params': self.dynamics_gp.feature_extractor.parameters(), 'weight_decay': 1e-4},
                {'params': self.dynamics_gp.mean_module.parameters()},
                {'params': self.dynamics_gp.covar_module.parameters(), 'lr': cfg.lr},
                {'params': self.dynamics_likelihood.parameters(), 'lr': cfg.lr},
                {'params': self.rew_gp.feature_extractor.parameters(), 'weight_decay': 1e-4},
                {'params': self.rew_gp.mean_module.parameters()},
                {'params': self.rew_gp.covar_module.parameters(), 'lr': cfg.lr},
                {'params': self.rew_likelihood.parameters(), 'lr': cfg.lr},
            ],
            lr=self.cfg.lr
        )

        # Turn to eval
        self.dynamics_gp.eval()
        self.dynamics_likelihood.eval()
        self.rew_gp.eval()
        self.rew_likelihood.eval()

        # Construct batched cache matrix M_0
        self.construct_M0()

    @torch.no_grad()
    def construct_M0(self):
        grid_coor = self.dynamics_gp.covar_module.grid[0] # assumme grid is the same for both dynamics and reward
        Z = torch.cartesian_prod(grid_coor, grid_coor) # obtain inducing points
        
        k_ZZ = self.dynamics_gp.covar_module(Z).to_dense().detach()
        noises = self.dynamics_gp.likelihood.noise.detach().unsqueeze(-1)
        self.dm_M = utils.construct_M0(k_ZZ, noises).to(self.device)
        self.dm_Wv = torch.zeros(self.cfg.obs_dim, Z.shape[0], 1).to(self.device) # caching W^T v \in R^m
        self.dm_MWv = torch.zeros(self.cfg.obs_dim, Z.shape[0], 1).to(self.device) # caching M W^T v \in R^m

        k_ZZ = self.rew_gp.covar_module(Z).to_dense().detach()
        noises = self.rew_gp.likelihood.noise.detach().unsqueeze(-1)
        self.rew_M = utils.construct_M0(k_ZZ, noises).to(self.device)
        self.rew_Wv = torch.zeros(1, Z.shape[0], 1).to(self.device) # caching W^T v \in R^m
        self.rew_MWv = torch.zeros(1, Z.shape[0], 1).to(self.device) # caching M W^T v \in R^m

        # Load memory to cache
        if self.current_step > 0:
            # subsample memory to maintain computational efficiency
            import tqdm
            subsample_size = min(self.current_step + 1, self.cfg.mem_subsample_size)
            indices = torch.randperm(self.current_step + 1)[:subsample_size]
            progress_bar = tqdm.tqdm(range(0, subsample_size), desc="Loading Memory to Caches")
            for i in progress_bar:
                idx = indices[i]
                input = torch.cat([self.memory["obs"][idx], self.memory["act"][idx]], dim=-1)
                obs = self.memory["obs"][idx]
                next_obs = self.memory["next_obs"][idx]
                rew = self.memory["rew"][idx]

                self.update_dm_cache(input, next_obs - obs)
                self.update_rew_cache(input, rew)

            # # TODO: Make the following code applicable
            # # Need to address limitation of dimensions for update_dm_cache / update_rew_cache
            # import time
            # start_time = time.time()
            # inputs = torch.cat([self.memory["obs"][:self.current_step+1], self.memory["act"][:self.current_step+1]], dim=-1)
            # # print(inputs.shape, self.memory["next_obs"][:self.current_step+1].shape, self.memory["rew"][:self.current_step+1].shape)
            # self.update_dm_cache(inputs, self.memory["next_obs"][:self.current_step+1] - self.memory["obs"][:self.current_step+1])
            # self.update_rew_cache(inputs, self.memory["rew"][:self.current_step+1])
            # end_time = time.time()
            # print(f"Loading Memory to Caches: {end_time - start_time:.2f}s")

    @torch.no_grad()
    def update_dm_cache(self, input, target):
        """
        TODO: Leverage the sparsity to speedup computation
        TODO: (Important) Fix exploding entries.
        """
        gp_input = self.dynamics_gp.feature_extractor(input)
        gp_input = gp_input.reshape(self.cfg.obs_dim, 2, -1).permute(0, 2, 1)
        gp_input = self.dynamics_gp.scale_to_bounds(gp_input)

        idx, value = self.dynamics_gp.covar_module._compute_grid(gp_input) # idx, value are of the shape (output_dim, 1, K) where K is a constant integer
        idx, value = idx.squeeze(1), value.squeeze(1)
        v = target - self.dynamics_model(input) # the correction (residual) term
        w_t = utils.get_w_x(idx, value, self.grid_size**2).to_dense().unsqueeze(-1) 
        Mw_t = torch.bmm(self.dm_M, w_t) 
        self.dm_M = self.dm_M + 1 / (1 + torch.bmm(w_t.permute(0, 2, 1), Mw_t)) * torch.bmm(Mw_t, Mw_t.permute(0, 2, 1)) 
        self.dm_Wv = self.dm_Wv + w_t * v.reshape(-1, 1, 1)
        self.dm_MWv = torch.bmm(self.dm_M, self.dm_Wv)

        print("dot product: ", 1 + torch.bmm(w_t.permute(0, 2, 1), Mw_t))
        print("w_t: ", w_t.norm())
        print("Mw_t: ", Mw_t.norm())
        print("M: ", self.dm_M.norm())
        if torch.isnan(self.dm_MWv).any():
            exit()

    @torch.no_grad()
    def update_rew_cache(self, input, rew):
        """
        TODO: Leverage the sparsity to speedup computation
        """
        gp_input = self.rew_gp.feature_extractor(input)
        gp_input = gp_input.reshape(1, 2, -1).permute(0, 2, 1)
        gp_input = self.rew_gp.scale_to_bounds(gp_input)

        idx, value = self.rew_gp.covar_module._compute_grid(gp_input)
        idx, value = idx.squeeze(1), value.squeeze(1)
        v = rew - self.rew_fn(input)
        w_t = utils.get_w_x(idx, value, self.grid_size**2).to_dense().unsqueeze(-1)
        Mw_t = torch.bmm(self.rew_M, w_t)
        self.rew_M = self.rew_M +  1 / (1 + torch.bmm(w_t.permute(0, 2, 1), Mw_t)) * torch.bmm(Mw_t, Mw_t.permute(0, 2, 1))
        self.rew_Wv = self.rew_Wv + w_t * v
        self.rew_MWv = torch.bmm(self.rew_M, self.rew_Wv)

    @torch.no_grad()
    def correction(self, obs, act, rew, next_obs, step):
        """
        Update cache matrix M_t and W^T v and their product for both dynamics and reward.
        """
        super().correction(obs, act, rew, next_obs, step)

        obs = torch.tensor(obs).to(self.device)
        act = act.detach().to(self.device)
        next_obs = torch.tensor(next_obs).to(self.device)
        rew = torch.tensor(rew).to(self.device)

        input = torch.cat([obs, act], dim=-1)

        self.update_dm_cache(input, next_obs)
        self.update_rew_cache(input, rew)        

    @torch.no_grad()
    def corrected_next(self, z, a):
        """
        TODO: Leverage the sparsity to speedup computation
        """
        x = torch.cat([z, a], dim=-1)

        # original predictions
        z_pred = z.clone().detach() + self.dynamics_model(x) 
        r_pred = self.rew_fn(x)

        # dynamics correction
        gp_dm_input = self.dynamics_gp.feature_extractor(x)
        gp_dm_input = gp_dm_input.reshape(self.cfg.obs_dim, 2, -1).permute(0, 2, 1)
        gp_dm_input = self.dynamics_gp.scale_to_bounds(gp_dm_input)

        idx, value = self.dynamics_gp.covar_module._compute_grid(gp_dm_input) # (output_dim, batch_size, K)
        if not torch.any(torch.isnan(value)):
            w_x = utils.get_w_x(idx, value, self.grid_size**2).to_dense() # (output_dim, batch_size, num_inducing)
            z_corr = torch.bmm(w_x, self.dm_MWv).squeeze(-1).t() # (batch_size, output_dim)
        else:
            print("NaN detected in value. Skipping dynamics correction.")
            print("Input: ", x)
            print("Latent: ", self.dynamics_gp.feature_extractor(x))
            z_corr = 0.0
            exit()

        # reward correction
        gp_rew_input = self.rew_gp.feature_extractor(x)
        gp_rew_input = gp_rew_input.reshape(1, 2, -1).permute(0, 2, 1)
        gp_rew_input = self.rew_gp.scale_to_bounds(gp_rew_input)

        # TODO: Device-side assert triggered here, need to fix
        # Currently, running on CPU
        idx, value = self.rew_gp.covar_module._compute_grid(gp_rew_input)
        if not torch.any(torch.isnan(value)):
            w_x = utils.get_w_x(idx, value, self.grid_size**2).to_dense()
            r_corr = torch.bmm(w_x, self.rew_MWv).squeeze(-1).t()
        else:
            print("NaN detected in value. Skipping reward correction.")
            print("Input: ", x)
            print("Latent: ", self.dynamics_gp.feature_extractor(x))
            r_corr = 0.0
            exit()

        return z_pred + z_corr, r_pred + r_corr

    @torch.no_grad()
    def estimate_value(self, z, actions, horizon):
        """
        Use the corrected_next method to estimate the value.
        """
        G, discount = 0, 1
        for t in range(horizon):
            pred_z, reward = self.corrected_next(z, actions[t])
            if torch.isnan(pred_z).any():
                print("NaN detected in corrected_next.")
                print("Input: ", torch.cat([z, actions[t]], dim=-1))
                print("Prediction: ", pred_z)
                exit()
            G += discount * reward
            discount *= self.cfg.discount
        G += discount * self.Q(z, self.pi(z, self.cfg.min_std)) # terminal value
        return G

    def to_eval(self):
        super().to_eval()

        self.dynamics_gp.eval()
        self.dynamics_likelihood.eval()
        self.rew_gp.eval()
        self.rew_likelihood.eval()

    def to_train(self):
        super().to_train()

        self.dynamics_gp.train()
        self.dynamics_likelihood.train()
        self.rew_gp.train()
        self.rew_likelihood.train()

    def update_gp_prior(self):
        self.to_train()

        # Compute GP loss over memory
        inputs = torch.cat([self.memory["obs"][:self.current_step+1], self.memory["act"][:self.current_step+1]], dim=-1)
        dm_targets = self.memory["next_obs"][:self.current_step+1] - self.memory["obs"][:self.current_step+1]
        rew_targets = self.memory["rew"][:self.current_step+1]

        # Subsample data to maintain computational efficiency (TODO: Use InfoRS instead of random sampling)
        subsample_size = min(inputs.size(0), self.cfg.gp_subsample_size)
        idxs = torch.randperm(inputs.size(0))[:subsample_size]
        inputs, dm_targets, rew_targets = inputs[idxs], dm_targets[idxs], rew_targets[idxs]
        
        inputs, dm_targets, rew_targets = inputs.t(), dm_targets.t(), rew_targets.t()

        self.dynamics_gp.set_train_data(inputs, dm_targets, strict=False)
        self.rew_gp.set_train_data(inputs, rew_targets, strict=False)

        for _ in range(self.cfg.gp_update_per_iter):
            self.gp_optim.zero_grad(set_to_none=True)

            dm_gp_loss = -self.dynamics_gp_mll(self.dynamics_gp(inputs), dm_targets).mean()
            rew_gp_loss = -self.rew_gp_mll(self.rew_gp(inputs), rew_targets).mean()

            # Optimize GP
            gp_loss = dm_gp_loss + rew_gp_loss
            gp_loss.backward()
            self.gp_optim.step()

        self.to_eval()

        return gp_loss.item()



class GPTDMPC(Agent):
    """
    Replace the dynamics model and reward function with SVDKL.
    Only access the mean function at inference time.

    Remark. Not working to the best of my knowledge.

    TODO: 
        1. Implement correction and reset_correction method
    """
    def __init__(self, cfg):
        warnings.warn("GPTDMPC is not working as expected (and is deprecated). Use TDMPC/MOPOC instead.")

        import gpytorch

        super().__init__(cfg)

        self.dynamics_model = utils.SVDKL(cfg.obs_dim + cfg.action_dim, cfg.mlp_dim, cfg.obs_dim).to(self.device)
        self.dm_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=cfg.obs_dim).to(self.device)
        self.dm_mll = gpytorch.mlls.VariationalELBO(self.dm_likelihood, self.dynamics_model.gp_layer, num_data=int(eval(cfg.train_steps)))

        self.rew_fn = utils.SVDKL(cfg.obs_dim + cfg.action_dim, cfg.mlp_dim, 1).to(self.device)
        self.rew_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=1).to(self.device)
        self.rew_mll = gpytorch.mlls.VariationalELBO(self.rew_likelihood, self.rew_fn.gp_layer, num_data=int(eval(cfg.train_steps)))

        self.optim = torch.optim.Adam(
            [
                {'params': self.dynamics_model.feature_extractor.parameters()},
                {'params': self.dynamics_model.gp_layer.hyperparameters(), 'lr': cfg.lr * 0.01},
                {'params': self.dynamics_model.gp_layer.variational_parameters()},
                {'params': self.q1_net.parameters()},
                {'params': self.q2_net.parameters()},
                {'params': self.rew_fn.feature_extractor.parameters()},
                {'params': self.rew_fn.gp_layer.hyperparameters(), 'lr': cfg.lr * 0.01},
                {'params': self.rew_fn.gp_layer.variational_parameters()},
            ],
            lr=self.cfg.lr
        )

        # Turn to eval
        self.dynamics_model.eval()
        self.dm_likelihood.eval()
        self.rew_fn.eval()
        self.rew_likelihood.eval()

    def to_eval(self):
        super().to_eval()
        self.dm_likelihood.eval()
        self.rew_likelihood.eval()

    def to_train(self):
        super().to_train()
        self.dm_likelihood.train()
        self.rew_likelihood.train()

    def next(self, z, a):
        """
        Difference to the original method: use SVDKL, only access the mean function at inference time.
        """
        x = torch.cat([z, a], dim=-1)
        training = self.dynamics_model.training
        if training:
            pred_x = self.dynamics_model(x).mean
            pred_r = self.rew_fn(x).mean
        else:
            with gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(30):
                pred_x = self.dynamics_model(x).mean
                pred_r = self.rew_fn(x).mean
        return pred_x, pred_r
    
    def update(self, replay_buffer, step):
        with torch.autograd.set_detect_anomaly(True):
            obs, next_obses, action, reward, idxs, weights = replay_buffer.sample()

            # Reset action to a leaf node
            action = action.detach()

            self.optim.zero_grad(set_to_none=True)
            self.std = utils.linear_schedule(self.cfg.std_schedule, step)
            
            # Turn to train
            self.to_train()

            # Do not encode representation
            z = [None] * (self.cfg.horizon + 1)
            z[0] = obs.clone()

            r_pred = [None] * self.cfg.horizon

            batch_size = obs.size(0)
            consistency_loss, reward_loss, value_loss, priority_loss = (torch.zeros(batch_size, device=obs.device),) * 4
            
            for t in range(self.cfg.horizon):
                # Predictions
                Q1, Q2 = self.q1_net(z[t], action[t]), self.q2_net(z[t], action[t])
                z[t+1], r_pred[t] = self.next(z[t], action[t]) # z[t+1] is the prediction

                next_obs = next_obses[t]
                next_z = next_obs # no encoding (supposedly from target encoder)
                td_target = self._td_target(next_obs, reward[t]).detach()

                # Losses
                rho = (self.cfg.rho ** t)
                # consistency_loss += rho * torch.mean(utils.mse(z[t+1], next_z), dim=1)
                # reward_loss += rho * utils.mse(r_pred[t], reward[t]).squeeze(1)
                value_loss += rho * (utils.mse(Q1, td_target) + utils.mse(Q2, td_target)).squeeze(1)
                priority_loss += rho * (utils.l1(Q1, td_target) + utils.l1(Q2, td_target)).squeeze(1)

            input = torch.cat([z[0], action[0]], dim=-1)
            dm_loss = -self.dm_mll(self.dynamics_model(input), next_obses[0])
            rew_loss = -self.rew_mll(self.rew_fn(input), reward[0])

            # Optimize model
            total_loss = self.cfg.value_coef * value_loss.clamp(max=1e4)

            weighted_loss = (total_loss * weights).mean() + self.cfg.consistency_coef * dm_loss.clamp(max=1e4) + self.cfg.reward_coef * rew_loss.clamp(max=1e4)
            weighted_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon)) 
            weighted_loss.backward()

            # Clip gradients
            # torch.nn.utils.clip_grad_norm_(self.dynamics_model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
            torch.nn.utils.clip_grad_norm_(self.q1_net.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
            torch.nn.utils.clip_grad_norm_(self.q2_net.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
            # torch.nn.utils.clip_grad_norm_(self.rew_fn.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
            torch.nn.utils.clip_grad_norm_(self.dynamics_model.feature_extractor.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
            torch.nn.utils.clip_grad_norm_(self.dynamics_model.gp_layer.hyperparameters(), self.cfg.grad_clip_norm * 0.1, error_if_nonfinite=False)
            torch.nn.utils.clip_grad_norm_(self.dynamics_model.variational_parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
            torch.nn.utils.clip_grad_norm_(self.rew_fn.feature_extractor.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
            torch.nn.utils.clip_grad_norm_(self.rew_fn.gp_layer.hyperparameters(), self.cfg.grad_clip_norm * 0.1, error_if_nonfinite=False)
            torch.nn.utils.clip_grad_norm_(self.rew_fn.variational_parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)

            self.optim.step()
            
            replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())

            # Update policy + target network
            pi_loss = self.update_pi([z_t.detach() for z_t in z])
            if step % self.cfg.update_freq == 0:
                self.ema(self.cfg.tau)

            # Turn to eval
            self.to_eval()

            # warm-up (for any necessary precomputation)
            self.dynamics_model(torch.randn(1, self.cfg.obs_dim + self.cfg.action_dim, device=self.device))
            self.rew_fn(torch.randn(1, self.cfg.obs_dim + self.cfg.action_dim, device=self.device))

            return {
                'consistency_loss': float(consistency_loss.mean().item()),
                'reward_loss': float(reward_loss.mean().item()),
                'value_loss': float(value_loss.mean().item()),
                'pi_loss': pi_loss,
                'total_loss': float(total_loss.mean().item()),
                'weighted_loss': float(weighted_loss.mean().item())
            }