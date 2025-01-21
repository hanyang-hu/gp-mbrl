import utils
from copy import deepcopy
import torch
import gpytorch
import numpy as np
import linear_operator
import gc


class Agent():
    """
    Base class for online planning agents.
    The planning and loss comptation methods are adapted from https://github.com/nicklashansen/tdmpc/
    """
    def __init__(self, cfg):
        self.cfg = cfg
        obs_dim = cfg.latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.std = utils.linear_schedule(cfg.std_schedule, 0)

        # Encoder
        self.encoder = utils.Encoder(cfg).to(self.device)
        self.encoder_target = deepcopy(self.encoder)

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
            "obs": torch.zeros((train_steps, 3, cfg.img_size, cfg.img_size)).to(self.device),
            "act": torch.zeros((train_steps, cfg.action_dim)).to(self.device),
            "next_obs": torch.zeros((train_steps, 3, cfg.img_size, cfg.img_size)).to(self.device),
            "rew": torch.zeros((train_steps, 1)).to(self.device)
        }
        self.current_step = 0

        # Optimizer
        self.optim = torch.optim.Adam(
            [
                {'params': self.encoder.parameters()},
                {'params': self.dynamics_model.parameters()},
                {'params': self.q1_net.parameters()},
                {'params': self.q2_net.parameters()},
                {'params': self.rew_fn.parameters()}
            ],
            lr=self.cfg.lr
        )

        # Turn to eval
        self.encoder.eval()
        self.encoder_target.eval()
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
                'encoder': self.encoder.state_dict(),
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
            z = self.encoder(obs).repeat(num_pi_trajs, 1)
            for t in range(horizon):
                pi_actions[t] = self.pi(z, self.cfg.min_std)
                z, _ = self.next(z, pi_actions[t])

        # Initialize state and parameters
        z = self.encoder(obs).repeat(self.cfg.num_samples+num_pi_trajs, 1)
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
        self.encoder.train()
        self.dynamics_model.train()
        self.q1_net.train()
        self.q2_net.train()
        self.pi_net.train()
        self.rew_fn.train()

    def to_eval(self):
        self.encoder.eval()
        self.dynamics_model.eval()
        self.q1_net.eval()
        self.q2_net.eval()
        self.pi_net.eval()
        self.rew_fn.eval()

    def ema(self, tau):
        utils.ema(self.q1_net, self.q1_target, tau)
        utils.ema(self.q2_net, self.q2_target, tau)
        utils.ema(self.encoder, self.encoder_target, tau)

    @torch.no_grad()
    def _td_target(self, next_obs, reward):
        """Compute the TD-target from a reward and the observation at the following time step."""
        next_z = self.encoder(next_obs)
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

            # Representation
            z = [None] * (self.cfg.horizon + 1)
            z[0] = self.encoder(obs)

            batch_size = obs.size(0)
            consistency_loss, reward_loss, value_loss, priority_loss = (torch.zeros(batch_size, device=obs.device),) * 4
            
            for t in range(self.cfg.horizon):
                # Predictions
                Q1, Q2 = self.q1_net(z[t], action[t]), self.q2_net(z[t], action[t])
                z[t+1], reward_pred = self.next(z[t], action[t]) # z[t+1] is the prediction

                next_obs = next_obses[t]
                next_z = self.encoder_target(next_obs).detach()
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
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
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

            # torch.cuda.empty_cache()
            # gc.collect()

            return {
                'consistency_loss': float(consistency_loss.mean().item()),
                'reward_loss': float(reward_loss.mean().item()),
                'value_loss': float(value_loss.mean().item()),
                'pi_loss': pi_loss,
                'total_loss': float(total_loss.mean().item()),
                'weighted_loss': float(weighted_loss.mean().item())
            }
        

class GPTDMPC(TDMPC):
    def __init__(self, cfg, num_inducing_points=256):
        super().__init__(cfg)

        # Deep Dynamics Kernel
        self.dynamics_likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([cfg.obs_dim])).to(self.device)
        self.dynamics_gp = utils.DKLOVC(
            cfg.obs_dim + cfg.action_dim, cfg.mlp_dim, cfg.obs_dim,
            likelihood=self.dynamics_likelihood,
            num_inducing_points=num_inducing_points
        ).to(self.device)
        self.dynamics_gp_mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.dynamics_gp.likelihood, self.dynamics_gp)

        # Deep Reward Kernel
        self.rew_likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([1])).to(self.device)
        self.rew_gp = utils.DKLOVC(
            cfg.obs_dim + cfg.action_dim, cfg.mlp_dim, 1,
            likelihood=self.rew_likelihood,
            num_inducing_points=num_inducing_points
        ).to(self.device)
        self.rew_gp_mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.rew_gp.likelihood, self.rew_gp)

        self.gp_optim = torch.optim.Adam(
            [
                {'params': self.dynamics_gp.feature_extractor.parameters(), 'weight_decay': 1e-4},
                {'params': self.dynamics_gp.mean_module.parameters()},
                {'params': self.dynamics_gp.covar_module.parameters(), 'lr': cfg.lr * 0.5},
                {'params': self.dynamics_likelihood.parameters(), 'lr': cfg.lr},
                {'params': self.rew_gp.feature_extractor.parameters(), 'weight_decay': 1e-4},
                {'params': self.rew_gp.mean_module.parameters()},
                {'params': self.rew_gp.covar_module.parameters(), 'lr': cfg.lr * 0.5},
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
            # self.gp_optim.zero_grad(set_to_none=True)
            
            # Turn to train
            self.to_train()

            # Do not encode representation
            z = [None] * (self.cfg.horizon + 1)
            z[0] = obs.clone()
            # gp_z = [None] * (self.cfg.horizon + 1)
            # gp_z[0] = obs.clone()

            batch_size = obs.size(0)
            consistency_loss, reward_loss, value_loss, priority_loss = (torch.zeros(batch_size, device=obs.device),) * 4
            # gp_loss = 0.0
            
            for t in range(self.cfg.horizon):
                # Predictions
                Q1, Q2 = self.q1_net(z[t], action[t]), self.q2_net(z[t], action[t])
                z[t+1], reward_pred = self.next(z[t], action[t]) # z[t+1] is the prediction

                next_obs = next_obses[t]
                next_z = next_obs # no encoding (supposedly from target encoder)
                td_target = self._td_target(next_obs, centered_reward[t]).detach()

                # gp_inputs = torch.cat([gp_z[t], action[t]], dim=-1)
                # dm_targets = next_obs - self.dynamics_model(torch.cat([gp_z[t], action[t]], dim=-1)).detach()
                # dm_targets = dm_targets.t()
                # self.dynamics_gp.set_train_data(gp_inputs, dm_targets, strict=False)
                # gp_output = self.dynamics_gp(gp_inputs)
                # gp_z[t+1] = gp_output.mean.t() + self.dynamics_model(torch.cat([gp_z[t], action[t]], dim=-1)).detach() # gp_z[t+1] is the GP corrected mean prediction
                # mll = -self.dynamics_gp_mll(gp_output, dm_targets).mean()

                # Losses
                rho = (self.cfg.rho ** t)
                consistency_loss += rho * torch.mean(utils.mse(z[t+1], next_z), dim=1)
                reward_loss += rho * utils.mse(reward_pred, centered_reward[t]).squeeze(1)
                value_loss += rho * (utils.mse(Q1, td_target) + utils.mse(Q2, td_target)).squeeze(1)
                priority_loss += rho * (utils.l1(Q1, td_target) + utils.l1(Q2, td_target)).squeeze(1)
                # gp_loss += rho * -mll

            # Optimize model
            total_loss = self.cfg.consistency_coef * consistency_loss.clamp(max=1e4) + \
                        self.cfg.reward_coef * reward_loss.clamp(max=1e4) + \
                        self.cfg.value_coef * value_loss.clamp(max=1e4)

            weighted_loss = (total_loss * weights).mean()
            weighted_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon)) 
            weighted_loss.backward()

            # inputs = torch.cat([obs, action[0]], dim=-1)
            # dm_targets = next_obses[0] - obs - self.dynamics_model(inputs).detach()
            # dm_targets = dm_targets.t()
            # self.dynamics_gp.set_train_data(inputs, dm_targets, strict=False)
            # gp_output = self.dynamics_gp(inputs)
            # gp_loss = -self.dynamics_gp_mll(gp_output, dm_targets).mean()

            # gp_loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.dynamics_model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
            torch.nn.utils.clip_grad_norm_(self.q1_net.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
            torch.nn.utils.clip_grad_norm_(self.q2_net.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
            torch.nn.utils.clip_grad_norm_(self.rew_fn.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)

            self.optim.step()
            # self.gp_optim.step()
            
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

    def update_gp_prior(self):
        # Clear (latent) training inputs
        self.dynamics_gp.latent_train_inputs = None
        self.rew_gp.latent_train_inputs = None

        self.to_train()

        # Compute GP loss over memory
        inputs = torch.cat([self.memory["obs"][:self.current_step+1], self.memory["act"][:self.current_step+1]], dim=-1)
        dm_targets = self.memory["next_obs"][:self.current_step+1] - self.memory["obs"][:self.current_step+1] - self.dynamics_model(inputs)
        rew_targets = self.memory["rew"][:self.current_step+1] - self.rew_fn(inputs) 

        # Subsample data to maintain computational efficiency (TODO: Use InfoRS instead of random sampling)
        subsample_size = min(inputs.size(0), self.cfg.gp_subsample_size)
        # weights = torch.linspace(1, 5, steps=inputs.size(0)) # preference for new memories
        # idxs = torch.multinomial(weights, subsample_size, replacement=False)
        idxs = torch.randperm(inputs.size(0))[:subsample_size]
        inputs, dm_targets, rew_targets = inputs[idxs], dm_targets[idxs], rew_targets[idxs]
        
        # inputs, dm_targets, rew_targets = inputs.t(), dm_targets.t(), rew_targets.t()
        dm_targets, rew_targets = dm_targets.t(), rew_targets.t()

        self.dynamics_gp.set_train_data(inputs, dm_targets, strict=False)
        # self.rew_gp.set_train_data(inputs, rew_targets, strict=False)

        for _ in range(self.cfg.gp_update_per_iter):
            self.gp_optim.zero_grad(set_to_none=True)

            dm_gp_loss = -self.dynamics_gp_mll(self.dynamics_gp(inputs), dm_targets).mean()
            # rew_gp_loss = -self.rew_gp_mll(self.rew_gp(inputs), rew_targets).mean()

            # Optimize GP
            gp_loss = dm_gp_loss #+ rew_gp_loss
            gp_loss.backward()
            self.gp_optim.step()

        self.to_eval()

        return gp_loss.item()
    
    @torch.no_grad()
    def compute_cache(self, replay_buffer):
        print("Computing cache...")
        self.cache = True 

        # Fetch data from memory
        window_size = self.current_step+1 # all data in the memory
        obs = self.memory["obs"][max(0, self.current_step+1-window_size):self.current_step+1]
        act = self.memory["act"][max(0, self.current_step+1-window_size):self.current_step+1]
        next_obs = self.memory["next_obs"][max(0, self.current_step+1-window_size):self.current_step+1]
        rew = self.memory["rew"][max(0, self.current_step+1-window_size):self.current_step+1]
        # temp_batch_size = replay_buffer.cfg.batch_size
        # replay_buffer.cfg.batch_size = window_size
        # obs, next_obses, action, reward, _, _ = replay_buffer.sample() # sample from the replay buffer
        # act, next_obs, rew = action[0], next_obses[0], reward[0]
        # replay_buffer.cfg.batch_size = temp_batch_size

        # Compute input and targets
        inputs = torch.cat([obs, act], dim=-1)
        dm_targets = next_obs - obs - self.dynamics_model(inputs)
        rew_targets = rew - self.rew_fn(inputs)
        dm_targets, rew_targets = dm_targets.t(), rew_targets.t()

        # Set train data and compute cache
        with linear_operator.settings.cholesky_jitter(1e-2):
            self.dynamics_gp.clear_cache(inputs, dm_targets)  
            print("Dynamics GP Rank: ", self.dynamics_gp.m_u.shape)

            # self.rew_gp.clear_cache(inputs, rew_targets)
            # print("Reward GP Rank: ", self.rew_gp.m_u.shape)

    @torch.no_grad()
    def correction(self, obs, act, rew, next_obs, step):
        super().correction(obs, act, rew, next_obs, step)

        # obs = torch.tensor(obs).to(self.device)
        # act = act.detach().to(self.device)
        # next_obs = torch.tensor(next_obs).to(self.device)
        # rew = torch.tensor(rew).to(self.device)

        # input = torch.cat([obs, act], dim=-1)
        # dm_target = next_obs - obs - self.dynamics_model(input)
        # rew_target = rew - self.rew_fn(input)
        # # dm_target, rew_target = dm_target.t(), rew_target.t()
        # input, dm_target, rew_target = input.unsqueeze(-1).t(), dm_target.unsqueeze(-1), rew_target.unsqueeze(-1)

        # with linear_operator.settings.cholesky_jitter(1e-3):
        #     try:
        #         self.dynamics_gp.update_cache(input, dm_target)
        #         self.rew_gp.update_cache(input, rew_target)  
        #     except:
        #         warnings.warn("Cholesky decomposition failed. Skipping cache update.")  

    @torch.no_grad()
    def corrected_next(self, z, a):
        x = torch.cat([z, a], dim=-1)

        # original predictions
        z_pred = z.clone().detach() + self.dynamics_model(x) 
        r_pred = self.rew_fn(x)

        if not self.cache:
            return z_pred, r_pred

        # gp orrections
        z_corr = self.dynamics_gp.mean_inference(x).t()
        # r_corr = self.rew_gp.mean_inference(x).t()

        return z_pred + z_corr, r_pred #+ r_corr

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

