import utils
from copy import deepcopy
import torch
import numpy as np


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
        self.dm_target = deepcopy(self.dynamics_model)

        # (Double) Q function
        self.q1_net = utils.QNet(obs_dim, cfg.action_dim, cfg.mlp_dim).to(self.device)
        self.q2_net = utils.QNet(obs_dim, cfg.action_dim, cfg.mlp_dim).to(self.device)
        self.q1_target = deepcopy(self.q1_net)
        self.q2_target = deepcopy(self.q2_net)

        # Policy network
        self.pi_net = utils.PolicyNet(obs_dim, cfg.mlp_dim, cfg.action_dim, cfg.action_lower_bound, cfg.action_upper_bound).to(self.device)
        self.pi_target = deepcopy(self.pi_net)
        self.pi_optim = torch.optim.Adam(self.pi_net.parameters(), lr=cfg.lr)

        # Reward network
        self.rew_fn = utils.MLP(obs_dim + cfg.action_dim, cfg.mlp_dim, 1).to(self.device)
        self.rew_fn_target = deepcopy(self.rew_fn)

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
        self.pi_target.eval()
        self.rew_fn.eval()
        self.rew_fn_target.eval()

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
        raise NotImplementedError

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
                z = self.dynamics_model(torch.cat([z, pi_actions[t]], dim=-1))

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
    
    def correction(self, obs, act, rew, next_obs, done):
        return # Default: no correction
    
    def reset_correction(self):
        return

    def next(self, z, a):
        x = torch.cat([z, a], dim=-1)
        return self.dynamics_model(x), self.rew_fn(x)
    
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
        utils.ema(self.dynamics_model, self.dm_target, tau)
        utils.ema(self.q1_net, self.q1_target, tau)
        utils.ema(self.q2_net, self.q2_target, tau)
        utils.ema(self.pi_net, self.pi_target, tau)
        utils.ema(self.rew_fn, self.rew_fn_target, tau)

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

    @torch.no_grad()
    def estimate_value(self, z, actions, horizon):
        G, discount = 0, 1
        for t in range(horizon):
            z, reward = self.next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount
        G += discount * self.Q(z, self.pi(z, self.cfg.min_std)) # terminal value
        return G
    
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

            batch_size = obs.size(0)
            consistency_loss, reward_loss, value_loss, priority_loss = (torch.zeros(batch_size, device=obs.device),) * 4
            
            for t in range(self.cfg.horizon):
                # Predictions
                Q1, Q2 = self.q1_net(z[t], action[t]), self.q2_net(z[t], action[t])
                z[t+1], reward_pred = self.next(z[t], action[t]) # z[t+1] is the prediction

                next_obs = next_obses[t]
                next_z = next_obs # no encoding (supposedly from target encoder)
                td_target = self._td_target(next_obs, reward[t]).detach()

                # Losses
                rho = (self.cfg.rho ** t)
                consistency_loss += rho * torch.mean(utils.mse(z[t+1], next_z), dim=1)
                reward_loss += rho * utils.mse(reward_pred, reward[t]).squeeze(1)
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


class MOPOC(Agent):
    """
    Model-based Planning with Online Correction agent.
    TODO: 
        1. Moidfy estimate_value
        2. Modify update method
        3. Implement correction amd reset_correction method
    """
    pass