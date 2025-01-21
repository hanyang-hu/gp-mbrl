import torch
import gpytorch
import numpy as np
import random
import time
import os
import gc
from omegaconf import OmegaConf
import gymnasium as gym
import torch.backends
import tqdm
import warnings
import cv2

from agent import TDMPC, GPTDMPC
from utils import Episode, ReplayBuffer


def update_metric(metrics, new_metrics):
    """Update a dictionary of metrics with new metrics."""
    for k, v in new_metrics.items():
        if k in metrics:
            metrics[k].append(v)
        else:
            metrics[k] = [v,]


def train(cfg_path = "./default.yaml", seed=None):
    cfg = OmegaConf.load(cfg_path)
    
    # Set random seeds for reproducibility
    if seed is not None:
        cfg.seed = seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if cfg.determinism:
        torch.use_deterministic_algorithms(True) # set CUBLAS_WORKSPACE_CONFIG=:4096:8 (for Windows)

    env = gym.make(cfg.task, render_mode="rgb_array")
    obs, _ = env.reset(seed=cfg.seed)

    vis_obs = env.render()
    vis_obs = cv2.resize(vis_obs, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
    # vis_obs = torch.tensor(vis_obs, dtype=torch.float32)
    cfg.obs_shape = list(vis_obs.shape)
    
    cfg.action_dim = env.action_space.shape[0]
    cfg.action_lower_bound = (env.action_space.low).tolist()
    cfg.action_upper_bound = (env.action_space.high).tolist()
    agent = TDMPC(cfg)
    # agent = GPTDMPC(cfg)
    buffer = ReplayBuffer(cfg)

    # Run training (adapted from https://github.com/nicklashansen/tdmpc/)
    train_metrics = {}
    episode_idx, start_time = 0, time.time()
    for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):
        # Collect trajectories
        obs, _ = env.reset(seed=cfg.seed+step)
        vis_obs = env.render()
        vis_obs = cv2.resize(vis_obs, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        # vis_obs = torch.tensor(vis_obs, dtype=torch.float32)
        episode = Episode(cfg, vis_obs)

        st = time.time()
        agent.to_eval() # fix statistics such as layernorm
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(128):
            while not episode.done:
                action = agent.plan(vis_obs, step=step, t0=episode.first)
                reward, done = 0.0, False
                for _ in range(cfg.action_repeat):
                    next_obs, r, d, _, _ = env.step(action.detach().cpu().numpy())
                    next_vis_obs = env.render()
                    next_vis_obs = cv2.resize(next_vis_obs, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
                    # next_vis_obs = torch.tensor(next_vis_obs, dtype=torch.float32)
                    reward += r # accumulate reward over action repeat
                    done = done or d
                    if done:
                        break
                agent.correction(vis_obs, action, reward, next_vis_obs, step+len(episode))
                obs = next_obs
                vis_obs = next_vis_obs
                episode += (vis_obs, action, reward, done)
            assert len(episode) == cfg.episode_length
            agent.to_train()
            buffer += episode
        et = time.time()
        print(f"Collecting episode {episode_idx} took {et-st:.2f}s")

        # Update model
        if step >= cfg.seed_steps:
            num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
            progress_bar = tqdm.tqdm(range(num_updates), desc=f"Episode {episode_idx}")
            for _ in progress_bar:
                loss = agent.update(buffer, step)
                post_dict = {"Weighted Loss": loss["weighted_loss"]}
                progress_bar.set_postfix(post_dict)
            
            if isinstance(agent, GPTDMPC):
                # Update kernel hyperparameters
                progress_bar = tqdm.tqdm(range(cfg.gp_update_num), desc=f"Episode {episode_idx} (Prior Update)")
                for _ in progress_bar:
                    loss = agent.update_gp_prior()
                    post_dict = {"GP Loss": loss}
                    progress_bar.set_postfix(post_dict)

                agent.compute_cache(buffer) # do not train GP prior separately

                # Clear memory
                torch.cuda.empty_cache()
                gc.collect()

        # Log training episode
        episode_idx += 1
        env_step = int(step*cfg.action_repeat)
        common_metrics = {
            'episode': episode_idx,
            'step': step,
            'env_step': env_step,
            'total_time': time.time() - start_time,
            'episode_reward': episode.cumulative_reward,
        }
        update_metric(train_metrics, common_metrics)
        try:
            print(f"Episode {episode_idx}:\n    Env Step: {env_step+len(episode)},\n    Total Time: {time.time() - start_time:.2f}s,\n    Episode Reward: {common_metrics['episode_reward']:.2f}\n    Horizon: {agent._prev_mean.shape}")
            print(f"    Reward Center: {agent.rew_ctr:.2f}")
        except:
            print(f"Episode {episode_idx}:\n    Env Step: {env_step+len(episode)},\n    Total Time: {time.time() - start_time:.2f}s,\n    Episode Reward: {common_metrics['episode_reward']:.2f}")
            print(f"    Reward Center: {agent.rew_ctr:.2f}")

    print('Training completed successfully')

    # Save model
    if cfg.save_path:
        os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
        agent.save(cfg.save_path)
        print(f'Model saved to {cfg.save_path}')
    # Save training metrics to "./results/{cfg.task}/metrics_{cfg.exp_name}_{cfg.seed}.csv"
    os.makedirs(f"./results/{cfg.task}", exist_ok=True)
    with open(f"./results/{cfg.task}/metrics_{cfg.exp_name}_{cfg.seed}.csv", "w") as f:
        f.write("episode,step,env_step,total_time,episode_reward\n")
        for i in range(episode_idx):
            f.write(f"{train_metrics['episode'][i]},{train_metrics['step'][i]},{train_metrics['env_step'][i]},{train_metrics['total_time'][i]},{train_metrics['episode_reward'][i]}\n")
    print(f"Training metrics saved to ./results/{cfg.task}/metrics_{cfg.exp_name}_{cfg.seed}.csv")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    import argparse

    parser = argparse.ArgumentParser(description="Train TD-MPC")
    parser.add_argument("--cfg_path", type=str, default="./default.yaml", help="Path to config file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    train(args.cfg_path, args.seed)