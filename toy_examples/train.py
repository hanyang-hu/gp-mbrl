import torch
import numpy as np
import time
import os
from omegaconf import OmegaConf
import gymnasium as gym
import tqdm

from agent import TDMPC
from utils import Episode, ReplayBuffer


def update_metric(metrics, new_metrics):
    """Update a dictionary of metrics with new metrics."""
    for k, v in new_metrics.items():
        if k in metrics:
            metrics[k].append(v)
        else:
            metrics[k] = [v,]


def evaluate(env, agent, num_episodes, step, episode_length, action_repeat, render):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards = []
    for i in range(num_episodes):
        obs, _ = env.reset()
        done, ep_reward, t = False, 0, 0
        agent.reset_correction()
        while not done and t < episode_length:
            action = agent.plan(obs, eval_mode=True, step=step, t0=t==0)
            for _ in range(action_repeat):
                next_obs, reward, done, _, _ = env.step(action.detach().cpu().numpy())
            agent.correction(obs, action, next_obs, reward, done)
            if render:
                env.render()
            obs = next_obs
            ep_reward += reward
            t += 1
        episode_rewards.append(ep_reward)
    return np.nanmean(episode_rewards)


def train(cfg_path = "./default.yaml"):
    cfg = OmegaConf.load(cfg_path)
    
    # Set random seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = gym.make(cfg.task, render_mode="rgb_array")
    cfg.obs_dim = env.observation_space.shape[0]
    cfg.action_dim = env.action_space.shape[0]
    cfg.action_lower_bound = (env.action_space.low).tolist()
    cfg.action_upper_bound = (env.action_space.high).tolist()
    agent = TDMPC(cfg)
    buffer = ReplayBuffer(cfg)

    # Run training (adapted from https://github.com/nicklashansen/tdmpc/)
    train_metrics = {}
    episode_idx, start_time = 0, time.time()
    for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):
        # Collect trajectory
        obs, _ = env.reset(seed=cfg.seed)
        agent.reset_correction() # do nothing for TD-MPC
        episode = Episode(cfg, obs)
        while not episode.done:
            action = agent.plan(obs, step=step, t0=episode.first)
            for _ in range(cfg.action_repeat):
                next_obs, reward, done, _, _ = env.step(action.detach().cpu().numpy())
            agent.correction(obs, action, next_obs, reward, done) # do nothing for TD-MPC
            obs = next_obs
            episode += (obs, action, reward, done)
        assert len(episode) == cfg.episode_length
        buffer += episode

        # Update model
        if step >= cfg.seed_steps:
            num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
            progress_bar = tqdm.tqdm(range(num_updates), desc=f"Episode {episode_idx}")
            for _ in progress_bar:
                loss = agent.update(buffer, step)
                progress_bar.set_postfix({"Weighted Loss": loss["weighted_loss"]})

        # Log training episode
        episode_idx += 1
        env_step = int(step*cfg.action_repeat)
        common_metrics = {
            'episode': episode_idx,
            'step': step,
            'env_step': env_step,
            'total_time': time.time() - start_time,
            'episode_reward': episode.cumulative_reward
        }
        update_metric(train_metrics, common_metrics)

        # Evaluate and visualize agent periodically
        if cfg.eval and env_step != 0 and env_step % cfg.eval_freq == 0:
            with torch.no_grad():
                eval_env = gym.make(cfg.task, render_mode="human")
                evaluate(eval_env, agent, cfg.eval_episodes, step, int(eval(cfg.val_episode_length)), action_repeat=cfg.action_repeat, render=True)
            print(f"Evaluation:\n    Episode: {episode_idx}, \n    Step: {step},\n    Env Step: {env_step},\n    Total Time: {time.time() - start_time:.2f}s,\n    Episode Reward: {common_metrics['episode_reward']:.2f}")

    print('Training completed successfully')

    # Save model
    if cfg.save_path:
        os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
        agent.save(cfg.save_path)
        print(f'Model saved to {cfg.save_path}')
    # Save training metrics to "./results/{cfg.task}/metrics.csv"
    # create the directory if it does not exist
    os.makedirs(f"./results/{cfg.task}", exist_ok=True)
    with open(f"./results/{cfg.task}/metrics.csv", "w") as f:
        f.write("episode,step,env_step,total_time,episode_reward\n")
        for i in range(episode_idx):
            f.write(f"{train_metrics['episode'][i]},{train_metrics['step'][i]},{train_metrics['env_step'][i]},{train_metrics['total_time'][i]},{train_metrics['episode_reward'][i]}\n")
    print(f"Training metrics saved to ./results/{cfg.task}/metrics.csv")

if __name__ == "__main__":
    train()