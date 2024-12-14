import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root_dir = "./results/"

td_mpc_no_latent_filenames = [
    "/Pendulum-v1/metrics_td_mpc_no_latent_41.csv",
    "/Pendulum-v1/metrics_td_mpc_no_latent_42.csv",
    "/Pendulum-v1/metrics_td_mpc_no_latent_43.csv",
    "/Pendulum-v1/metrics_td_mpc_no_latent_44.csv",
    "/Pendulum-v1/metrics_td_mpc_no_latent_45.csv",
]

td_mpc_no_latent_ori_filenames = [
    "/Pendulum-v1/metrics_td_mpc_no_latent_ori_41.csv",
    "/Pendulum-v1/metrics_td_mpc_no_latent_ori_42.csv",
    "/Pendulum-v1/metrics_td_mpc_no_latent_ori_43.csv",
    "/Pendulum-v1/metrics_td_mpc_no_latent_ori_44.csv",
    "/Pendulum-v1/metrics_td_mpc_no_latent_ori_45.csv",
]

td_mpc_no_latent_ctr_filenames = [
    "/Pendulum-v1/metrics_td_mpc_no_latent_ctr_41.csv",
    "/Pendulum-v1/metrics_td_mpc_no_latent_ctr_42.csv",
    "/Pendulum-v1/metrics_td_mpc_no_latent_ctr_43.csv",
    "/Pendulum-v1/metrics_td_mpc_no_latent_ctr_44.csv",
    "/Pendulum-v1/metrics_td_mpc_no_latent_ctr_45.csv",
]

# read files
td_mpc_no_latent = [pd.read_csv(root_dir + filename) for filename in td_mpc_no_latent_filenames]
td_mpc_no_latent_ori = [pd.read_csv(root_dir + filename) for filename in td_mpc_no_latent_ori_filenames]
td_mpc_no_latent_ctr = [pd.read_csv(root_dir + filename) for filename in td_mpc_no_latent_ctr_filenames]

# compute mean and std of the episode rewards
td_mpc_no_latent_mean = np.mean([df["episode_reward"] for df in td_mpc_no_latent], axis=0)
td_mpc_no_latent_std = np.std([df["episode_reward"] for df in td_mpc_no_latent], axis=0)
td_mpc_no_latent_ori_mean = np.mean([df["episode_reward"] for df in td_mpc_no_latent_ori], axis=0)
td_mpc_no_latent_ori_std = np.std([df["episode_reward"] for df in td_mpc_no_latent_ori], axis=0)
td_mpc_no_latent_ctr_mean = np.mean([df["episode_reward"] for df in td_mpc_no_latent_ctr], axis=0)
td_mpc_no_latent_ctr_std = np.std([df["episode_reward"] for df in td_mpc_no_latent_ctr], axis=0)

# plot
plt.figure(figsize=(10, 5))
plt.plot(td_mpc_no_latent_mean, label="TD-MPC with residual connection")
plt.fill_between(range(len(td_mpc_no_latent_mean)), td_mpc_no_latent_mean - td_mpc_no_latent_std, td_mpc_no_latent_mean + td_mpc_no_latent_std, alpha=0.3)
plt.plot(td_mpc_no_latent_ori_mean, label="TD-MPC w/o residual connection")
plt.fill_between(range(len(td_mpc_no_latent_ori_mean)), td_mpc_no_latent_ori_mean - td_mpc_no_latent_ori_std, td_mpc_no_latent_ori_mean + td_mpc_no_latent_ori_std, alpha=0.3)
# plt.plot(td_mpc_no_latent_ctr_mean, label="TD-MPC with reward centering")
# plt.fill_between(range(len(td_mpc_no_latent_ctr_mean)), td_mpc_no_latent_ctr_mean - td_mpc_no_latent_ctr_std, td_mpc_no_latent_ctr_mean + td_mpc_no_latent_ctr_std, alpha=0.3)

plt.xlabel("Episode")
plt.ylabel("Episode Return")
plt.title("Pendulum-v1")
plt.legend()
plt.show()