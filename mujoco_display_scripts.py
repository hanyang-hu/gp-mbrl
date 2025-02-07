import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root_dir = "./results/"

task = "Pusher-v5"

exp_name, kernel = "gp_td_mpc", "NA"
td_mpc_filenames = [f"{task}/metrics_{exp_name}_{kernel}_{seed}.csv" for seed in range(1, 6)]

exp_name, kernel = "gp_td_mpc", "RBF"
gp_td_mpc_RBF_filenames = [f"{task}/metrics_{exp_name}_{kernel}_{seed}.csv" for seed in range(1, 6)]

exp_name, kernel = "gp_td_mpc", "Matern"
gp_td_mpc_matern_filenames = [f"{task}/metrics_{exp_name}_{kernel}_{seed}.csv" for seed in range(1, 6)]

exp_name, kernel = "gp_td_mpc_dkl", "RBF"
gp_td_mpc_dkl_RBF_filenames = [f"{task}/metrics_{exp_name}_{kernel}_{seed}.csv" for seed in range(1, 6)]

exp_name, kernel = "gp_td_mpc_dkl", "Matern"
gp_td_mpc_dkl_matern_filenames = [f"{task}/metrics_{exp_name}_{kernel}_{seed}.csv" for seed in range(1, 6)]

# read files
td_mpc = [pd.read_csv(root_dir + filename) for filename in td_mpc_filenames]
gp_td_mpc_RBF = [pd.read_csv(root_dir + filename) for filename in gp_td_mpc_RBF_filenames]
gp_td_mpc_matern = [pd.read_csv(root_dir + filename) for filename in gp_td_mpc_matern_filenames]
gp_td_mpc_dkl_RBF = [pd.read_csv(root_dir + filename) for filename in gp_td_mpc_dkl_RBF_filenames]
gp_td_mpc_dkl_matern = [pd.read_csv(root_dir + filename) for filename in gp_td_mpc_dkl_matern_filenames]

# Truncate the first few random rollouts
td_mpc = [df.iloc[4:] for df in td_mpc]
gp_td_mpc_RBF = [df.iloc[4:] for df in gp_td_mpc_RBF]
gp_td_mpc_matern = [df.iloc[4:] for df in gp_td_mpc_matern]
gp_td_mpc_dkl_RBF = [df.iloc[4:] for df in gp_td_mpc_dkl_RBF]
gp_td_mpc_dkl_matern = [df.iloc[4:] for df in gp_td_mpc_dkl_matern]

# compute mean of the episode rewards
td_mpc_mean = np.mean([df["episode_reward"] for df in td_mpc], axis=0)
gp_td_mpc_RBF_mean = np.mean([df["episode_reward"] for df in gp_td_mpc_RBF], axis=0)
gp_td_mpc_matern_mean = np.mean([df["episode_reward"] for df in gp_td_mpc_matern], axis=0)
gp_td_mpc_dkl_RBF_mean = np.mean([df["episode_reward"] for df in gp_td_mpc_dkl_RBF], axis=0)
gp_td_mpc_dkl_matern_mean = np.mean([df["episode_reward"] for df in gp_td_mpc_dkl_matern], axis=0)

timesteps = td_mpc[0]["env_step"]

# plot
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(timesteps, td_mpc_mean, label="TD-MPC", color="black", marker="o")
ax1.plot(timesteps, gp_td_mpc_RBF_mean, label="GP-TD-MPC (RBF)", color="red", marker="s")
ax1.plot(timesteps, gp_td_mpc_matern_mean, label="GP-TD-MPC (Matérn-3/2)", color="lime", marker="s")
# ax1.plot(timesteps, gp_td_mpc_dkl_RBF_mean, label="GP-TD-MPC (DKL + RBF)", color="maroon", marker="v")
# ax1.plot(timesteps, gp_td_mpc_dkl_matern_mean, label="GP-TD-MPC (DKL + Matérn-3/2)", color="green", marker="v")

ax1.set_xlabel("Timesteps")
ax1.set_ylabel("Total Reward")
ax1.set_title("Pendulum-v1")

# legend
ax1.legend()

plt.show()


"""
export CUBLAS_WORKSPACE_CONFIG=:16:8

python train.py --seed 1 --cfg_path "./pusher.yaml"
python train.py --seed 2 --cfg_path "./pusher.yaml"
python train.py --seed 3 --cfg_path "./pusher.yaml"
python train.py --seed 4 --cfg_path "./pusher.yaml"
python train.py --seed 5 --cfg_path "./pusher.yaml"

"""