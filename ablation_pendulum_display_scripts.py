import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root_dir = "./results/"

task = "ablation-Pendulum-v1"

exp_name, kernel = "gp_td_mpc", "NA"
td_mpc_filenames = [f"{task}/metrics_{exp_name}_{kernel}_{seed}.csv" for seed in range(1, 6)]

exp_name, kernel = "gp_td_mpc_dkl_2d", "RBF"
gp_td_mpc_dkl_RBF_filenames = [f"{task}/metrics_{exp_name}_{kernel}_{seed}.csv" for seed in range(1, 6)]

exp_name, kernel = "gp_td_mpc_dkl_fps", "RBF"
gp_td_mpc_dkl_fps_RBF_filenames = [f"{task}/metrics_{exp_name}_{kernel}_{seed}.csv" for seed in range(1, 6)]

exp_name, kernel = "gp_td_mpc_dkl_ski", "RBF"
gp_td_mpc_dkl_ski_filenames = [f"{task}/metrics_{exp_name}_{kernel}_{seed}.csv" for seed in range(1, 6)]

# read files
td_mpc = [pd.read_csv(root_dir + filename) for filename in td_mpc_filenames]
gp_td_mpc_dkl_RBF = [pd.read_csv(root_dir + filename) for filename in gp_td_mpc_dkl_RBF_filenames]
gp_td_mpc_dkl_fps_RBF = [pd.read_csv(root_dir + filename) for filename in gp_td_mpc_dkl_fps_RBF_filenames]
gp_td_mpc_dkl_ski = [pd.read_csv(root_dir + filename) for filename in gp_td_mpc_dkl_ski_filenames]

# Truncate the first few random rollouts
td_mpc = [df.iloc[4:] for df in td_mpc]
gp_td_mpc_dkl_RBF = [df.iloc[4:] for df in gp_td_mpc_dkl_RBF]
gp_td_mpc_dkl_fps_RBF = [df.iloc[4:] for df in gp_td_mpc_dkl_fps_RBF]
gp_td_mpc_dkl_ski = [df.iloc[4:] for df in gp_td_mpc_dkl_ski]

# compute mean of the episode rewards
td_mpc_mean = np.mean([df["episode_reward"].cummax() for df in td_mpc], axis=0)
gp_td_mpc_dkl_RBF_mean = np.mean([df["episode_reward"].cummax() for df in gp_td_mpc_dkl_RBF], axis=0)
gp_td_mpc_dkl_fps_RBF_mean = np.mean([df["episode_reward"].cummax() for df in gp_td_mpc_dkl_fps_RBF], axis=0)
gp_td_mpc_dkl_ski_mean = np.mean([df["episode_reward"].cummax() for df in gp_td_mpc_dkl_ski], axis=0)

timesteps = td_mpc[0]["env_step"]


# Set larger fonts
plt.rcParams.update({'font.size': 12})

# Plot performance and number of inducing points in subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.tight_layout(pad=5.0)

# Plot performance
ax1.plot(timesteps, td_mpc_mean, label="TD-MPC", color="black", marker="o")
ax1.plot(timesteps, gp_td_mpc_dkl_RBF_mean, label="GP-TD-MPC (DKL + RBF)", color="maroon", marker="v")
ax1.plot(timesteps, gp_td_mpc_dkl_fps_RBF_mean, label="GP-TD-MPC (DKL + FPS + RBF)", color="blueviolet", marker="^")
ax1.plot(timesteps, gp_td_mpc_dkl_ski_mean, label="GP-TD-MPC (DKL + SKI)", color="darkorange", marker="D")

ax1.set_xlabel("Timesteps")
ax1.set_ylabel("Total Reward")
ax1.set_title("Performance Comparison (Pendulum-v1)")

# Plot average runtime
td_mpc_mean_runtime = np.mean([df["total_time"] for df in td_mpc], axis=0)
gp_td_mpc_dkl_RBF_mean_runtime = np.mean([df["total_time"] for df in gp_td_mpc_dkl_RBF], axis=0)
gp_td_mpc_dkl_fps_RBF_mean_runtime = np.mean([df["total_time"] for df in gp_td_mpc_dkl_fps_RBF], axis=0)
gp_td_mpc_dkl_ski_mean_runtime = np.mean([df["total_time"] for df in gp_td_mpc_dkl_ski], axis=0)

ax2.plot(timesteps, td_mpc_mean_runtime, label="TD-MPC", color="black", marker="o")
ax2.plot(timesteps, gp_td_mpc_dkl_RBF_mean_runtime, label="GP-TD-MPC (DKL + RBF)", color="maroon", marker="v")
ax2.plot(timesteps, gp_td_mpc_dkl_fps_RBF_mean_runtime, label="GP-TD-MPC (DKL + FPS + RBF)", color="blueviolet", marker="^")
ax2.plot(timesteps, gp_td_mpc_dkl_ski_mean_runtime, label="GP-TD-MPC (DKL + SKI)", color="darkorange", marker="D")

ax2.set_xlabel("Timesteps")
ax2.set_ylabel("Total Time")
ax2.set_title("Runtime Comparison (Pendulum-v1)")

# Adjust legend and show plot
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=4)
plt.show()
