import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root_dir = "./results/"

td_mpc_filenames = [
    "/Pendulum-v1/metrics_td_mpc_1.csv",
    "/Pendulum-v1/metrics_td_mpc_2.csv",
    "/Pendulum-v1/metrics_td_mpc_3.csv",
    "/Pendulum-v1/metrics_td_mpc_4.csv",
    "/Pendulum-v1/metrics_td_mpc_5.csv",
    "/Pendulum-v1/metrics_td_mpc_6.csv",
    "/Pendulum-v1/metrics_td_mpc_7.csv",
    "/Pendulum-v1/metrics_td_mpc_8.csv",
    "/Pendulum-v1/metrics_td_mpc_9.csv",
    "/Pendulum-v1/metrics_td_mpc_10.csv",
]

gp_td_mpc_filenames = [
    "/Pendulum-v1/metrics_gp_td_mpc_w_rew_pc_1.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_w_rew_pc_2.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_w_rew_pc_3.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_w_rew_pc_4.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_w_rew_pc_5.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_w_rew_pc_6.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_w_rew_pc_7.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_w_rew_pc_8.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_w_rew_pc_9.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_w_rew_pc_10.csv",
]

gp_td_mpc_wo_rew_filenames = [
    "/Pendulum-v1/metrics_gp_td_mpc_wo_rew_pc_1.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_wo_rew_pc_2.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_wo_rew_pc_3.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_wo_rew_pc_4.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_wo_rew_pc_5.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_wo_rew_pc_6.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_wo_rew_pc_7.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_wo_rew_pc_8.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_wo_rew_pc_9.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_wo_rew_pc_10.csv",
]

gp_td_mpc_matern32_filenames = [
    "/Pendulum-v1/metrics_gp_td_mpc_matern32_pc_1.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_matern32_pc_2.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_matern32_pc_3.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_matern32_pc_4.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_matern32_pc_5.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_matern32_pc_6.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_matern32_pc_7.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_matern32_pc_8.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_matern32_pc_9.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_matern32_pc_10.csv",
]

gp_td_mpc_spectral_mixture_filenames = [
    "/Pendulum-v1/metrics_gp_td_mpc_spectral_mixture_1.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_spectral_mixture_2.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_spectral_mixture_3.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_spectral_mixture_4.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_spectral_mixture_5.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_spectral_mixture_6.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_spectral_mixture_7.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_spectral_mixture_8.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_spectral_mixture_9.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_spectral_mixture_10.csv",
]

# read files
td_mpc = [pd.read_csv(root_dir + filename) for filename in td_mpc_filenames]
gp_td_mpc = [pd.read_csv(root_dir + filename) for filename in gp_td_mpc_filenames]
gp_td_mpc_wo_rew = [pd.read_csv(root_dir + filename) for filename in gp_td_mpc_wo_rew_filenames]
gp_td_mpc_matern32 = [pd.read_csv(root_dir + filename) for filename in gp_td_mpc_matern32_filenames]
gp_td_mpc_spectral_mixture = [pd.read_csv(root_dir + filename) for filename in gp_td_mpc_spectral_mixture_filenames]

# Only maintain the previous 2000 timesteps
td_mpc = [df.iloc[:11] for df in td_mpc]
gp_td_mpc = [df.iloc[:11] for df in gp_td_mpc]
gp_td_mpc_wo_rew = [df.iloc[:11] for df in gp_td_mpc_wo_rew]
gp_td_mpc_matern32 = [df.iloc[:11] for df in gp_td_mpc_matern32]
gp_td_mpc_spectral_mixture = [df.iloc[:11] for df in gp_td_mpc_spectral_mixture]

# compute mean and std of the episode rewards

td_mpc_mean = np.mean([df["episode_reward"].cummax() for df in td_mpc], axis=0)
gp_td_mpc_mean = np.mean([df["episode_reward"].cummax() for df in gp_td_mpc], axis=0)
gp_td_mpc_wo_rew_mean = np.mean([df["episode_reward"].cummax() for df in gp_td_mpc_wo_rew], axis=0)
gp_td_mpc_matern32_mean = np.mean([df["episode_reward"].cummax() for df in gp_td_mpc_matern32], axis=0)
gp_td_mpc_spectral_mixture_mean = np.mean([df["episode_reward"].cummax() for df in gp_td_mpc_spectral_mixture], axis=0)

td_mpc_mad = np.median([df["episode_reward"].cummax() for df in td_mpc], axis=0)
gp_td_mpc_mad = np.median([df["episode_reward"].cummax() for df in gp_td_mpc], axis=0)
gp_td_mpc_wo_rew_mad = np.median([df["episode_reward"].cummax() for df in gp_td_mpc_wo_rew], axis=0)
gp_td_mpc_matern32_mad = np.median([df["episode_reward"].cummax() for df in gp_td_mpc_matern32], axis=0)
gp_td_mpc_spectral_mixture_mad = np.median([df["episode_reward"].cummax() for df in gp_td_mpc_spectral_mixture], axis=0)

td_mpc_mean2 = np.mean([df["episode_reward"] for df in td_mpc], axis=0)
gp_td_mpc_mean2 = np.mean([df["episode_reward"] for df in gp_td_mpc], axis=0)
gp_td_mpc_wo_rew_mean2 = np.mean([df["episode_reward"] for df in gp_td_mpc_wo_rew], axis=0)
gp_td_mpc_matern32_mean2 = np.mean([df["episode_reward"] for df in gp_td_mpc_matern32], axis=0)
gp_td_mpc_spectral_mixture_mean2 = np.mean([df["episode_reward"] for df in gp_td_mpc_spectral_mixture], axis=0)

td_mpc_mad2 = np.median([df["episode_reward"] for df in td_mpc], axis=0)
gp_td_mpc_mad2 = np.median([df["episode_reward"] for df in gp_td_mpc], axis=0)
gp_td_mpc_wo_rew_mad2 = np.median([df["episode_reward"] for df in gp_td_mpc_wo_rew], axis=0)
gp_td_mpc_matern32_mad2 = np.median([df["episode_reward"] for df in gp_td_mpc_matern32], axis=0)
gp_td_mpc_spectral_mixture_mad2 = np.median([df["episode_reward"] for df in gp_td_mpc_spectral_mixture], axis=0)

timesteps = td_mpc[0]["env_step"]

# plot
fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(15, 5), layout="constrained")
plt.subplots_adjust(wspace=0.3)

# First subplot
ax1.plot(timesteps, gp_td_mpc_mean, label="GP-TD-MPC", color='red', marker='v')
ax1.plot(timesteps, gp_td_mpc_wo_rew_mean, label="GP-TD-MPC w/o Reward Correction", color='blue', marker='D')
ax1.plot(timesteps, gp_td_mpc_matern32_mean, label="GP-TD-MPC (with Matérn kernel)", color='green', marker='s')
ax1.plot(timesteps, gp_td_mpc_spectral_mixture_mean, label="GP-TD-MPC (with Spectral Mixture kernel)", color='purple', marker='x')
ax1.plot(timesteps, td_mpc_mean, label="TD-MPC", color='black', marker='o') 
ax1.set_xlabel("Number of Timesteps", fontsize=12)
ax1.set_ylabel("Cummulative Maximum of Return", fontsize=12)

# Second subplot
ax2.plot(timesteps, gp_td_mpc_mean2, label="GP-TD-MPC", color='red', marker='v')
ax2.plot(timesteps, gp_td_mpc_wo_rew_mean2, label="GP-TD-MPC w/o Reward Correction", color='blue', marker='D')
ax2.plot(timesteps, gp_td_mpc_matern32_mean2, label="GP-TD-MPC (with Matérn kernel)", color='green', marker='s')
ax2.plot(timesteps, gp_td_mpc_spectral_mixture_mean2, label="GP-TD-MPC (with Spectral Mixture kernel)", color='purple', marker='x')
ax2.plot(timesteps, td_mpc_mean2, label="TD-MPC", color='black', marker='o') 
ax2.set_xlabel("Number of Timesteps", fontsize=12)
ax2.set_ylabel("Return", fontsize=12)

# # First subplot
# ax1.plot(timesteps, gp_td_mpc_mad, label="GP-TD-MPC", color='red', marker='v')
# ax1.plot(timesteps, gp_td_mpc_wo_rew_mad, label="GP-TD-MPC w/o Reward Correction", color='blue', marker='D')
# ax1.plot(timesteps, gp_td_mpc_matern32_mad, label="GP-TD-MPC (with Matérn kernel)", color='green', marker='s')
# ax1.plot(timesteps, gp_td_mpc_spectral_mixture_mad, label="GP-TD-MPC (with Spectral Mixture kernel)", color='purple', marker='x')
# ax1.plot(timesteps, td_mpc_mad, label="TD-MPC", color='black', marker='o')
# ax1.set_xlabel("Number of Timesteps", fontsize=12)
# ax1.set_ylabel("Cummulative Maximum of Return", fontsize=12)

# # Second subplot
# ax2.plot(timesteps, gp_td_mpc_mad2, label="GP-TD-MPC", color='red', marker='v')
# ax2.plot(timesteps, gp_td_mpc_wo_rew_mad2, label="GP-TD-MPC w/o Reward Correction", color='blue', marker='D')
# ax2.plot(timesteps, gp_td_mpc_matern32_mad2, label="GP-TD-MPC (with Matérn kernel)", color='green', marker='s')
# ax2.plot(timesteps, gp_td_mpc_spectral_mixture_mad2, label="GP-TD-MPC (with Spectral Mixture kernel)", color='purple', marker='x')
# ax2.plot(timesteps, td_mpc_mad2, label="TD-MPC", color='black', marker='o')
# ax2.set_xlabel("Number of Timesteps", fontsize=12)
# ax2.set_ylabel("Return", fontsize=12)

# Common legend
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01), bbox_transform=plt.gcf().transFigure, ncol=5, fontsize=12)
plt.tight_layout(rect=[0, 0.07, 1, 1])

plt.show()

# Compute total runtime and std
td_mpc_runtime = np.mean([df["total_time"].iloc[-1] for df in td_mpc])
td_mpc_runtime_std = np.std([df["total_time"].iloc[-1] for df in td_mpc])
gp_td_mpc_runtime = np.mean([df["total_time"].iloc[-1] for df in gp_td_mpc])
gp_td_mpc_runtime_std = np.std([df["total_time"].iloc[-1] for df in gp_td_mpc])

print(f"TD-MPC runtime: {td_mpc_runtime:.2f} ± {td_mpc_runtime_std:.2f}")
print(f"GP-TD-MPC runtime: {gp_td_mpc_runtime:.2f} ± {gp_td_mpc_runtime_std:.2f}")


"""
set CUBLAS_WORKSPACE_CONFIG=:16:8

python train.py --seed 1
python train.py --seed 2
python train.py --seed 3
python train.py --seed 4
python train.py --seed 5

python train.py --seed 6
python train.py --seed 7
python train.py --seed 8
python train.py --seed 9
python train.py --seed 10

"""