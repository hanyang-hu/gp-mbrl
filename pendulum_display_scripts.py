import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root_dir = "./results/"

task = "Pendulum-v1"

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

exp_name, kernel = "gp_td_mpc_dkl", "SM"
gp_td_mpc_SM_filenames = [f"{task}/metrics_{exp_name}_{kernel}_{seed}.csv" for seed in range(1, 6)]

# read files
td_mpc = [pd.read_csv(root_dir + filename) for filename in td_mpc_filenames]
gp_td_mpc_RBF = [pd.read_csv(root_dir + filename) for filename in gp_td_mpc_RBF_filenames]
gp_td_mpc_matern = [pd.read_csv(root_dir + filename) for filename in gp_td_mpc_matern_filenames]
gp_td_mpc_dkl_RBF = [pd.read_csv(root_dir + filename) for filename in gp_td_mpc_dkl_RBF_filenames]
gp_td_mpc_dkl_matern = [pd.read_csv(root_dir + filename) for filename in gp_td_mpc_dkl_matern_filenames]
gp_td_mpc_dkl_SM = [pd.read_csv(root_dir + filename) for filename in gp_td_mpc_SM_filenames]

# Truncate the first few random rollouts
td_mpc = [df.iloc[4:] for df in td_mpc]
gp_td_mpc_RBF = [df.iloc[4:] for df in gp_td_mpc_RBF]
gp_td_mpc_matern = [df.iloc[4:] for df in gp_td_mpc_matern]
gp_td_mpc_dkl_RBF = [df.iloc[4:] for df in gp_td_mpc_dkl_RBF]
gp_td_mpc_dkl_matern = [df.iloc[4:] for df in gp_td_mpc_dkl_matern]
gp_td_mpc_dkl_SM = [df.iloc[4:] for df in gp_td_mpc_dkl_SM]

# compute mean of the episode rewards
td_mpc_mean = np.mean([df["episode_reward"] for df in td_mpc], axis=0)
gp_td_mpc_RBF_mean = np.mean([df["episode_reward"] for df in gp_td_mpc_RBF], axis=0)
gp_td_mpc_matern_mean = np.mean([df["episode_reward"] for df in gp_td_mpc_matern], axis=0)
gp_td_mpc_dkl_RBF_mean = np.mean([df["episode_reward"] for df in gp_td_mpc_dkl_RBF], axis=0)
gp_td_mpc_dkl_matern_mean = np.mean([df["episode_reward"] for df in gp_td_mpc_dkl_matern], axis=0)
gp_td_mpc_dkl_SM_mean = np.mean([df["episode_reward"] for df in gp_td_mpc_dkl_SM], axis=0)

timesteps = td_mpc[0]["env_step"]

# plot
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(timesteps, td_mpc_mean, label="TD-MPC", color="black", marker="o")
ax1.plot(timesteps, gp_td_mpc_RBF_mean, label="GP-TD-MPC (RBF)", color="red", marker="s")
ax1.plot(timesteps, gp_td_mpc_matern_mean, label="GP-TD-MPC (Matérn-3/2)", color="lime", marker="s")
ax1.plot(timesteps, gp_td_mpc_dkl_RBF_mean, label="GP-TD-MPC (DKL + RBF)", color="maroon", marker="v")
ax1.plot(timesteps, gp_td_mpc_dkl_matern_mean, label="GP-TD-MPC (DKL + Matérn-3/2)", color="green", marker="v")
ax1.plot(timesteps, gp_td_mpc_dkl_SM_mean, label="GP-TD-MPC (DKL + SM)", color="blue", marker="x")

ax1.set_xlabel("Timesteps")
ax1.set_ylabel("Total Reward")
ax1.set_title("Pendulum-v1")

# legend
ax1.legend()

plt.show()

# Compute total runtime and std
td_mpc_runtime = np.mean([df["total_time"].iloc[-1] for df in td_mpc])
td_mpc_runtime_std = np.std([df["total_time"].iloc[-1] for df in td_mpc])
gp_td_mpc_RBF_runtime = np.mean([df["total_time"].iloc[-1] for df in gp_td_mpc_RBF])
gp_td_mpc_RBF_runtime_std = np.std([df["total_time"].iloc[-1] for df in gp_td_mpc_RBF])
gp_td_mpc_matern_runtime = np.mean([df["total_time"].iloc[-1] for df in gp_td_mpc_matern])
gp_td_mpc_matern_runtime_std = np.std([df["total_time"].iloc[-1] for df in gp_td_mpc_matern])
gp_td_mpc_dkl_RBF_runtime = np.mean([df["total_time"].iloc[-1] for df in gp_td_mpc_dkl_RBF])
gp_td_mpc_dkl_RBF_runtime_std = np.std([df["total_time"].iloc[-1] for df in gp_td_mpc_dkl_RBF])
gp_td_mpc_dkl_matern_runtime = np.mean([df["total_time"].iloc[-1] for df in gp_td_mpc_dkl_matern])
gp_td_mpc_dkl_matern_runtime_std = np.std([df["total_time"].iloc[-1] for df in gp_td_mpc_dkl_matern])
gp_td_mpc_dkl_SM_runtime = np.mean([df["total_time"].iloc[-1] for df in gp_td_mpc_dkl_SM])
gp_td_mpc_dkl_SM_runtime_std = np.std([df["total_time"].iloc[-1] for df in gp_td_mpc_dkl_SM])


print(f"TD-MPC: {td_mpc_runtime:.2f} ± {td_mpc_runtime_std:.2f}")
print(f"GP-TD-MPC (RBF): {gp_td_mpc_RBF_runtime:.2f} ± {gp_td_mpc_RBF_runtime_std:.2f}")
print(f"GP-TD-MPC (Matérn-3/2): {gp_td_mpc_matern_runtime:.2f} ± {gp_td_mpc_matern_runtime_std:.2f}")
print(f"GP-TD-MPC (DKL + RBF): {gp_td_mpc_dkl_RBF_runtime:.2f} ± {gp_td_mpc_dkl_RBF_runtime_std:.2f}")
print(f"GP-TD-MPC (DKL + Matérn-3/2): {gp_td_mpc_dkl_matern_runtime:.2f} ± {gp_td_mpc_dkl_matern_runtime_std:.2f}")
print(f"GP-TD-MPC (DKL + SM): {gp_td_mpc_dkl_SM_runtime:.2f} ± {gp_td_mpc_dkl_SM_runtime_std:.2f}")

"""
set CUBLAS_WORKSPACE_CONFIG=:16:8


python train.py --seed 1 --kernel "NA"
python train.py --seed 2 --kernel "NA"
python train.py --seed 3 --kernel "NA"
python train.py --seed 4 --kernel "NA"
python train.py --seed 5 --kernel "NA"

python train.py --seed 1 --kernel "RBF"
python train.py --seed 2 --kernel "RBF"
python train.py --seed 3 --kernel "RBF"
python train.py --seed 4 --kernel "RBF"
python train.py --seed 5 --kernel "RBF"

python train.py --seed 1 --kernel "Matern"
python train.py --seed 2 --kernel "Matern"
python train.py --seed 3 --kernel "Matern"
python train.py --seed 4 --kernel "Matern"
python train.py --seed 5 --kernel "Matern"

python train.py --seed 1 --kernel "SM"
python train.py --seed 2 --kernel "SM"
python train.py --seed 3 --kernel "SM"
python train.py --seed 4 --kernel "SM"
python train.py --seed 5 --kernel "SM"

"""