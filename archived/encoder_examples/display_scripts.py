import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root_dir = "./results/"

td_mpc_filenames = [
    "/Pendulum-v1/metrics_td_mpc_no_res_41.csv",
    # "/Pendulum-v1/metrics_td_mpc_no_res_42.csv",
    "/Pendulum-v1/metrics_td_mpc_no_res_43.csv",
    "/Pendulum-v1/metrics_td_mpc_no_res_44.csv",
    "/Pendulum-v1/metrics_td_mpc_no_res_45.csv",
]

gp_td_mpc_filenames = [
    "/Pendulum-v1/metrics_gp_td_mpc_ap_41.csv",
    # "/Pendulum-v1/metrics_gp_td_mpc_ap_42.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_ap_43.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_ap_44.csv",
    "/Pendulum-v1/metrics_gp_td_mpc_ap_45.csv",
]

# read files
td_mpc = [pd.read_csv(root_dir + filename) for filename in td_mpc_filenames]
gp_td_mpc = [pd.read_csv(root_dir + filename) for filename in gp_td_mpc_filenames]

# truncate to the minimum episode length
min_len = min([len(df) for df in td_mpc + gp_td_mpc])
# td_mpc = [df.iloc[:min_len] for df in td_mpc]
gp_td_mpc = [df.iloc[:min_len] for df in gp_td_mpc]

# compute mean and std of the episode rewards
td_mpc_mean = np.mean([df["episode_reward"] for df in td_mpc], axis=0)
td_mpc_std = np.std([df["episode_reward"] for df in td_mpc], axis=0)
gp_td_mpc_mean = np.mean([df["episode_reward"] for df in gp_td_mpc], axis=0)
gp_td_mpc_std = np.std([df["episode_reward"] for df in gp_td_mpc], axis=0)
# td_mpc_median = np.median([df["episode_reward"] for df in td_mpc], axis=0)
# td_mpc_mae = np.mean([np.abs(df["episode_reward"] - td_mpc_median) for df in td_mpc], axis=0)
# gp_td_mpc_median = np.median([df["episode_reward"] for df in gp_td_mpc], axis=0)
# gp_td_mpc_mae = np.mean([np.abs(df["episode_reward"] - gp_td_mpc_median) for df in gp_td_mpc], axis=0)

# plot
plt.figure(figsize=(10, 5))
# plt.plot(td_mpc[0]["episode"], td_mpc_median, label="TD-MPC")
# plt.fill_between(td_mpc[0]["episode"], td_mpc_median - td_mpc_mae, td_mpc_median + td_mpc_mae, alpha=0.3)
# plt.plot(gp_td_mpc[0]["episode"], gp_td_mpc_median, label="TD-MPC (no residual dynamics)")
# plt.fill_between(gp_td_mpc[0]["episode"], gp_td_mpc_median - gp_td_mpc_mae, gp_td_mpc_median + gp_td_mpc_mae, alpha=0.3)
plt.plot(td_mpc[0]["episode"], td_mpc_mean, label="TD-MPC")
plt.fill_between(td_mpc[0]["episode"], td_mpc_mean - td_mpc_std, td_mpc_mean + td_mpc_std, alpha=0.3)
plt.plot(gp_td_mpc[0]["episode"], gp_td_mpc_mean, label="GP-TD-MPC")
plt.fill_between(gp_td_mpc[0]["episode"], gp_td_mpc_mean - gp_td_mpc_std, gp_td_mpc_mean + gp_td_mpc_std, alpha=0.3)
# plt.axvline(x=16, color='black', linestyle='--')
plt.xlabel("Episode")
plt.ylabel("Total Return")
plt.title("Pendulum-v1")
plt.legend()
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

python train.py --seed 45
python train.py --seed 44
python train.py --seed 43
python train.py --seed 42
python train.py --seed 41

"""