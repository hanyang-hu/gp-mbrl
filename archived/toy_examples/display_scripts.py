import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root_dir = "./results/"

td_mpc_filenames = [
    "/Pendulum-v1/metrics_td_mpc_2_41.csv",
    "/Pendulum-v1/metrics_td_mpc_2_42.csv",
    # "/Pendulum-v1/metrics_td_mpc_2_43.csv",
    "/Pendulum-v1/metrics_td_mpc_2_44.csv",
    "/Pendulum-v1/metrics_td_mpc_2_45.csv",
]

td_mpc_ovc_filenames = [
    "/Pendulum-v1/metrics_td_mpc_ovc_2_41.csv",
    "/Pendulum-v1/metrics_td_mpc_ovc_2_42.csv",
    # "/Pendulum-v1/metrics_td_mpc_ovc_2_43.csv",
    "/Pendulum-v1/metrics_td_mpc_ovc_2_44.csv",
    "/Pendulum-v1/metrics_td_mpc_ovc_2_45.csv",
]

# read files
td_mpc = [pd.read_csv(root_dir + filename) for filename in td_mpc_filenames]
td_mpc_ovc = [pd.read_csv(root_dir + filename) for filename in td_mpc_ovc_filenames]

# compute mean and std of the episode rewards
td_mpc_median = np.median([df["episode_reward"] for df in td_mpc], axis=0)
td_mpc_mae = np.mean([np.abs(df["episode_reward"] - td_mpc_median) for df in td_mpc], axis=0)
td_mpc_ovc_median = np.median([df["episode_reward"] for df in td_mpc_ovc], axis=0)
td_mpc_ovc_mae = np.mean([np.abs(df["episode_reward"] - td_mpc_ovc_median) for df in td_mpc_ovc], axis=0)

# plot
plt.figure(figsize=(10, 5))
plt.plot(td_mpc[0]["episode"], td_mpc_median, label="TD-MPC (no latent)")
plt.fill_between(td_mpc[0]["episode"], td_mpc_median - td_mpc_mae, td_mpc_median + td_mpc_mae, alpha=0.3)
plt.plot(td_mpc_ovc[0]["episode"], td_mpc_ovc_median, label="TD-MPC (no latent) with OVC")
plt.fill_between(td_mpc_ovc[0]["episode"], td_mpc_ovc_median - td_mpc_ovc_mae, td_mpc_ovc_median + td_mpc_ovc_mae, alpha=0.3)
# plt.axvline(x=16, color='black', linestyle='--')
plt.xlabel("Episode")
plt.ylabel("Total Return")
plt.title("Pendulum-v1")
plt.legend()
plt.show()

# Compute total runtime and std
td_mpc_runtime = np.mean([df["total_time"].iloc[-1] for df in td_mpc])
td_mpc_runtime_std = np.std([df["total_time"].iloc[-1] for df in td_mpc])
td_mpc_ovc_runtime = np.mean([df["total_time"].iloc[-1] for df in td_mpc_ovc])
td_mpc_ovc_runtime_std = np.std([df["total_time"].iloc[-1] for df in td_mpc_ovc])

print(f"td_mpc (no latent) runtime: {td_mpc_runtime:.2f} ± {td_mpc_runtime_std:.2f}")
print(f"MOPOC-svgp runtime: {td_mpc_ovc_runtime:.2f} ± {td_mpc_ovc_runtime_std:.2f}")


"""
set CUBLAS_WORKSPACE_CONFIG=:16:8

python train.py --seed 41
python train.py --seed 42
python train.py --seed 44
python train.py --seed 45

python train.py --seed 43

"""