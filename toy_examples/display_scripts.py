import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root_dir = "./results/"

td_mpc_no_latent_filenames = [
    "/Pendulum-v1/metrics_td_mpc_no_latent_41.csv",
    # "/Pendulum-v1/metrics_td_mpc_no_latent_42.csv",
    # "/Pendulum-v1/metrics_td_mpc_no_latent_43.csv",
    # "/Pendulum-v1/metrics_td_mpc_no_latent_44.csv",
    # "/Pendulum-v1/metrics_td_mpc_no_latent_45.csv",
]

mopoc_v0_filenames = [
    "/Pendulum-v1/metrics_mopoc_v1_41.csv",
    # "/Pendulum-v1/metrics_mopoc_v1_42.csv",
    # "/Pendulum-v1/metrics_mopoc_v1_43.csv",
    # "/Pendulum-v1/metrics_mopoc_v1_44.csv",
    # "/Pendulum-v1/metrics_mopoc_v1_45.csv",
]

# read files
td_mpc_no_latent = [pd.read_csv(root_dir + filename) for filename in td_mpc_no_latent_filenames]
mopoc_v0 = [pd.read_csv(root_dir + filename) for filename in mopoc_v0_filenames]

# compute mean and std of the episode rewards
td_mpc_no_latent_median = np.median([df["episode_reward"] for df in td_mpc_no_latent], axis=0)
td_mpc_no_latent_mae = np.mean([np.abs(df["episode_reward"] - td_mpc_no_latent_median) for df in td_mpc_no_latent], axis=0)
mopoc_v0_median = np.median([df["episode_reward"] for df in mopoc_v0], axis=0)
mopoc_v0_mae = np.mean([np.abs(df["episode_reward"] - mopoc_v0_median) for df in mopoc_v0], axis=0)

# plot
plt.figure(figsize=(10, 5))
plt.plot(td_mpc_no_latent[0]["episode"], td_mpc_no_latent_median, label="TD-MPC (no latent)")
plt.fill_between(td_mpc_no_latent[0]["episode"], td_mpc_no_latent_median - td_mpc_no_latent_mae, td_mpc_no_latent_median + td_mpc_no_latent_mae, alpha=0.3)
plt.plot(mopoc_v0[0]["episode"], mopoc_v0_median, label="MOPOC-v0")
plt.fill_between(mopoc_v0[0]["episode"], mopoc_v0_median - mopoc_v0_mae, mopoc_v0_median + mopoc_v0_mae, alpha=0.3)
plt.xlabel("Episode")
plt.ylabel("Total Return")
plt.title("Pendulum-v1")
plt.legend()
plt.show()

# Compute total runtime and std
td_mpc_no_latent_runtime = np.mean([df["total_time"].iloc[-1] for df in td_mpc_no_latent])
td_mpc_no_latent_runtime_std = np.std([df["total_time"].iloc[-1] for df in td_mpc_no_latent])
mopoc_v0_runtime = np.mean([df["total_time"].iloc[-1] for df in mopoc_v0])
mopoc_v0_runtime_std = np.std([df["total_time"].iloc[-1] for df in mopoc_v0])

print(f"TD-MPC (no latent) runtime: {td_mpc_no_latent_runtime:.2f} ± {td_mpc_no_latent_runtime_std:.2f}")
print(f"MOPOC-v0 runtime: {mopoc_v0_runtime:.2f} ± {mopoc_v0_runtime_std:.2f}")


"""
python train.py --seed 41
python train.py --seed 42
python train.py --seed 43
python train.py --seed 44
python train.py --seed 45

"""