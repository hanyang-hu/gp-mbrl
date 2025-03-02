import numpy as np
import pandas as pd

root_dir = "./results/"

task_list = {
    "Pendulum-v1",
    "Reacher-v5",
    "Pusher-v5",
    "important-data-Swimmer-v5-256from4096",
    "Swimmer-v5",
    "HalfCheetah-v5",
}

exp_name_list = {
    "gp_td_mpc",
    "gp_td_mpc_dkl"
}

kernel_list = {
    "NA",
    "RBF",
    "Matern",
    "SM",
}

seed_list = {
    1, 2, 3, 4, 5
}

# Load data (also consider the case where the file does not exist), only interested in total runtime
data = {}
for task in task_list:
    data[task] = {}
    for exp_name in exp_name_list:
        data[task][exp_name] = {}
        for kernel in kernel_list:
            temp_list = []
            for seed in seed_list:
                try:
                    df = pd.read_csv(f"{root_dir}{task}/metrics_{exp_name}_{kernel}_{seed}.csv")
                    temp_list.append(df["total_time"].iloc[-1])
                except:
                    temp_list = None
                    break
            if temp_list is not None:
                data[task][exp_name][kernel] = sum(temp_list) / len(temp_list)
            else:
                data[task][exp_name][kernel] = None

# Print data with 2 decimal places
for task, exp_data in data.items():
    print(f"Task: {task}")
    for exp_name, kernel_data in exp_data.items():
        print(f"  Experiment: {exp_name}")
        for kernel, value in kernel_data.items():
            if value is not None:
                print(f"    Kernel: {kernel}, Total Runtime: {value:.2f}")
            else:
                print(f"    Kernel: {kernel}, Total Runtime: None")