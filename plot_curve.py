import pandas as pd
import os

import numpy as np
from matplotlib import pyplot as plt

import seaborn as sns; sns.set()

def moving_avg(data: np.array, window: int):
    res = np.zeros(data.shape[0])
    for index in range(data.shape[0]):
        if index == 0:
            res[index] = data[index]
            continue

        if index < window-1:
            res[index] = np.mean(data[:index])
        else:
            res[index] = np.mean(data[index-window+1:index])

    return res

scheduler_file = "scheduler_speed_base_abilene"
dir = f"results/sample_agent/{scheduler_file}/abc/sample_config"

experiments = [
    "2023-09-03_12-01-56_seed4704",
    "2023-09-03_17-24-33_seed2965",
    "2023-09-03_17-40-52_seed6649",
]

episodes = 150
window_size = 70
results = np.zeros((len(experiments), episodes))
for i, exp in enumerate(experiments):
    file_path = os.path.join(dir, exp, "rewards.csv")
    df = pd.read_csv(file_path, header=None)
    results[i, :] = np.array(df[0])


# CI calculation
mean = moving_avg(np.mean(results, axis=0), window_size)
std = moving_avg(np.std(results, axis=0), 2*window_size)
ci = 1.96*std/np.sqrt(len(experiments))

x_range = range(episodes)

plt.plot(x_range, mean, color="red")
plt.fill_between(x_range, mean-0.5*ci, mean+0.5*ci, color="red", alpha=0.3)
#plt.legend(["train mean", "train 95%CI" ])
plt.xlabel("episode")
plt.ylabel("reward")
plt.title(f"")
plt.show()