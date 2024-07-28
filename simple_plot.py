import matplotlib.pyplot as plt
import math
import numpy as np

q_max = -10#, -100
r_max = 1#, 0

n_runs = 4
algos = [
    "ButtonMonitor_1",
    # "iStatelessBinaryMonitor_p0.75",
    # "iStatelessBinaryMonitor_p0.5",
    # "iStatelessBinaryMonitor_p0.25",
    "ButtonMonitor_0.1",
    # "iStatelessBinaryMonitor_p0.01"
]
for algo in algos:
    runs = []
    for i in range(n_runs):
        x = np.load(f"data/iGym-Grid/RiverSwim-6-v0/{algo}/{algo.replace("p", "")}_{q_max}_{r_max}_{i}.npz")
        runs.append(x["test/return"])
    # print(np.argmin(np.asarray(runs).sum(-1)))
    # exit()
    smoothed = []
    for run in runs:
        val = [run[0]]
        for tmp in run[1:]:
            val.append(0.9 * val[-1] + 0.1 * tmp)
        smoothed.append(val)
    mean_return = np.mean(np.asarray(smoothed), axis=0)
    std_return = np.std(np.asarray(smoothed), axis=0)
    lower_bound = mean_return - 1.96 * std_return / math.sqrt(n_runs)
    upper_bound = mean_return + 1.96 * std_return / math.sqrt(n_runs)
    plt.fill_between(np.arange(len(mean_return)),
                     lower_bound,
                     upper_bound,
                     alpha=0.25
                     )
    prob = float(algo[-1]) * 100
    plt.plot(np.arange(len(mean_return)),
             mean_return,
             alpha=1,
             label=f"{prob}%",
             linewidth=3
             )
plt.axhline(.447, linestyle='--', label="cautious", c="magenta")
# plt.axhline(0.941, linestyle='--', label="cautious", c="olive")
plt.xlabel("training steps (x 100)")
plt.ylabel("discounted test return")
plt.title(f" button monitor - over {len(runs)} runs")
plt.grid()
plt.legend()
# for i in range(30):
#     plt.plot(np.arange(len(mean_return)),
#              smoothed[i],
#              label=algo,
#              linewidth=3
#              )
plt.show()
