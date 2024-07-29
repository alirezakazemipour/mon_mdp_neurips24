import matplotlib.pyplot as plt
import math
import numpy as np

n_runs = 100
algos = [
    # "MDP",
    ("NeverObsButtonMonitor_1.0", "blue", "100%"),
    # (""
     # "StatelessBinaryMonitor_0.75", "red", "75%"),
    # (""
     # "StatelessBinaryMonitor_0.5", "green", "50%"),
    # (""
     # "StatelessBinaryMonitor_0.25", "orange", "25%"),
    ("NeverObsButtonMonitor_0.1", "brown", "10%")
]

plt.style.use('ggplot')

fig, ax = plt.subplots()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

SMALL_SIZE = 8
MEDIUM_SIZE = 24
BIGGER_SIZE = 26

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=17)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

for conf in algos:
    algo, color, legend = conf
    runs = []
    for i in range(n_runs):
        x = np.load(f"data/simone/{algo}/{algo}_{-10}_{1}_{i}.npz")
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
    ax.fill_between(np.arange(len(mean_return)),
                        lower_bound,
                        upper_bound,
                        alpha=0.25
                        )
        # prob = min(float(algo[22:26].replace("_", "")) * 100, 100)
    ax.plot(np.arange(len(mean_return)),
                mean_return,
                alpha=1,
                linewidth=3,
                c=color,
                label=legend
                )
    # plt.fill_between(np.arange(len(mean_return)),
    #                  20 - 4.5,
    #                  20 + 4.5,
    #                  alpha=0.15,
    #                  color="magenta"
    #                  )
    # plt.axhline(.447, linestyle='--', label="cautious", c="magenta")
    # plt.axhline(0.941, linestyle='--', label="cautious", c="olive")
plt.xlabel("training steps (x100)", weight="bold", fontsize=18)
plt.axhline(0.447, linestyle="--", color="k", linewidth=3, label="cautious")
ax.set_ylabel("Discounted Test Return", weight="bold",  fontsize=18)
plt.title(f"DESF", weight="bold")
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
# plt.savefig("/Users/alirezakazemipour/Desktop/button_grid.pdf", format="pdf", bbox_inches="tight")
plt.show()

