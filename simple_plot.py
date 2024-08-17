import matplotlib.pyplot as plt
import math
import numpy as np

plt.style.use('ggplot')
fig, ax = plt.subplots()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

SMALL_SIZE = 8
MEDIUM_SIZE = 24
BIGGER_SIZE = 26

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=17)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
plt.title(f"EOP", weight="bold")

n_runs = 100
monitor = "Ask"
env = "Gridworld-TwoRoom-Quicksand-3x5-v0"

info = {"RiverSwim-6-v0": {"Ask": (199.14, "optimal"),
                           "Button": (192.72, "optimal"),
                           },
        "Gridworld-Penalty-3x3-v0": {"Ask": (9.415, "cautious"),
                                     "Button": (8.878, "cautious"),
                                     },
        "Gridworld-Corridor-3x4-v0": {"Ask": (9.409, "optimal"),
                                      "Button": (8.972, "optimal"),
                                      },
        "Gridworld-Empty-Distract-6x6-v0": {"Ask": (9.411, "cautious"),
                                            "Button": (8.057, "cautious"),
                                            },
        "Gridworld-TwoRoom-Quicksand-3x5-v0": {"Ask": (9.044, "cautious"),
                                               "Button": (8.413, "cautious"),
                                               },
        # "Gridworld-Quicksand-Distract-4x4-v0": {"Ask": (, "optimal"),
        #                                         "Button": (, "optimal"),
        #                                         },
        }

algos = [
    (f"{monitor}Monitor_1.0", "blue", "100%"),
    (f"{monitor}Monitor_0.75", "red", "75%"),
    (f"{monitor}Monitor_0.5", "green", "50%"),
    (f"{monitor}Monitor_0.25", "orange", "25%"),
    (f"{monitor}Monitor_0.1", "brown", "10%"),
    (f"{monitor}Monitor_0.01", "magenta", "1%")
]

assert n_runs == 100

for conf in algos:
    algo, color, legend = conf
    ref, opt_caut = info[env][monitor]

    runs = []
    for i in range(n_runs):
        x = np.load(f"data/iGym-Grid/{env}/{algo}/{algo.replace('_', '__')}_{i}.npz")
        runs.append(x["test/return"])
        smoothed = []
        for run in runs:
            val = [run[0]]
            for tmp in run[1:]:
                val.append(0.9 * val[-1] + 0.1 * tmp)
            smoothed.append(val)
    data = np.asarray(smoothed)
    mean_return = np.mean(data, axis=0)
    std_return = np.std(data, axis=0)
    lower_bound = mean_return - 1.96 * std_return / math.sqrt(n_runs)
    upper_bound = mean_return + 1.96 * std_return / math.sqrt(n_runs)
    ax.fill_between(np.arange(len(mean_return)),
                        lower_bound,
                        upper_bound,
                        alpha=0.25,
                    color=color
                        )
    ax.plot(np.arange(len(mean_return)),
                mean_return,
                alpha=1,
                linewidth=3,
                c=color,
                label=legend
                )
plt.axhline(ref, linestyle="--", color="k", linewidth=3, label=f"{opt_caut}")
ax.set_ylabel("Discounted Test Return", weight="bold",  fontsize=18)
plt.title(f"DESF", weight="bold")
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
plt.xlabel("training steps (x100)", weight="bold", fontsize=18)

plt.savefig(f"/Users/alirezakazemipour/Desktop/{monitor}_{env}.pdf",
            format="pdf",
            bbox_inches="tight"
            )
# plt.show()

