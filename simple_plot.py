import matplotlib.pyplot as plt
import math
import numpy as np
import itertools

plt.style.use('ggplot')

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
# plt.title(f"EOP", weight="bold")

n_runs = 30
monitor = "iLevelMonitor_nl3", "iNMonitor_nm4", "iStatelessBinaryMonitor", "iButtonMonitor", "iRandomNonZeroMonitor"
env = (
    "RiverSwim-6-v0",
    # "Gridworld-Penalty-3x3-v0",
    "Gridworld-Corridor-3x4-v0",
    "Gridworld-Empty-Distract-6x6-v0",
    # "Gridworld-TwoRoom-Quicksand-3x5-v0",
    "Gridworld-Quicksand-Distract-4x4-v0",
)
env_mon_combo = itertools.product(env, monitor)

info = {"RiverSwim-6-v0": {"iStatelessBinaryMonitor": (19.91, "optimal"),
                           "iButtonMonitor": (19.05, "optimal"),
                           "iLevelMonitor_nl3": (19.91, "optimal"),
                           "iNMonitor_nm4": (19.91, "optimal"),
                           "iRandomNonZeroMonitor": (19.91, "optimal")
                           },
        "Gridworld-Empty-Distract-6x6-v0": {"iStatelessBinaryMonitor": (0.904, "optimal"),
                                            "iButtonMonitor": (0.812, "optimal"),
                                            "iLevelMonitor_nl3": (0.904, "optimal"),
                                            "iNMonitor_nm4": (0.904, "optimal"),
                                            "iRandomNonZeroMonitor": (0.904, "optimal"),
                                            },
        "Gridworld-Corridor-3x4-v0": {"iStatelessBinaryMonitor": (0.764, "optimal"),
                                      "iButtonMonitor": (0.672, "optimal"),
                                      "iLevelMonitor_nl3": (0.764, "optimal"),
                                      "iNMonitor_nm4": (0.764, "optimal"),
                                      "iRandomNonZeroMonitor": (0.764, "optimal"),
                                      },
        "Gridworld-Penalty-3x3-v0": {"iStatelessBinaryMonitor": (0.941, "optimal"),
                                     "iButtonMonitor": (0.849, "optimal"),
                                     "iLevelMonitor_nl3": (0.941, "optimal"),
                                     "iNMonitor_nm4": (0.941, "optimal"),
                                     "iRandomNonZeroMonitor": (0.941, "optimal"),
                                     },
        "Gridworld-TwoRoom-Quicksand-3x5-v0": {"iStatelessBinaryMonitor": (0.941, "optimal"),
                                               "iButtonMonitor": (0.849, "optimal"),
                                               "iLevelMonitor_nl3": (0.941, "optimal"),
                                               "iNMonitor_nm4": (0.941, "optimal"),
                                               "iRandomNonZeroMonitor": (0.941, "optimal"),
                                               },
        "Gridworld-Quicksand-Distract-4x4-v0": {"iStatelessBinaryMonitor": (0.914, "optimal"),
                                                "iButtonMonitor": (0.821, "optimal"),
                                                "iLevelMonitor_nl3": (0.914, "optimal"),
                                                "iNMonitor_nm4": (0.914, "optimal"),
                                                "iRandomNonZeroMonitor": (0.914, "optimal"),
                                                },
        }

for env, monitor in env_mon_combo:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    algos = [
        (f"{monitor}", "blue", "100%"),
        # (f"{monitor}_0.75", "red", "75%"),
        # (f"{monitor}_0.5", "green", "50%"),
        # (f"{monitor}_0.25", "orange", "25%"),
        # (f"{monitor}_0.1", "brown", "10%"),
        # (f"{monitor}_0.01", "magenta", "1%")
    ]

    assert n_runs == 30

    for conf in algos:
        algo, color, legend = conf
        ref, opt_caut = info[env][monitor]
        runs = []
        for i in range(n_runs):
            x = np.load(f"data/iGym-Grid/{env}/{algo}/q_visit_-10.0_-10.0_1.0_1.0_1.0_0.0_0.01_{i}.npz")["test/return"]
            runs.append(x)
        # print(np.argmin(np.array(runs).sum(-1)))
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
        # for run in runs[2:3]:
        #     plt.plot(np.arange(len(mean_return)), run)
    plt.axhline(ref, linestyle="--", color="k", linewidth=3, label=f"{opt_caut}")
    ax.set_ylabel("Discounted Test Return", weight="bold", fontsize=18)
    ax.legend(loc='lower right', ncol=2, bbox_to_anchor=(1, 0))
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    plt.title(f"{env}_{monitor}")
    plt.xlabel("training steps (x100)", weight="bold", fontsize=18)
    ax.set_xticks(np.arange(0, len(mean_return) + 0.1, 50))

    # plt.show()
    plt.savefig(f"/Users/alirezakazemipour/Desktop/{monitor}_{env}.pdf",
                format="pdf",
                bbox_inches="tight"
                )
    plt.close()