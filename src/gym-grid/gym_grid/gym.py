from gymnasium.envs.registration import register


def register_envs():
    register(
        id="Gridworld-Corridor",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=200,
        kwargs={
            "grid": "corridor",
        },
    )

    register(
        id="Gridworld-Bottleneck",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "bottleneck",
        },
    )

    register(
        id="Gridworld-Empty-2x2-v0",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=10,
        kwargs={
            "grid": "2x2_empty",
        },
    )

    register(
        id="Gridworld-Empty-3x3-v0",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "3x3_empty",
        },
    )

    register(
        id="Gridworld-Loop",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "loop",
        },
    )

    register(
        id="Gridworld-Empty-10x10-v0",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=100,
        kwargs={
            "grid": "10x10_empty",
        },
    )

    register(
        id="Gridworld-Empty",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "empty",
        },
    )

    register(
        id="Gridworld-Penalty-3x3-v0",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "3x3_penalty",
        },
    )

    register(
        id="Gridworld-Stochastic-3x3-v0",
        entry_point="gym_grid.gridworld:StochasticMiniGrid",
        max_episode_steps=10,
        kwargs={
            "grid": "3x3_stochastic",
        },
    )

    register(
        id="Gridworld-Quicksand-4x4-v0",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "4x4_quicksand",
        },
    )

    register(
        id="Gridworld-Hazard",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "hazard",
        },
    )

    register(
        id="Gridworld-TwoRoom-3x5",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "two_room_3x5",
        },
    )

    register(
        id="Gridworld-OneWay",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "one_way",
        },
    )
    register(
        id="Gridworld-Full-5x5-v0",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "5x5_full",
        },
    )

    register(
        id="Gridworld-TwoRoom-2x11",
        entry_point="gym_grid.gridworld:GridworldMiddleStart",
        max_episode_steps=200,
        kwargs={
            "grid": "two_room_2x11",
        },
    )

    register(
        id="Gridworld-Barrier-5x5-v0",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "5x5_barrier",
        },
    )

    register(
        id="RiverSwim",
        entry_point="gym_grid.gridworld:RiverSwim",
        max_episode_steps=200,
        kwargs={
            "grid": "river_swim",
        },
    )