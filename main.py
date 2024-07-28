import gymnasium
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import os
import numpy as np
from pprint import pprint  # noqa: F401
import git

from src.utils import dict_to_id
from src.experiment import Experiment
from src.wrappers import monitor_wrappers
import src.actor
import src.critic


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # pprint(config)

    group = dict_to_id(cfg.environment) + "/" + str(cfg.monitor.id) + "_" + str(cfg.monitor.prob)
    base_folder = group
    run_id = "_".join(
        [
            str(cfg.monitor.id),
            str(cfg.monitor.prob),
            str(cfg.agent.critic.q0_max),
            str(cfg.agent.critic.r0_max),
            str(cfg.experiment.rng_seed),
        ]
    )

    if cfg.experiment.datadir is not None:
        filepath = os.path.join(cfg.experiment.datadir, base_folder)
        os.makedirs(filepath, exist_ok=True)
        filepath = os.path.join(filepath, run_id)
        if os.path.isfile(filepath + ".npz"):
            print("   [RUN ALREADY DONE]")
            return

    wandb.init(
        group=group,
        config=config,
        settings=wandb.Settings(
            start_method="thread",
            _disable_stats=True,
            _disable_meta=True,
        ),
        **cfg.wandb,
    )

    if cfg.environment.id in ["Gym-Grid/Gridworld-TwoRoom-Distract-Middle-2x11-v0"]:
        if cfg.monitor.id in ["ButtonMonitor"]:
            with open_dict(cfg):
                cfg.monitor.button_cell_id = 16

    # Decay learning rate in stochastic monitors
    if cfg.monitor.id in ["NMonitor"]:
        cfg.agent.critic.lr.min_value = min(0.1, cfg.agent.critic.lr.min_value)
        cfg.agent.critic.lr_visit.min_value = min(0.1, cfg.agent.critic.lr_visit.min_value)

    env = gymnasium.make(**cfg.environment)
    if "reward_noise_std" in cfg.environment.keys():
        cfg.environment.reward_noise_std = 0.0  # test without noise so we need 1 episode only
    env_test = gymnasium.make(**cfg.environment)
    env = getattr(monitor_wrappers, cfg.monitor.id)(env, **cfg.monitor)
    env_test = getattr(monitor_wrappers, cfg.monitor.id)(env_test, **cfg.monitor)

    sizes = (
        env.observation_space["env"].n,
        env.observation_space["mon"].n,
        env.action_space["env"].n,
        env.action_space["mon"].n,
    )

    critic = getattr(src.critic, cfg.agent.critic.id)(*sizes, **cfg.agent.critic)
    actor = getattr(src.actor, cfg.agent.actor.id)(critic, **cfg.agent.actor)
    experiment = Experiment(env, env_test, actor, critic, **cfg.experiment)
    data = experiment.train()

    if cfg.experiment.datadir is not None:
        np.savez(filepath, **data)

    if cfg.experiment.debugdir is not None:
        from plot_gridworld_agent import plot_agent

        filepath = os.path.join(cfg.experiment.debugdir, base_folder)
        os.makedirs(filepath, exist_ok=True)
        filepath = os.path.join(filepath, run_id)
        os.makedirs(filepath, exist_ok=True)
        plot_agent(actor, critic, filepath)

    wandb.finish()


if __name__ == "__main__":
    run()
