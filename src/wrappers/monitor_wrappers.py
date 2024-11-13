import gymnasium
from gymnasium import spaces
import numpy as np
from abc import abstractmethod
import pygame


class Monitor(gymnasium.Wrapper):
    """
    Generic class for monitors that DO NOT depend on the environment state.
    Monitors that DO depend on the environment state need to be customized
    according to the environment.

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        observability (float): probability that the monitor works properly.
            If < 1.0, then there is a chance that the environment reward is
            unobservable regardless of the state and action.
    """

    def __init__(self, env, observability=1.0, **kwargs):
        gymnasium.Wrapper.__init__(self, env)
        self.observability = observability

    @abstractmethod
    def _monitor_step(self, action, env_reward):
        pass

    @abstractmethod
    def _monitor_set_state(self, state):
        pass

    @abstractmethod
    def _monitor_get_state(self):
        pass

    def set_state(self, state):
        self.env.unwrapped.set_state(state["env"])
        self._monitor_set_state(state["mon"])

    def get_state(self):
        return {"env": self.env.unwrapped.get_state(), "mon": self._monitor_get_state()}

    def reset(self, seed=None, **kwargs):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        env_obs, env_info = self.env.reset(seed=seed, **kwargs)
        self.monitor_state = self.observation_space["mon"].sample()
        return {"env": env_obs, "mon": self.monitor_state}, env_info

    def render(self):
        """
        Make the screen flash if the agent observes a reward.
        Works only if the environment is rendered with pygame.
        """

        surf = pygame.display.get_surface()
        if surf is not None:
            surf.fill((255, 255, 255, 128), rect=None, special_flags=0)
            pygame.display.update()

    def step(self, action):
        """
        This type of monitors DO NOT depend on the environment state.
        Therefore, we first execute self.env.step() and then self._monitor_step().
        Everything else works as in classic Gymnasium environments, but state,
        actions, and rewards are dictionaries. That is, the agent expects

            actions = {"env": action_env, "mon": action_mon}

        and returns

            state = {"env": state_env, "mon": state_mon}
            reward = {"env": reward_env, "mon": reward_mon, "proxy": reward_proxy}
            terminated = env_terminated or monitor_terminated

        Truncated and info remain the same as self.env.step().
        """
        (
            env_obs,
            env_reward,
            env_terminated,
            env_truncated,
            env_info,
        ) = self.env.step(action["env"])

        (
            monitor_obs,
            proxy_reward,
            monitor_reward,
            monitor_terminated,
        ) = self._monitor_step(action, env_reward)

        obs = {"env": env_obs, "mon": monitor_obs}
        reward = {"env": env_reward, "mon": monitor_reward, "proxy": proxy_reward}
        terminated = env_terminated or monitor_terminated
        truncated = env_truncated

        if self.observability < 1.0 and self.np_random.random() > self.observability:
            reward["proxy"] = np.nan

        if self.render_mode == "human" and not np.isnan(reward["proxy"]):
            self.render()

        return obs, reward, terminated, truncated, env_info


class FullMonitor(Monitor):
    """
    This monitor always shows the true reward, regardless of its state and action.
    The monitor reward is always 0.
    This is a 'trivial Mon-MDP', i.e., it is equivalent to a classic MDP.

    Args:
        env (gymnasium.Env): the Gymnasium environment.
    """

    def __init__(self, env, **kwargs):
        Monitor.__init__(self, env, **kwargs)
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(1),
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(1),
        })  # fmt: skip

    def _monitor_set_state(self, state):
        return

    def _monitor_get_state(self):
        return np.array(0)

    def _monitor_step(self, action, env_reward):
        return self._monitor_get_state(), env_reward, 0.0, False


class AskMonitor(Monitor):
    """
    Simple monitor where the action is "turn on monitor" / "do nothing".
    The monitor is always off. The reward is seen only when the agent asks for it.
    The monitor reward is a constant penalty given if the agent asks to see the reward.

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        monitor_cost (float): cost for asking the monitor for rewards.
    """

    def __init__(self, env, monitor_cost=0.2, **kwargs):
        Monitor.__init__(self, env, **kwargs)
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(2),
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(1),
        })  # fmt: skip
        self.monitor_cost = monitor_cost
        self.prob = kwargs["prob"]
        self.forbidden_states = kwargs["forbidden_states"] if kwargs["forbidden_states"] is not None else []

    def _monitor_set_state(self, state):
        return

    def _monitor_get_state(self):
        return np.array(0)

    def _monitor_step(self, action, env_reward):
        env_next_obs = self.env.unwrapped.get_state()

        p = self.np_random.random()
        if action["mon"] == 1:
            if p < self.prob and env_next_obs not in self.forbidden_states:
                proxy_reward = env_reward
            else:
                proxy_reward = np.nan
            monitor_reward = -self.monitor_cost
        else:
            proxy_reward = np.nan
            monitor_reward = 0.0
        return self._monitor_get_state(), proxy_reward, monitor_reward, False


class ButtonMonitor(Monitor):
    """
    Monitor for Gridworlds.
    The monitor is turned on/off by doing LEFT (environment action) where a button is.
    If the monitor is on, the agent receives negative monitor rewards and observes
    the environment rewards.
    Ending an episode with the monitor on results in a large penalty.
    The monitor on/off state at the beginning of an episode is random.
    The position of the button can be specified by an argument (top-left by default).

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        monitor_cost (float): cost for monitor being active,
        monitor_end_cost (float): cost for ending an episode (by termination,
            not truncation) with the monitor active,
        button_cell_id (int): position of the monitor,
        env_action_push (int): the environment action to turn the monitor on/off.
    """

    def __init__(self, env, monitor_cost=0.2, **kwargs):
        Monitor.__init__(self, env, **kwargs)
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(1),  # no monitor action
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(2),  # monitor on/off
        })  # fmt: skip
        self.button_cell_id = kwargs["button_cell_id"]
        self.monitor_state = 0
        self.monitor_cost = monitor_cost
        self.prob = kwargs["prob"]
        self.forbidden_states = kwargs["forbidden_states"] if kwargs["forbidden_states"] is not None else []
        self.button_flip_act = kwargs["button_flip_act"]

    def _monitor_set_state(self, state):
        self.monitor_state = state

    def _monitor_get_state(self):
        return np.array(self.monitor_state)

    def step(self, action):
        env_obs = self.env.unwrapped.get_state()
        (
            env_next_obs,
            env_reward,
            env_terminated,
            env_truncated,
            env_info,
        ) = self.env.step(action["env"])

        monitor_reward = 0.0
        proxy_reward = np.nan
        p = self.np_random.random()

        if self.monitor_state == 1:
            if p < self.prob and env_next_obs not in self.forbidden_states:
                proxy_reward = env_reward
            else:
                proxy_reward = np.nan
            monitor_reward += -self.monitor_cost

        if action["env"] == self.button_flip_act and env_obs == self.button_cell_id:
            if self.monitor_state == 1:
                self.monitor_state = 0
            elif self.monitor_state == 0:
                self.monitor_state = 1
        monitor_terminated = False

        obs = {"env": env_next_obs, "mon": self._monitor_get_state()}
        reward = {"env": env_reward, "mon": monitor_reward, "proxy": proxy_reward}
        terminated = env_terminated or monitor_terminated
        truncated = env_truncated

        if self.render_mode == "human" and not np.isnan(reward["proxy"]):
            self.render()

        return obs, reward, terminated, truncated, env_info


class NSupporterMonitor(Monitor):
    """
    There are N monitors. At every time step, a random monitor is on.
    If the agent's action matches the monitor state, the agent observes the
    environment reward but receives a negative monitor reward.
    Otherwise it does not observe the environment reward, but receives a smaller
    positive monitor reward.
    For example, if state = 2 and action = 2, the agent observes the environment
    reward and gets reward_monitor = -0.2.
    If state = 2 and action != 2, the agent does not observe the reward but
    gets reward_monitor = 0.001.

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        n_monitors (int): number of monitors,
        monitor_cost (float): cost for observing the reward,
        monitor_bonus (float): reward for not observing the reward.
    """

    def __init__(self, env, monitor_cost=0.2, monitor_bonus=0.001, **kwargs):
        Monitor.__init__(self, env, **kwargs)
        n_monitors = kwargs["n_monitors"]
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(n_monitors),
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(n_monitors),
        })  # fmt: skip
        self.monitor_state = 0
        self.monitor_cost = monitor_cost
        self.monitor_bonus = monitor_bonus
        self.prob = kwargs["prob"]
        self.forbidden_states = kwargs["forbidden_states"] if kwargs["forbidden_states"] is not None else []

    def _monitor_set_state(self, state):
        self.monitor_state = state

    def _monitor_get_state(self):
        return np.array(self.monitor_state)

    def _monitor_step(self, action, env_reward):
        assert (action["mon"] < self.action_space["mon"].n), "illegal monitor action"

        env_next_obs = self.env.unwrapped.get_state()

        p = self.np_random.random()
        if action["mon"] == self.monitor_state:
            if p < self.prob and (env_next_obs not in self.forbidden_states):
                proxy_reward = env_reward
            else:
                proxy_reward = np.nan
            monitor_reward = -self.monitor_cost
        else:
            proxy_reward = np.nan
            monitor_reward = self.monitor_bonus

        self.monitor_state = self.observation_space["mon"].sample()
        return self._monitor_get_state(), proxy_reward, monitor_reward, False


class NExpertMonitor(Monitor):
    """
    There are N monitors. At every time step, a random monitor is on.
    If the agent's action matches the monitor state, the agent observes the
    environment reward but receives a negative monitor reward.
    Otherwise it does not observe the environment reward, but receives a smaller
    positive monitor reward.
    For example, if state = 2 and action = 2, the agent observes the environment
    reward and gets reward_monitor = -0.2.
    If state = 2 and action != 2, the agent does not observe the reward but
    gets reward_monitor = 0.001.

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        n_monitors (int): number of monitors,
        monitor_cost (float): cost for observing the reward,
        monitor_bonus (float): reward for not observing the reward.
    """

    def __init__(self, env, monitor_cost=0.2, monitor_bonus=0.001, **kwargs):
        Monitor.__init__(self, env, **kwargs)
        self.n_monitors = kwargs["n_monitors"]
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(self.n_monitors + 1),
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(self.n_monitors),
        })  # fmt: skip
        self.monitor_state = 0
        self.monitor_cost = monitor_cost
        self.monitor_bonus = monitor_bonus
        self.prob = kwargs["prob"]
        self.forbidden_states = kwargs["forbidden_states"] if kwargs["forbidden_states"] is not None else []

    def _monitor_set_state(self, state):
        self.monitor_state = state

    def _monitor_get_state(self):
        return np.array(self.monitor_state)

    def _monitor_step(self, action, env_reward):
        assert (action["mon"] < self.action_space["mon"].n), "illegal monitor action"

        env_next_obs = self.env.unwrapped.get_state()

        p = self.np_random.random()
        if action["mon"] == self.monitor_state:
            if p < self.prob and (env_next_obs not in self.forbidden_states):
                proxy_reward = env_reward
            else:
                proxy_reward = np.nan
            monitor_reward = -self.monitor_cost
        else:
            proxy_reward = np.nan
            monitor_reward = -self.monitor_bonus

        if action["mon"] == self.n_monitors:
            proxy_reward = np.nan
            monitor_reward = 0

        self.monitor_state = self.observation_space["mon"].sample()
        return self._monitor_get_state(), proxy_reward, monitor_reward, False


class LevelMonitor(Monitor):
    """
    The monitor has N levels, from 0 to N - 1.
    The initial level is random, and it increases if the agent's action matches
    the current level.
    For example, if state = 2 and action = 2, then next_state = 3.
    If the agent executes the wrong action, the level resets to 0.
    Actions 0 to N - 1 are costly.
    Action N does nothing and costs nothing.
    Environment rewards will become visible only when the monitor level is max,
    i.e., when state = N - 1.
    To keep it maxxed, the agent has to keep doing action = N - 1 (paying a cost)
    or do action = N (no cost).

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        n_levels (int): number of levels,
        monitor_cost (float): cost for leveling up the monitor state.
    """

    def __init__(self, env, monitor_cost=0.2, **kwargs):
        Monitor.__init__(self, env, **kwargs)
        n_levels = kwargs["n_levels"]
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(n_levels + 1),  # last action is "do nothing"
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(n_levels),
        })  # fmt: skip
        self.monitor_state = 0
        self.monitor_cost = monitor_cost
        self.prob = kwargs["prob"]
        self.forbidden_states = kwargs["forbidden_states"] if kwargs["forbidden_states"] is not None else []

    def _monitor_set_state(self, state):
        self.monitor_state = state

    def _monitor_get_state(self):
        return np.array(self.monitor_state)

    def _monitor_step(self, action, env_reward):
        assert (action["mon"] < self.action_space["mon"].n), "illegal monitor action"

        env_next_obs = self.env.unwrapped.get_state()

        monitor_reward = 0.0
        proxy_reward = np.nan
        p = self.np_random.random()

        if self.monitor_state == self.observation_space["mon"].n - 1:
            if p < self.prob and (env_next_obs not in self.forbidden_states):
                proxy_reward = env_reward

        if action["mon"] == self.action_space["mon"].n - 1:
            pass  # last action is "do nothing"
        else:
            monitor_reward = -self.monitor_cost  # pay cost
            if action["mon"] == self.monitor_state:
                self.monitor_state += 1  # raise level
                if self.monitor_state > self.observation_space["mon"].n - 1:  # level is already max
                    self.monitor_state = self.observation_space["mon"].n - 1
            else:
                self.monitor_state = 0  # reset level

        return self._monitor_get_state(), proxy_reward, monitor_reward, False


class RandomNonZeroMonitor(Monitor):
    """
    This monitor randomly makes non-zero rewards unobservable.
    There are no monitor states and actions.
    The monitor reward is always 0.

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        prob (float): the probability that the reward is unobservable.
    """

    def __init__(self, env, prob=0.5, **kwargs):
        Monitor.__init__(self, env, **kwargs)
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(1),
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(1),
        })  # fmt: skip
        self.prob = prob
        self.forbidden_states = kwargs["forbidden_states"] if kwargs["forbidden_states"] is not None else []

    def _monitor_set_state(self, state):
        return

    def _monitor_get_state(self):
        return np.array(0)

    def _monitor_step(self, action, env_reward):
        env_next_obs = self.env.unwrapped.get_state()
        monitor_reward = 0.0
        if env_reward != 0 and self.np_random.random() < self.prob or (env_next_obs in self.forbidden_states):
            proxy_reward = np.nan
        else:
            proxy_reward = env_reward
        return self._monitor_get_state(), proxy_reward, monitor_reward, False


class RandomMonitor(Monitor):
    """
    This monitor randomly makes rewards unobservable.
    Each reward has a different probability of being observed, which is fixed
    when the environment is created.
    There are no monitor states and actions.
    The monitor reward is always 0.

    Args:
        env (gymnasium.Env): the Gymnasium environment.
    """

    def __init__(self, env, **kwargs):
        Monitor.__init__(self, env, **kwargs)
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(1),
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(1),
        })  # fmt: skip
        self.prob = kwargs["prob"]
        self.forbidden_states = kwargs["forbidden_states"] if kwargs["forbidden_states"] is not None else []

    def _monitor_set_state(self, state):
        return

    def _monitor_get_state(self):
        return np.array(0)

    def _monitor_step(self, action, env_reward):
        env_next_obs = self.env.unwrapped.get_state()
        monitor_reward = 0.0
        if self.np_random.random() < self.prob or (
                env_next_obs in self.forbidden_states):
            proxy_reward = np.nan
        else:
            proxy_reward = env_reward
        return self._monitor_get_state(), proxy_reward, monitor_reward, False
