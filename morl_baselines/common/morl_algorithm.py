from abc import ABC, abstractmethod
from typing import Union, Optional

import numpy as np
import torch as th
import mo_gym
from mo_gym import eval_mo
import gym
from gym import spaces

from torch.utils.tensorboard import SummaryWriter


class MOPolicy(ABC):
    """
    An MORL policy, has an underlying learning structure which can be:
    - used to get a greedy action via eval()
    - updated using some experiences via update()

    Note that the learning structure can embed multiple policies (for example using a Conditioned Network). In this case,
    eval() requires a weight vector as input.
    """
    def __init__(self, id: Optional[int] = None, device: Union[th.device, str] = "auto") -> None:
        self.id = id
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu") if device == "auto" else device
        self.global_step = 0

    @abstractmethod
    def eval(self, obs: np.ndarray, w: Optional[np.ndarray]) -> Union[int, np.ndarray]:
        """Gives the best action for the given observation

        Args:
            obs (np.array): Observation
            w (optional np.array): weight for scalarization

        Returns:
            np.array or int: Action
        """

    def policy_eval(self, eval_env, weights: np.ndarray, writer: SummaryWriter):
        """
        Runs a policy evaluation (typically on one episode) on eval_env and logs some metrics using writer.
        :param eval_env: evaluation environment
        :param weights: weights to use in the evaluation
        :param writer: wandb writer
        :return: a tuple containing the evaluations
        """

        # TODO, make eval_mo generic to scalarization?
        scalarized_reward, scalarized_discounted_reward, vec_reward, discounted_vec_reward = eval_mo(self, eval_env, weights)
        if self.id is None:
            idstr = ""
        else:
            idstr = f"_{self.id}"

        writer.add_scalar(f"eval{idstr}/scalarized_reward", scalarized_reward, self.global_step)
        writer.add_scalar(f"eval{idstr}/scalarized_discounted_reward", scalarized_discounted_reward, self.global_step)
        for i in range(vec_reward.shape[0]):
            writer.add_scalar(f"eval{idstr}/vec_{i}", vec_reward[i], self.global_step)
            writer.add_scalar(f"eval{idstr}/discounted_vec_{i}", discounted_vec_reward[i], self.global_step)

        return (
            scalarized_reward,
            scalarized_discounted_reward,
            vec_reward,
            discounted_vec_reward
        )

    @abstractmethod
    def update(self):
        """Update algorithm's parameters"""


class MOAgent(ABC):
    """
    An MORL Agent, can contain one or multiple MOPolicies.
    Contains helpers to extract features from the environment, setup logging etc.
    """
    def __init__(self, env: Optional[gym.Env], device: Union[th.device, str] = "auto") -> None:
        self.extract_env_info(env)
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu") if device == "auto" else device

        self.global_step = 0
        self.num_episodes = 0

    def extract_env_info(self, env):
        """
        Extracts all the features of the environment: observation space, action space, ...
        """
        # Sometimes, the environment is not instantiated at the moment the MORL algorithms is being instantiated.
        # So env can be None. It is the reponsibility of the implemented MORLAlgorithm to call this method in those cases
        if env is not None:
            self.env = env
            self.observation_shape = self.env.observation_space.shape
            self.observation_dim = self.env.observation_space.shape[0]
            self.action_space = env.action_space
            if isinstance(self.env.action_space, (spaces.Discrete, spaces.MultiBinary)):
                self.action_dim = self.env.action_space.n
            else:
                self.action_dim = self.env.action_space.shape[0]
            self.reward_dim = self.env.reward_space.shape[0]


    @abstractmethod
    def get_config(self) -> dict:
        """Generates dictionary of the algorithm parameters configuration

        Returns:
            dict: Config
        """

    def setup_wandb(self, project_name: str, experiment_name: str):
        self.experiment_name = experiment_name
        import wandb

        wandb.init(
            project=project_name,
            sync_tensorboard=True,
            config=self.get_config(),
            name=self.experiment_name,
            monitor_gym=True,
            save_code=True,
        )
        self.writer = SummaryWriter(f"/tmp/{self.experiment_name}")
        # The default "step" of wandb is not the actual time step (gloabl_step) of the MDP
        wandb.define_metric("*", step_metric="global_step")

    def close_wandb(self):
        import wandb
        self.writer.close()
        wandb.finish()