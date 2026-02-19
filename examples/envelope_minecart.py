import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.wrappers import MORecordEpisodeStatistics

from morl_baselines.multi_policy.envelope.envelope import Envelope


def main():
    def make_env():
        env = mo_gym.make("minecart-v0")
        env = MORecordEpisodeStatistics(env, gamma=0.98)
        # env = mo_gym.LinearReward(env)
        return env

    env = make_env()
    eval_env = make_env()
    # RecordVideo(make_env(), "videos/minecart/", episode_trigger=lambda e: e % 1000 == 0)

    agent = Envelope(
        env,
        max_grad_norm=1.0,#0.1 CHECK WAS TOO LOW
        learning_rate=2e-4,# 3e-4 CHECK WAS LOW 
        # gamma=0.98,
        batch_size=32,
        net_arch=[256, 256, 256, 256],
        buffer_size=int(1.5e6),
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay_steps=100000,
        # initial_homotopy_lambda=0.0,
        # final_homotopy_lambda=1.0,
        # homotopy_decay_steps=50000,
        learning_starts=100,
        envelope=True,
        # gradient_updates=1,
        # target_net_update_freq=1000,  # 1000,  # 500 reduce by gradient updates
        tau=0.1,
        log=True,
        project_name="MORL-Baselines",
        experiment_name="Envelope",
    )

    agent.train(
        total_timesteps=100000,
        total_episodes=None,
        weight=None,
        eval_env=eval_env,
        ref_point=np.array([-1, -1, -200.0]),
        # known_pareto_front=env.unwrapped.pareto_front(gamma=0.98),
        # num_eval_weights_for_front=100,
        # eval_freq=1000,
        # reset_num_timesteps=False,
        # reset_learning_starts=False,
    )


if __name__ == "__main__":
    main()
