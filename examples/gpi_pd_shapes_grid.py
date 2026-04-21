import fire
import mo_gymnasium as mo_gym
import numpy as np

from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD
from mo_gymnasium.wrappers import MORecordEpisodeStatistics
from mo_gymnasium.envs.shapes_grid.shapes_grid import DIFFICULTY

# from gymnasium.wrappers.record_video import RecordVideo


def main(algo: str, gpi_pd: bool, g: int, wandb_mode: str = "online", timesteps_per_iter: int = 10000, seed: int = 0):
    gpi_pd = str(gpi_pd).lower() == "true" #TODO add argparse
    def make_env():
        extra_kwargs = {}
        extra_kwargs["difficulty"] = DIFFICULTY["EASY"]
        env = mo_gym.make("shapes-grid-v0", **extra_kwargs)
        env = MORecordEpisodeStatistics(env, gamma=0.98)
        # env = mo_gym.LinearReward(env)
        return env

    env = make_env()
    eval_env = make_env()
    # RecordVideo(make_env(), "videos/minecart/", episode_trigger=lambda e: e % 1000 == 0)


    agent = GPIPD(
        env,
        seed=seed,
        num_nets=2,
        max_grad_norm=None,
        learning_rate=3e-4,
        gamma=0.98,
        batch_size=128,
        net_arch=[256, 256, 256, 256],
        buffer_size=int(1e6),
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay_steps=200000,
        learning_starts=100,
        alpha_per=0.6,
        min_priority=0.01,
        per=gpi_pd,
        gpi_pd=gpi_pd,
        use_gpi=True,
        gradient_updates=g,
        target_net_update_freq=200,
        tau=1,
        dyna=gpi_pd,
        dynamics_uncertainty_threshold=1.5,
        dynamics_net_arch=[256, 256, 256],
        dynamics_buffer_size=int(1e5),
        dynamics_rollout_batch_size=25000,
        dynamics_train_freq=lambda t: 250,
        dynamics_rollout_freq=250,
        dynamics_rollout_starts=5000,
        dynamics_rollout_len=1,
        real_ratio=0.5,
        log=True,
        wandb_mode=wandb_mode,
        project_name="MORL-Baselines",
        experiment_name="GPI-PD",
    )

    agent.train(
        total_timesteps=15 * timesteps_per_iter,
        eval_env=eval_env,
        ref_point=np.array([-5.0, -5.0, -5.0]),
        known_pareto_front=None,
        weight_selection_algo=algo,# here
        timesteps_per_iter=timesteps_per_iter,
        checkpoints=True,
    )


if __name__ == "__main__":
    fire.Fire(main)
