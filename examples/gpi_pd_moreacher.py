import fire
import mo_gymnasium as mo_gym
import numpy as np
import os
os.environ["MUJOCO_GL"] = "egl"
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD
from mo_gymnasium.wrappers import MORecordEpisodeStatistics

# from gymnasium.wrappers.record_video import RecordVideo


def main(algo: str = "gpi-ls", gpi_pd: bool = False, g: int = 1, exp_type: str = None, wandb_mode: str = "online", log: bool = True, total_timesteps: int = 200000, timesteps_per_iter: int = 10000, seed: int = 0, exp_notes: str = ""):
    gpi_pd = str(gpi_pd).lower() == "true"
    log = str(log).lower() == "true"
    def make_env():
        env = mo_gym.make("mo-reacher-v4")
        env = MORecordEpisodeStatistics(env, gamma=0.99)
        return env

    env = make_env()
    eval_env = make_env()  # RecordVideo(make_env(), "videos/minecart/", episode_trigger=lambda e: e % 1000 == 0)
    experiment_name = "GPI_MO-Reacher_"+str(total_timesteps)+"_"+exp_notes+"_"+str(seed)
    agent = GPIPD(
        env,
        seed=seed,
        num_nets=2,
        # max_grad_norm=None,
        learning_rate=1e-4,
        gamma=0.99,
        batch_size=128,
        net_arch=[256, 256, 256, 256],
        buffer_size=int(1e6),
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay_steps=50000,
        learning_starts=5000,
        alpha_per=0.6,
        min_priority=0.01,
        per=gpi_pd,
        gpi_pd=gpi_pd,
        use_gpi=True,
        gradient_updates=g,
        target_net_update_freq=1000,
        tau=1,
        dyna=gpi_pd,
        dynamics_uncertainty_threshold=1.5,
        dynamics_net_arch=[256, 256, 256],
        dynamics_normalize_inputs=False,
        dynamics_buffer_size=int(1e5),
        dynamics_rollout_batch_size=25000,
        dynamics_train_freq=lambda t: 250,
        dynamics_rollout_freq=250,
        dynamics_rollout_starts=5000,
        dynamics_rollout_len=1,
        real_ratio=0.1,
        log=log,
        wandb_mode=wandb_mode,
        project_name="MORL-Baselines",
        experiment_name=experiment_name,
        group = experiment_name.rsplit("_", 1)[0]
    )

    agent.train(
        total_timesteps=total_timesteps,
        eval_env=eval_env,
        ref_point=np.array([-100, -100, -100, -100]),
        known_pareto_front=None,
        weight_selection_algo=algo,# here
        timesteps_per_iter=timesteps_per_iter,
        checkpoints=True,
        save_freq=100000,
    )


if __name__ == "__main__":
    fire.Fire(main)
