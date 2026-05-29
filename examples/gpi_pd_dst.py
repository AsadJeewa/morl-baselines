import fire
import mo_gymnasium as mo_gym
import numpy as np

from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD
from mo_gymnasium.wrappers import MORecordEpisodeStatistics
from morl_baselines.common.weights import equally_spaced_weights, random_weights, extrema_weights, equally_spaced_train_and_eval_weights

def main(algo: str, gpi_pd: bool, g: int, experiment_type: str = None, wandb_mode: str = "online", log: bool = True, total_timesteps: int = 150000, timesteps_per_iter: int = 10000, seed: int = 0, exp_notes: str = ""):
    gpi_pd = str(gpi_pd).lower() == "true" 
    log = str(log).lower() == "true"
    def make_env():
        env = mo_gym.make("deep-sea-treasure-v0")
        env = MORecordEpisodeStatistics(env, gamma=0.98)
        # env = mo_gym.LinearReward(env)
        return env

    env = make_env()
    eval_env = make_env()
    dim = env.reward_dim
    # RecordVideo(make_env(), "videos/minecart/", episode_trigger=lambda e: e % 1000 == 0)


    agent = GPIPD(
        env,
        seed=seed,
        num_nets=2,
        max_grad_norm=None,
        learning_rate=3e-4,
        gamma=0.98,
        batch_size=128,
        net_arch=[256, 256],
        buffer_size=int(1e6),
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay_steps=total_timesteps*0.6,
        learning_starts=1000,
        alpha_per=0.6,
        min_priority=0.01,
        per=gpi_pd,
        gpi_pd=gpi_pd,
        use_gpi=True,
        gradient_updates=g,
        target_net_update_freq=100,
        tau=1,
        dyna=gpi_pd,
        dynamics_uncertainty_threshold=1.5,
        dynamics_net_arch=[256, 256],
        dynamics_buffer_size=int(1e5),
        dynamics_rollout_batch_size=25000,
        dynamics_train_freq=lambda t: 250,
        dynamics_rollout_freq=250,
        dynamics_rollout_starts=1000,
        dynamics_rollout_len=3,
        real_ratio=0.5,
        log=log,
        wandb_mode=wandb_mode,
        project_name="MORL-Baselines",
        experiment_name="GPI_DST_"+str(experiment_type)+"_"+str(total_timesteps)+"_"+exp_notes,
    )

    agent.train(
        total_timesteps=total_timesteps,
        eval_env=eval_env,
        ref_point=np.array([0.0, -50.0]),
        known_pareto_front=env.unwrapped.pareto_front(gamma=0.99),
        weight_selection_algo=algo,# here
        timesteps_per_iter=timesteps_per_iter,
        # eval_freq=1000,
        checkpoints=True,
        save_freq=20000,
    )



if __name__ == "__main__":
    fire.Fire(main)
