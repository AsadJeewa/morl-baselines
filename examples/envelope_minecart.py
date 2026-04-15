import fire
import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.wrappers import MORecordEpisodeStatistics
#from mo_gymnasium.wrappers.vector import MOSyncVectorEnv
from morl_baselines.multi_policy.envelope.envelope import Envelope
from morl_baselines.common.weights import equally_spaced_weights, random_weights, extrema_weights

def main(total_timesteps: int, wandb_mode: str = "offline", seed: int = 0):
    def make_env():
        env = mo_gym.make("minecart-v0")
        env = MORecordEpisodeStatistics(env, gamma=0.98)
        # env = MOSyncVectorEnv(env)
        return env

    EXPERIMENT = "sparse"  
    # options: "sparse", "interpolation", "extrapolation", "dist_shift"
    
    env = make_env()
    eval_env = make_env()
    dim = env.reward_dim
    if EXPERIMENT == "sparse":
        train_weights = random_weights(dim=dim, n=3, dist="dirichlet", seed=42)
        eval_weights = equally_spaced_weights(dim=dim, n=100)

    elif EXPERIMENT == "interpolation":
        train_weights = random_weights(dim=dim, n=5, dist="dirichlet", seed=42)
        eval_weights = equally_spaced_weights(dim=dim, n=100)

    elif EXPERIMENT == "extrapolation":
        train_weights = np.ones((1, dim)) / dim # central
        eval_weights = extrema_weights(dim=dim)

    elif EXPERIMENT == "dist_shift":
        train_weights = random_weights(dim=dim, n=50, dist="gaussian", seed=42)
        eval_weights = random_weights(dim=dim, n=100, dist="dirichlet", seed=123)
    # RecordVideo(make_env(), "videos/minecart/", episode_trigger=lambda e: e % 1000 == 0)

    agent = Envelope(
        env,
        seed=seed,
        max_grad_norm=1.0,#0.1 CHECK WAS TOO LOW
        learning_rate=2e-4,# 3e-4 CHECK WAS LOW 
        gamma=0.98,
        batch_size=32,
        net_arch=[256, 256, 256, 256],
        buffer_size=int(1.5e6),
        initial_epsilon=0.8,
        final_epsilon=0.5,
        epsilon_decay_steps=15000,
        initial_homotopy_lambda=0.9,
        final_homotopy_lambda=0.85,
        homotopy_decay_steps=10000,
        learning_starts=1000,
        envelope=True,
        gradient_updates=5,
        target_net_update_freq=1000,  # 1000,  # 500 reduce by gradient updates
        tau=0.1,
        log=True,
        wandb_mode=wandb_mode,
        project_name="MORL-Baselines",
        experiment_name="Envelope",
    )

    agent.train(
        total_timesteps=total_timesteps,
        total_episodes=None,
        weight_list=None,
        eval_env=eval_env,
        ref_point=np.array([-1, -1, -200.0]),
        known_pareto_front=env.unwrapped.pareto_front(gamma=0.98),
        eval_weights = None,
        # num_eval_weights_for_front=100,
        eval_freq=1000,
        # reset_num_timesteps=False,
        # reset_learning_starts=False,
        checkpoints=True,
        save_freq=10000,
    )


if __name__ == "__main__":
    fire.Fire(main)
