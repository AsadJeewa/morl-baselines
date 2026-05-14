import fire
import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.wrappers import MORecordEpisodeStatistics
#from mo_gymnasium.wrappers.vector import MOSyncVectorEnv
from morl_baselines.multi_policy.envelope.envelope import Envelope
from morl_baselines.common.weights import equally_spaced_weights, random_weights, extrema_weights

def main(experiment: str = "interEasy", total_timesteps: int = 100000, wandb_mode: str = "online", log: bool = True, seed: int = 0):
    log = str(log).lower() == "true" 
    def make_env():
        env = mo_gym.make("deep-sea-treasure-v0")
        env = MORecordEpisodeStatistics(env, gamma=0.98)
        # env = MOSyncVectorEnv(env)
        return env

    env = make_env()
    eval_env = make_env()
    dim = env.reward_dim
    
    if experiment == "interEasy":
        train_weights = equally_spaced_weights(dim=dim, n=20)
        eval_weights = equally_spaced_weights(dim=dim, n=100)

    elif experiment == "interMedium":
        train_weights = equally_spaced_weights(dim=dim, n=10)
        eval_weights = equally_spaced_weights(dim=dim, n=100)

    elif experiment == "interDifficult":
        train_weights = equally_spaced_weights(dim=dim, n=5)
        eval_weights = equally_spaced_weights(dim=dim, n=100)
    
    agent = Envelope(
        env,
        seed=seed,
        max_grad_norm=1.0,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=128,
        net_arch=[128,128],
        buffer_size=int(1e5),
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay_steps=20000,
        initial_homotopy_lambda=0.0,
        final_homotopy_lambda=1.0,
        homotopy_decay_steps=50000,
        learning_starts=1000,
        envelope=True,
        gradient_updates=1,
        target_net_update_freq=200, 
        tau=1.0,
        log=log,
        wandb_mode=wandb_mode,
        project_name="MORL-Baselines",
        experiment_name="Envelope-DST",
    )

    agent.train(
        total_timesteps=total_timesteps,
        total_episodes=None,
        weight_list=train_weights,
        eval_env=eval_env,
        ref_point=np.array([0.0, -50.0]),
        known_pareto_front=env.unwrapped.pareto_front(gamma=0.99),
        eval_weights = eval_weights,
        num_eval_weights_for_front=50,
        eval_freq=1000,
        reset_num_timesteps=True,
        reset_learning_starts=False,
        checkpoints=True,
        save_freq=20000,
    )


if __name__ == "__main__":
    fire.Fire(main)
