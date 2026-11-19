import fire
import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.wrappers import MORecordEpisodeStatistics
#from mo_gymnasium.wrappers.vector import MOSyncVectorEnv
from morl_baselines.multi_policy.envelope.envelope import Envelope
from morl_baselines.common.weights import equally_spaced_weights, random_weights, extrema_weights, equally_spaced_train_and_eval_weights

def main(total_timesteps: int = 500000, exp_type: str = "default", wandb_mode: str = "online", log: bool = True, seed: int = 0, use_argmax_for_envelope: bool = False, use_train_weights_for_envelope: bool = False, exp_notes: str = "", depth: int = 6, learning_rate: float = 2e-4, gradient_updates: int = 5, batch_size: int = 32, tau: float = 0.1, initial_epsilon: float = 1.0, final_epsilon: float = 0.05, epsilon_decay_fraction: float = 0.6):
    log = str(log).lower() == "true"    
    def make_env():
        env = mo_gym.make("fruit-tree-v0",depth=depth)
        env = MORecordEpisodeStatistics(env, gamma=0.99)
        # env = MOSyncVectorEnv(env)
        return env

    # exp_type options: "sparse", "interpolation", "extrapolation", "dist_shift"
    
    env = make_env()
    eval_env = make_env()
    dim = env.reward_dim
   
    experiment_name="Envelope_FruitTree_"+str(total_timesteps)+"_"+exp_type+"_"+exp_notes+"_"+str(seed)
    agent = Envelope(
        env,
        seed=seed,
        max_grad_norm=1.0,#0.1 CHECK WAS TOO LOW
        learning_rate=learning_rate,# 3e-4 CHECK WAS LOW 
        gamma=0.99,
        batch_size=batch_size,
        net_arch=[256, 256],
        buffer_size=int(1e5),
        initial_epsilon=initial_epsilon,
        final_epsilon=final_epsilon,
        epsilon_decay_steps=int(total_timesteps * epsilon_decay_fraction),
        initial_homotopy_lambda=0.9,
        final_homotopy_lambda=0.85,
        homotopy_decay_steps=10000,
        learning_starts=1000,
        envelope=True,
        gradient_updates=gradient_updates,
        target_net_update_freq=1000,  # 1000,  # 500 reduce by gradient updates
        tau=tau,
        log=log,
        wandb_mode=wandb_mode,
        project_name="MORL-Baselines",
        experiment_name=experiment_name,
        group = experiment_name.rsplit("_", 1)[0]
    )

    agent.train(
        total_timesteps=total_timesteps,
        total_episodes=None,
        # train_weights=train_weights,
        use_argmax_for_envelope=use_argmax_for_envelope,
        use_train_weights_for_envelope=use_train_weights_for_envelope,  
        eval_env=eval_env,
        ref_point=np.zeros(env.unwrapped.reward_dim),
        known_pareto_front=env.unwrapped.pareto_front(gamma=0.99),
        # eval_weights = eval_weights,
        num_eval_weights_for_front=50,
        eval_freq=5000,
        # reset_num_timesteps=False,
        # reset_learning_starts=False,
        checkpoints=True,
        save_freq=100000,
    )


if __name__ == "__main__":
    fire.Fire(main)