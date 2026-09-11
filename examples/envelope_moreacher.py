import fire
import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.wrappers import MORecordEpisodeStatistics
import os
os.environ["MUJOCO_GL"] = "egl"
#from mo_gymnasium.wrappers.vector import MOSyncVectorEnv
from morl_baselines.multi_policy.envelope.envelope import Envelope
from morl_baselines.common.weights import equally_spaced_weights, random_weights, extrema_weights, equally_spaced_train_and_eval_weights

def main(total_timesteps: int=500000, exp_type: str = "default", wandb_mode: str = "online", log: bool = True, seed: int = 0, use_argmax_for_envelope: bool = False, use_train_weights_for_envelope: bool = False, exp_notes: str = ""):
    log = str(log).lower() == "true"    
    def make_env():
        env = mo_gym.make("mo-reacher-v4")
        env = MORecordEpisodeStatistics(env, gamma=0.99)
        # env = MOSyncVectorEnv(env)
        return env

    # exp_type options: "sparse", "interpolation", "extrapolation", "dist_shift"
    
    env = make_env()
    eval_env = make_env()
    dim = env.reward_dim
   
    train_weights=None
    eval_weights = None
    if exp_type is not None:
        if exp_type.lower() == "intereasy":
            train_weights, eval_weights = equally_spaced_train_and_eval_weights(dim=dim, n_train=20, n_eval=100,seed=seed)
        elif exp_type.lower() == "intermedium":
            train_weights, eval_weights = equally_spaced_train_and_eval_weights(dim=dim, n_train=10, n_eval=100,seed=seed)
        elif exp_type.lower() == "interdifficult":
            train_weights, eval_weights = equally_spaced_train_and_eval_weights(dim=dim, n_train=5, n_eval=100,seed=seed)

    experiment_name="Envelope_MO-Reacher_"+str(total_timesteps)+"_"+exp_type+"_"+exp_notes+"_"+str(seed)
    agent = Envelope(
        env,
        seed=seed,
        max_grad_norm=1.0,#0.1 CHECK WAS TOO LOW
        learning_rate=1e-4,# 3e-4 CHECK WAS LOW 
        gamma=0.99,
        batch_size=128,
        net_arch=[256, 256, 256, 256],
        buffer_size=int(2e6),
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay_steps=50000,
        initial_homotopy_lambda=0,
        final_homotopy_lambda=1,
        # homotopy_decay_steps=0,
        learning_starts=5000,
        per=True,
        envelope=True,
        gradient_updates=1,
        target_net_update_freq=1000,  # 1000,  # 500 reduce by gradient updates
        tau=1,
        log=log,
        wandb_mode=wandb_mode,
        project_name="MORL-Baselines",
        experiment_name=experiment_name,
        group = experiment_name.rsplit("_", 1)[0]
    )

    agent.train(
        total_timesteps=total_timesteps,
        total_episodes=None,
        train_weights=None,
        use_argmax_for_envelope=use_argmax_for_envelope,
        use_train_weights_for_envelope=use_train_weights_for_envelope,  
        eval_env=eval_env,
        ref_point=np.array([-100, -100, -100, -100]),
        known_pareto_front=None,
        eval_weights = None,
        num_eval_weights_for_front=100,
        eval_freq=1000,
        # reset_num_timesteps=False,
        # reset_learning_starts=False,
        checkpoints=True,
        save_freq=100000,
    )


if __name__ == "__main__":
    fire.Fire(main)
