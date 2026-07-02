import fire
import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.wrappers import MORecordEpisodeStatistics
from morl_baselines.multi_policy.envelope.envelope import Envelope
from morl_baselines.common.weights import equally_spaced_weights, random_weights, extrema_weights, equally_spaced_train_and_eval_weights

def main(exp_type: str = None, total_timesteps: int = 100000, wandb_mode: str = "online", log: bool = True, seed: int = 0, use_argmax_for_envelope: bool = False, use_train_weights_for_envelope: bool = False, exp_notes: str = ""):
    log = str(log).lower() == "true" 
    def make_env():
        env = mo_gym.make("deep-sea-treasure-v0")
        env = MORecordEpisodeStatistics(env, gamma=0.98)
        # env = MOSyncVectorEnv(env)
        return env

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
            
    agent = Envelope(
        env,
        seed=seed,
        max_grad_norm=1.0,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=256,
        net_arch=[256,256],
        buffer_size=int(5e4),
        initial_epsilon=0.5,
        final_epsilon=0.01,
        epsilon_decay_steps=total_timesteps*0.6,
        initial_homotopy_lambda=0.2,
        final_homotopy_lambda=0.2,
        homotopy_decay_steps=total_timesteps,
        learning_starts=1000,
        envelope=True,
        gradient_updates=2,
        target_net_update_freq=1000, 
        tau=1,
        log=log,
        wandb_mode=wandb_mode,
        project_name="MORL-Baselines",
        experiment_name="Envelope_DST_"+str(total_timesteps)+"_"+exp_type+"_"+exp_notes,
    )

    agent.train(
        total_timesteps=total_timesteps,
        total_episodes=None,
        train_weights=train_weights,
        use_argmax_for_envelope=use_argmax_for_envelope,
        use_train_weights_for_envelope=use_train_weights_for_envelope,  
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
