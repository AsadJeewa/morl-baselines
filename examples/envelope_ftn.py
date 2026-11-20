import fire
import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.wrappers import MORecordEpisodeStatistics
from morl_baselines.multi_policy.envelope.envelope import Envelope

# Settings follow Yang et al. (2019), "A Generalized Algorithm for MORL and Policy Adaptation":
#   lr 1e-3, batch 32, replay memory 4000, target network copied every 100 steps,
#   MLP with 4 hidden layers {16,32,64,32} x (dim(S)+m) = x 8 -> [128, 256, 512, 256],
#   N_omega = 32 sampled preferences per update, prioritised replay + double Q-learning,
#   one update per step once a batch is available, homotopy lambda 0 -> 1, 5000 episodes.
# NOT stated in the paper for FTN (assumptions, marked below): gamma, epsilon schedule,
#   homotopy schedule shape, gradient clipping.


def main(
    total_timesteps: int = 300000,         # 5000 episodes x 6 steps (depth 6)
    exp_type: str = "paper",
    wandb_mode: str = "online",
    log: bool = True,
    seed: int = 0,
    use_argmax_for_envelope: bool = False,
    use_train_weights_for_envelope: bool = False,
    exp_notes: str = "",
    depth: int = 6,
    learning_rate: float = 1e-3,
    gradient_updates: int = 1,
    batch_size: int = 32,
    num_sample_w: int = 32,
    buffer_size: int = 4000,
    tau: float = 1.0,
    target_net_update_freq: int = 100,
    initial_epsilon: float = 0.5,         # assumption: dialog task in the paper used 0.5 -> 0
    final_epsilon: float = 0.01,
    epsilon_decay_fraction: float = 1.0,
    homotopy_decay_fraction: float = 1.0,  # assumption: lambda rises 0 -> 1 over the whole run (linear here)
    num_eval_weights_for_front: int = 200,
    eval_freq: int = 1000,
):
    log = str(log).lower() == "true"

    def make_env():
        env = mo_gym.make("fruit-tree-v0", depth=depth)
        env = MORecordEpisodeStatistics(env, gamma=0.99)
        return env

    env = make_env()
    eval_env = make_env()

    experiment_name = "Envelope_FruitTree_" + str(total_timesteps) + "_" + exp_type + "_" + exp_notes + "_" + str(seed)
    agent = Envelope(
        env,
        seed=seed,
        max_grad_norm=1.0,                 # not in the paper; kept as in your previous script
        learning_rate=learning_rate,
        gamma=0.99,                        # assumption
        batch_size=batch_size,
        net_arch=[128, 256, 512, 256],
        buffer_size=buffer_size,
        initial_epsilon=initial_epsilon,
        final_epsilon=final_epsilon,
        epsilon_decay_steps=int(total_timesteps * epsilon_decay_fraction),
        initial_homotopy_lambda=0.0,
        final_homotopy_lambda=1.0,
        homotopy_decay_steps=int(total_timesteps * homotopy_decay_fraction),
        learning_starts=batch_size,        # "after at least a batch of experiences are stored"
        envelope=True,
        num_sample_w=num_sample_w,
        per=True,
        gradient_updates=gradient_updates,
        target_net_update_freq=target_net_update_freq,
        tau=tau,                           # 1.0 = hard copy every target_net_update_freq steps
        log=log,
        project_name="MORL-Baselines",
        experiment_name=experiment_name,
        group=experiment_name.rsplit("_", 1)[0],
    )

    agent.train(
        total_timesteps=total_timesteps,
        total_episodes=None,
        use_argmax_for_envelope=use_argmax_for_envelope,
        use_train_weights_for_envelope=use_train_weights_for_envelope,
        eval_env=eval_env,
        ref_point=np.zeros(env.unwrapped.reward_dim),
        known_pareto_front=env.unwrapped.pareto_front(gamma=0.99),
        num_eval_weights_for_front=num_eval_weights_for_front,
        eval_freq=eval_freq,
        checkpoints=True,
        save_freq=10000,
    )


if __name__ == "__main__":
    fire.Fire(main)