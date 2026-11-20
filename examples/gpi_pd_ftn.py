import fire
import mo_gymnasium as mo_gym
import numpy as np
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD
from mo_gymnasium.wrappers import MORecordEpisodeStatistics

# GPI-LS settings follow Alegre et al. (2023), Appendix 2.2 (the Minecart experiments, the closest
# deep discrete-action setup; the paper has no FTN experiments):
#   MLP 4 x 256 with Dropout + LayerNorm, Adam lr 3e-4, mini-batch 256, G = 20 gradient updates per
#   environment step, epsilon-greedy annealed linearly 1 -> 0.05 over the first 50,000 steps,
#   GPI-LS = model-free with a plain uniform replay buffer (no Dyna, no prioritisation).
# NOT stated in the paper (assumptions, marked): gamma for FTN, buffer size, target update,
#   learning start. timesteps_per_iter is smaller than the paper's Minecart value (about 10k) because the
#   FTN front has 64 points and GPI-LS adds roughly one weight per iteration.


def main(
    algo: str = "gpi-ls",
    total_timesteps: int = 300000,
    timesteps_per_iter: int = 2000,
    wandb_mode: str = "online",
    log: bool = True,
    seed: int = 0,
    exp_notes: str = "",
    depth: int = 6,
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    gradient_updates: int = 20,
    initial_epsilon: float = 1.0,
    final_epsilon: float = 0.05,
    epsilon_decay_steps: int = 50000,
    buffer_size: int = int(1e5),             # assumption
    learning_starts: int = 1000,             # assumption
    target_net_update_freq: int = 100,       # assumption (hard copy: tau = 1)
    num_eval_weights_for_front: int = 200,
    eval_freq: int = 5000,
):
    log = str(log).lower() == "true"

    def make_env():
        env = mo_gym.make("fruit-tree-v0", depth=depth)
        env = MORecordEpisodeStatistics(env, gamma=0.99)
        return env

    env = make_env()
    eval_env = make_env()

    experiment_name = "GPI_FruitTree_" + str(total_timesteps) + "_paper_" + exp_notes + "_" + str(seed)
    agent = GPIPD(
        env,
        seed=seed,
        num_nets=2,
        max_grad_norm=None,
        learning_rate=learning_rate,
        gamma=0.99,                          # assumption
        batch_size=batch_size,
        net_arch=[256, 256, 256, 256],
        buffer_size=buffer_size,
        initial_epsilon=initial_epsilon,
        final_epsilon=final_epsilon,
        epsilon_decay_steps=epsilon_decay_steps,
        learning_starts=learning_starts,
        per=False,                           # GPI-LS: uniform replay
        gpi_pd=False,                        # no GPI prioritisation
        use_gpi=True,
        dyna=False,                          # GPI-LS is model-free
        gradient_updates=gradient_updates,
        target_net_update_freq=target_net_update_freq,
        tau=1.0,
        drop_rate=0.01,
        layer_norm=True,
        log=log,
        wandb_mode=wandb_mode,               # fork-specific argument, as in your previous script
        project_name="MORL-Baselines",
        experiment_name=experiment_name,
        group=experiment_name.rsplit("_", 1)[0],   # fork-specific argument, as in your previous script
    )

    agent.train(
        total_timesteps=total_timesteps,
        eval_env=eval_env,
        ref_point=np.zeros(env.unwrapped.reward_dim),
        known_pareto_front=env.unwrapped.pareto_front(gamma=0.99),
        num_eval_weights_for_front=num_eval_weights_for_front,
        eval_freq=eval_freq,
        weight_selection_algo=algo,
        timesteps_per_iter=timesteps_per_iter,
        checkpoints=True,
        save_freq=100000,                    # fork-specific argument, as in your previous script
    )


if __name__ == "__main__":
    fire.Fire(main)