import gymnasium as gym
import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers import MORecordEpisodeStatistics
import numpy as np
# from gymnasium.utils.play import play
from cleanrl_utils.utils import get_base_env
from morl_baselines.multi_policy.envelope.envelope import Envelope
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD
from morl_baselines.common.weights import equally_spaced_weights
from morl_baselines.common.plot_utils import plot_preferences #TODO pairwise
import torch
import time
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
import fire
import ast

def main(algo:str, seed: int = 0, env_id: str = "minecart-v0", num_neurons: int = 256, num_layers: int = 4, checkpoint_file: str = None, exp_note: str = ""):
    RENDER_DELAY = 0
    net_arch = [int(num_neurons)] * int(num_layers)
    if checkpoint_file:
        checkpoint_location = str(Path("examples/weights") / f"{Path(checkpoint_file).name}.tar")

    algo = algo.lower()
    is_gpi = "gpi" in algo
    is_env = "env" in algo
    if not (is_gpi or is_env):
        raise ValueError("Only Envelope and GPI are supported.")
    
    env = mo_gym.make(env_id)
    env = MORecordEpisodeStatistics(env, gamma=0.98)
    obs, info = env.reset()
    done = False
    envelope = True
    num_obj = get_base_env(env.env).reward_dim
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = None
    checkpoint = None
    config = {}

    if checkpoint_file:
        checkpoint = torch.load(checkpoint_location,map_location=device)
        config = checkpoint.get("config", {})
    if is_env:
        net_arch = config.get("net_arch", net_arch)
        envelope = config.get("envelope", envelope)
        agent = Envelope(
            env,
            net_arch=net_arch,
            envelope=envelope,
            log=False,
            device=device,
        )
    elif is_gpi:
        net_arch = config.get("net_arch", net_arch)
        num_nets = config.get("num_nets", 2)
        gpi_pd = config.get("gpi_pd", False)
        use_gpi = config.get("use_gpi", True)
        per = config.get("per", gpi_pd)
        # layer_norm = config.get("layer_norm", layer_norm)
        # drop_rate = config.get("drop_rate", drop_rate)
        agent = GPIPD(
            env,
            net_arch=net_arch,
            num_nets=num_nets,
            gpi_pd=gpi_pd,
            use_gpi=use_gpi,
            per=per,
            dyna=gpi_pd,
            # layer_norm=layer_norm,
            # drop_rate=drop_rate,
            log=False,
            device=device,
        )
    if checkpoint_file is not None:
        agent.load(checkpoint_location, load_replay_buffer=False)
    if is_gpi:
        for q_net in agent.q_nets:
            q_net.eval()
    if is_env:
        agent.q_net.eval()
        agent.target_q_net.eval()

    num_eval_weights = 100
    weights = equally_spaced_weights(dim=env.reward_dim, n=num_eval_weights, seed=seed+1000)
    all_weights = []
    all_returns = []

    for idx, w_np in enumerate(weights):

        obs, info = env.reset()
        done = False

        ep_return = np.zeros(agent.reward_dim, dtype=np.float32)

        while not done:

            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            w_tensor = torch.tensor(w_np, dtype=torch.float32)
            with torch.no_grad():
                if is_gpi:
                    action = agent.gpi_action(obs_tensor, w_tensor)
                elif is_env:
                    q = agent.q_net(obs_tensor.unsqueeze(0), w_tensor.unsqueeze(0))
                    action = (q * w_tensor.unsqueeze(1)).sum(dim=2).argmax(dim=1).item()
            obs, vec_reward, terminated, truncated, info = env.step(action)
            time.sleep(RENDER_DELAY)

            ep_return += vec_reward
            done = terminated or truncated

        all_weights.append(w_np)
        all_returns.append(ep_return)

        print(
            f"{idx+1}/{num_eval_weights}",
            "w =", np.round(w_np, 3),
            "return =", np.round(ep_return, 3)
        )


    all_weights = np.array(all_weights)
    all_returns = np.array(all_returns)

    # flip cost objective (Minecart-specific)
    # all_returns[:, 2] = -all_returns[:, 2]

    print("\n=== Correlations ===")

    for obj in range(agent.reward_dim):

        p_corr, _ = pearsonr(all_weights[:, obj], all_returns[:, obj])
        s_corr, _ = spearmanr(all_weights[:, obj], all_returns[:, obj])

        print(f"Obj {obj} | Pearson: {p_corr:.3f} | Spearman: {s_corr:.3f}")


    fig, axes = plt.subplots(1, num_obj, figsize=(5 * num_obj, 4))

    for i in range(num_obj):

        x = all_weights[:, i]
        y = all_returns[:, i]

        axes[i].scatter(x, y)

        coeffs = np.polyfit(x, y, 1)
        line = np.poly1d(coeffs)

        xs = np.linspace(x.min(), x.max(), 100)
        axes[i].plot(xs, line(xs))

        axes[i].set_xlabel(f"w[{i}]")
        axes[i].set_ylabel(f"r[{i}]")
        axes[i].set_title(f"Obj {i}")

    plt.tight_layout()
    plt.savefig("weight_return_scatter_"+exp_note+".png")
    plot_preferences(agent, algo,env, n_points=30, exp_note=exp_note)


if __name__ == "__main__":
    fire.Fire(main)