import secrets

import pandas as pd
import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers import MORecordEpisodeStatistics
import numpy as np
# from cleanrl_utils.utils import get_base_env
# from gymnasium.utils.play import play
from morl_baselines.multi_policy.envelope.envelope import Envelope
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD
from morl_baselines.common.weights import equally_spaced_weights
from morl_baselines.common.plot_utils import plot_preferences, plot_correlations #TODO pairwise
from morl_baselines.common.evaluation import log_all_multi_policy_metrics, compute_all_controllability_metrics
from morl_baselines.common.pareto import filter_pareto_dominated
import wandb

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import fire
import ast
from gymnasium.wrappers import RecordVideo
import os
from tqdm import tqdm

def main(algo: str, seed: int = 0, env_id: str = "minecart-v0", 
         ref_point: str = "[-1,-1,-200]",
         use_wandb: bool = False,
         project_name: str = "MORL-Baselines",
         num_eval_episodes: int = 10, num_neurons: int = 256, num_layers: int = 4, 
         checkpoint_file: str = None, exp_note: str = "", n_points=30, 
         record_video: bool = False, right_angled: bool = True, 
         mine_config: str = "mine_config.json"):
    RENDER_DELAY = 0
    right_angled = str(right_angled).lower() == "true"
    ref_point = np.array(ast.literal_eval(ref_point))
    net_arch = [int(num_neurons)] * int(num_layers)
    if checkpoint_file:
        checkpoint_location = checkpoint_location = f"examples/weights/{Path(checkpoint_file).name}.tar"
    algo = (algo or "random").lower()
    is_gpi = "gpi" in algo
    is_env = "env" in algo
    is_random = not (is_gpi or is_env)    
    kwargs = {}
    if record_video:
        kwargs["render_mode"] = "rgb_array"
    if mine_config and "minecart" in env_id.lower():
        kwargs["config"] = mine_config
        
    # 3. Create the environment safely
    env = mo_gym.make(env_id, **kwargs)
    env = MORecordEpisodeStatistics(env, gamma=0.98)
    if record_video:
        video_dir = f"./videos/{env_id}/{algo}_{exp_note}"
        os.makedirs(video_dir, exist_ok=True)

        env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda ep: True 
            #  episode_trigger=lambda e: e % 1000 == 0
        )

    obs, info = env.reset()
    done = False
    envelope = True
    # base_env = get_base_env(env.env)
    num_obj = env.unwrapped.reward_dim
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = None
    checkpoint = None
    config = {}

    if checkpoint_file:
        checkpoint = torch.load(checkpoint_location,map_location=device)
        config = checkpoint.get("config", {})
        training_seed = checkpoint.get("seed", seed)
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
    if not os.path.exists(f"results/{env_id}"):
        os.makedirs(f"results/{env_id}", exist_ok=True)

    weights = equally_spaced_weights(dim=num_obj, n=num_eval_weights, seed=seed+1000)
    all_weights, all_returns = evaluate_policy(agent=agent, env=env, weights=weights, algo=algo, num_eval_episodes=num_eval_episodes)

    run_id = secrets.token_urlsafe(4)[:6]
    # plot_correlations(env, algo, all_weights, all_returns, exp_note=exp_note)
    plot_preferences(run_id=run_id,seed=training_seed,agent=agent, env=env, algo=algo, n_points=n_points, exp_note=exp_note, right_angled=right_angled)

    metrics = compute_all_controllability_metrics(all_weights, all_returns)
    print("Preference controllability:", metrics["preference_controllability"])
    print("Local sensitivity:", metrics["local_sensitivity"])
    print("Objective controllability:", [v for k, v in metrics.items() if k.startswith("objective_controllability")])

    filtered_front = list(filter_pareto_dominated(all_returns))

    if use_wandb:
        wandb.init(
            project=project_name,
            name=f"{algo}_{env_id}_{exp_note}_seed{training_seed}",
        )

    log_all_multi_policy_metrics(
        current_front=filtered_front,
        hv_ref_point=ref_point,
        reward_dim=num_obj,
        global_step=0,
        n_sample_weights=50,
    )

    if use_wandb:
        wandb.log({
            "eval/preference_controllability": metrics["preference_controllability"],
            "eval/local_sensitivity": metrics["local_sensitivity"],
            **{f"eval/{k}": v for k, v in metrics.items() if k.startswith("objective_controllability")},
        })
        wandb.finish()

    data = {
        "run_id": run_id,
        "training_seed": training_seed,
        "preference_controllability": metrics["preference_controllability"],
        "local_sensitivity": metrics["local_sensitivity"],
    }
    for k, v in metrics.items():
        if k.startswith("objective_controllability"):
            data[k] = v

    df = pd.DataFrame([data])
    filepath = f"results/{env_id}/metrics_{algo}_{exp_note}.csv"
    df.to_csv(filepath, mode="a", index=False, header=not os.path.isfile(filepath))
    env.close()

def evaluate_policy(agent, env, weights, algo, num_eval_episodes):
    algo_lower = algo.lower()
    is_gpi = "gpi" in algo_lower
    is_env = "env" in algo_lower

    all_weights = []
    all_returns = []
    
    i = 0
    total_episodes = len(weights) * num_eval_episodes
    pbar = tqdm(total=total_episodes, desc="Evaluating")
    for w_np in weights:
        print(i, " of ", len(weights))
        i += 1

        episode_returns = []

        for episode in range(num_eval_episodes):
            obs, _ = env.reset()
            done = False
            ep_return = np.zeros(env.unwrapped.reward_dim, dtype=np.float32)

            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                w_tensor = torch.tensor(w_np, dtype=torch.float32)

                with torch.no_grad():
                    if is_gpi:
                        action = agent.gpi_action(obs_tensor, w_tensor)

                    elif is_env:
                        q = agent.q_net(
                            obs_tensor.unsqueeze(0),
                            w_tensor.unsqueeze(0)
                        )
                        action = torch.einsum(
                            "bar,r->ba", q, w_tensor
                        ).argmax(dim=1).item()

                    else:
                        action = env.action_space.sample()

                obs, r, terminated, truncated, _ = env.step(action)
                ep_return += r
                done = terminated or truncated
            pbar.update(1)
            episode_returns.append(ep_return)

        # Average the episodes for this preference
        mean_return = np.mean(episode_returns, axis=0)

        all_weights.append(w_np)
        all_returns.append(mean_return)
    pbar.close()
    return np.array(all_weights), np.array(all_returns)

if __name__ == "__main__":
    fire.Fire(main)