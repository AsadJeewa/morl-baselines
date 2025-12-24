import torch
import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers import MORecordEpisodeStatistics
from cleanrl.moppo_decomp import Agent
from cleanrl.weighted_hrl_moppo_decomp import Controller
from mo_gymnasium.wrappers.vector import MOSyncVectorEnv
from cleanrl_utils import get_base_env, make_env
import numpy as np
from tqdm import tqdm
import time
import csv
from morl_baselines.common.performance_indicators import (
    hypervolume,
    igd,
    cardinality,
    expected_utility,
    maximum_utility_loss,
    sparsity,
)
from morl_baselines.common.pareto import filter_pareto_dominated
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # only needed for 3D plots
#from morl_baselines.common.evaluation import log_all_multi_policy_metrics

# --- Load checkpoint ---
#controller_checkpoint_path = "../cleanrl/model/shapes-grid/weights_v0__shapes-grid-v0__weights_hrl_moppo_decomp__1__1762263777/checkpoint_2400.pt"
#controller_checkpoint_path = "../cleanrl/model/shapes-grid/larger_grid_HIGH_level_1__shapes-grid-v0__hrl_moppo_decomp__1__1760607246/checkpoint_800.pt"
controller_checkpoint_path = "../cleanrl/model/shapes-grid/hrl_weights_toy__shapes-grid-v0__weights_hrl_moppo_decomp__1__1763476770/checkpoint_500.pt"
#agents_checkpoint_path = "../cleanrl/model/shapes-grid/larger_grid_low_level__shapes-grid-v0__moppo_decomp__1__1760103414/checkpoint_2440.pt"
agents_checkpoint_path = "../cleanrl/model/shapes-grid/toy2_long__shapes-grid-v0__moppo_decomp__1__1763475033/checkpoint_1010.pt"
                                                           
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agents_checkpoint = torch.load(agents_checkpoint_path, map_location=device)
controller_checkpoint = torch.load(controller_checkpoint_path, map_location=device)
RENDER_DELAY = 1

env_id = "shapes-grid-v0"
difficulty="easy"

# --- Create agent and load weights ---
env = MOSyncVectorEnv([make_env(env_id, difficulty=difficulty)])  # single vector env
base_env = get_base_env(env.envs[0])
num_objectives = base_env.reward_dim

controller = Controller(env).to(device)
controller.load_state_dict(controller_checkpoint["controller"])
controller.eval()
obj_duration = controller_checkpoint["obj_duration"]
agents = [Agent(env).to(device) for _ in range(num_objectives)]
for idx, agent in enumerate(agents):
    agent.load_state_dict(agents_checkpoint["agents"][idx])
    agent.eval()  # evaluation mode

# --- Evaluation parameters ---
num_seeds = 10
episodes_per_seed = 10
all_vector_rewards = []

# --- Log file ---
log_file = "hl_action_log.csv"
csv_writer = csv.writer(open(log_file, "w", newline=""))
csv_writer.writerow(["Seed", "Episode", "Step", "HighLevelAction"])

# --- Main evaluation loop ---
# for seed in tqdm(range(num_seeds), desc="Seeds"):
for seed in range(num_seeds):
    env = MOSyncVectorEnv([make_env(env_id,idx,seed=seed,difficulty=difficulty,render=True)])
    base_env = get_base_env(env.envs[0])
    
    for ep in range(episodes_per_seed):
    # for ep in tqdm(range(episodes_per_seed), desc=f"Seed {seed}"):
        obs, _ = env.reset(seed=seed)
        done = False
        ep_vector_reward = np.zeros(num_objectives)
        step_counter = 0
        hrl_counter = 0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            
            # Generate a weight vector for this episode
            # current_weights = torch.tensor(
            #     np.random.dirichlet(np.ones(num_objectives), size=1),
            #     dtype=torch.float32
            # ).to(device)
            current_weights = (torch.ones(1, num_objectives, dtype=torch.float32) / num_objectives).to(device)
            # current_weights = w = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32).to(device)
            with torch.no_grad():
                #TODO RESET SO HL DOES NOT USE SPECIALISED OBS
                hl_action, _, _, _ = controller.get_action_and_value(obs_tensor, current_weights)
                # print(hl_action)
                hl_action_idx = hl_action.item()
                hrl_counter+=1
                base_env.log_info = hl_action
                base_env.set_specialisation(hl_action_idx + 1)
            
            csv_writer.writerow([seed, ep, step_counter, hl_action_idx])
            step_counter += 1
            
            for k in range(obj_duration):
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
                action, _, _, _ = agents[hl_action_idx].get_action_and_value(obs_tensor)
                action_np = action.cpu().numpy()
                
                obs, reward, terminated, truncated, _ = env.step(action_np)
                reward = np.array(reward).flatten()
                ep_vector_reward += reward
                
                time.sleep(RENDER_DELAY)
                done = terminated[0] or truncated[0]
                if done:
                    break

        all_vector_rewards.append(ep_vector_reward)
    env.close()
csv_writer.close()
# --- Compute Pareto front metrics ---
all_vector_rewards = np.vstack(all_vector_rewards)  # shape: (num_episodes, num_objectives)

# Mean + std per objective
mean_vector_reward = np.mean(all_vector_rewards, axis=0)
std_vector_reward = np.std(all_vector_rewards, axis=0)

#hv_ref_point = all_vector_rewards.min(axis=0) - 0.1
hv_ref_point = np.zeros(3)

# print(all_vector_rewards)
'''
# --- Use MORL-Baselines utilities ---
metrics = log_all_multi_policy_metrics(
    all_vector_rewards,
    hv_ref_point=hv_ref_point,
    reward_dim=num_objectives,
    global_step=0,
    n_sample_weights=50,
    wandb_run=None, # disable W&B logging
)
print("\nPareto Metrics (via MORL-Baselines):")
for k, v in metrics.items():
    print(f"{k}: {v:.3f}")
'''

pareto_front = filter_pareto_dominated(all_vector_rewards.tolist())  # list of 1D arrays

# Hypervolume
#ref_point = all_vector_rewards.min(axis=0) - 0.1  # slightly worse than worst
#hv_indicator = HV(ref_point=hv_ref_point)
#hypervolume = hv_indicator.do(all_vector_rewards)
hv = hypervolume(hv_ref_point,pareto_front)
# Cardinality
card = cardinality(pareto_front)

# EUM (expected utility mean) using uniform random weights
num_samples = 1000
weights = np.random.dirichlet(np.ones(num_objectives), size=num_samples) # different to eval weights during training
eum = expected_utility(pareto_front, weights, utility=lambda w, x: np.dot(w, x))

# Maximum Utility Loss
mul = maximum_utility_loss(
    pareto_front,
    reference_set=all_vector_rewards.tolist(),  # or known_front if available
    weights_set=weights,
    utility=lambda w, x: np.dot(w, x),
)

# Sparsity
sp = sparsity(pareto_front)

# --- Print results ---
for i, reward in enumerate(mean_vector_reward):
    print(f"Objective {i}: {reward:.3f} ± {std_vector_reward[i]:.3f}")

print(f"Hypervolume: {hv:.3f}")
print(f"Cardinality (Pareto size): {card}")
print(f"Expected Utility Mean (EUM): {eum:.3f}")
print(f"Maximum Utility Loss (MUL): {mul:.3f}")
print(f"Sparsity (diversity): {sp:.3f}")
# print(f"IGD: {igd_val:.3f}")  # if known front available


# Pareto Front
num_objectives = all_vector_rewards.shape[1]
pareto_array = np.array(pareto_front)
# print(pareto_array)
fig_name = "pareto_checkpoint.png"
if num_objectives == 2:
    plt.figure(figsize=(6,6))
    plt.scatter(pareto_array[:,0], pareto_array[:,1], c='red', label='Pareto Front')
    plt.xlabel('Objective 0')
    plt.ylabel('Objective 1')
    plt.title('Pareto Front (2 Objectives)')
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_name)  # saves figure to file
    plt.close()  # closes the figure
elif num_objectives == 3:
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pareto_array[:,0], pareto_array[:,1], pareto_array[:,2], c='red', label='Pareto Front')
    ax.set_xlabel('Objective 0')
    ax.set_ylabel('Objective 1')
    ax.set_zlabel('Objective 2')
    ax.set_title('Pareto Front (3 Objectives)')
    plt.savefig(fig_name)  # saves figure to file
    plt.close()  # closes the figure


else:
    print("Pareto visualization not implemented for more than 3 objectives. Consider using pairwise scatter plots or dimensionality reduction.")
    