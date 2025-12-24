import torch
import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers import MORecordEpisodeStatistics
from cleanrl.moppo_decomp import Agent
from cleanrl.hrl_moppo_decomp import Controller
import gymnasium as gym
import numpy as np
from mo_gymnasium.wrappers.vector import MOSyncVectorEnv
from cleanrl_utils import get_base_env, make_env
from tqdm import tqdm
import time
import csv
# --- Load checkpoint --
controller_checkpoint_path = "../cleanrl/model/shapes-grid/cnn_high_level_easy_output_weights__shapes-grid-v0__hrl_moppo_decomp__1__1766410483/checkpoint_50.pt"
agents_checkpoint_path = "../cleanrl/model/shapes-grid/SAVE/cnn_low_level_easy__shapes-grid-v0__moppo_decomp__1__1766138143/checkpoint_410.pt"  # adjust path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agents_checkpoint = torch.load(agents_checkpoint_path, map_location=device)
controller_checkpoint = torch.load(controller_checkpoint_path, map_location=device)
RENDER_DELAY = 1

env_id = "shapes-grid-v0"
difficulty="easy"

env = MOSyncVectorEnv([make_env(env_id, difficulty=difficulty)]) #expect list of callable of gym env
# --- Create agent and load weights ---
base_env = get_base_env(env.envs[0])
num_objectives = base_env.reward_dim
controller = Controller(env).to(device)
controller.load_state_dict(controller_checkpoint["controller"])
obj_duration = controller_checkpoint["obj_duration"]
agents = [Agent(env).to(device) for i in range(num_objectives)]
for idx, agent in enumerate(agents):
        agent.load_state_dict(agents_checkpoint["agents"][idx])
        agent.eval()  # evaluation mode

# --- Reset environment ---
obs, _ = env.reset(seed=42)
done = False

step_counter = 0
num_seeds = 2
episodes_per_seed = 2
all_vector_rewards = []
all_scalar_rewards = []

log_file = "hl_action_log.csv"
with open(log_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Step", "HighLevelAction"])

    for seed in tqdm(range(num_seeds)):
        env = MOSyncVectorEnv([make_env(env_id,seed=seed,difficulty=difficulty,render=True) for _ in range(1)])  # single env
        base_env = get_base_env(env.envs[0])
        for ep in tqdm(range(episodes_per_seed), desc=f"Seed {seed}"):
            obs, _ = env.reset(seed=seed)           
            done = False
            ep_reward = 0.0
            while not done:
                # obs is already batched for vector env
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

                with torch.no_grad():
                    #TODO RESET SO HL DOES NOT USE SPECIALISED OBS
                    hl_action, hl_logprob, hl_entropy, hl_value = controller.get_action_and_value(obs_tensor)
                    base_env.log_info = hl_action
                    spec_obs = base_env.set_specialisation(hl_action.item()+1)
                    # Log the high-level action
                    writer.writerow([step_counter, hl_action.item()])
                    step_counter += 1
                    
                    for k in range(obj_duration): #each env is seperate
                        obs_tensor = torch.tensor(spec_obs, dtype=torch.float32).to(device)
                        action, _, _, _ = agents[hl_action].get_action_and_value(obs_tensor) #take action based on speciliased state
                        # Convert to numpy and step
                        action_np = action.cpu().numpy()
                        obs, reward, terminated, truncated, info = env.step(action_np)
                        time.sleep(RENDER_DELAY)
                        ep_reward += reward
                        # Only one environment in batch, so index 0
                        done = terminated[0] or truncated[0]
                        if done:
                            break
            all_vector_rewards.append(ep_reward)
            all_scalar_rewards.append(np.mean(ep_reward))
        env.close()
f.close()
mean_scalar_reward = np.mean(all_scalar_rewards)
std_scalar_reward = np.std(all_scalar_rewards)
average_vector_rewards = np.vstack(all_vector_rewards)
mean_vector_reward = np.mean(average_vector_rewards,axis=0)
std_vector_reward = np.std(average_vector_rewards,axis=0)
for i, reward in enumerate(mean_vector_reward):
    print(f"Average Reward Obj{i}: {mean_vector_reward[i]:.3f} ± {std_vector_reward[i]:.3f}")
print(f"Average Scalar Reward: {mean_scalar_reward:.3f} ± {std_scalar_reward:.3f}")