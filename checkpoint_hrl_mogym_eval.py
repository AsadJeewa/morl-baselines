import torch
import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers import MORecordEpisodeStatistics, SingleRewardWrapper
from cleanrl.moppo_decomp import Agent
from cleanrl.hrl_moppo_decomp import Controller
import gymnasium as gym
import numpy as np
from mo_gymnasium.wrappers.vector import MOSyncVectorEnv
from cleanrl_utils.utils import get_base_env
from tqdm import tqdm
import time
# --- Load checkpoint ---
controller_checkpoint_path = "checkpoint/four-room-easy/controller_checkpoint_300.pt"  # adjust path
agents_checkpoint_path = "checkpoint/four-room-easy/agents_checkpoint_4880.pt"  # adjust path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agents_checkpoint = torch.load(agents_checkpoint_path, map_location=device)
controller_checkpoint = torch.load(controller_checkpoint_path, map_location=device)



# --- Environment factory ---
def make_env(env_id, obj_idx, render=False, seed=None):
    def thunk():
        if render:
            env = mo_gym.make(env_id, render_mode="human")
        else:
            env = mo_gym.make(env_id)
        env = MORecordEpisodeStatistics(env, gamma=0.98)
        # env = SingleRewardWrapper(env, obj_idx)
        if seed is not None:
            env.reset(seed=seed)
        return env

    return thunk

idx = 0
env_id = "four-room-easy-v0"
env = MOSyncVectorEnv([make_env(env_id, idx)]) #expect list of callable of gym env
obj_duration = 8 #TODO get from model
# --- Create agent and load weights ---
base_env = get_base_env(env.envs[0])
num_objectives = base_env.reward_dim
controller = Controller(env).to(device)
controller.load_state_dict(controller_checkpoint["controller"])
# obj_duration= controller.load_state_dict(controller_checkpoint["obj_duration"])
agents = [Agent(env).to(device) for i in range(num_objectives)]
for idx, agent in enumerate(agents):
        agent.load_state_dict(agents_checkpoint["agents"][idx])
        agent.eval()  # evaluation mode

# --- Reset environment ---
obs, _ = env.reset(seed=42)
done = False

num_seeds = 2
episodes_per_seed = 2
all_vector_rewards = []
all_scalar_rewards = []
for seed in tqdm(range(num_seeds)):
    env = MOSyncVectorEnv([make_env(env_id,idx,render=True,seed=seed) for _ in range(1)])  # single env
    base_env = get_base_env(env.envs[0])
    for ep in tqdm(range(episodes_per_seed)):
        obs, _ = env.reset(seed=seed)
        done = False
        ep_reward = 0.0
        while not done:
            # obs is already batched for vector env
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

            with torch.no_grad():
                hl_action, hl_logprob, hl_entropy, hl_value = controller.get_action_and_value(obs_tensor)
                base_env.log_info = hl_action
                for k in range(obj_duration): #each env is seperate
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
                    action, _, _, _ = agents[hl_action].get_action_and_value(obs_tensor)
                    # Convert to numpy and step
                    action_np = action.cpu().numpy()
                    obs, reward, terminated, truncated, info = env.step(action_np)
                    time.sleep(1)
                    ep_reward += reward
                    # Only one environment in batch, so index 0
                    done = terminated[0] or truncated[0]
        all_vector_rewards.append(ep_reward)
        all_scalar_rewards.append(np.mean(ep_reward))
        env.close()
mean_scalar_reward = np.mean(all_scalar_rewards)
std_scalar_reward = np.std(all_scalar_rewards)
average_vector_rewards = np.vstack(all_vector_rewards)
mean_vector_reward = np.mean(average_vector_rewards,axis=0)
std_vector_reward = np.std(average_vector_rewards,axis=0)
for i, reward in enumerate(mean_vector_reward):
    print(f"Average Reward Obj{i}: {mean_vector_reward[i]:.3f} ± {std_vector_reward[i]:.3f}")
print(f"Average Scalar Reward: {mean_scalar_reward:.3f} ± {std_scalar_reward:.3f}")