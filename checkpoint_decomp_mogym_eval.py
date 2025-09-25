import torch
import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers import MORecordEpisodeStatistics, SingleRewardWrapper
from cleanrl.moppo_decomp import Agent
import gymnasium as gym
import numpy as np
from cleanrl_utils.utils import get_base_env

# --- Load checkpoint ---
# checkpoint_path = "checkpoint/four-room-test/checkpoint_2480.pt"  # adjust path
checkpoint_path = "checkpoint/four-room-easy/agents_checkpoint_4880.pt"  # adjust path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(checkpoint_path, map_location=device)



# --- Environment factory ---
def make_env(env_id, obj_idx, render=False, seed=None):
    def thunk():
        if render:
            env = mo_gym.make(env_id, render_mode="human")
        else:
            env = mo_gym.make(env_id)
        env = MORecordEpisodeStatistics(env, gamma=0.98)
        env = SingleRewardWrapper(env, obj_idx)
        if seed is not None:
            env.reset(seed=seed)
        return env

    return thunk

idx = 1
env_id = "four-room-easy-v0"
env = gym.vector.SyncVectorEnv([make_env(env_id, idx)]) #expect list of callable of gym env

# --- Create agent and load weights ---
agent = Agent(env)
agent.load_state_dict(checkpoint["agents"][idx])
agent.eval()  # evaluation mode

# --- Reset environment ---
obs, _ = env.reset(seed=42)
done = False

num_seeds = 3
episodes_per_seed = 500
all_rewards = []
for seed in range(num_seeds):
    env = gym.vector.SyncVectorEnv([make_env(env_id,idx,render=True,seed=seed) for _ in range(1)])  # single env
    base_env = get_base_env(env.envs[0])
    spec_obs = base_env.update_specialisation(idx+1)
    for ep in range(episodes_per_seed):
        obs, _ = env.reset(seed=seed)
        done = False
        total_reward = 0.0
        while not done:
            # obs is already batched for vector env
            obs_tensor = torch.tensor(spec_obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
            # Convert to numpy and step
            action_np = action.cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            spec_obs = base_env.get_spec_obs()
            total_reward += reward
            # Only one environment in batch, so index 0
            done = terminated[0] or truncated[0]
        all_rewards.append(total_reward)
    env.close()
mean_reward = np.mean(all_rewards)
std_reward = np.std(all_rewards)
print(f"Average Reward: {mean_reward:.3f} ± {std_reward:.3f}")