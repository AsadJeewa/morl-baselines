import torch
import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers import MORecordEpisodeStatistics, SingleRewardWrapper
from cleanrl.moppo_decomp import Agent
import gymnasium as gym
import numpy as np
from cleanrl_utils import get_base_env, make_env
from tqdm import tqdm
# --- Load checkpoint ---
# checkpoint_path = "checkpoint/four-room-test/checkpoint_2480.pt"  # adjust path
# checkpoint_path = "../cleanrl/model/shapes-grid/fin_moppo_env__shapes-grid-v0__moppo_decomp__1__1760003876/checkpoint_2160.pt"  # adjust path
# checkpoint_path = "../cleanrl/model/shapes-grid/larger_grid_low_level__shapes-grid-v0__moppo_decomp__1__1760103414/checkpoint_2440.pt"  # adjust path
# checkpoint_path = "../cleanrl/model/shapes-grid/toy_1__shapes-grid-v0__moppo_decomp__1__1763472383/checkpoint_120.pt"  # adjust path
# checkpoint_path = "../cleanrl/model/shapes-grid/toy_2_penalty__shapes-grid-v0__moppo_decomp__1__1763473551/checkpoint_280.pt"  # adjust path
# checkpoint_path = "../cleanrl/model/shapes-grid/toy2_long__shapes-grid-v0__moppo_decomp__1__1763475033/checkpoint_1010.pt"  # adjust path
# checkpoint_path = "../cleanrl/model/shapes-grid/hard_low_level_pen__shapes-grid-v0__moppo_decomp__1__1763737176/checkpoint_2440.pt"  # adjust path
# checkpoint_path = "../cleanrl/model/shapes-grid/SAVE/cnn_low_level__shapes-grid-v0__moppo_decomp__1__1766134932/checkpoint_70.pt"  # adjust path
checkpoint_path = "../cleanrl/model/shapes-grid/SAVE/cnn_low_level_easy__shapes-grid-v0__moppo_decomp__1__1766138143/checkpoint_410.pt"  # adjust path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(checkpoint_path, map_location=device)

idx = 0
env_id = "shapes-grid-v0"
difficulty="easy"
env = gym.vector.SyncVectorEnv([make_env(env_id,idx=idx, difficulty=difficulty)]) #expect list of callable of gym env

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
for seed in tqdm(range(num_seeds)):
    env = gym.vector.SyncVectorEnv([make_env(env_id,idx=idx,seed=seed,difficulty=difficulty,render=True) for _ in range(1)])  # single env
    base_env = get_base_env(env.envs[0])
    base_env.set_specialisation(idx+1)
    for ep in tqdm(range(episodes_per_seed)):
        obs, _ = env.reset(seed=seed)
        done = False
        total_reward = 0.0
        while not done:
            # obs is already batched for vector env
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
            # Convert to numpy and step
            action_np = action.cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            total_reward += reward
            # Only one environment in batch, so index 0
            done = terminated[0] or truncated[0]
            if done:
                break
        all_rewards.append(total_reward)
    env.close()
mean_reward = np.mean(all_rewards)
std_reward = np.std(all_rewards)
print(f"Average Reward: {mean_reward:.3f} ± {std_reward:.3f}")