import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
# from gymnasium.utils.play import play
from cleanrl_utils.utils import get_base_env

# env = mo_gym.make('resource-gathering-v0', render_mode="human")
env_id = "four-room-easy-v0"
env = mo_gym.make(env_id, render_mode="human")
env = mo_gym.wrappers.SingleRewardWrapper(env, 2)
# play(env,zoom=3)#only mario it seems

obs, info = env.reset()
# but vector_reward is a numpy array!
#next_obs, vector_reward, terminated, truncated, info = env.step(your_agent.act(obs))
# actions = np.array([1,1,1,1,1,1,1,2,2,2,2,2,2,2])

done = False
steps = 0
num_obj = get_base_env(env.env).reward_dim
sum_ep_rewards = np.zeros(num_obj, dtype=np.float32)

num_ep = 10
for ep in range(1, num_ep + 1):
    while not done:
        action = np.random.randint(env.action_space.n)
        obs, vec_reward, terminated, truncated, info = env.step(action)#(env.action_space.sample())
        sum_ep_rewards += vec_reward
        steps+=1
        done = terminated or truncated
    print("SUM: ",sum_ep_rewards)
    print("END EPISODE")
    sum_ep_rewards = 0
    done = False

# Optionally, you can scalarize the reward function with the LinearReward wrapper
#env = mo_gym.wrappers.LinearReward(env, weight=np.array([0.8, 0.2, 0.2]))