import os
import gymnasium as gym
import mo_gymnasium as mo_gym
from gymnasium.wrappers import RecordVideo


def main():
    env_id = "deep-sea-treasure-v0"

    env = mo_gym.make(env_id, render_mode="rgb_array")

    video_dir = "./videos_test"
    os.makedirs(video_dir, exist_ok=True)

    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda ep: ep == 0  # ONLY first episode
    )

    obs, info = env.reset()

    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        done = terminated or truncated

    env.close()

    print("Done. Total reward:", total_reward)
    print("Video saved in:", video_dir)


if __name__ == "__main__":
    main()