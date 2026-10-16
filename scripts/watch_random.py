import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")

obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Obs: {obs}, Reward: {reward:.2f}")
    if terminated or truncated:
        obs, info = env.reset()

env.close()
