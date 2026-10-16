import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create the environment
env = make_vec_env("LunarLander-v3", n_envs=1)

# Create the PPO agent (small MLP policy)
model = PPO("MlpPolicy", env, verbose=1)

# Train for 1 million steps (takes ~15-40 min on a laptop)
model.learn(total_timesteps=1_000_000)

# Save the model
model.save("../models/ppo_lunarlander")
print("Training finished and model saved!")
