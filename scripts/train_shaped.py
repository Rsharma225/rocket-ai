import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from reward_wrapper import CustomRewardWrapper


def make_env():
    env = gym.make("LunarLander-v3")
    env = CustomRewardWrapper(env, time_penalty=0.05, touchdown_bonus=25.0)
    return env


env = make_vec_env(make_env, n_envs=1)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("../models/ppo_lunarlander_shaped")
print("Shaped model saved!")
