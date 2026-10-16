import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, time_penalty=0.05, touchdown_bonus=25.0):
        super().__init__(env)
        self.time_penalty = time_penalty
        self.touchdown_bonus = touchdown_bonus
        self._both_legs_grounded = False

    def reset(self, **kwargs):
        self._both_legs_grounded = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        reward -= self.time_penalty

        both_legs_grounded = bool(obs[6] and obs[7])
        if both_legs_grounded and not self._both_legs_grounded:
            horizontal_speed = abs(obs[2])
            vertical_speed = abs(obs[3])
            angle = abs(obs[4])
            angular_speed = abs(obs[5])

            landing_quality = (
                max(0.0, 1.0 - horizontal_speed / 0.5)
                * max(0.0, 1.0 - vertical_speed / 0.5)
                * max(0.0, 1.0 - angle / 0.2)
                * max(0.0, 1.0 - angular_speed / 0.5)
            )
            reward += self.touchdown_bonus * landing_quality

        self._both_legs_grounded = both_legs_grounded

        return obs, reward, terminated, truncated, info
