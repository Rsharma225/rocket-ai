import gymnasium as gym
from stable_baselines3 import PPO
import pygame


class ScoreOverlay(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episode_reward = 0.0
        self.font = None

    def reset(self, **kwargs):
        self.episode_reward = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward

        # Draw score on the window
        if self.env.render_mode == "human":
            if self.font is None:
                pygame.font.init()
                self.font = pygame.font.SysFont("Arial", 24)

            # Get the pygame surface that Gymnasium is using
            screen = pygame.display.get_surface()
            if screen is not None:
                text = self.font.render(
                    f"Score: {self.episode_reward:.1f}", True, (255, 255, 255))
                screen.blit(text, (10, 10))
                pygame.display.flip()

        return obs, reward, terminated, truncated, info


# Load your best model
model = PPO.load("../models/ppo_lunarlander.zip")

env = gym.make("LunarLander-v3", render_mode="human")
env = ScoreOverlay(env)

obs, info = env.reset()

for _ in range(5000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print(f"Episode finished with score: {env.episode_reward:.1f}")
        obs, info = env.reset()

env.close()
