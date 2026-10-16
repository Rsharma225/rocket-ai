import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt


def evaluate_detailed(model_path, n_episodes=50):
    model = PPO.load(model_path)
    env = gym.make("LunarLander-v3")

    scores = []
    times_in_air = []
    perfect_landings = 0

    for i in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        scores.append(total_reward)
        times_in_air.append(steps)

        # Perfect landing = score >= 200
        if total_reward >= 200:
            perfect_landings += 1

        print(f"Episode {i+1}: Score = {total_reward:.1f} | Steps = {steps}")

    env.close()

    return {
        "scores": np.array(scores),
        "times": np.array(times_in_air),
        "perfect_rate": perfect_landings / n_episodes * 100
    }


# Change these names if your files are different
normal = evaluate_detailed("../models/ppo_lunarlander", n_episodes=40)
shaped = evaluate_detailed("../models/ppo_lunarlander_shaped", n_episodes=40)

# Plotting the results
# ========== CREATE THE 4 CHARTS (Nicer version) ==========
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Lunar Lander – Model Comparison",
             fontsize=16, fontweight='bold', y=0.98)

colors = ["#2ecc71", "#3498db"]  # nice green & blue

# 1. Average Score
axes[0, 0].bar(["Normal PPO", "Shaped PPO"],
               [normal["scores"].mean(), shaped["scores"].mean()],
               color=colors, width=0.55, edgecolor="black", linewidth=0.7)
axes[0, 0].set_title("Average Score", fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel("Score", fontsize=11)
axes[0, 0].set_ylim(0, max(normal["scores"].mean(),
                    shaped["scores"].mean()) * 1.15)
for i, v in enumerate([normal["scores"].mean(), shaped["scores"].mean()]):
    axes[0, 0].text(i, v + 5, f"{v:.0f}", ha='center', fontweight='bold')

# 2. Average Time in Air
axes[0, 1].bar(["Normal PPO", "Shaped PPO"],
               [normal["times"].mean(), shaped["times"].mean()],
               color=colors, width=0.55, edgecolor="black", linewidth=0.7)
axes[0, 1].set_title("Average Time in Air", fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel("Steps", fontsize=11)
for i, v in enumerate([normal["times"].mean(), shaped["times"].mean()]):
    axes[0, 1].text(i, v + 8, f"{v:.0f}", ha='center', fontweight='bold')

# 3. Perfect Landings %
axes[1, 0].bar(["Normal PPO", "Shaped PPO"],
               [normal["perfect_rate"], shaped["perfect_rate"]],
               color=colors, width=0.55, edgecolor="black", linewidth=0.7)
axes[1, 0].set_title("Perfect Landings (Score ≥ 200)",
                     fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel("% of Episodes", fontsize=11)
axes[1, 0].set_ylim(0, 105)
for i, v in enumerate([normal["perfect_rate"], shaped["perfect_rate"]]):
    axes[1, 0].text(i, v + 2, f"{v:.0f}%", ha='center', fontweight='bold')

# 4. Score Distribution
axes[1, 1].hist(normal["scores"], bins=12, alpha=0.75, label="Normal PPO",
                color=colors[0], edgecolor="black", linewidth=0.5)
axes[1, 1].hist(shaped["scores"], bins=12, alpha=0.75, label="Shaped PPO",
                color=colors[1], edgecolor="black", linewidth=0.5)
axes[1, 1].set_title("Score Distribution", fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel("Score", fontsize=11)
axes[1, 1].set_ylabel("Number of Episodes", fontsize=11)
axes[1, 1].legend(frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("four_charts_nice.png", dpi=200, bbox_inches="tight")
plt.show()

print("\nNice chart saved as 'four_charts.png'")
