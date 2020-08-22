import matplotlib.pyplot as plt
import numpy as np

rewards = np.loadtxt("csvs/rewards.csv", delimiter=",")
rewards_mean = np.mean(rewards, axis=0)
plus_rewards = np.loadtxt("csvs/plus_rewards.csv", delimiter=",")
plus_rewards_mean = np.mean(plus_rewards, axis=0)
action_plus_rewards = np.loadtxt("csvs/action_plus_rewards.csv", delimiter=",")
action_plus_rewards_mean = np.mean(action_plus_rewards, axis=0)

t = np.arange(0, len(rewards_mean), 1)
plt.plot(t, rewards_mean, label="Dyna-Q")
plt.plot(t, plus_rewards_mean, label="Dyna-Q+")
plt.plot(t, action_plus_rewards_mean, label="Dyna-Q+ in Policy")

plt.legend()
plt.ylabel("Cumulative Reward")
plt.xlabel("Time Step")
# plt.show()
plt.savefig("images/cumulative_reward.png")