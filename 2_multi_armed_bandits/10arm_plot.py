import matplotlib.pyplot as plt
import numpy as np

epsilons = [0, 0.01, 0.1]
arr = np.loadtxt("10arm_average_rewards_t.csv", delimiter=",")
arr = np.transpose(arr)
t = list(np.arange(arr.shape[1]))
reward = arr[:3]
optimal = arr[3:]

fig, axs = plt.subplots(2, 1, figsize=(5, 10))

for i, epsilon in enumerate(epsilons):
    axs[0].plot(t, reward[i], label=rf"$\epsilon$ = {epsilon}", linewidth=0.7)
    axs[1].plot(t, optimal[i], label=rf"$\epsilon$ = {epsilon}", linewidth=0.7)

axs[0].set_ylabel("Average Reward")
axs[0].set_xlabel("Timestep")
axs[0].legend()

axs[1].set_ylabel("% Optimal Action")
axs[1].set_xlabel("Timestep")
axs[1].set_ylim([0, 1])
axs[1].legend()
plt.show()