import matplotlib.pyplot as plt
import numpy as np

uni = np.loadtxt("csvs/uniform_b3_values_10000.csv", delimiter=",")
onp = np.loadtxt("csvs/on-policy_b3_values_10000.csv", delimiter=",")

n = np.arange(0, 100 * 50, 50)

plt.figure(figsize=(9, 6))
plt.plot(n, uni[:100], label="Uniform", color="tab:green")
plt.plot(n, onp, label="On-policy", color="tab:blue")

plt.xlim([0, 100*50])
plt.xlabel("Computation Time")
plt.ylabel("Value of start state under greedy policy")
plt.legend()
plt.title("Trajectory Sampling, 10000 states, b=3")
plt.savefig("images/trajectory_sampling_10000.png", dpi=300)
