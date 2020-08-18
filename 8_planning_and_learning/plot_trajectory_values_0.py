import matplotlib.pyplot as plt
import numpy as np

uni_1 = np.loadtxt("csvs/uniform_b1_values_1000.csv", delimiter=",")
uni_3 = np.loadtxt("csvs/uniform_b3_values_1000.csv", delimiter=",")
uni_10 = np.loadtxt("csvs/uniform_b10_values_1000.csv", delimiter=",")
onp_1 = np.loadtxt("csvs/on-policy_b1_values_1000.csv", delimiter=",")
onp_3 = np.loadtxt("csvs/on-policy_b3_values_1000.csv", delimiter=",")
onp_10 = np.loadtxt("csvs/on-policy_b10_values_1000.csv", delimiter=",")

n = np.arange(0, 100 * 50, 50)

plt.figure(figsize=(9, 6))
plt.plot(n, uni_1[:100], label="Uniform, b=1", color="lightblue")
plt.plot(n, onp_1, label="On-policy, b=1", color="tab:blue")
plt.plot(n, uni_3[:100], label="Uniform, b=3", color="lightgreen")
plt.plot(n, onp_3, label="On-policy, b=3", color="tab:green")
plt.plot(n, uni_10[:100], label="Uniform, b=10", color="lightcoral")
plt.plot(n, onp_10, label="On-policy, b=10", color="tab:red")

plt.xlim([0, 100*50])
plt.xlabel("Computation time")
plt.ylabel("Value of start state under greedy policy")
plt.legend()
plt.title("Trajectory Sampling, 1000 states")
plt.savefig("images/trajectory_sampling_1000.png", dpi=300)