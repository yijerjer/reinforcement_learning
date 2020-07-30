import matplotlib.pyplot as plt
import numpy as np

sample_av = np.loadtxt("csvs/parameter_sample_average.csv", delimiter=",")
fixed_step = np.loadtxt("csvs/parameter_fixed_step.csv", delimiter=",")
sample_av = np.transpose(sample_av)
fixed_step = np.transpose(fixed_step)

plt.plot(sample_av[0], sample_av[1], label="Sample average")
plt.plot(fixed_step[0], fixed_step[1], label="Fixed step")
plt.ylabel("Average Reward")
plt.xlabel(r"$\epsilon$")
plt.xscale("log", basex=2)
plt.title("Parameter plot for stationary problem")
plt.legend()

plt.savefig("images/parameter_epsilon.png")