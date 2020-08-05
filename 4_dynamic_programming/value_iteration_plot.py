import matplotlib.pyplot as plt
import numpy as np

values_4 = np.loadtxt("csvs/val_iter_state_values_p40.csv", delimiter=",")
values_25 = np.loadtxt("csvs/val_iter_state_values_p25.csv", delimiter=",")
values_55 = np.loadtxt("csvs/val_iter_state_values_p55.csv", delimiter=",")
policy_4 = np.loadtxt("csvs/val_iter_policy_p40.csv", delimiter=",")
policy_25 = np.loadtxt("csvs/val_iter_policy_p25.csv", delimiter=",")
policy_55 = np.loadtxt("csvs/val_iter_policy_p55.csv", delimiter=",")

fig = plt.figure(figsize=(6, 16))

ax_1 = plt.subplot(4, 1, 1)
ax_1.plot(values_25[-1][:-2], label=f"p = 0.25")
ax_1.plot(values_4[-1][:-2], label=f"p = 0.4")
ax_1.plot(values_55[-1][:-2], label=f"p = 0.55")

ax_1.legend()

ax_2 = plt.subplot(4, 1, 2)
ax_2.bar(np.arange(1, 100, 1), policy_25)
ax_2.set_title("p = 0.25")
ax_2.set_ylim([0, 55])

ax_3 = plt.subplot(4, 1, 3)
ax_3.bar(np.arange(1, 100, 1), policy_4)
ax_3.set_title("p = 0.4")
ax_3.set_ylim([0, 55])

ax_4 = plt.subplot(4, 1, 4)
ax_4.bar(np.arange(1, 100, 1), policy_55)
ax_4.set_title("p = 0.55")
ax_4.set_ylim([0, 55])

plt.suptitle("Gambler's Problem, State Values and Optimal Policies")
plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
plt.savefig("images/gambler.png")
plt.show()