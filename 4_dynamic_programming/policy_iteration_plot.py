import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

fig = plt.figure(figsize=(10, 20))

for iter in range(5):
    policy = np.loadtxt(f"csvs/policy_{iter}.csv", delimiter=",")
    values = np.loadtxt(f"csvs/state_values_{iter}.csv", delimiter=",")

    ax_0 = plt.subplot(5, 2, (iter * 2) + 1)
    ax_0.imshow(policy, cmap="RdBu")
    ax_0.set_xticks(range(0, 21, 5))
    ax_0.set_yticks(range(0, 21, 5))
    ax_0.set_xlabel("Cars at location B")
    ax_0.set_ylabel("Cars at location A")
    for i, row in enumerate(policy):
        for j, val in enumerate(row):
            ax_0.text(j, i, int(val), color='w', fontsize=8, va="center", ha="center")
    
    A = range(0, 21)
    B = range(0, 21)
    X, Y = np.meshgrid(A, B)
    Z = values
    ax_1 = plt.subplot(5, 2, (iter * 2) + 2, projection='3d')
    ax_1.plot_surface(X, Y, Z, cmap='viridis')
    ax_1.set_xticks(range(0, 21, 5))
    ax_1.set_yticks(range(0, 21, 5))
    ax_1.set_xlabel("Cars at location B")
    ax_1.set_ylabel("Cars at location A")
    
plt.suptitle("Car Rental Policy and State Value, Modified Reward", size="large")
plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, wspace=0, hspace=0.2)
plt.savefig("images/car_rental.png", dpi=500)
# plt.show()