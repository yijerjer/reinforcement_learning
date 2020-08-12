import matplotlib.pyplot as plt
import numpy as np

wind = np.loadtxt("csvs/wind_0.csv", delimiter=",")
path = np.loadtxt("csvs/9_action_stochastic_path_episode_1000.csv", delimiter=",")
path = path.astype(int)
wind = wind.astype(int)

rows = 7; columns = 10
grid = np.zeros((7, 10))
for cell in path:
    grid[cell[0]][cell[1]] = 2

grid[3][0] = 1
grid[3][7] = 3
# grid = np.flip(grid, axis=0)

plt.matshow(grid, cmap="Blues", origin="lower")
plt.yticks([])
plt.xticks([])
for i in range(0, len(path) - 1):
    curr_cell = path[i]
    next_cell = path[i+1]
    move = next_cell - curr_cell

    plt.arrow(curr_cell[1], curr_cell[0], move[1], move[0], lw=2, head_width=0.1, length_includes_head=True, color='whitesmoke')

for i, row in enumerate(wind):
    for j, val in enumerate(row):
        plt.text(j, i, wind[i][j], color='dimgrey', size="large", ha="center", va="center")

plt.title("Optimal Policy for 4 Available Actions", size="x-large")
plt.savefig("images/9_actions_stochastic_policy.png", dpi=500)
# plt.show()
