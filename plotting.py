import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from grid import Grid, Action


def plot_original_grid(grid):
    rewards = []
    for row in grid.grid[1:-1]:
        row_rewards = []
        for cell in row[1:-1]:
            row_rewards.append(cell.reward)
        rewards.append(row_rewards)

    rewards = rewards[::-1]

    plt.matshow(rewards, cmap='ocean')
    ax = plt.gca()

    # Major ticks
    ax.set_xticks(np.arange(0, len(grid.grid[0])-2, 1))
    ax.set_yticks(np.arange(0, len(grid.grid)-2, 1))
    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, len(grid.grid[0])-1, 1))
    ax.set_yticklabels(np.arange(1, len(grid.grid)-1, 1))

    # Minor ticks
    ax.set_xticks(np.arange(.5, len(grid.grid[0])-2, 1), minor=True)
    ax.set_yticks(np.arange(.5, len(grid.grid)-2, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.invert_yaxis()
    plt.show()


def quatromatrix(left, bottom, right, top, ax=None, triplotkw={}, tripcolorkw={}):
    left = left[::-1]
    bottom = bottom[::-1]
    right = right[::-1]
    top = top[::-1]

    if not ax: ax=plt.gca()
    n = left.shape[0]; m=left.shape[1]

    a = np.array([[0,0],[0,1],[.5,.5],[1,0],[1,1]])
    tr = np.array([[0,1,2], [0,2,3],[2,3,4],[1,2,4]])

    A = np.zeros((n*m*5,2))
    Tr = np.zeros((n*m*4,3))

    for i in range(n):
        for j in range(m):
            k = i*m+j
            A[k*5:(k+1)*5,:] = np.c_[a[:,0]+j, a[:,1]+i]
            Tr[k*4:(k+1)*4,:] = tr + k*5

    C = np.c_[ left.flatten(), bottom.flatten(),
              right.flatten(), top.flatten()   ].flatten()

    triplot = ax.triplot(A[:,0], A[:,1], Tr, **triplotkw)
    tripcolor = ax.tripcolor(A[:,0], A[:,1],
                                Tr,
                                facecolors=C,
                                **tripcolorkw)
    return tripcolor


def plot_grid(grid: Grid):
    left = []
    right = []
    top = []
    bottom = []
    for row in grid.grid[1:-1]:
        left_row = []
        right_row = []
        top_row = []
        bottom_row = []
        for cell in row[1:-1]:
            top_row.append(cell.q_values[Action.N])
            right_row.append(cell.q_values[Action.E])
            bottom_row.append(cell.q_values[Action.S])
            left_row.append(cell.q_values[Action.W])
        left.append(left_row)
        right.append(right_row)
        top.append(top_row)
        bottom.append(bottom_row)

    # print('left: ', left)
    # print('right: ', right)
    # print('top: ', top)
    # print('bottom: ', bottom)

    left = np.array(left)
    right = np.array(right)
    top = np.array(top)
    bottom = np.array(bottom)

    # Plotting Source: https://stackoverflow.com/questions/44666679/something-like-plt-matshow-but-with-triangles

    fig, ax=plt.subplots()
    fig.set_size_inches(12, 12)
    quatromatrix(left, bottom, right, top, ax=ax,
                 triplotkw={"color": "k", "lw": 1},
                 tripcolorkw={"cmap": "ocean"})


    for (i, j), z in np.ndenumerate(left[::-1]):
        ax.text(j+0.2, i+0.5, '{:0.1f}'.format(z), ha='center', va='center')
    for (i, j), z in np.ndenumerate(right[::-1]):
        ax.text(j+1-0.2, i+0.5, '{:0.1f}'.format(z), ha='center', va='center')
    for (i, j), z in np.ndenumerate(top[::-1]):
        ax.text(j+0.5, i+0.8, '{:0.1f}'.format(z), ha='center', va='center')
    for (i, j), z in np.ndenumerate(bottom[::-1]):
        ax.text(j+0.5, i+0.2, '{:0.1f}'.format(z), ha='center', va='center')

    ax.margins(0)
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(left.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(left.shape[0]) + 0.5, minor=False)

    ax.set_xticklabels(np.arange(1, left.shape[1] + 1))
    ax.set_yticklabels(np.arange(1, left.shape[0] + 1))

    #ax.invert_yaxis()
    #ax.xaxis.tick_top()

    plt.show()


from grid import opposites as opposite_of

def plot_optimal_policy(grid):
    optimal_policy = []

    for row in grid.grid[1:-1]:
        optimal_row = []
        for cell in row[1:-1]:
            best_action = max(cell.q_values, key=cell.q_values.get)
            optimal_cell = {k: 0 for k in cell.q_values}
            optimal_cell[opposite_of[best_action]] = 1
            optimal_row.append(optimal_cell)
        optimal_policy.append(optimal_row)

    left = [[cell[Action.W] for cell in row] for row in optimal_policy]
    right = [[cell[Action.E] for cell in row] for row in optimal_policy]
    top = [[cell[Action.N] for cell in row] for row in optimal_policy]
    bottom = [[cell[Action.S] for cell in row] for row in optimal_policy]

    left = np.array(left)
    right = np.array(right)
    top = np.array(top)
    bottom = np.array(bottom)

    # Plotting Source: https://stackoverflow.com/questions/44666679/something-like-plt-matshow-but-with-triangles

    fig, ax=plt.subplots()
    fig.set_size_inches(12, 12)
    quatromatrix(left, bottom, right, top, ax=ax,
                 triplotkw={"color": "k", "lw": 1},
                 tripcolorkw={"cmap": "Greys_r"})

    ax.margins(0)
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(left.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(left.shape[0]) + 0.5, minor=False)

    ax.set_xticklabels(np.arange(1, left.shape[1] + 1))
    ax.set_yticklabels(np.arange(1, left.shape[0] + 1))

    #ax.invert_yaxis()
    #ax.xaxis.tick_top()
    a = False
    plt.show()
