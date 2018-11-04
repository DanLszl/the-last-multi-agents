import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from grid import Grid, Action


def plot_original_grid(grid: Grid, ax):
    colors = []
    for row in grid.grid[1:-1]:
        row_colors = []
        for cell in row[1:-1]:
            row_colors.append(cell.color)
        colors.append(row_colors)

    colors = colors[::-1]

    ax.matshow(colors, cmap='RdYlGn')

    ax.tick_params(bottom=False, top=True, left=True, right=False,
                    labelbottom=False, labeltop=True, labelleft=True, labelright=False)

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
    return ax


def quatromatrix(left, bottom, right, top, ax=None, triplotkw={}, tripcolorkw={}):
    left = left[::-1]
    bottom = bottom[::-1]
    right = right[::-1]
    top = top[::-1]

    if not ax:
        ax = plt.gca()
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


def plot_grid(grid: Grid, ax):
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

    quatromatrix(left, bottom, right, top, ax=ax,
                 triplotkw={"color": "k", "lw": 1},
                 tripcolorkw={"cmap": "ocean"})  # magma, ocean_r


    for (i, j), z in np.ndenumerate(left[::-1]):
        ax.text(j+0.2, i+0.5, '{:0.1f}'.format(z), ha='center', va='center')
    for (i, j), z in np.ndenumerate(right[::-1]):
        ax.text(j+1-0.2, i+0.5, '{:0.1f}'.format(z), ha='center', va='center')
    for (i, j), z in np.ndenumerate(top[::-1]):
        ax.text(j+0.5, i+0.8, '{:0.1f}'.format(z), ha='center', va='center')
    for (i, j), z in np.ndenumerate(bottom[::-1]):
        ax.text(j+0.5, i+0.2, '{:0.1f}'.format(z), ha='center', va='center')

    ax.tick_params(bottom=False, top=True, left=True, right=False,
                    labelbottom=False, labeltop=True, labelleft=True, labelright=False)

    ax.margins(0)
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(left.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(left.shape[0]) + 0.5, minor=False)

    ax.set_xticklabels(np.arange(1, left.shape[1] + 1))
    ax.set_yticklabels(np.arange(1, left.shape[0] + 1))

    #ax.invert_yaxis()
    #ax.xaxis.tick_top()


def plot_optimal_policy(grid: Grid, ax):
    optimal_policy = []

    for row in grid.grid[1:-1]:
        optimal_row = []
        for cell in row[1:-1]:
            if cell.can_be_starting_cell:
                max_q_value = max(cell.q_values.values())
                best_actions = [i for i, j in cell.q_values.items() if j == max_q_value]
            else:
                best_actions = []

            optimal_row.append(best_actions)
        optimal_policy.append(optimal_row)

    ax = plot_original_grid(grid, ax)

    arrow = lambda x, y, dx, dy: \
              ax.arrow(x=x, y=y, dx=dx, dy=dy,
                       length_includes_head=True,
                       width=(dx+dy)/10, head_length=0.2, head_width=0.2, linewidth=0)

    # uparrow = lambda i,j: arrow(x=i, y=j, dx=0, dy=1)
    # downarrow = lambda i,j: arrow(x=i, y=j, dx=0, dy=-1)
    # rightarrow = lambda i,j: arrow(x=i, y=j, dx=1, dy=0)
    # leftarrow = lambda i,j: arrow(x=i, y=j, dx=-1, dy=0)

    uparrow = lambda i,j: arrow(x=i, y=j, dx=0, dy=0.5)
    downarrow = lambda i,j: arrow(x=i, y=j, dx=0, dy=-0.5)
    rightarrow = lambda i,j: arrow(x=i, y=j, dx=0.5, dy=0)
    leftarrow = lambda i,j: arrow(x=i, y=j, dx=-0.5, dy=0)

    action_to_arrow = {
        Action.N: uparrow,
        Action.S: downarrow,
        Action.E: rightarrow,
        Action.W: leftarrow
    }

    for i, row in enumerate(optimal_policy[::-1]):
        for j, best_actions in enumerate(row):
            for best_action in best_actions:
                action_to_arrow[best_action](j, i)


def plot(grid, filename=None, show_plot=True, plot_orientation='horizontal'):
    if plot_orientation == 'horizontal':
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(len(grid.grid[0])*2, len(grid.grid)))
    else:
        fig, axes = plt.subplots(nrows=2, ncols=1, tight_layout=True, figsize=(len(grid.grid[0]), len(grid.grid)*2))

    ax = axes[0]
    plot_grid(grid, ax)

    ax = axes[1]
    plot_optimal_policy(grid, ax)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.009, hspace=0.009)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300) # bbox_inches='tight',

    if show_plot:
        plt.show()
