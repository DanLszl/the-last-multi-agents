import numpy as np
import matplotlib.pyplot as plt
from grid import Grid, Action

def quatromatrix(left, bottom, right, top, ax=None, triplotkw={},tripcolorkw={}):
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
    tripcolor = ax.tripcolor(A[:,0], A[:,1], Tr, facecolors=C, **tripcolorkw)
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

    quatromatrix(left, bottom, right, top, ax=ax,
                 triplotkw={"color": "k", "lw": 1},
                 tripcolorkw={"cmap": "gray"})

    ax.margins(0)
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(left.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(left.shape[0]) + 0.5, minor=False)

    ax.set_xticklabels(np.arange(1, left.shape[1] + 1))
    ax.set_yticklabels(np.arange(1, left.shape[0] + 1))

    #ax.invert_yaxis()
    #ax.xaxis.tick_top()


    plt.show()
