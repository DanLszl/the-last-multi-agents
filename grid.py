from enum import Enum
import numpy as np
from typing import List


'''
Grid definition:
W: wall
+: empty cell
T: treasure
O: Pit
C: Cliff
S: Starting cell
Example:
'''

grid_def = '''
++++++++
++WWWW++
+++++W++
+++++W++
+++++W++
++++O+++
+WWW++++
+++++++T
'''.strip()

cliffwalking_def = '''
++++++++++
++++++++++
++++++++++
++++++++++
++++++++++
SCCCCCCCCT
'''.strip()

arguments = {
    '+': dict(name='+', reward=-1, can_step_on_it=True, is_end_state=False, color=0),
    'W': dict(name='W', reward=-1, can_step_on_it=False, is_end_state=False, color=-5),
    'O': dict(name='O', reward=-20, can_step_on_it=True, is_end_state=True, color=-10),
    'T': dict(name='T', reward=10, can_step_on_it=True, is_end_state=True, color=10),
    'C': dict(name='C', reward=-100, can_step_on_it=True, is_end_state=True, color=-10),
    'S': dict(name='S', reward=-1, can_step_on_it=True, is_end_state=False, color=5),
    'Padding': dict(name='|', reward=-1, can_step_on_it=False, is_end_state=False, color=0)
}


# %%
def create_cell(name):
    return Cell(**arguments[name])


# %%
class Action(Enum):
    N = 0
    E = 1
    S = 2
    W = 3


opposites = {
    Action.N: Action.S,
    Action.S: Action.N,
    Action.E: Action.W,
    Action.W: Action.E
}



# %%
class Cell:
    def __init__(self, name, reward, can_step_on_it, is_end_state, color):
        self._name = name
        self._reward = reward
        self._can_step_on_it = can_step_on_it
        self._is_end_state = is_end_state
        self._q_values = self._init_q_values()
        self._color = color

    def _init_q_values(self):
        if self._is_end_state:
            # Need this for the bootstrap update rule
            return {a: 0 for a in Action}
            return {a: self._reward for a in Action}
        else:
            return {a: 0 for a in Action}

    def add_neighbors(self, N, E, S, W):
        self.neighbors = {Action.N: N, Action.E: E, Action.S: S, Action.W: W}

    def take_action(self, action: Action):
        # TODO think this trhough, if you introduce new kind of cells
        if self.neighbors[action].can_step_on_it:
            return self.neighbors[action]
        else:
            return self

    def update_q_value(self, action: Action, new_q_value: float):
        self._q_values[action] = new_q_value

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward(self) -> int:
        return self._reward

    @property
    def can_step_on_it(self) -> bool:
        return self._can_step_on_it

    @property
    def is_end_state(self) -> bool:
        return self._is_end_state

    @property
    def q_values(self):
        return self._q_values

    @property
    def color(self):
        return self._color

    @property
    def can_be_starting_cell(self) -> bool:
        return self.can_step_on_it and not self.is_end_state

    def __repr__(self) -> str:
        return self._name


# %%
class Grid:
    def __init__(self, grid_def):
        self.grid_def = grid_def
        self.starting_cell = None
        self._parse_grid_def()
        self._create_padding()
        self._create_neighborhoods()

    def _parse_grid_def(self):
        self.grid = []
        for line in self.grid_def.split('\n'):
            self.grid.append(self._parse_line(line))

    def _parse_line(self, line) -> List[Cell]:
        grid_line = []
        for char in line:
            new_cell = create_cell(char)
            grid_line.append(new_cell)
            if new_cell.name == 'S':
                self.starting_cell = new_cell
        return grid_line

    def _create_padding(self):
        grid_width = len(self.grid[0])
        upper_boundary = [create_cell('Padding') for _ in range(grid_width)]
        lower_boundary = [create_cell('Padding') for _ in range(grid_width)]
        self.grid.insert(0, upper_boundary)
        self.grid.append(lower_boundary)

        for line in self.grid:
            line.insert(0, create_cell('Padding'))
            line.append(create_cell('Padding'))

    def _create_neighborhoods(self):
        for i in range(1, len(self.grid)-1):
            for j in range(1, len(self.grid[0])-1):
                N = self.grid[i-1][j]
                E = self.grid[i][j+1]
                S = self.grid[i+1][j]
                W = self.grid[i][j-1]
                self.grid[i][j].add_neighbors(N, E, S, W)

    def get_starting_position(self) -> Cell:
        if self.starting_cell is not None:
            return self.starting_cell
        else:
            return self.get_initial_random_position()

    def get_initial_random_position(self) -> Cell:
        # 1 because of the padding
        x = np.random.randint(1, len(self.grid)-1)
        y = np.random.randint(1, len(self.grid[0])-1)

        if self.grid[x][y].can_be_starting_cell:
            return self.grid[x][y]
        else:
            return self.get_initial_random_position()

    def pretty_print(self):
        raise NotImplementedError
        count = 0
        print("-" * 82)
        for position_x in reversed(range(self.grid_size)):
            for position_y in range(self.grid_size):
                print("|\t{0:.2f}\t".format(self.coordinates[(position_x, position_y)].value), end = '')
                count += 1
            print("|")
            print()
            print("-" * 82)
            print()
