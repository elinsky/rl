import gym
import gym_minigrid
import numpy as np
from collections import namedtuple

named_observation = namedtuple('observation', field_names=['grid', 'agent_direction'])
named_grid = namedtuple('grid', field_names=['row_0', 'row_1', 'row_2', 'row_3', 'row_4'])
named_row = namedtuple('row', field_names=['col_0', 'col_1', 'col_2', 'col_3', 'col_4'])


class ImmutableObjectDirectionWrapper(gym.core.ObservationWrapper):
    """
    Observation wrapper that returns the grid of objects along
    with the agent's direction.
    """

    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation) -> tuple:
        def _name_grid(grid):
            row_0 = named_row(grid[0][0], grid[0][1], grid[0][2], grid[0][3], grid[0][4])
            row_1 = named_row(grid[1][0], grid[1][1], grid[1][2], grid[1][3], grid[1][4])
            row_2 = named_row(grid[2][0], grid[2][1], grid[2][2], grid[2][3], grid[2][4])
            row_3 = named_row(grid[3][0], grid[3][1], grid[3][2], grid[3][3], grid[3][4])
            row_4 = named_row(grid[4][0], grid[4][1], grid[4][2], grid[4][3], grid[4][4])
            a_grid = named_grid(row_0, row_1, row_2, row_3, row_4)
            return a_grid

        object_grid = observation['image'][:, :, 0]
        object_grid_tuple = tuple(map(tuple, object_grid))
        a_named_grid = _name_grid(object_grid_tuple)
        named_obs = named_observation(a_named_grid, self.env.agent_dir)
        return named_obs
