import numpy as np
from environments.gridworlds.gridworld import *


class BlocksWorld(GridWorld):
    """
    Simple squared room with no walls
    """
    def __init__(self, n=2, **kwargs):
        nx = n
        ny = n
        init_locs = np.array([[0, 0]])
        term_locs = np.array([[n-1, n-1]])
        reward = np.array([100]) 

        super().__init__(nx, ny, init_locs, term_locs=term_locs, **kwargs)
        self.add_grid_rewards(reward, term_locs)