import numpy as np
from numpy.core.multiarray import array as array
from environments.gridworlds.gridworld import GridWorld
from environments.gridworlds.gridworlds_classic import *


class TorusWorld(GridWorld):
    """
    A torus-shaped gridworld
    """
    def grid_transition(self, l1, a):
        """
        arguments:
            - l1: 2d location on the grid
            - a:  identifier of an action
        returns:
            - lists_l2: all the possible locations that can be reached by performing a in l1
            - probls_l2 the probabilities associated with the locations in lists_l2
        """
        l2 = l1 + self.get_action(a)

        # In a torus, the left (top) edge is connected to the right (bottom) one
        l2[0] = l2[0] % self.nx
        l2[1] = l2[1] % self.ny

        lists_l2 = np.array([l2])
        probs_l2 = np.array([1])
        return lists_l2, probs_l2


class TwoRoomsTorus(TwoRooms, TorusWorld):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SimpleTorus(SimpleGrid, TorusWorld):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
