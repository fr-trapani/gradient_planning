import numpy as np
from environments.gridworlds.gridactions import *
from environments.gridworlds.gridworld import *
from environments.gridworlds.gridworlds_classic import *
from environments.poseworlds.poseworld import *


class EuclideanPoseWorld(PoseWorld):

    def __init__(self, 
                 nx, 
                 ny, 
                 init_locs, 
                 p_init_locs=None, 
                 no=8, 
                 term_locs=np.ndarray([0,3]), 
                 walls=np.ndarray([0,2]), 
                 actions=TurningActionSet(), 
                 name="euclidean_poseworld", 
                 **kwargs):
        
        super().__init__(nx, 
                         ny, 
                         init_locs, 
                         p_init_locs=p_init_locs, 
                         no=no, 
                         term_locs=term_locs, 
                         walls=walls, 
                         actions=actions, 
                         action_cost=self.grid_action_cost, 
                         name=name, 
                         **kwargs)


    def grid_action_cost(self, s1, _, s2):
        p1, p2 = self.decode(s1), self.decode(s2)
        l1, l2 = p1[1:], p2[1:]
        return - np.sqrt((l1[0] - l2[0])**2 + (l1[1] - l2[1])**2)


class EuclideanGridWorld(GridWorld):

    def __init__(self, 
                 nx, 
                 ny, 
                 init_locs, 
                 p_init_locs=None, 
                 term_locs=np.ndarray([0,2]), 
                 walls=np.ndarray([0,2]), 
                 actions=KingActionSet(), 
                 name="euclidean_gridworld", 
                 **kwargs):
        
        super().__init__(nx, 
                         ny, 
                         init_locs, 
                         p_init_locs=p_init_locs, 
                         term_locs=term_locs, 
                         walls=walls, 
                         actions=actions, 
                         action_cost=self.grid_action_cost, 
                         name=name, 
                         **kwargs)


    def grid_action_cost(self, s1, _, s2):
        l1, l2 = self.decode(s1), self.decode(s2)
        return - np.sqrt((l1[0] - l2[0])**2 + (l1[1] - l2[1])**2)    


class TwoRoomsEuclidePoseWorld(TwoRooms, EuclideanPoseWorld):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class TwoRoomsEuclideGridWorld(TwoRooms, EuclideanGridWorld):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class FourRoomEuclidePoseWorld(FourRoom, EuclideanPoseWorld):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class FourRoomEuclideGridWorld(FourRoom, EuclideanGridWorld):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class ForkedRoomEuclidePoseWorld(ForkedRoom, EuclideanPoseWorld):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class ForkedRoomEuclideGridWorld(ForkedRoom, EuclideanGridWorld):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)