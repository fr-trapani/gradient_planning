import numpy as np
from environments.gridworlds.gridworld import *


class MiniWorld(GridWorld):
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


class SimpleGrid(GridWorld):
    """
    Simple squared room with no walls
    """
    def __init__(self, **kwargs):
        nx = 10
        ny = 10
        init_locs = np.array([[1, 1]])
        term_locs = np.array([[8, 8]])
        reward = np.array([100]) 

        super().__init__(nx, ny, init_locs, term_locs=term_locs, **kwargs)
        self.add_grid_rewards(reward, term_locs)


class TwoStartsRoom(GridWorld):
    """
    Simple squared room with no walls
    """
    def __init__(self, **kwargs):
        nx = 10
        ny = 13
        init_locs = np.array([[1, 1], [1, 11]])
        p_init_locs = np.array([.25, .75])
        term_locs = np.array([[8, 6]])
        reward = np.array([100]) 

        super().__init__(nx, ny, init_locs, p_init_locs=p_init_locs, term_locs=term_locs, **kwargs)
        self.add_grid_rewards(reward, term_locs)


class MidWall(GridWorld):
    """
    grid world with mid division
    
    """
    def __init__(self, n_square=5, **kwargs):
        
        self.n_square = n_square
        
        nx = 5*n_square + 1
        ny = 3*n_square + 1

        init_locs = np.array([[n_square, 2*n_square]])
        term_locs = np.array([[3*n_square, n_square]])
        reward = np.array([100]) 

        walls_map = np.full([nx, ny], False, dtype=bool)
        walls_map[2*n_square, n_square : 2*n_square] = True
        walls_map[n_square : 2*n_square, n_square] = True
        walls_map[2*n_square:4*n_square, 2*n_square] = True
        walls = np.argwhere(walls_map)
        
        super().__init__(nx, ny, init_locs, term_locs=term_locs, walls=walls, **kwargs)
        self.add_grid_rewards(reward, term_locs)


class FourRoom(GridWorld):
    """
    grid world with four rooms connected by doors
    
    """
    def __init__(self, n_square=3, init_locs=None, p_init_locs=None, term_locs=None, reward=np.array([100]), **kwargs):
        
        self.n_square = n_square
        
        nx = 2*n_square + 3
        ny = 2*n_square + 3

        if init_locs is None:
            init_locs = np.array([[int(n_square/2 + 1), int(n_square/2 + 1)]])
        if term_locs is None:
            term_locs = np.array([[int(n_square*3/2 + 2), int(n_square*3/2 + 2)]])

        # Walls
        walls_map = np.full([nx, ny], False, dtype=bool)
        walls_map[0,:] = True
        walls_map[n_square+1,:] = True
        walls_map[nx-1,:] = True
        walls_map[:,0] = True
        walls_map[:,n_square+1] = True
        walls_map[:,ny-1] = True

        # Doors
        walls_map[n_square+1, int(n_square/2 + 1)] = False
        walls_map[n_square+1, int(n_square*3/2 + 2)] = False
        walls_map[int(n_square/2 + 1), n_square+1] = False
        walls_map[int(n_square*3/2 + 2), n_square+1] = False

        walls = np.argwhere(walls_map)
        
        super().__init__(nx, ny, init_locs, p_init_locs=p_init_locs, term_locs=term_locs, walls=walls, **kwargs)
        self.add_grid_rewards(reward, term_locs)


        # state_sets
        self.state_sets = []
        self.state_set_colors = []
        self.state_set_labels = []

        room_1 = np.full([nx, ny], False, dtype=bool)
        room_1[1:(1+n_square), 1:(1+n_square)] = True
        
        room_2 = np.full([nx, ny], False, dtype=bool)   
        room_2[(2+n_square):(2+n_square*2), 1:(1+n_square)] = True

        room_3 = np.full([nx, ny], False, dtype=bool)   
        room_3[1:(1+n_square), (2+n_square):(2+n_square*2)] = True

        room_4 = np.full([nx, ny], False, dtype=bool)   
        room_4[(2+n_square):(2+n_square*2), (2+n_square):(2+n_square*2)] = True

        doors = np.array([[n_square+1, int(n_square/2 + 1)], [n_square+1, int(n_square*3/2 + 2)], [int(n_square/2 + 1), n_square+1], [int(n_square*3/2 + 2), n_square+1]])

        self.state_sets.append(self.encode(np.argwhere(room_1)))
        self.state_set_colors.append("blue")
        self.state_set_labels.append("room1")

        self.state_sets.append(self.encode(np.argwhere(room_2)))
        self.state_set_colors.append("red")
        self.state_set_labels.append("room2")

        self.state_sets.append(self.encode(np.argwhere(room_3)))
        self.state_set_colors.append("orange")
        self.state_set_labels.append("room3")

        self.state_sets.append(self.encode(np.argwhere(room_4)))
        self.state_set_colors.append("magenta")
        self.state_set_labels.append("room4")   

        self.state_sets.append(self.encode(doors))
        self.state_set_colors.append("brown")
        self.state_set_labels.append("doors")   

class FourRoomAlternate(FourRoom):
    """
    grid world with four rooms connected by doors
    
    """
    def __init__(self, n_square=3, **kwargs):
        
        init_locs = np.array([[int(n_square/2 + 1), int(n_square/2)]])
        term_locs = np.array([[int(n_square*3/2 + 3), int(n_square*3/2 + 2)]])
        super().__init__(n_square=n_square, init_locs=init_locs, p_init_locs=None, term_locs=term_locs, **kwargs)


class FourRoom2Goals(FourRoom):
    """
    grid world with four rooms connected by doors
    
    """
    def __init__(self, n_square=7, **kwargs):
        
        init_locs = np.array([[int(n_square/2 + 1), int(n_square/2 + 1)]])
        term_locs = np.array([ [int(n_square*3/2 + 2), int(n_square/2 + 1)], [int(n_square/2 + 1), int(n_square*3/2 + 2)] ])
        reward=np.array([100, 100])
        super().__init__(n_square=n_square, init_locs=init_locs, p_init_locs=None, term_locs=term_locs, reward=reward, **kwargs)


class OneDoor(GridWorld):
    """
    grid world with two rooms connected by a door
    
    """
    def __init__(self, 
                 n_square = 11, 
                 init_locs=np.array([[2,2]]), 
                 p_init_locs=None, 
                 term_locs=np.array([[20, 2]]), 
                 rew_locs=np.array([[20, 2]]), 
                 reward_vals = np.array([100]), 
                 **kwargs):

        self.n_square = n_square     
        nx =  2 * n_square+1
        ny = n_square
        
        walls_map = np.full([nx, ny], False, dtype=bool)
        walls_map[n_square, 0:n_square//2] = True
        walls_map[n_square, n_square//2 + 1:n_square] = True
        walls = np.argwhere(walls_map)

        super().__init__(nx, ny, init_locs, p_init_locs=p_init_locs, term_locs=term_locs, walls=walls, **kwargs)
        self.add_grid_rewards(reward_vals, rew_locs)


class ForkedRoom(GridWorld):
    """
    grid world with three rooms 
    """
    def __init__(self, n_square=5, **kwargs):

        self.n_square = n_square
        
        nx = 4 * self.n_square + 3
        ny = 3 * self.n_square + 1

        init_locs = np.array([[2*n_square + n_square//2 + 2, 3*n_square - 2]])
        rew_term_locs = np.array([[n_square//2, 3*n_square - 2]])
        reward = np.array([100])
        
        # walls defined by lower left corner and upper right corner [[x1,y1], [x2,y2]]
        walls_map = np.full([nx, ny], False, dtype=bool)
        walls_map[0 : n_square-1, n_square] = True
        walls_map[n_square+2 : 3 * n_square + 1, n_square] = True
        walls_map[3*n_square + 4 : 3 * 4*n_square + 2, n_square] = True
        walls_map[n_square, n_square+2 : 3*n_square + 1] = True
        walls_map[2*n_square + 1, n_square : 3*n_square + 1] = True
        walls_map[3*n_square + 2, n_square+2 : 3*n_square + 1] = True
        walls = np.argwhere(walls_map)

        super().__init__(nx, ny, init_locs, term_locs=rew_term_locs, walls=walls, **kwargs)
        self.add_grid_rewards(reward, rew_term_locs)


class Agarwal(GridWorld):
    """
    Simple squared room with no walls
    """
    def __init__(self, **kwargs):
        nx = 3
        ny = 2

        init_locs = np.array([[0, 1]])
        term_locs = np.array([[0, 0], [1, 0], [2, 1]])
        rew_locs = np.array([[1, 0]])
        rew_vals = np.array([100])
        wall_locs = np.array([[2, 0]])
        super().__init__(nx, ny, init_locs, term_locs=term_locs, action_cost=0, walls=wall_locs, **kwargs)
        self.add_grid_rewards(rew_vals, rew_locs)


class Russo(GridWorld):
    """
    Simple squared room with no walls
    """
    def __init__(self, **kwargs):

        actions = IdleActionSet()
        nx = 3
        ny = 2

        init_locs = np.array([[0, 0]])
        term_locs = np.array([[2, 1]])

        super().__init__(nx, ny, init_locs, term_locs=term_locs, actions=actions, **kwargs)
