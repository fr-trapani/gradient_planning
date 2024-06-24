from environments.gridworlds.gridworld import GridWorld
from environments.gridworlds.gridactions import *
import numpy as np


## Idiothetic Functions

def grid2pose(positions, no):
    """
    transforms a list of positions into a list of poses, adding all orientations to each position 
    """
    n, d = positions.shape
    if d > 2:
        return positions, np.arange(n)
    else:
        poses = np.ndarray([n, no, 3], dtype=int)
        idxs = np.ndarray([n, no], dtype=int)

        for i_n in range(n):
            for i_o in range(no):
                poses[i_n, i_o, :] = np.append(i_o, positions[i_n, :])
                idxs[i_n, i_o] = i_n
        poses = np.reshape(poses, [n*no, 3])           
        idxs = np.reshape(idxs, [n*no])  
        return poses, idxs         


## Idiothetic Worlds

class PoseWorld(GridWorld):
    """
    A discrete, finite state Markov Decision Process.
    States represent position and orientation in a 2D grid. 

    attributes:
        n_states:           INT:                                number of states        
        n_actions:          INT:                                number of actions        
        A:                  BOOL[n_state, n_action]:            valid state-action pairs   
        P:                  FLOAT[n_state, n_action, n_state]:  world function P(s1, a, s2) := p(s2 | s1, a)
        R:                  INT[n_state, n_action, n_state]:    reward function R(s1, a, s2)   
        states_start:       INT[n_s0]:                          list of initial states  
        states_terminal:    INT[n_sE]:                          list of terminal states  

        nx, ny:             INT:                                size of the 2D state grid
        no:                 INT:                                number of possible orientations
        walls:              INT[n_walls, 2]:                    location of walls in the 2D state grid              
        actions:            GridActionSet                       an object representing all the possible actions and their costs
        grid:               int[nx, ny]:                        matrix mapping [ns] states to [nx, ny] locations 
        """
    
    def __init__(self, 
                nx,
                ny, 
                init_locs, 
                p_init_locs=None, 
                no=4,
                term_locs=np.ndarray([0,3]), 
                walls=np.ndarray([0,2]), 
                actions=PoseActionSet(),
                action_cost=-1,
                name="poseworld",
                **kwargs):
        '''
        Abstract class representing a discrete, finite state Markov Decision Process with states organized in 2D grid

        args:
            nx, ny:         int                     square size of the grid
            actions:        dict(string:[n_a * 2])  all the [n_a] possible action (movements), encoded as the offset applied to the current state location
            init_locs:      int[n_i * 3]            orientation and grid coordinates of the [n_i] starting positions
            init_locs:      float[n_i]              probability distribution over the [n_i] starting positions
            term_locs:      int[n_t * 3]            orientation and grid coordinates of the [n_t] terminal positions
            walls_grid:     int[n_w * 2]            grid coordinates of the [n_w] wall locations      
            actions:        GridActionSet           an object representing all the possible actions and their costs
        '''
        self.no = no
        init_poses, init_idxs = grid2pose(init_locs, self.no)
        p_init_poses = p_init_locs[init_idxs] if (p_init_locs is not None) else None
        term_poses, _ = grid2pose(term_locs, self.no)
        super().__init__(nx, ny, init_poses, p_init_locs=p_init_poses, term_locs=term_poses, walls=walls, actions=actions, action_cost=action_cost, name=name, **kwargs)


    def _init_grid(self, walls):
        """
        constructs the 3D grid matrix, representing a mapping between state IDs and state position (intended as 2D coordinates + orientation)
        """
        grid = np.ones([self.no, self.nx, self.ny], dtype=int)
        for w in walls:
            grid[:, *w] = -1  # invalid locations are represented with -1s
        grid[grid > -1] = np.cumsum(grid[grid > -1]) - 1  # valid locations are represented by an int identifier in [0, n_state)
        return grid                           


    def grid_transition(self, p1, a):
        """
        arguments:
            - p1: int[3] representing 2D location + orientation on the grid
            - a:  identifier of an action
        returns:
            - lists_l2: all the possible locations that can be reached by performing a in l1
            - probls_l2 the probabilities associated with the locations in lists_l2
        """
        # l1[0] represents the orientation in the grid space
        # l1[1] and l1[2] represent the x,y coordinates in the grid space
        o1 = p1[0]
        l1 = p1[1:]
        
        # action[0] represent the angular displacement
        # action[1] represent the linear displacement (number of grid tiles)
        action = self.get_action(a)
        Δo = action[0]
        Δd = action[1]

        # apply angular displacement
        o2 = (o1 + Δo)
        # ensure orientation is in [0 self.no)
        o2 = (o2 + self.no) % self.no 
        
        # before applying the linear displacement, we need to rotate the displacement vector according to current orientation
        # o = 0 corresponds to "east" orientation, so the default linear displacement corresponds to an upwards movement (x=+, y=0)  
        Δl = np.array([Δd, 0], dtype=int)

        # compute rotation matrix
        θ = o2/self.no *2*np.pi
        R = np.array([(np.cos(θ), -np.sin(θ)), (np.sin(θ), np.cos(θ))])

        # apply rotation
        Δl = np.round(R @ Δl).astype(int)

        # apply linear displacement
        l2 = l1 + Δl

        p2 = np.append(o2, l2)
        lists_p2 = np.array([p2])
        probs_p2 = np.array([1])
        return lists_p2, probs_p2
   

    def add_grid_rewards(self, rew_values, rew_poses, overwrite=False):
        """
        Assigns rewards rew_values[i] to positions rew_poses[i].
        If rew_poses does not have orientation, the reward is assigned to all orientations
        """
        rew_poses, init_idxs = grid2pose(rew_poses, self.no)
        rew_values = rew_values[init_idxs]
        super().add_grid_rewards(rew_values, rew_poses, overwrite=overwrite)
