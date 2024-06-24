from environments.sparseworld import SparseWorld
from environments.gridworlds.gridactions import *
import numpy as np


## Grid Worlds

class GridWorld(SparseWorld):
    """
    A discrete, finite state Markov Decision Process with states organized in 2D grid

    attributes:
        n_states:           INT:                                number of states        
        n_actions:          INT:                                number of actions        
        A:                  BOOL[n_state, n_action]:            valid state-action pairs   
        P:                  FLOAT[n_state, n_action, n_state]:  world function P(s1, a, s2) := p(s2 | s1, a)
        R:                  INT[n_state, n_action, n_state]:    reward function R(s1, a, s2)   
        states_start:       INT[n_s0]:                          list of initial states  
        p_states_start:     FLOAT[n_s0]:                        list of probabilities over initial states  
        states_terminal:    INT[n_sE]:                          list of terminal states  

        nx, ny              INT:                                size of the 2D state grid
        walls:              INT[n_walls, 2]:                    location of walls in the 2D state grid              
        actions:            GridActionSet                       an object representing all the possible actions and their costs
        grid                INT[nx, ny]:                        matrix mapping [ns] states to [nx, ny] locations 
        """

    def __init__(self, 
                nx,
                ny, 
                init_locs, 
                p_init_locs=None,
                term_locs=np.ndarray([0,2]), 
                walls=np.ndarray([0,2]), 
                actions=BasicActionSet(),
                action_cost = -1,
                name="gridworld",
                **kwargs):
        '''
        Abstract class representing a discrete, finite state Markov Decision Process with states organized in 2D grid

        args:
            nx, ny: square size of the grid
            actions:            GridActionSet           an object representing all the possible actions and their costs
            init_locs:          int[n_i * 2]            grid coordinates of the [n_i] possible starting locations
            p_init_locs:        float[n_i]              probability distribution over [n_i] possible starting locations
            term_locs:          int[n_t * 2]            grid coordinates of the [n_t] terminal locations
            walls_grid:         int[n_w * 2]            grid coordinates of the [n_w] wall locations      
        '''
        self.nx = nx
        self.ny = ny
        self.actions = actions
        self.grid = self._init_grid(walls)
        self.n_state = np.sum(self.grid > -1)
        self.n_action = self.actions.n_action
        states_start = self.encode(init_locs)
        states_terminal = self.encode(term_locs)
        A, P = self._init_world_model(states_terminal)
        super().__init__(A, P, states_start, p_init_locs, action_cost, self.n_state, self.n_action, name=name, **kwargs)


    def _init_grid(self, walls):
        """
        constructs the 2D grid matrix, representing a mapping between state IDs and state locations
        """
        grid = np.ones([self.nx, self.ny], dtype=int)
        # invalid locations are represented with -1s
        for w in walls:
            grid[*w] = -1
        # valid locations are assigned with an integer identifier in [0, n_state)
        grid[grid > -1] = np.cumsum(grid[grid > -1]) - 1
        return grid                           


    def _init_world_model(self, states_terminal):
        """
        constructs the A[s1, a] and P[s1, a, s2] matrices, representing validity of state-action pairs and world model, respectively
        """
        A = np.full([self.n_state, self.n_action], False)
        P = np.zeros([self.n_state, self.n_action, self.n_state])
        for s1 in range(self.n_state):
            # terminal states connect to themselves through any action
            if s1 in states_terminal:
                A[s1,:] = True
                P[s1, :, s1] = 1
            else:
                for a in range(self.n_action):
                    l1 = self.decode(s1)
                    l2s, p2s,  = self.grid_transition(l1, a)
                    for l2, p2 in zip(l2s, p2s):   
                        if self.valid_position(l2):
                            s2 = self.encode(l2)
                            A[s1, a] = True
                            P[s1, a, s2] = p2
                    # normalize P again since some invalid actions might have been removed
                    if A[s1, a]:
                        P[s1, a, :] /= np.sum(P[s1, a, :])
        if np.any(np.sum(A, axis=1) == 0):
            raise Exception("some state has no valid actions")
        return A, P


    def grid_transition(self, l1, a):
        """
        arguments:
            - l1: 2d location on the grid
            - a:  identifier of an action
        returns:
            - lists_l2: all the possible locations that can be reached by performing a in l1
            - probls_l2 the probabilities associated with the locations in lists_l2
        """
        lists_l2 = np.array([l1 + self.get_action(a)])
        probs_l2 = np.array([1])
        return lists_l2, probs_l2


    def add_grid_rewards(self, rew_values, rew_loc_start, rew_actions=None, rew_loc_target=None, overwrite=False):
        """
        Assigns rewards rew_values[i] to event triplet <rew_loc_start[i], rew_actions[i] ,rew_loc_target[i].
        If rew_actions or rew_loc_target is None, the reward is assigned to the whole set of actions or targets.
        """
        rew_states_start = self.encode(rew_loc_start)
        if rew_loc_target is None:
            rew_states_target = None
        else:
            rew_states_target = self.encode(rew_loc_target)
        self.add_rewards(rew_values, rew_states_start, rew_actions, rew_states_target, overwrite)


    def get_action(self, id):
        """
        id: the label (str) or index (int) of the action to retrieve
        returns the corresponding action
        """
        return self.actions.get_action(id)


    def encode(self, pp):
        """
        from state position to state index 
        """  
        if len(pp) == 0:
            return np.array([])
        if not isinstance(pp, np.ndarray):
            pp = np.array(pp)
        if len(pp.shape) == 1:
            assert(self.valid_position(pp))
            return self.grid[*pp]
        else:
            ss = np.ndarray(pp.shape[0], dtype=int)
            for i, p in enumerate(pp):
                assert(self.valid_position(p))
                ss[i] = self.grid[*p]
            return ss
        

    def decode(self, ss):
        """
        from state index to state position 
        """ 
        if np.isscalar(ss):
            assert((ss >= 0) & (ss < self.n_state))
            return np.argwhere(self.grid == ss)[0]
        else:
            assert np.all((ss >= 0) & (ss < self.n_state))
            return np.array([np.argwhere(self.grid == s)[0] for s in ss])


    def valid_position(self, p):
        """
        check if state position is valid 
        """ 
        if np.all(p >= 0) & np.all(p < self.grid.shape):
            return (self.grid[*p] != -1)
        else:
            return False


    def valid_position_vec(self):
        """
        A vector bool[n_state] representing valid vs non valid positions
        """ 
        return self.grid > -1