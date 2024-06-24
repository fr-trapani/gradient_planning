import numpy as np
from environments.gridworlds.gridworld import *

class ErgodicWorld(GridWorld):
    """
    An ergodic gridworld:
        - there are no terminal states (absorbing states)
        - all states are recurrent
        - the outcome of each action is stocastic 
    """

    def __init__(self, nx, ny, init_locs, p0=0.9, actions=IdleActionSet(), **kwargs):
        self.p0 = p0
        super().__init__(nx=nx, ny=ny, init_locs=init_locs, term_locs=np.array([]), actions=actions, ** kwargs)


    def grid_transition(self, l1, a):
        """
        arguments:
            - l1: 2d location on the grid
            - a:  identifier of an action
        returns:
            - lists_l2: all the possible locations that can be reached by performing a in l1
            - probls_l2 the probabilities associated with the locations in lists_l2
        """
        lists_l2 = []
        probs_l2 = []

        for action in self.actions.action_values:
            l2 = l1 + action
            p = self.p0 if np.all(self.get_action(a) == action) else (1 - self.p0) / (self.n_action - 1)
            
            lists_l2.append(l2)
            probs_l2.append(p)

        lists_l2 = np.array(lists_l2)
        probs_l2 = np.array(probs_l2)
        return lists_l2, probs_l2
    

    
    
class ActionRewardGrid(GridWorld):
    """
    A maze with action dependent reward and stocastic actions
    """
    def __init__(self, **kwargs):
        nx = 15
        ny = 7

        init_locs = np.array([[2, 3]])
        term_locs = np.array([[12, 3]])

        rew_s_locs = np.array([[10, 3], [12, 3]])
        rew_s_vals = np.array([-300, 25])

        rew_sa_locs = np.array([[11, 3], [10, 4], [10, 2], [11, 5], [11, 1], [12, 5], [12, 1]])
        rew_sa_acts = np.array([1,        1,       1,       0,       2,       0,       2])
        rew_sa_vals = np.array([100,     -300,    -300,    -300,    -300,    -300,    -300])

        rew_sas_locs = np.array([[3, 1], [4, 1], [5, 1], [3, 5], [4, 5], [5, 5]])
        rew_sas_acts = np.array([0, 0, 0, 2, 2, 2])
        rew_sas_tgts = np.array([[3, 0], [4, 0], [5, 0], [3, 6], [4, 6], [5, 6]])
        rew_sas_vals = np.array([-10, -10, -10, -10, -10, -10])


        super().__init__(nx, ny, init_locs, term_locs=term_locs, **kwargs)
        self.add_grid_rewards(rew_s_vals, rew_s_locs)
        self.add_grid_rewards(rew_sa_vals, rew_sa_locs, rew_sa_acts)
        self.add_grid_rewards(rew_sas_vals, rew_sas_locs, rew_sas_acts, rew_sas_tgts)


    def grid_transition(self, l1, a):
        """
        arguments:
            - l1: 2d location on the grid
            - a:  identifier of an action
        returns:
            - lists_l2: all the possible locations that can be reached by performing a in l1
            - probls_l2 the probabilities associated with the locations in lists_l2
        """
        lists_l2 = np.array([l1 + self.get_action(a), l1])
        probs_l2 = np.array([0.95, 0.05])
        return lists_l2, probs_l2
    

class WindyWorld(GridWorld):

    def __init__(self, nx, ny, init_locs, wind_prob=0.3, wind_dir=np.array([0, 1]), **kwargs):
        """
        A GridWorld where the outcome of actions is not deterministic.
        The target position p2 is shifted in the direction [wind_dir] with probability [wind_prob].
        """   
        self.wind_prob = wind_prob
        self.wind_dir = wind_dir
        super().__init__(nx, ny, init_locs, **kwargs)
        

    def grid_transition(self, l1, a):
        """
        arguments:
            - l1: 2d location on the grid
            - a:  identifier of an action
        returns:
            - lists_l2: all the possible locations that can be reached by performing a in l1
            - probls_l2 the probabilities associated with the locations in lists_l2
        """
        lists_l2 = np.array([])
        probs_l2 = np.array([])
        
        l2A = l1 + self.get_action(a)
        if self.valid_position(l2A):            
            l2B = l2A + self.wind_dir
            if self.valid_position(l2B):
                lists_l2 = np.array([l2A, l2B])
                probs_l2 = np.array([1-self.wind_prob, self.wind_prob])
            else:
                lists_l2 = np.array([l2A])
                probs_l2 = np.array([1])

        return lists_l2, probs_l2


class WindyCliff(WindyWorld):
    """
    A dangerous GridWorld with a long and narrow passage close to a cliff 
    """   

    def __init__(self, wind_prob=0.3, **kwargs):

        nx = 15
        ny = 7
        wind_dir=np.array([0, 1])
        init_locs = np.array([[1, 1]])

        treasure_loc = np.array([[13, 1]])
        treasure_val = np.full(treasure_loc.shape[0], 100)

        cliff_map = np.full([nx, ny], False, dtype=bool)
        cliff_map[1:14, 3] = True
        cliff_locs = np.argwhere(cliff_map)
        cliff_vals = np.full(cliff_locs.shape[0], -100)


        super().__init__(nx, ny, init_locs, term_locs=treasure_loc, wind_prob=wind_prob, wind_dir=wind_dir,**kwargs)    
        self.add_grid_rewards(treasure_val, treasure_loc)
        self.add_grid_rewards(cliff_vals, cliff_locs)


