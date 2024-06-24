import numpy as np
from environments.world import World


class SparseWorld(World):
    """
    Abstract class representing a discrete, finite state Markov Decision Process with sparse rewards

    attributes:
        n_states            INT:                                number of states        
        n_actions           INT:                                number of actions        
        A:                  BOOL[n_state, n_action]:            valid state-action pairs   
        P:                  FLOAT[n_state, n_action, n_state]:  world function P(s1, a, s2) := p(s2 | s1, a)
        R:                  INT[n_state, n_action, n_state]:    complete reward function R(s1, a, s2) accounting for action costs and rewards   
        R0:                 INT[n_state, n_action, n_state]:    default reward function R0(s1, a, s2) accounting for action costs only
        states_start        INT[n_s0]:                          list of initial states  
        p_states_start:     float[n_s0]                         probability distribution over initial states
        states_terminal:    INT[n_sE]:                          list of terminal states  
     """
    

    def __init__(self, A, P, S0, p0=None, action_cost=0, n_state=None, n_action=None, name="world", **kwargs):
        """
        arguments:
            A:              BOOL[n_state, n_action]:            valid state-action pairs   
            P:              FLOAT[n_state, n_action, n_state]:  world function P(s1, a, s2) := p(s2 | s1, a)
            S0              INT[n_s0]:                          list of initial states  
            p0:             FLOAT[n_s0]                         probability distribution over initial states
            action_cost:    INT | INT func(s1, a, s2):          cost (negative reward) associated with each action   
            n_states        INT:                                number of states        
            n_actions       INT:                                number of actions   
        """
        R0_func = action_cost if callable(action_cost) else (lambda s1, a, s2 : action_cost)
        super().__init__(A, P, R0_func, S0, p0, n_state, n_action, name, **kwargs)
        self.R0 = self.R.copy()        
        self.state_rewards_only = True
        

    def reset_rewards(self):
        """
        Resets the reward matrix removing all rewards and keeping only action costs 
        """
        self.R = self.R0.copy()
        self.state_rewards_only = True
        
    
    def add_rewards(self, rew_values, rew_states, rew_actions=None, rew_targets=None, overwrite=False):
        """
        Assigns rewards rew_values[i] to event triplet <rew_states1[i], rew_actions[i] ,rew_targets[i].
        If rew_actions or rew_targets is None, the reward is assigned to the whole set of actions or targets.
        """
        for i, v in enumerate(rew_values):
            s = rew_states[i]

            if rew_actions is None:
                if overwrite:
                    self.R[:, :, s] = v
                else:
                    self.R[:, :, s] += v

            elif rew_targets is None:
                a = rew_actions[i]
                if overwrite:
                    self.R[s, a, :] = v
                else:
                    self.R[s, a, :] += v
                self.state_rewards_only = False

            else:
                a = rew_actions[i]
                t = rew_targets[i]
                if overwrite:
                    self.R[s, a, t] = v
                else:
                    self.R[s, a, t] += v
                self.state_rewards_only = False


        # triplets that are not reachable should have zero reward.
        self.R[self.P == 0] = 0
        self.R[self.states_terminal] = 0

    
    def get_state_rewards(self):
        """
        Returns all the rewards that are only state dependent (i.e. can be obtained upon reaching a state regardless of the previous state and action):
            - states: np array of state IDs corresponding to a state-dependent reward
            - rewards np array of rewards associated with [states]
        """
        states = np.array([], dtype=int)
        rewards = np.array([])
        
        R_sparse = (self.R - self.R0)
        valid_events = (self.P > 0)
        valid_events[self.states_terminal, :, :] = False
        
        for s in range(self.n_state):
            
            s_valid = valid_events[:, :, s]
            s_rewards_sparse = R_sparse[:,  :, s]
            s_rewards = s_rewards_sparse[s_valid]

            if len(s_rewards) > 0:
                if np.all(s_rewards == s_rewards[0]) and (s_rewards[0] != 0) :
                    states = np.append(states, s)
                    rewards = np.append(rewards, s_rewards[0])

        return states, rewards
    

    def get_state_action_rewards(self):
        """
        Returns all the rewards that are only state-action dependent (i.e. can be obtained upon executing action a in state s regardless of the outcome):
            - states: np array of state IDs corresponding to a state-action-dependent reward
            - actions: np array of action IDs corresponding to a state-action-dependent reward
            - rewards np array of rewards associated with [states]-[actions]
        """

        states = np.array([], dtype=int)
        actions = np.array([], dtype=int)
        rewards = np.array([])

        if not self.state_rewards_only:

            R = self.R.copy()

            # Remove from R all state-dependent rewards
            states, _ = self.get_state_rewards()
            if len(states):
                R[:, :, states] = self.R0[:, :, states]

            states = np.array([], dtype=int)
            actions = np.array([], dtype=int)
            rewards = np.array([])

            for s in range(self.n_state):
                for a in range(self.n_action):
                    rs = (R - self.R0)[s, a, self.P[s, a] > 0]
                    if len(rs):
                        if np.all(rs == rs[0]) and (rs[0] != 0) :
                            states = np.append(states, s)
                            actions = np.append(actions, a)
                            rewards = np.append(rewards, rs[0])

        return states, actions, rewards


    def get_state_action_state_rewards(self):
        """
        Returns all the reward triplets <state,action,state> that are not just state-dependent or state-action dependent 
        """

        states_1 = np.array([], dtype=int)
        actions = np.array([], dtype=int)
        states_2 = np.array([], dtype=int)
        rewards = np.array([])

        if not self.state_rewards_only:

            R = self.R.copy() 
            R[self.P == 0] = self.R0[self.P == 0]

            # Remove from R all state- and state-action- dependent rewards
            states2, _ = self.get_state_rewards()
            if len(states2):
                R[:, :, states2] = self.R0[:, :, states2]

            states1, actions, _ = self.get_state_action_rewards()
            if len(states1):
                R[states1, actions, :] = self.R0[states1, actions, :]

            idxs = (R != self.R0)
            states_1 = np.where(idxs)[0]
            actions = np.where(idxs)[1]
            states_2 = np.where(idxs)[2]
            rewards = R[idxs]

        return states_1, actions, states_2, rewards