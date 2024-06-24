import numpy as np


class World:
    """
    Class representing a discrete, finite state Markov Decision Process

    attributes:
        n_states:           INT:                                number of states        
        n_actions:          INT:                                number of actions        
        A:                  BOOL[n_state, n_action]:            valid state-action pairs   
        P:                  FLOAT[n_state, n_action, n_state]:  world function P(s1, a, s2) := p(s2 | s1, a)
        R:                  INT[n_state, n_action, n_state]:    reward function R(s1, a, s2)   
        states_start:       INT[n_s0]:                          list of initial states  
        p_states_start:     FLOAT[n_s0]:                        list of probabilities over initial states  
        states_terminal:    INT[n_sE]:                          list of terminal states  
     """

    def __init__(self, A, P, R, S0, p0=None, n_state=None, n_action=None, name="world", **kwargs):
        '''
        Class representing a discrete, finite state Markov Decision Process

        args:
            A:              func(state)                | bool[n_state, n_action]:           valid state-action pairs   
            P:              func(state, action) | FLOAT[n_state, n_action, n_state]:        world function P(s1, a, s2) := p(s2 | s1, a)
            R:              func(state, action, state) | INT[n_state, n_action, n_state]:   reward function R(s1, a, s2) 
            S0:             int[n_s0]                                                       the initial states
            p0:             float[n_s0]                                                     probability distribution over initial states
            n_state:        int[1]                                                          total number of states. State IDs are the integer numbers within range(n_state)
            n_action:       int[1]                                                          total number of actions. Action IDs are the integer numbers within range(n_action)
        '''
        if (n_state is None) | (n_action is None):
            if not callable(A):
                n_state, n_action = A.shape
            elif not callable(P):
                n_state, n_action, _ = P.shape
            elif not callable(R):
                n_state, n_action, _ = R.shape
            else:
                raise Exception("number of state and/or number of actions are not defined")
        
        self.name=name
        self.n_state = n_state
        self.n_action = n_action

        # Build the matrix of valid state-action pairs
        if callable(A):
            self.A = np.full([self.n_state, self.n_action], False)
            for s in range(self.n_state):
                self.A[s] = A(s, **kwargs)
        else:
            assert A.shape == (self.n_state, self.n_action) 
            self.A = A

        # Build the world model
        if callable(P):
            self.P = np.zeros([self.n_state, self.n_action, self.n_state])
            for s in range(self.n_state):
                for a in np.where(self.A[s])[0]:
                    self.P[s, a] = P(s, a, **kwargs)
        else:
            assert P.shape == (self.n_state, self.n_action, self.n_state) 
            self.P = P

        # Build the reward matrix
        if  callable(R):
            self.R = np.zeros([self.n_state, self.n_action, self.n_state])
            for s1 in self.states_transient:
                for a in np.where(self.A[s1])[0]:
                    for s2 in np.where(self.P[s1, a])[0]:
                        self.R[s1, a, s2] = R(s1, a, s2, **kwargs)      
        else:
            assert R.shape == (self.n_state, self.n_action, self.n_state) 
            self.R = R

        # Build initial and terminal states
        self.states_start = S0
        self.p_states_start = p0 if (p0 is not None) else np.full(len(S0), 1/len(S0))      


    @property
    def states_terminal(self):
        """
        returns the set of terminal states
        """
        return np.where(np.all((np.sum(self.P, axis=1) > 0) == np.eye(self.n_state), axis=1))[0]


    @property
    def states_transient(self):
        """
        returns the set of non-terminal states
        """
        return np.where(np.any((np.sum(self.P, axis=1) > 0) != np.eye(self.n_state), axis=1))[0]
    

    def choose_initial_state(self):
        """
        Returns an initial state s in [self.states_start] sampled from the distribution [p_states_start]
        """
        return np.random.choice(self.states_start, p=self.p_states_start)


    def model(self, s, a):
        """
        Returns:
            - a [n_state] vector representing probability distribution p(s2 | S, A) for a given state S and action A
            - a [n_state] vector representing the reward values associated with states s2 for a=A and s1=S
        """
        assert self.valid_action(s, a)
        return self.P[s, a], self.R[s, a]


    def step(self, s1, a):
        """
        Provides the outcome of taking actions a in state s.
        Returns:
            - the new state s2
            - the obtained reward r
            - whether the new state is terminal
        """
        p_s2, r_s2 = self.model(s1, a)
        s2 = np.random.choice(self.n_state, p=p_s2)
        r = r_s2[s2]
        terminal = s2 in self.states_terminal
        return s2, r, terminal


    def valid_action(self, s1, a):
        """
        check whether a given <state-action> pair is allowed
        """
        try:
            return self.A[s1, a]
        except:
            return False
        

    