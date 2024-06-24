import numpy as np
from abc import ABCMeta, abstractmethod


class BaseTabular(metaclass=ABCMeta):
    '''
    Discrete Reinforcement Learning Agent performing Geometric Path Programming in Grid World environments.
        args:
            env: gridworld environment
            gamma: time discount factor
    '''

    def __init__(self, env, name="tabular", **kwargs):

        self.env = env
        self.name = name
        self.state = self.env.choose_initial_state()
        self.__dict__.update(kwargs)


    def reset(self):
        self.state = self.env.choose_initial_state()


    @property
    def n_state(self):
        return self.env.n_state


    @property
    def n_action(self):
        return self.env.n_action


    @property
    def A(self):
        return self.env.A


    def policy(self, s):
        return self.policy_vec()[s, :]
    

    @abstractmethod
    def policy_vec(self):
        pass
    

    @abstractmethod
    def reset(self):
        pass


    def choose_action(self, state, greedy=False):
        """
        Chooses and returns an action a given a state s according to the policy p(a|s)
        """
        if greedy:
            return np.argmax(self.policy(state))
        else:
            return np.random.choice(self.n_action, p=self.policy(state))
    

    def trajectory(self, s0=None, n_steps=1000, greedy=False, move=False):
        '''
        Generate a trajectory of length [n_steps] starting from state [s0]
        '''

        # Reinitialize the position
        if s0 is None:
            s0 = self.state

        # Initialize sequences of states, actions and rewards composing the trajectory
        ss = np.zeros(n_steps + 1, dtype=int)
        aa = np.zeros(n_steps, dtype=int)
        rr = np.zeros(n_steps, dtype=int)

        keep_rolling = True
        arrived = False
        ss[0] = s0
        i = 0

        # Roll out
        while keep_rolling and not arrived:
            ai = self.choose_action(ss[i], greedy=greedy)
            si_plus, ri, done = self.env.step(ss[i], ai)

            ss[i + 1] = si_plus
            aa[i] = ai
            rr[i] = ri
            i += 1

            # Stop when a terminal state is reached... (termination)
            if done:
                arrived = True
                ss = ss[:i + 1]
                aa = aa[:i]
                rr = rr[:i]

            # ...or when the maximum number of steps is reached (truncation)
            if  i >= n_steps:
                keep_rolling = False

        # update position
        if move:
            self.state = ss[-1]

        return ss, aa, rr, arrived


    @abstractmethod
    def learn(self):
        pass