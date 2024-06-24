import numpy as np
from tqdm import tqdm
from agents.basetabular import BaseTabular


class DynaQ(BaseTabular):

    def __init__(self, env, gamma=0.99, epsilon=0.1, name="dynaq", soft_policy=True, **kwargs):

        super().__init__(env, name=name, **kwargs)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.soft_policy = soft_policy
        
        self.Q = None
        self.M = None
        self.R = None
        self.M_visited = None

        self.Q_t = None
        self.p_t = None
        self.epochs = None    

        self.reset()


    def reset(self, clear_params=True, clear_history=True):

        if clear_params:
            self.P = self.A / np.sum(self.A, axis=1, keepdims=True)
            self.Q = np.zeros([self.n_state,self.n_action], dtype=float)
            self.M = np.zeros([self.n_state, self.n_action, self.n_state], dtype=float)
            self.R = np.zeros([self.n_state, self.n_action, self.n_state], dtype=float)
            self.M_visited = np.full([self.n_state, self.n_action], False, dtype=bool)

        if clear_history:
            self.Q_t = np.zeros([0, self.n_state, self.n_action], dtype=float)
            self.p_t = np.zeros([0, self.n_state, self.n_action], dtype=float)
            self.epochs = np.zeros(0)


    def policy_vec(self):
        return self.P


    def V(self, s=None):
        return np.max(self.Q, axis=1)
    

    def value(self, s=None):
        return self.V_vec[s]
    

    def model(self, s, a):
        if self.M_visited[s, a]:
            return self.M[s, a, :] / np.sum(self.M[s, a, :])
        else:
            return None
        
    
    def replay(self):
        sa_visited = np.argwhere(self.M_visited)
        [s1, a] = sa_visited[np.random.randint(len(sa_visited))]
        s2 = np.random.choice(self.n_state, p=self.model(s1, a))
        r = self.R[s1, a, s2]
        return s1, a, s2, r


    def Q_update(self, s1, a, s2, r, alpha):
        self.Q[s1, a] += alpha * (r + self.gamma * np.max(self.Q[s2, :]) - self.Q[s1 ,a])


    def P_update(self, s):

        if self.soft_policy:
            self.P[s, self.A[s]] = np.exp(self.Q[s, self.A[s]]) / np.sum(np.exp(self.Q[s, self.A[s]]))

        else:
            best_actions = np.argwhere(self.Q[s, self.A[s]] == np.amax(self.Q[s, self.A[s]]))

            if best_actions == self.A[s]:
                self.P[s, self.A[s]] = 1/len(self.A[s])

            else:
                p_greedy = (1 - self.epsilon) / len(best_actions)
                p_others = self.epsilon / (self.A[s] - len(best_actions))
                self.P[s, self.A[s]] = p_others
                self.P[s, best_actions] = p_greedy
                
    
    def M_update(self, s1, a, s2, r):
        self.M_visited[s1, a] = True
        self.M[s1, a, s2] += 1
        self.R[s1, a, s2] = r


    def learn(self, n_episodes=1000, n_steps=100, n_replay=10, alpha=0.1, alpha_func=None, clear_history=False, **kwargs):

        # clear history
        if clear_history:
            self.reset(clear=False)

        # initialize history
        Q_t = np.zeros([n_episodes, self.n_state, self.n_action], dtype=float)
        p_t = np.zeros([n_episodes, self.n_state, self.n_action], dtype=float)

        # learning loop
        for i_episode in tqdm(range(n_episodes)):

            s1 = self.env.choose_initial_state()
            done = False

            for i_step in range(n_steps):

                a = np.random.choice(self.n_action, p = self.policy(s1))
                s2, r, done = self.env.step(s1, a)

                # update alpha if computed dynamically
                if alpha_func is not None:
                    alpha = alpha_func(self)

                # Bellman Update
                self.Q_update(s1, a, s2, r, alpha)
                self.P_update(s1)

                # Update Model
                self.M_update(s1, a, s2, r)

                # Replays
                for i_rep in range(n_replay):

                    s1_r, a_r, s2_r, r_r = self.replay()

                    # update alpha if computed dynamically
                    if alpha_func is not None:
                        alpha = alpha_func(self)

                    # Bellman Update
                    self.Q_update(s1_r, a_r, s2_r, r_r, alpha)
                    self.P_update(s1_r)

                if done:
                    break
                s1 = s2
            
            

            # store history
            Q_t[i_episode] = self.Q
            p_t[i_episode] = self.P

               
        # append history
        self.Q_t = np.concatenate([self.Q_t, Q_t])
        self.p_t = np.concatenate([self.p_t, p_t])
        self.epochs = np.append(self.epochs, n_episodes + 1)



class Reinforce(BaseTabular):
 
    def __init__(self, env, gamma=0.99, name="reinforce", **kwargs):

        super().__init__(env, name=name, **kwargs)

        self.gamma = gamma
        
        # Initialized in reset, updated in trajectory
        self.v = None           
        self.theta = None       # policy 

        # Initialized in reset, updated in learn
        self.theta_t = None     
        self.p_t = None     # policy p(A|S) distribution over time
        self.epochs = None      

        self.reset()


    def reset(self, clear_params=True, clear_history=True):

        if clear_params:
            # updated in trajectory
            self.theta = np.log(self.A / np.sum(self.A, axis=1, keepdims=True))

        if clear_history:
            # updated in gpp_learn
            self.theta_t = np.zeros((0, self.n_state, self.n_action))   
            self.p_t = np.zeros((0, self.n_state, self.n_action))   # policy p(A|S) distribution over time
            self.epochs = np.zeros(0)                               # training epochs


    def policy_vec(self):
        return np.exp(self.theta) / np.sum(np.exp(self.theta), axis=1, keepdims=True)
       

    def learn(self, 
              n_episodes=1000, 
              n_steps=100, 
              alpha=0.1, 
              alpha_func=None, 
              clear_history=False, 
              **kwargs):

        # clear history
        if clear_history:
            self.reset(clear_params=False)

        # initialize history
        theta_t = np.zeros((n_episodes, self.n_state, self.n_action))
        p_t = np.zeros((n_episodes, self.n_state, self.n_action))

        # learning loop
        for e in tqdm(range(n_episodes)):

            # update alpha if computed dynamically
            if alpha_func is not None:
                alpha = alpha_func(self)

            S0 = self.env.choose_initial_state()
            ss, aa, rr, arrived = self.trajectory(s0=S0, n_steps=n_steps, greedy=False, move=False)
            tt = np.arange(len(rr))

            for t in tt:

                t_vec = tt[t:] - t
                r_vec = rr[t:]
                G = (self.gamma**t_vec) @ r_vec

                s = ss[t]
                a = aa[t]

                grad_log = np.zeros([self.n_state, self.n_action])
                grad_log[s] -= self.policy(s)
                grad_log[s, a] += 1 

                self.theta += alpha * (self.gamma**t) * G * grad_log 

            # store history
            theta_t[e] = self.theta
            p_t[e] = self.policy_vec()

        # append history
        self.theta_t = np.concatenate([self.theta_t, theta_t])
        self.p_t = np.concatenate([self.p_t, p_t])
        self.epochs = np.append(self.epochs, t + 1)
