import numpy as np
from tqdm import tqdm
from agents.basetabular import BaseTabular
from abc import ABCMeta, abstractmethod
from utils.policy_functions import *


class PGP(BaseTabular, metaclass=ABCMeta):
    '''
    Discrete Reinforcement Learning Agent performing Path Gradient Planning in Grid World environments.
        args:
            env: gridworld environment
            gamma: time discount factor
    '''

    def __init__(self, env, gamma=0.99, p0_func=p0_onehot, theta0_func=theta0_uniform, name="pgp", **kwargs):

        super().__init__(env, name=name, **kwargs)

        self.gamma = gamma
        self.p0_func = p0_func
        self.theta0_func = theta0_func
        
        # Initialized in reset, updated in trajectory
        self.theta = None       # policy parameter
        self.p0 = None          
        
        # Initialized in reset, updated in learn
        self.theta_t = None     # policy parameter over time
        self.sr0_t = None       # E[SR]_p0 over time
        self.v_t = None         # value V(s) over time
        self.epochs = None      

        self.reset()


    def reset(self, clear_params=True, clear_history=True):
        '''
        Resets the policy table, setting all probability distributions p(A|s) to uniform distributions
        '''

        if clear_params:
            # updated in trajectory
            self.theta = self.theta0_func(self)
            self.p0 = self.p0_func(self, self.env.states_start, self.env.p_states_start)
            super().reset()

        if clear_history:
            # updated in gpp_learn
            self.theta_t = np.zeros((0, self.n_state, self.n_action))   # policy parameter over time
            self.sr0_t = np.zeros((0, self.n_state))                    # E[SR]_p0 over time
            self.v_t = np.zeros(0)                                      # value V(s) over time
            self.epochs = np.zeros(0)                                   # training epochs


    def load_history_policy(self, t=-1):
        '''
        Loads the policy from an intermediate traininst step [t]
        '''
        self.theta = self.theta_t[t, :, :]


    @property
    def T(self):
        '''
        Builds the Transition Matrix
        '''
        π = self.policy_vec()
        P = self.env.P
        T = np.einsum("ij, ijk -> ik", π, P, optimize=True)
        return T


    @property
    def SR(self):
        '''
        Builds the Successor-Representation Matrix D
        '''
        D_inv = np.eye(self.n_state) - self.gamma * self.T
        SR = np.linalg.inv(D_inv)
        return SR
    
    
    @property
    def E(self):
        '''
        Builds the State-Action Successor-Representation Matrix E
        '''
        P = self.env.P
        D = self.SR
        return P@D


    @property
    def V(self):
        '''
        Returns the value function V(s).
        '''
        D = self.SR
        π = self.policy_vec()
        r = self.env.R
        V = np.einsum("oi, ij, ijk -> o", D, π, r, optimize=True)
        return V


    @property
    def EV(self):
        '''
        Returns the expected value of V(s0).
        '''
        return self.p0 @ self.V


    @property
    def CorrCount_SS(self):
        '''
        Returns a state to state counter correlation matrix CC_i,j with i,j ∈ S 
        '''    
        p0 = self.p0
        D = self.SR
        δs = np.eye(self.n_state)

        # CC_i,j  = Σ_0 [ p_0 ⋅ D_0i ⋅ D_ij + p_0 ⋅ D_0j ⋅ D_ji ] + D_ij ⋅ δ_ij
        CC_1 = np.einsum('o, oi, ij -> ij', p0, D, D, optimize='optimal')
        CC_2 = np.einsum('o, oj, ji -> ij', p0, D, D, optimize='optimal')
        CC_3 = D * δs
        return CC_1 + CC_2 + CC_3
    

    @property
    def CorrCount_AA(self):
        '''
        Returns a state-action to state-action counter correlation matrix CC_<ij>,<kl> with i,k ∈ S, j,l ∈ A
        '''
        p0 = self.p0
        D = self.SR
        E = self.E
        π = self.policy_vec()
        δs = np.eye(self.n_state)
        δa = np.eye(self.n_action)
        
        # CC_<ij>,<kl>  = p_0 ⋅ D_0i ⋅ π_ij ⋅ δ_ik ⋅ δ_jl + p_0 ⋅ [D_0i ⋅ PD_<ij>,k + D_0k ⋅ PD_<kl>,i] ⋅ π_ij⋅ π_kl
        CC_1 = np.einsum('o, oi, ij, ik, jl -> ijkl', p0, D, π, δs, δa, optimize='optimal')
        CC_2 = np.einsum('o, oi, ijk, ij, kl -> ijkl', p0, D, E, π, π, optimize='optimal')
        CC_3 = np.einsum('o, ok, kli, ij, kl -> ijkl', p0, D, E, π, π, optimize='optimal')
        return CC_1 + CC_2 + CC_3
    

    @property
    def CorrCount_AS(self):
        '''
        Returns a state-action to state counter correlation matrix CC_<ij>,<k> with i,k ∈ S, j ∈ A
        '''
        p0 = self.p0
        D = self.SR
        E = self.E
        π = self.policy_vec()
        # CC_<ij>,k  = p_0 ⋅ D_0i ⋅ π_ij ⋅ PD_<ij>,k + p_0 ⋅ D_0,k ⋅ D_k,i ⋅ π_ij
        CC_1 = np.einsum('o, oi, ij, ijk -> ijk', p0, D, π, E, optimize='optimal')
        CC_2 = np.einsum('o, ok, ki, ij -> ijk', p0, D, D, π, optimize='optimal')
        return CC_1 + CC_2


    @property
    def CorrCount_SA(self):
        '''
            Returns a state to state-action counter correlation matrix CC_<i>,<j,k> with i,j ∈ S, k ∈ A
        '''
        CC_as = self.CorrCount_AS
        CC_sa = np.moveaxis(CC_as, [2], [0])
        return CC_sa


    def policy_gradient(self, λ=0, **kwargs):
        '''
        Computes the gradient of the value function V(s) with respect to the policy p(a|s).
        '''
        # GRADIENT OF THE EXPECTED VALUE: 
        # ∇_π_ij E[V(o)]    =     Σ_osa   ∇_π_ij [ p_o ⋅ D_os ⋅ π_sa ⋅ R_sa ]
        #                   =     Σ_osay  p_o ⋅ γ ⋅ D_oi ⋅ P_ijy ⋅ D_ys ⋅ π_sa ⋅ R_sa
        #                      +  Σ_o     p_o ⋅ D_oi ⋅ R_ij
        #
        # with o,s,i,y ∈ S  and  a,j ∈ A 
        
        # VARIABLES:
        # γ: the time discount factor
        # R_ij: the reward obtained at state i with action j
        # π_ij: probability p(a=j | s1=i)
        # D_ik: the successor matrix
        # P_ijk: probability vector p(s2=k | s1=i, a=j)

        # PARAMETERS
        γ = self.gamma
        p0 = self.p0
        π = self.policy_vec()
        D = self.SR
        P = self.env.P
        R = np.sum(self.env.R * self.env.P, axis = 2)

        # dEV1 = Σ_osay  p_o ⋅ γ ⋅ D_oi ⋅ P_ijy ⋅ D_ys ⋅ π_sa ⋅ R_sa
        # dEV2 = Σ_o p_o ⋅ D_oi ⋅ R_ij
        dEV1 = γ * np.einsum('o, oi, ijy, ys, sa, sa -> ij', p0, D, P, D, π, R, optimize='optimal')
        dEV2 = np.einsum('o, oi, ij -> ij', p0, D, R, optimize='optimal')
        dEV = dEV1 + dEV2    
        
        # KLD regularization:
        # ∇_π_ij KLD(π | π_0)    =  ∇_π_ij [ Σ_sa π_sa ⋅ log(π_sa / π_0) ]
        #                        =  log(π_ij / π_0)    +    π_ij ⋅ ∇_π_ij [ log(π_ij) - log(π_0) ]  
        #                        =  log(π_ij / π_0)    +    π_ij ⋅ [ 1/π_ij - 0 ]  
        #                        =  log(π_ij)  -  log(π_0)  +  1  
        π0 = self.A / np.sum(self.A, axis=1, keepdims=True)

        # set a threshold on policy values to avoid log(0) = -inf divergence
        π_thresh = 1e-2
        π[self.A & (π < π_thresh)] = π_thresh

        # only compute divergence on valid actions
        L = np.zeros(π.shape)
        L[self.A] = np.log(π[self.A] / π0[self.A])  +  1
        
        g = dEV - λ * L
        return g


    def policy_hessian(self, **kwargs):
        """
        Compute the hessian of the value V(s) with respect to the policy p(a|s)
        """
        γ = self.gamma
        p0 = self.p0
        π = self.policy_vec()
        D = self.SR
        P = self.env.P
        R = np.sum(self.env.R * self.env.P, axis = 2)

        h1 = γ^2 * np.einsum('o, ok, klm, mi, ijn, ns, sa, sa -> klij', p0, D, P, D, P, D, π, R, optimize='optimal')
        h2 = γ^2 * np.einsum('o, oi, ijn, nk, klm, ms, sa, sa -> klij', p0, D, P, D, P, D, π, R, optimize='optimal')
        h3 = γ   * np.einsum('o, oi, ijn, nk, kl -> klij', p0, D, P, D, R, optimize='optimal')
        h4 = γ   * np.einsum('o, ok, klm, mi, ij -> klij', p0, D, P, D, R, optimize='optimal')

        H = h1 + h2 + h3 + h4
        return H
    

    def learn(self, n_steps=1000, 
              alpha=0.1, 
              alpha_func=None, 
              min_theta=np.log(0.01), 
              keep_history=True,
              **kwargs):
        '''
        Optimizes the policy theta through gradient descend.
        '''

        # initialize history
        v_t = np.zeros(n_steps)
        sr0_t = np.zeros((n_steps, self.n_state))
        theta_t = np.zeros((n_steps, self.n_state, self.n_action))

        # learning loop
        for t in tqdm(range(n_steps)):

            # update alpha if computed dynamically
            if alpha_func is not None:
                alpha = alpha_func(self, **kwargs)

            # compute gradient
            g = self.gradient(**kwargs) 

            # update theta
            self.update(g, alpha, **kwargs)

            # clamp policy if needed
            if min_theta is not None:
                self.theta[self.A] = np.maximum(self.theta[self.A], min_theta)

            # store history
            v_t[t] = self.EV
            sr0_t[t] = self.p0 @ self.SR
            theta_t[t] = self.theta

        # append history
        if keep_history:
            self.v_t = np.concatenate([self.v_t, v_t])
            self.sr0_t = np.concatenate([self.sr0_t, sr0_t])
            self.theta_t = np.concatenate([self.theta_t, theta_t])
            self.epochs = np.append(self.epochs, t + 1)


    @abstractmethod
    def gradient(self, **kwargs):
        '''
        Computes the gradient of the value function V(s) with respect to the policy parameters theta.
        '''
        pass


    @abstractmethod
    def update(self, gradient, alpha, **kwargs):
        '''
        Updates the policy parameters theta
        '''
        pass
