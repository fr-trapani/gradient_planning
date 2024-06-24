import numpy as np
from agents.pgp.pgp import PGP
from utils.policy_functions import *


class SoftMaxPGP(PGP):
    '''
    GPP implementation that uses soft-max to normalize the policy parameter theta
    '''
    def __init__(self, env, gamma=0.99, p0_func=p0_onehot, theta0_func=theta0_uniform, name="pgp_softmax", **kwargs):
        super().__init__(env, gamma, p0_func, theta0_func, name, **kwargs)

    
    def policy_vec(self):
        """
        Returns the probability vector p(A|s)
        """
        return  np.exp(self.theta) / np.sum(np.exp(self.theta), axis=1, keepdims=True)


    def FIM(self, two_dims=True):
        '''
            Returns the Fisher Information matrix FIM_i,j,k,l with i,k ∈ S, j,l ∈ A
        '''
        π = self.policy_vec()
        CC_aa = self.CorrCount_AA
        CC_as = self.CorrCount_AS
        CC_ss = self.CorrCount_SS
        
        I_aa = CC_aa
        I_as = np.einsum("ijk, kl -> ijkl", CC_as, π, optimize=True)
        I_sa = np.einsum("kli, ij -> ijkl", CC_as, π, optimize=True)
        I_ss = np.einsum("ik, ij, kl -> ijkl", CC_ss, π, π, optimize=True)
        FIM = I_aa - I_sa - I_as + I_ss

        if two_dims:
            nsa = self.n_state * self.n_action
            FIM = FIM.reshape([nsa, nsa])
            FIM = FIM[self.A.reshape(nsa), :]
            FIM = FIM[:, self.A.reshape(nsa)]

        return FIM
    

    def gradient(self, natural=False, normalize=True, **kwargs):
        '''
        Computes the gradient of the value function V(s) with respect to the policy parameters theta.
        '''
        π = self.policy_vec()
        g_π = self.policy_gradient(**kwargs)
        
        # ∇_θ_kl E[V(o)]    =  Σ_j ∇_π_kj E[V(o)] ⋅ ∇_θ_kl π_kj
        #                   =  Σ_j g_j ⋅ π_kj ⋅ ( δ_jl - π_kl )
        #                   =  π_kl ⋅ ( g_l - Σ_j π_kj ⋅ g_j )
        g_θ = (g_π - np.sum(g_π * π, axis=1,keepdims=True)) * π

        if natural:
            f = np.linalg.inv(self.FIM())
            ng_θ = f @ g_θ[self.A]
            g_θ[self.A] = ng_θ
           
        if normalize:
            g_norm = np.linalg.norm(g_θ)
            if (g_norm > 0):
                g_θ = g_θ/g_norm 

        return g_θ


    def update(self, gradient, alpha, **kwargs):
        '''
        Updates the policy parameters theta
        '''
        self.theta[self.A] += (alpha * gradient)[self.A]
        # self.theta -= np.log(np.sum(np.exp(self.theta), axis=1, keepdims=True))