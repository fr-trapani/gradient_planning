import numpy as np
import numpy.matlib as mb

# alpha functions
def situational_alpha(agent, p0=None, alpha_norm=None, alpha_mean=0.1, situational=0.5, normalization="L2"):
    """
    Implementation of situational step-size function for GPP training.
    The step-size tensor alpha is computed at each agent training step,
    depending on the current state and successor matrix.
    """
    # prior of initial state s0 p(s0)
    if p0 is None:
        p0 = agent.p0

    # choose normalization function
    if normalization == "L2":
        norm_func = l2_normalization
    else:
        norm_func = l1_normalization

    # compute situational alpha based on the SR matrix and the initial position    
    alpha_situational = mb.repmat(p0 @ agent.SR, agent.n_action, 1).T
    alpha_situational = norm_func(alpha_situational, 1)

    # compute flat alpha, constant for each time-step and state-action pair
    alpha_flat = np.ones([agent.n_state, agent.n_action])
    alpha_flat = norm_func(alpha_flat, 1)

    # blending of the two
    alpha = alpha_situational * situational + alpha_flat * (1 - situational)

    # normalization
    if alpha_norm is None:
            alpha_mean_matrix = np.full([agent.n_state, agent.n_action], alpha_mean)
            alpha_norm = np.sqrt(np.sum(np.power(alpha_mean_matrix, 2)))
    alpha = norm_func(alpha, alpha_norm)
    
    return alpha


# θ(0) functions
def theta0_uniform(agent):
    return np.log(agent.A / np.sum(agent.A, axis=1, keepdims=True))

def theta0_biased(agent, biased_action=0, bias_mult=8):  
    π = agent.A.astype(float)
    π[:, biased_action] *= bias_mult
    π /= np.sum(π, axis=1, keepdims=True)
    θ = np.log(π)
    return θ

# p(s0) functions
def p0_situational(agent, s0s=None, p0s=None, situational=0.9):
    """
    Computes a custom distribution of state probabilities combining uniform and SR-based distributions
    """
    p0_local = p0_SR(agent, s0s, p0s)
    p0_global = p0_uniform(agent)
    p0 = p0_local*situational + p0_global*(1 - situational)
    return p0

def p0_uniform(agent, s0s=None, p0s=None):
    return np.full(agent.n_state, 1/agent.n_state)

def p0_SR(agent, s0s=None, p0s=None):
    t0 = p0_onehot(agent, s0s, p0s) @ agent.SR
    p0 = t0 / np.sum(t0)
    return p0


def p0_SR2(agent, s0s=None, p0s=None):
    t0 = (p0_onehot(agent, s0s, p0s) @ agent.SR)**2
    p0 = t0 / np.sum(t0)
    return p0

def p0_SRexp(agent, s0s=None, p0s=None):
    t0 = np.exp(p0_onehot(agent, s0s, p0s) @ agent.SR)
    p0 = t0 / np.sum(t0)
    return p0

def p0_onehot(agent, s0s=None, p0s=None):
    if s0s is None:
        s0s = agent.env.states_start
    if p0s is None:
        p0s = 1
    p0 = index2vector(agent.n_state, s0s, p0s)
    p0 = p0 / np.sum(p0)
    return p0

# helper functions
def l2_normalization(v, norm=1):
    v = v / np.sqrt(np.sum(np.power(v, 2))) * norm
    return v

def l1_normalization(v, norm=1):
    v = v / np.sum(v) * norm
    return v

def index2vector(n, idxs, values=1):
    v = np.zeros(n)
    v[idxs] = 1
    return v