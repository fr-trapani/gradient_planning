import numpy as np

# Agent Functions:

def soften_softmax_policy(agent, p_min=0.1, p_max=1):
    """
    Softens the policy by ensuring p(a|s) >= epsilon for each s, a
    """
    agent.theta = soften_softmax(theta=agent.theta, p_min=p_min, p_max=p_max, A=agent.A)


def policy_history(agent):
    """
    Load all policies from training
    """
    policies = np.ndarray(agent.theta_t.shape)
    for t in range(policies.shape[0]):
        agent.load_history_policy(t)
        policies[t] = agent.policy_vec()
    agent.load_history_policy()
    return policies


def policy_divergence(agent, p0=None, derivative=False, normalize=False):
    """
    Computes the Kullback–Leibler divergence between the current policy and the initial policy
    """
    policies = policy_history(agent)
    divergences = kl_time_divergence(policies, p0)

    if derivative:
        divergences[1:, :] = divergences[1:, :] - divergences[:-1, :]  
        divergences[0, :] = 0

    if normalize:
        scale = np.max(np.abs(divergences), axis=0)
        divergences[:, scale < 0.001] = 0 
        divergences = divergences / scale
        divergences[np.isnan(divergences)] = 0

    return divergences


def counter_difference(agent, derivative=False, normalize=False, only_positive=False):
    """
    Computes the difference of the expected permanence in states across different policies
    """      
    occupancies = agent.sr0_t
    counter_differences =  occupancies - occupancies[0]

    if only_positive:
        counter_differences[counter_differences < 0] = 0
    
    if derivative:
        counter_differences[1:, :] = counter_differences[1:, :] - counter_differences[:-1, :]  
        counter_differences[0, :] = 0

    if normalize:
        scale = np.max(np.abs(counter_differences), axis=0)
        counter_differences[:, scale < 0.001] = 0 
        counter_differences = counter_differences / scale
        counter_differences[np.isnan(counter_differences)] = 0

    return counter_differences


def action_counter_difference(agent, derivative=False, normalize=False, only_positive=False):
    """
    Computes the difference of the expected permanence in states across different policies
    """      
    policies = policy_history(agent)
    occupancies = agent.sr0_t
    action_occupancies = np.einsum("ts, tsa -> tsa", occupancies, policies)
    counter_differences =  action_occupancies - action_occupancies[0]

    if only_positive:
        counter_differences[counter_differences < 0] = 0
    
    if derivative:
        counter_differences = counter_differences[1:, :, :] - counter_differences[:-1, :, :] 

    if normalize:
        scale = np.max(np.abs(counter_differences), axis=0)
        counter_differences[:, scale < 0.001] = 0 
        counter_differences = counter_differences / scale
        counter_differences[np.isnan(counter_differences)] = 0

    return counter_differences


def theta_gradients(agent, normalize=False):

   # Load all policies from training
    gradients = np.ndarray(agent.theta_t.shape)
    for t in range(gradients.shape[0]):
        agent.load_history_policy(t)
        gradients[t] = agent.gradient()

    # Put back most recent policy
    agent.load_history_policy()

    if normalize:
        scale = np.max(np.abs(gradients), axis=0)
        gradients[:, scale < 0.001] = 0 
        gradients = gradients / scale
        gradients[np.isnan(gradients)] = 0

    return gradients


def rank_states(agent, f=policy_divergence, value_min=0, n_histo_bins=5):
    """
    Computes the state ranks given the agent's policy
    """
    score = f(agent)
    dt_score = score[1:, :] - score[:-1, :]
    values = np.max(dt_score, axis=0) / np.max(dt_score)
    times = np.argmax(dt_score, axis=0)
    time_max = len(score)

    values[agent.env.states_terminal] = 0
    values[np.isnan(values)] = 0

    order = np.argsort(times)
    ranks = np.argsort(order)

    mask = values >= value_min
    values[np.logical_not(mask)] = 0
    times[np.logical_not(mask)] = np.max(times[mask]) + 1
    ranks[np.logical_not(mask)] = np.max(ranks[mask])

    hist = times
    hist[~mask] = np.mean(hist[mask])
    hist = hist - np.min(hist)
    hist = hist / np.max(hist) * (n_histo_bins-1) + 0.5
    hist = hist.astype(int)

    return values, ranks, times, hist, mask, time_max


#  Helper Functions:

def soften_softmax(theta, p_min=0, p_max=1, A=None):
    
    if A is None:
        A = np.full(theta.shape, True, dtype=bool)
    theta_clamped = np.full(theta.shape, -np.inf)

    for s in range(theta_clamped.shape[0]):
        aa = np.argwhere(A[s, :])
        if aa.size == 0:
            continue
        thetas = theta[s, aa]
        p_acts = np.exp(thetas) / np.sum(np.exp(thetas))

        # first clamp policies that are too small
        idx2clamp = p_acts < p_min
        idx2scale = p_acts >= p_min
        
        thetas[idx2clamp] = np.log(p_min)
        dp = 1 - np.sum(idx2clamp * p_min)
        thetas[idx2scale] = np.log(np.exp(thetas[idx2scale]) / np.sum(np.exp(thetas[idx2scale])) * dp)

        # then clamp policies that are too big
        idx2clamp = p_acts > p_max
        idx2scale = p_acts <= p_min
        
        thetas[idx2clamp] = np.log(p_max)
        dp = 1 - np.sum(idx2clamp * p_max)
        thetas[idx2scale] = np.log(np.exp(thetas[idx2scale]) / np.sum(np.exp(thetas[idx2scale])) * dp)

        theta_clamped[s, aa] = thetas

    return theta_clamped


def kl_time_divergence(ps, p0):
    if p0 is None:
        p0 = ps[0]
    pds = np.zeros(ps.shape[:-1])
    for i in range(ps.shape[0]):
        pds[i] = kl_divergence(ps[i], p0)
    return pds


def kl_divergence(p1, p2): 
    log_prob = np.zeros(p1.shape)
    log_prob[p2 > 0] = np.log(p1[p2 > 0]/p2[p2 > 0])
    kl_div = np.sum(p1 * log_prob, axis=-1)
    return kl_div
