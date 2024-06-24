import numpy as np
from matplotlib import pyplot as plt
from utils.policy_tools import *
from sklearn.decomposition import PCA

from agents.basetabular import BaseTabular


class AgentPlotter:

    def __init__(self, agents, agent_colors=None):
        """
        A class to plot useful information about agents and their policies
        Args:
            agent: a list of agents
        """
        self.set_agents(agents, agent_colors)

    
    def set_agents(self, agents, agent_colors=None):
        self.agents = agents if hasattr(agents, '__iter__') else [agents]
        self.agent_colors = agent_colors if (agent_colors is not None) else [plt.cm.get_cmap("hsv", len(self.agents) + 1)(a) for a in range(len(self.agents))]
    

    def get_agent(self, agent_id):
        if isinstance(agent_id, BaseTabular):
            agent = agent_id
        elif isinstance(agent_id, int):
            agent = self.agents[agent_id]
        elif isinstance(agent_id, str):
            agent = self.agents[ [agent.name == agent_id for agent in self.agents] ]
        else:
            agent=self.agents[0]
        return agent


    def set_states(self, states, state_labels=None, state_colors=None, state_alphas=None):
        self.states = states
        self.state_labels = state_labels if (state_labels is not None) else ["state {}".format(i) for i in range(len(states))]
        self.state_colors = state_colors if (state_colors is not None) else [plt.cm.get_cmap("hsv", len(states) + 1)(s) for s in range(len(states))]
        self.state_alphas = state_alphas if (state_alphas is not None) else np.ones(len(states))*0.4


    def set_state_actions(self, state_actions, state_action_labels=None, state_action_colors=None, state_action_alphas=None):
        self.state_actions = state_actions
        self.state_action_labels = state_action_labels if (state_action_labels is not None) else ["action {}".format(i) for i in range(len(state_actions))]
        self.state_action_colors = state_action_colors if (state_action_colors is not None) else [plt.cm.get_cmap("hsv", len(state_actions) + 1)(s) for s in range(len(state_actions))]
        self.state_action_alphas = state_action_alphas if (state_action_alphas is not None) else np.ones(len(state_actions))


    def set_world(self, world):
        self.set_states(world.state_sets, state_labels=world.state_set_labels, state_colors=world.state_set_colors)


    def plotValue(self, ax=None, derivative=False, normalize=False):
        """
        Plots how the value estimates V(s0) change across training steps
        """
                
        if ax is None:
            ax = plt.gca()

        for i, agent in enumerate(self.agents):

            value = agent.v_t
            if derivative:
                value = value[1:] - value[:-1]

            if normalize:
                value -= np.min(value)
                value /= np.max(value)
            
            plt.plot(value, label=agent.name, color=self.agent_colors[i])

            for t in np.cumsum(agent.epochs[:-1]):
                plt.axvline(x = t, color = 'k', linestyle = ':', label = None)

        plt.xlabel('Time steps [a.u.]');
        plt.ylabel('Value [a.u.]');
        plt.legend(loc='lower right')

        d_text = "derivative" if derivative else ""
        plt.title("{} V_s0(t)".format(d_text))


    def plotKLDivergence(self, agent_id=0, ax=None, plot_legend=False, state_idxs=None, **kwargs):
        """
        Plots the KL divergence between the current and the initial policies
        """

        a = self.get_agent(agent_id)

        if ax is None:
            ax = plt.gca()

        if state_idxs is None:
            state_idxs = np.arange(a.n_state)

        divergence = policy_divergence(a, **kwargs)

        for i, state_set in enumerate(self.states):
            state_set = np.intersect1d(state_set, state_idxs)
            plt.plot(divergence[:, state_set], label=self.state_labels[i], color=self.state_colors[i], alpha=self.state_alphas[i], linewidth=1)

        for t in np.cumsum(a.epochs[:-1]):
            plt.axvline(x = t, color = 'k', linestyle = ':', label = None)

        plt.xlim([0, len(divergence)])
        plt.xlabel('Time steps [a.u.]');
        plt.ylabel('Policy Divergence KL(π, π0) [a.u.]')
        if plot_legend:
            self.plot_legend()
        plt.title("KL(s, t) for agent {}".format(a.name))
        ax.set_box_aspect(1)


    def plotCounterDifference(self, agent_id=0, state_actions=False, ax=None, state_idxs=None, plot_legend=False, **kwargs):
        """
        Plots the diverence of state counters between the current and initial policies
        """    
        a = self.get_agent(agent_id)

        if ax is None:
            ax = plt.gca()

        if state_idxs is None:
            state_idxs = np.arange(a.n_state)

        if state_actions:
            cd = action_counter_difference(a, **kwargs)
            for i, (state_set, action_set) in enumerate(self.state_actions):
                for state, action in zip(state_set, action_set):
                    if state in state_idxs:
                        plt.plot(cd[:, state, action], label=self.state_action_labels[i], color=self.state_action_colors[i], alpha=self.state_action_alphas[i], linewidth=1)
        else:
            cd = counter_difference(a, **kwargs)
        for i, state_set in enumerate(self.states):
            state_set = np.intersect1d(state_set, state_idxs)
            plt.plot(cd[:, state_set], label=self.state_labels[i], color=self.state_colors[i], alpha=self.state_alphas[i], linewidth=1)

        for t in np.cumsum(a.epochs[:-1]):
            plt.axvline(x = t, color = 'k', linestyle = ':', label = None)

        plt.xlabel('Time steps [a.u.]');
        plt.ylabel('Actoin Counter Difference [a.u.]');
        if plot_legend:
            self.plot_legend()

        plt.title("ACD(s, t) for agent {}".format(a.name))
        ax.set_box_aspect(1)


    def plotThetaGradients(self, agent_id=0, ax=None, state_idxs=None, plot_legend=True, **kwargs):
        a = self.get_agent(agent_id)

        if ax is None:
            ax = plt.gca()

        if state_idxs is None:
            state_idxs = np.arange(a.n_state)

        grads = theta_gradients(a, **kwargs)

        for i, (state_set, action_set) in enumerate(self.state_actions):
            for state, action in zip(state_set, action_set):
                if state in state_idxs:
                    plt.plot(grads[:, state, action], label=self.state_action_labels[i], color=self.state_action_colors[i], alpha=self.state_action_alphas[i], linewidth=1)

        for t in np.cumsum(a.epochs[:-1]):
            plt.axvline(x = t, color = 'k', linestyle = ':', label = None)

        plt.xlim([0, len(grads)])
        plt.xlabel('Time steps [a.u.]');
        plt.ylabel('∇θ E[V] [a.u.]')
        if plot_legend:
            self.plot_legend()
            
        plt.title("∇θ E[V] [a.u.] for agent {}".format(a.name))
        ax.set_box_aspect(1)


    def plotStateRanks(self, agent_id=None, f=policy_divergence, ax=None, plot_legend=True, **kwargs):
        """
        Plots the KL divergence between the current and the initial policies
        """

        a = self.get_agent(agent_id)
        if ax is None:
            ax = plt.gca()

        values, ranks, times, hist, mask, time_max = rank_states(a, f=f)

        order = np.argsort(times)
        plt.plot(times[order], ranks[order], color="black", label=None)

        if self.states is not None:
            for i_s, ss in enumerate(self.states):
                for s in ss:
                    x = times[s]
                    y = ranks[s]
                    plt.scatter(x, y, color=self.state_colors[i_s], label=self.state_labels[i_s])
        
        plt.xlim([-time_max/10, time_max + time_max/10])
        plt.xlabel("Time-Steps [a.u.]")
        plt.ylabel("State Rank [a.u.]")
        if plot_legend:
            self.plot_legend()
        plt.title("State-optimization ranking for agent {}".format(a.name))
        ax.set_box_aspect(1)      


    def plotKLCDEmbedding(self, agent_id=None, ax=None, state_idxs=None, plot_legend=False, cd_args={}, kl_args={}, **kwargs):
        a = self.get_agent(agent_id)

        if ax is None:
            ax = plt.gca()

        if state_idxs is None:
            state_idxs = np.arange(a.n_state)

        divergence = policy_divergence(a, **kl_args)
        cd = counter_difference(a, **cd_args)

        for i, state_set in enumerate(self.states):
            state_set = np.intersect1d(state_set, state_idxs)
            plt.plot(divergence[:, state_set], cd[:, state_set], label=self.state_labels[i], color=self.state_colors[i], alpha=self.state_alphas[i], linewidth=1)
            plt.scatter(divergence[-1, state_set], cd[-1, state_set], label=self.state_labels[i], color=self.state_colors[i], alpha=self.state_alphas[i], linewidth=1)
            
        plt.ylabel('Actoin Counter Difference [a.u.]');
        plt.xlabel('Policy Divergence KL(π, π0) [a.u.]')
        if plot_legend:
            self.plot_legend()
        plt.title("CD-KL embedding for agent {}".format(a.name))
        ax.set_box_aspect(1)


    def plotPCAEmbedding(self, agent_id=None, ax=None, state_idxs=None, plot_legend=False, cd_args={}, kl_args={}, **kwargs):
        a = self.get_agent(agent_id)

        if ax is None:
            ax = plt.gca()

        if state_idxs is None:
            state_idxs = np.arange(a.n_state)

        kl = policy_divergence(a, **kl_args)
        cd = counter_difference(a, **cd_args)
        features = np.concatenate([kl, cd])
        features = features - np.mean(features)
        features = features / np.std(features)

        pca_10 = PCA(n_components=10)
        pca_10.fit(features[:, state_idxs].T)

        pca_2 = PCA(n_components=2)
        pca_2.fit(features[:, state_idxs].T)
        embedding_2D = pca_2.transform(features.T)

        plt.subplot(211)
        plt.plot(np.cumsum(pca_10.explained_variance_ratio_))
        plt.scatter(np.arange(pca_10.n_components_), np.cumsum(pca_10.explained_variance_ratio_))
        plt.tight_layout()
        plt.title("KL-CD PCA variance explained")
        plt.xlabel("# components [a.u.]")
        plt.ylabel("variance explained [%]")

        plt.subplot(212)
        for i, state_set in enumerate(self.states):
            state_set = np.intersect1d(state_set, state_idxs)
            plt.scatter(embedding_2D[state_set, 0], embedding_2D[state_set, 1], label=self.state_labels[i], color=self.state_colors[i], alpha=self.state_alphas[i], linewidth=1)
        plt.xlabel('PCA #1  [a.u.]');
        plt.ylabel('PCA #2  [a.u.]')
        if plot_legend:
            self.plot_legend()
        plt.title("KL-CD 2D PCA embedding")


    def plot_legend(self):

        handles, labels = plt.gca().get_legend_handles_labels()
        newLabels, newHandles = [], []
        for handle, label in zip(handles, labels):
            if label not in newLabels:
                newLabels.append(label)
                newHandles.append(handle)
        plt.legend(newHandles, newLabels, loc='lower right')
