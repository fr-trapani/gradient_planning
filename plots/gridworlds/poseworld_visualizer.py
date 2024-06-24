from plots.gridworlds.gridworld_visualizer import GridWorldVisualizer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import hsv_to_rgb
import copy


class PoseWorldVisualizer(GridWorldVisualizer):

    @property
    def grid(self):
        return np.any(self.env.valid_position_vec(), axis=0)

    def get_grid_position(self, state):
        return self.env.decode(state)[1:]
    
    def get_grid_orientation(self, state, rads=False):
        o = self.env.decode(state)[0]
        if rads:
            o = o/self.env.no *2*np.pi
        return o
    

    def __init__(self, 
                 environment, 
                 agent=None):
        '''
        A class to visualize PoseWorld MDPs, agents, and the corresponding policies and policy updates  
            args:
                env:    PoseWorld
                agent:  TabularAgent
        '''    
        super().__init__(environment=environment, agent=agent)


    def plot_states(self, 
                    states, 
                    colors=None, 
                    ax=None, 
                    full_color=True,
                    **kwargs):
        """
        Draws circles representing a set of states within the gridworld
        """  
        if ax is None:
            ax = plt.gca()

        if colors is None:
            colors = [plt.cm.get_cmap("hsv", len(states) + 1)(s) for s in range(len(states))]
    
        v1 = [-0.5, -0.5]
        v2 = [-0.5, +0.5]
        v3 = [0, 0]

        for state, color in zip(states, colors):
            
            c= self.get_grid_position(state)
            θ = self.get_grid_orientation(state, rads=True)

            R = np.array([(np.cos(θ), -np.sin(θ)), (np.sin(θ), np.cos(θ))])
            p1 = c + R @ v1
            p2 = c + R @ v2
            p3 = c + R @ v3

            if full_color:
                f_color = color
                e_color = "none"
            else:
                f_color = "none"
                e_color = color
                    
            vertices = np.array([p1, p2, p3])
            shape = Polygon(vertices, facecolor=f_color, edgecolor=e_color, alpha=1, linewidth=2, zorder=2)  
            ax.add_patch(shape)


    def plot_actions(self,
                     states=None,
                     actions=None, 
                     colors=None,
                     width_func=None, 
                     ax=None,
                     **kwargs):
        # raise Warning("actions cannot be plotted for pose worlds")
        pass


    def plot_policy(self, 
                    vec=None,
                    agent=None,
                    factor = 1, 
                    ax=None, 
                    cs=['green', 'red', 'black'], 
                    normalize=True, 
                    plot_maze=True, 
                    plot_grid=True, 
                    plot_axis=True,
                    action='forward'):
        """
        Plots the tabular policy or the policy gradient of the agent
        """
        if ax is None:
            ax = plt.gca()

        if agent is None:
            agent = self.agent

        # Plot the environment    
        if plot_maze:
            self.plot_maze(ax=ax, plot_grid=plot_grid, plot_axis=plot_axis)
        else:
            self._plot(ax, plot_grid=plot_grid, plot_axis=plot_axis)

        # get the vector to plot (the policy or its gradient)
        if vec is None:
            vec = agent.policy_vec()
                                    
        # In poseworld we only visualize one action at a time
        a = self.env.actions.get_action_index(action)
        Δl0 = np.array([1, 0], dtype=int)
        
        # plot the vector
        for s in self.env.states_transient:

            l1 = self.get_grid_position(s)
            θ = self.get_grid_orientation(s, rads=True)
            R = np.array([(np.cos(θ), -np.sin(θ)), (np.sin(θ), np.cos(θ))])
            Δl = R @ Δl0

            p = vec[s, a]
            if normalize:
                p = p / np.sum(vec[s])
                
            l2 = l1 + Δl*abs(p)*factor*0.4
            c = cs[2] if np.all(vec >= 0) else cs[1] if (p < 0) else cs[0]
            ax.plot([l1[0], l2[0]], [l1[1], l2[1]], c=c, linewidth=1.5)
        return ax


    def plot_gradient_norm(self, 
                           ax=None, 
                           agent=None,
                           grad=None, 
                           grad_kwargs={},
                           merge_func = np.max, 
                           policy_gradient=False, 
                           cmap='Blues', 
                           vmax=None, 
                           colorbar=False, 
                           plot_axis=True):
        """
        Visualizes the norm of the policy gradient as a grid image
        """
        if agent is None:
            agent = self.agent

        if ax is None:
            ax = plt.gca()
            
        if grad is None:
            grad = agent.policy_gradient(**grad_kwargs) if policy_gradient else agent.gradient(**grad_kwargs)

        g_norm = np.sqrt((grad**2).sum(axis=1))  
        g_matrix = np.reshape(g_norm, [self.env.no, -1])
        g_merged = merge_func(g_matrix, axis=0)     
        self.plot_grid(g_merged, ax=ax, cmap=cmap, vmax=vmax, colorbar=colorbar, plot_axis=plot_axis);


    def plot_successor_matrix(self, 
                              agent=None,
                              p0=None, 
                              axis=0, 
                              ax=None, 
                              merge_func = np.max, 
                              cmap='Blues', 
                              vmax=None, 
                              colorbar=False, 
                              plot_axis=True):
        """
        Visualizes a row/columns of the successor matrix
        """
        if agent is None:
            agent = self.agent

        if p0 is None:
            p0 = agent.p0
            
        if axis == 0:
            S = p0 @ agent.SR
            S[self.env.states_terminal] = 0
        else:
            S = agent.SR @ p0

        S_matrix = np.reshape(S, [self.env.no, -1])
        S_merged = merge_func(S_matrix, axis=0)   

        self.plot_grid(S_merged, ax=ax, cmap=cmap, vmax=vmax, colorbar=colorbar, plot_axis=plot_axis);

