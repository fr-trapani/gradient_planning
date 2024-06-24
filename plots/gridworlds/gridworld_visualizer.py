import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import hsv_to_rgb
import copy


class GridWorldVisualizer():

    @property
    def nx(self):
        return self.env.nx

    @property
    def ny(self):
        return self.env.ny
    
    @property
    def grid(self):
        return self.env.valid_position_vec()

    def get_grid_position(self, state):
        return self.env.decode(state)


    def __init__(self, 
                 environment, 
                 agent=None):
        '''
        A class to visualize GridWorld MDPs, agents, and the corresponding policies and policy updates  
            args:
                env:    GridWorld
                agent:  TabularAgent
        '''    
        self.env = environment
        self.agent = agent


    def _plot(self, 
              ax=None, 
              plot_grid=True, 
              plot_axis=True):
        """
        Basic functions that plots the axes and the labels
        """
        if ax is None:
            ax = plt.gca()
                
        # Plot the Grid
        if plot_grid:
            ax.set_xticks(np.arange(self.nx + 1) - .5, minor=True)
            ax.set_yticks(np.arange(self.ny + 1) - .5, minor=True)
            ax.grid(which="minor", color="black", linewidth=0.5, alpha=0.3)
            ax.tick_params(which="minor", bottom=False, left=False)

        if plot_axis:
            # remove all ticks
            ax.set_ylabel("Y")
            ax.set_xlabel("X")
            ax.set_xticks(np.arange(self.nx))
            ax.set_yticks(np.arange(self.ny))
            ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=90, ha='right')
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        
        ax.set_aspect('equal', 'box')


    def plot_states(self, 
                    states, 
                    colors=None, 
                    shape="circle", 
                    ax=None, 
                    full_color=True):
        """
        Draws circles representing a set of states within the gridworld
        """  
        if ax is None:
            ax = plt.gca()

        if colors is None:
            colors = [plt.cm.get_cmap("hsv", len(states) + 1)(s) for s in range(len(states))]
    
        for state, color in zip(states, colors):
            
            p = self.get_grid_position(state)

            if full_color:
                f_color = color
                e_color = "none"
            else:
                f_color = "none"
                e_color = color

            match shape:
                case "circle":
                    patch = plt.Circle(p, 0.5, facecolor=f_color, edgecolor=e_color, alpha=1, linewidth=2, zorder=2)  
                    ax.add_patch(patch)

                case "square":
                    patch = Rectangle(p - 0.5, 1, 1, facecolor=f_color, edgecolor=e_color, alpha=1, linewidth=2, zorder=2)  
                    ax.add_patch(patch)

                case _:
                    ax.plot([p[0] - 0.5, p[0] + 0.5], [p[1] - 0.5, p[1] + 0.5], c=color, linewidth=3)
                    ax.plot([p[0] - 0.5, p[0] + 0.5], [p[1] + 0.5, p[1] - 0.5], c=color, linewidth=3)


    def plot_actions(self, 
                     states,
                     actions, 
                     colors=None,
                     width_func=None, 
                     ax=None,
                     **kwargs):
        """
        Draws colored arrows representing a set of state-action pairs within the gridworld 
        """  
        if ax is None:
            ax = plt.gca()

        if colors is None:
            colors = [plt.cm.get_cmap("hsv", len(actions) + 1)(s) for s in range(len(actions))]
    
        for s, a, c in zip(states, actions, colors):

            l1 = self.get_grid_position(s)
            l2s, p2s = self.env.grid_transition(l1, a)
            el2s = np.average(l2s, axis=0, weights=p2s)
            displacement = (el2s - l1)

            if width_func is None:
                # if width is none, plot actions as arrows
                l2 = l1 + displacement * 0.5
                l3a = l1 + displacement * 0.4 + np.flip(displacement)*0.2
                l3b = l1 + displacement * 0.4 - np.flip(displacement)*0.2

                ax.plot([l1[0], l2[0]], [l1[1], l2[1]], c=c, linewidth=2)
                ax.plot([l2[0], l3a[0]], [l2[1], l3a[1]], c=c, linewidth=2)
                ax.plot([l2[0], l3b[0]], [l2[1], l3b[1]], c=c, linewidth=2)

            else:
                # otherwise plot rectangles 
                lr = l1 + displacement*1/3
                w = width_func(displacement)
                ax.add_patch(Rectangle((lr[0] - w[0]/2, lr[1] - w[1]/2), w[0], w[1], facecolor=c, alpha=1))


    def plot_maze(self,
                  ax=None, 
                  plot_grid=True,
                  plot_axis=True, 
                  max_rew=100, 
                  pos_rew_cmap="Greens", 
                  neg_rew_cmap="Reds", 
                  walls_cmap="Greys_r"):
        """
        Functions that plots the GridWorld, showing its walls, rewarding locations and terminal locations
        """    
        if ax is None:
            ax = plt.gca()

        rew_pos_colors = plt.get_cmap(pos_rew_cmap)(np.linspace(0.2, 0.8, max_rew+1))
        rew_neg_colors = plt.get_cmap(neg_rew_cmap)(np.linspace(0.2, 0.8, max_rew+1))

        # Plot Valid States
        alpha_grid = 1 - self.grid.astype(float)
        ax.imshow(self.grid.T, cmap=walls_cmap, vmin=0, vmax=1, interpolation='none', alpha=alpha_grid.T)

        # Plot Starting Positions
        s0 = self.env.states_start
        c0 = np.full(s0.shape, 'c', dtype=str)
        self.plot_states(s0, ax=ax, full_color=True, colors=c0)

        # Plot Rewards (state based)
        sR, rR = self.env.get_state_rewards()
        cR = [rew_pos_colors[min(int(r), max_rew)] if (r >= 0) else rew_neg_colors[min(int(-r), max_rew)] for r in rR]
        self.plot_states(sR, ax=ax, shape="square", full_color=True, colors=cR)

        # Plot Rewards (state-action based)
        sRA, aRA, rRA = self.env.get_state_action_rewards()
        cRA = [rew_pos_colors[min(int(r), max_rew)] if (r >= 0) else rew_neg_colors[min(int(-r), max_rew)] for r in rRA]
        width_func_RA = lambda d : np.flip(abs(d)) * 2/3 + 1/3
        self.plot_actions(sRA, aRA, cRA, width_func=width_func_RA)

        # Plot Rewards (state-action-target based)
        sRAS, aRAS, _, rRAS = self.env.get_state_action_state_rewards()
        cRAS = [rew_pos_colors[min(int(r), max_rew)] if (r >= 0) else rew_neg_colors[min(int(-r), max_rew)] for r in rRAS]
        width_func_RAS = lambda d : np.array([1/3, 1/3])
        self.plot_actions(sRAS, aRAS, cRAS, width_func=width_func_RAS)

        # Plot Terminal
        sE = self.env.states_terminal
        cE = np.full(sE.shape, 'k', dtype=str)
        self.plot_states(sE, ax=ax, shape="square", full_color=False, colors=cE)

        self._plot(ax, plot_grid=plot_grid, plot_axis=plot_axis)
        return ax  


    def plot_grid(self, 
                  vec, 
                  ax=None, 
                  cmap='Blues', 
                  interpolation='none',
                  vmin=None, 
                  vmax=None, 
                  colorbar=False, 
                  plot_grid=True, 
                  plot_axis=True):
        """
        Visualizes a matrix with shape [nx*ny] (like a policy or gradient) as a grid image
        """
        if ax is None:
            ax = plt.gca()

        grid = np.full([self.nx, self.ny], np.nan)
        grid[self.grid] = vec.astype(float)

        # walls
        cmap = copy.copy(plt.cm.get_cmap(cmap))
        cmap.set_bad(color='white')

        im = ax.imshow(grid.T, cmap=cmap, vmin=vmin, interpolation=interpolation, vmax=vmax)
        self._plot(ax, plot_grid=plot_grid, plot_axis=plot_axis)

        if colorbar:
            plt.colorbar(im, fraction=0.02, ax=ax); 

        return im
    

    def plot_alpha_grid(self, 
                        color_vec, 
                        alpha_vec, 
                        alpha_normalize=True, 
                        alpha_threshold=0, 
                        mask=None, ax=None, 
                        cmap='cool', 
                        nan_color='black', 
                        vmin=None, 
                        vmax=None, 
                        hmin=0.1, 
                        hmax=1, 
                        colorbar=False, 
                        plot_grid=True, 
                        plot_axis=True):
        """
        Visualizes a matrix with shape [nx*ny] (like a policy or gradient) as a grid image
        """
        if ax is None:
            ax = plt.gca()
        if mask is None:
            mask = np.full([self.env.n_state], True, dtype=bool)

        cmap = copy.copy(plt.cm.get_cmap(cmap))
        cmap.set_bad(color=nan_color)

        color_mat = np.full([self.nx, self.ny], np.nan)
        alpha_mat = np.zeros([self.nx, self.ny])

        for s in range(self.env.n_state):
            if alpha_vec[s] >= alpha_threshold and mask[s]:
                color_mat[*self.get_grid_position(s)] = color_vec[s]
                alpha_mat[*self.get_grid_position(s)] = alpha_vec[s]
            else:
                color_mat[*self.get_grid_position(s)] = np.nan
                alpha_mat[*self.get_grid_position(s)] = -1

        if alpha_normalize:        
            alpha_mat = alpha_mat / np.max(alpha_mat)
        alpha_mat = alpha_mat * (hmax-hmin) + hmin
        alpha_mat[alpha_mat < hmin] = 0

        im = ax.imshow(color_mat.T, alpha=alpha_mat.T, cmap=cmap, vmin=vmin, interpolation='none', vmax=vmax)
        self._plot(ax, plot_grid=plot_grid, plot_axis=plot_axis)
        if colorbar:
            plt.colorbar(im, fraction=0.02, ax=ax); 

        return im
    
    
    def plot_grids(self, 
                   vec, 
                   fig=None, 
                   subpl=None, 
                   idxs=None, 
                   cmap='Blues', 
                   vmin=0., 
                   vmax=None, 
                   colorbar=False, 
                   plot_grid=True, 
                   plot_axis=True, 
                   title="Iteration: "):

        if idxs is None:
            idxs= np.arange(len(vec))
        vec = vec[idxs]

        if subpl is None:
            subpl = [1+len(vec)//3, 3]

        if fig is None:
            fig, ax = plt.subplots(subpl[0], subpl[1], figsize=(subpl[1]*3, subpl[0]*3))

        else:
            ax = fig.subplots(subpl[0], subpl[1])
        ax = ax.flatten()

        for i in range(len(vec)):
            self.plot_grid(vec[i], ax=ax[i], cmap=cmap, vmin=vmin, vmax=vmax, colorbar=colorbar, plot_grid=plot_grid, plot_axis=plot_axis)
            ax[i].set_title(title + f"{idxs[i]}")

        for i in range(len(vec), subpl[0]*subpl[1]):
            fig.delaxes(ax[i])

        return fig, ax


    def plot_moment(self, 
                    moment, 
                    state, 
                    action=None, 
                    ax=None, 
                    cmap='Blues', 
                    colorbar=False, 
                    plot_axis=True):
        """
        Visualizes the norm of the policy gradient as a grid image
        """

        ns = self.env.n_state
        na = self.env.n_action

        if ax is None:
            ax = plt.gca()

        if moment.shape == (ns, ns):
            vec = moment[state, :]
            vec[state] = 0
            vec[self.env.states_terminal] = 0
            vec = vec/ np.linalg.norm(vec)

            self.plot_grid(vec, ax=ax, cmap=cmap, colorbar=colorbar, plot_axis=plot_axis)
            self.plot_states([state], colors=["orange"], shape="square")

        elif moment.shape == (ns, na, ns, na):
            vec = moment[state, action, :, :]
            vec[state, action] = 0
            vec[self.env.states_terminal, :] = 0
            vec = vec/ np.linalg.norm(vec)

            self.plot_policy(vec, normalize=False, plot_maze=False, plot_grid=True, plot_axis=plot_axis)
            self.plot_actions([state], [action], colors=["orange"])

        elif moment.shape == (ns, na, ns):
            vec = moment[state, action, :]
            vec[self.env.states_terminal] = 0
            vec = vec/ np.linalg.norm(vec)

            self.plot_grid(vec, ax=ax, cmap=cmap, colorbar=colorbar, plot_axis=plot_axis)
            self.plot_actions([state], [action], colors=["orange"])

        elif moment.shape == (ns, ns, na):
            vec = moment[state, :, :]
            vec[state] = 0
            vec[self.env.states_terminal] = 0
            vec = vec/ np.linalg.norm(vec)

            self.plot_policy(vec, normalize=False, plot_maze=False, plot_grid=True, plot_axis=plot_axis)
            self.plot_states([state], colors=["orange"], shape="square")
        else:
            raise Exception("Moment shape is inconsistent") 


    def plot_trajectory(self, 
                        agent=None, 
                        ax=None, 
                        ss=None, 
                        s0=None, 
                        n_steps=1000, 
                        greedy=False, 
                        plot_maze=True, 
                        plot_grid=True, 
                        plot_axis=True, 
                        jitter=0.1):
        """
        Plots the trajectory of the Agent within the environment 
        """  

        if ax is None:
            ax = plt.gca()

        if agent is None:
            agent = self.agent

        if s0 is None:
            s0 = self.env.choose_initial_state()
        
        if ss is None: 
            ss, _, _, _ = agent.trajectory(s0=s0, n_steps=n_steps, greedy=greedy)

        pp = np.array([self.get_grid_position(s) for s in ss])
        pp = pp + np.random.normal(loc=0.0, scale=jitter, size=pp.shape)

        # Plot the environment    
        if plot_maze:
            self.plot_maze(ax=ax, plot_grid=plot_grid, plot_axis=plot_axis)
        else:
            self._plot(ax, plot_grid=plot_grid, plot_axis=plot_axis)

        for i in range(len(pp) - 1):
            ax.plot([pp[i][0],pp[i+1][0]], 
                    [pp[i][1],pp[i+1][1]], color="black", linewidth=1.5)
                
        ax.plot(pp[0][0], pp[0][1], 'k', markersize=10, marker='P')
        ax.plot(pp[len(pp) - 1][0], pp[len(pp) - 1][1], 'k', markersize=10, marker='X')

        return ax


    def plot_trajectory_distribution(self, 
                                     ax=None, 
                                     agent=None, 
                                     n_samples=1000, 
                                     s0=None, 
                                     n_steps=1000, 
                                     min_hue=0.5, 
                                     max_hue=0.9, 
                                     plot_maze=True, 
                                     plot_grid=True, 
                                     plot_axis=True):
        """
        Plots an image representing the probability distribution p(S | policy)
        Saturation of squares represent the probability of transiting through that square  ( p(S | policy) )
        Color (hue) of squares represent the median normalized timestep at which that square is reached
        """  

        if ax is None:
            ax = plt.gca()
                    
        if agent is None:
            agent = self.agent
        
        # probability of reaching state s
        p_s = np.full([self.nx, self.ny], 0., dtype=float)
        # timesteps at which state s is reached 
        t_s = [[] for _ in range(self.env.n_state)]
        
        for _ in range(n_samples):
            if s0 is None:
                s0i = self.env.choose_initial_state()
            else:
                s0i = s0

            ss, _, _, _ = agent.trajectory(s0=s0i, n_steps=n_steps, greedy=False)
            for t, s in enumerate(ss):
                p_s[*self.get_grid_position(s)] += 1/n_samples
                t_s[s].append(t)

        #  MEDIAN timestep at which state s is reached 
        mt_s =  np.full([self.nx, self.ny], 0., dtype=float)
        for s, ts in enumerate(t_s):
            if len(ts) > 0:
                mt_s[*self.get_grid_position(s)] = np.mean(ts)  #max(set(ts), key = ts.count)
        
        #  median NORMALIZED timestep at which state s is reached 
        nmt_s = mt_s / np.max(mt_s[p_s > 0.3])

        # clip to 1
        p_s[p_s > 1] = 1
        nmt_s[nmt_s > 1] = 1

        # build the image using HSV channels
        hue = nmt_s.T * (max_hue - min_hue) + min_hue
        sat = p_s.T
        val = np.full([self.nx, self.ny], 1., dtype=float).T

        timed_trajectory_distr = hsv_to_rgb(np.stack([hue, sat, val], 2))
        im = ax.imshow(timed_trajectory_distr)

        # Plot the environment    
        if plot_maze:
            self.plot_maze(ax=ax, plot_grid=plot_grid, plot_axis=plot_axis)
        else:
            self._plot(ax, plot_grid=plot_grid, plot_axis=plot_axis)
            
        return im    


    def plot_policy(self, 
                    vec=None,  
                    agent=None, 
                    factor = 1, 
                    ax=None, 
                    cs=['green', 'red', 'black'], 
                    normalize=True, 
                    plot_maze=True, 
                    plot_grid=True, 
                    plot_axis=True):
        
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
        
        # plot the vector
        for s in self.env.states_transient:

            p = vec[s]
            if normalize:
                p = p / np.max(np.abs(p))

            for a in range(self.env.n_action):
                l1 = self.get_grid_position(s)
                Δl = self.env.get_action(a)
                c = cs[2] if np.all(vec >= 0) else cs[1] if (p[a] < 0) else cs[0]

                if np.linalg.norm(Δl) == 0: # stay action
                    ax.plot(*l1, 'o', alpha=0.3, markersize= abs(p[a])*factor*10, c=c)
                else:        
                    l2 = l1 + Δl * abs(p[a]) * factor * 0.4
                    ax.plot([l1[0], l2[0]], [l1[1], l2[1]], c=c, linewidth=1.5)
        return ax
    

    def plot_gradient(self, 
                      agent=None,
                      policy_gradient=False, 
                      factor = 1.0, 
                      normalize = False, 
                      ax=None, 
                      cs=['green', 'red', 'black'], 
                      plot_maze=True, 
                      plot_grid=True, 
                      plot_axis=True,
                      **kwargs):
        """
        Plots the policy gradient of the agent
        """
        if agent is None:
            agent = self.agent
        vec = agent.policy_gradient(**kwargs) if policy_gradient else agent.gradient(**kwargs)
        vec = vec / np.max(np.abs(vec))
        self.plot_policy(vec=vec, factor=factor, normalize=normalize, ax=ax, cs=cs, plot_maze=plot_maze, plot_grid=plot_grid, plot_axis=plot_axis)


    def plot_gradient_norm(self, 
                           ax=None, 
                           agent=None, 
                           grad=None, 
                           grad_kwargs={}, 
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
        self.plot_grid(g_norm, ax=ax, cmap=cmap, vmax=vmax, colorbar=colorbar, plot_axis=plot_axis);


    def plot_successor_matrix(self, 
                              agent=None, 
                              p0=None, 
                              axis=0, 
                              ax=None, 
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

        self.plot_grid(S, ax=ax, cmap=cmap, vmax=vmax, colorbar=colorbar, plot_axis=plot_axis);