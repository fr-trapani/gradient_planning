import numpy as np
from matplotlib import pyplot as plt
from utils.policy_functions import *
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from utils.policy_tools import kl_divergence

class GridWorldAnimator():


    def __init__(self, visualizer):
        '''
        A class to visualize animations of policies and gradients during training  
            args:
                env: GridWorld environment
                agent: the TabularAgent wandering in the environment
        '''    
        self.viz = visualizer


    def animate_training(self, func_names, func_args, ts_interval=100, fps=5):
        fig, ax = plt.subplots()
        ts_max = self.viz.agent.theta_t.shape[0] -1
        duration = (ts_max / ts_interval) / fps

        def make_frame(s):
            ax.clear()
            ts = np.floor(s*fps*ts_interval).astype(int)
            self.viz.agent.load_history_policy(ts)
            for f_name, f_args in zip(func_names, func_args):
                func = getattr(self.viz, f_name)
                func(ax=ax, **f_args)

            return mplfig_to_npimage(fig)    
        
        animation = VideoClip(make_frame, duration = duration)
        return animation


    def animate_successor_matrix(self, p0=None, axis=0, ts_interval=100, fps=5):
        func_name = "plot_successor_matrix"
        func_args = {"p0":p0, "axis":axis}
        return self.animate_training([func_name], [func_args], ts_interval=ts_interval, fps=fps)


    def animate_policy(self, ts_interval=100, fps=5):
        func_name = "plot_policy"
        func_args = {"plot_axis":False}
        return self.animate_training([func_name], [func_args], ts_interval=ts_interval, fps=fps)


    def animate_policy_sr(self, ts_interval=100, fps=5):
        func_name1 = "plot_successor_matrix"
        func_args1 = {}
        func_name2 = "plot_policy"
        func_args2 = {"plot_axis":False}
        return self.animate_training([func_name1, func_name2], [func_args1, func_args2], ts_interval=ts_interval, fps=fps)


    def animate_gradient(self, policy_gradient=False, natural=False, ts_interval=100, fps=5):
        func_name = "plot_gradient"
        func_args = {"policy_gradient":policy_gradient, "plot_axis":False, "grad_kwargs":{"natural":natural}}
        return self.animate_training([func_name], [func_args], ts_interval=ts_interval, fps=fps)


    def animate_gradient_norm(self, policy_gradient=False, normalize=True, natural=False, ts_interval=100, fps=5):
        func_name1 = "plot_gradient_norm"
        func_args1 = {"policy_gradient":policy_gradient, "plot_axis":False, "grad_kwargs":{"natural":natural, "normalize":normalize}}
        func_name2 = "plot_maze"
        func_args2 = {"plot_axis":False}
        return self.animate_training([func_name1, func_name2], [func_args1, func_args2], ts_interval=ts_interval, fps=fps)


    def animate_gradient_with_norm(self, policy_gradient=False, natural=False, ts_interval=100, fps=5):
        func_name1 = "plot_gradient_norm"
        func_name2 = "plot_gradient"
        func_args = {"policy_gradient":policy_gradient, "plot_axis":False, "grad_kwargs":{"natural":natural}}
        return self.animate_training([func_name1, func_name2], [func_args, func_args], ts_interval=ts_interval, fps=fps)


    def animate_policy_divergence(self, ts_interval=100, fps=5, **kwargs):
        fig, ax = plt.subplots()
        ts_max = self.viz.agent.theta_t.shape[0] -1
        duration = (ts_max / ts_interval) / fps

        self.viz.agent.load_history_policy(0)
        p0 = self.viz.agent.policy_vec()
        
        def make_frame(s):
            ax.clear()

            ts = np.floor(s*fps*ts_interval).astype(int)    
            self.viz.agent.load_history_policy(ts)
            p = self.viz.agent.policy_vec()
            kld = kl_divergence(p, p0)

            alpha_vec = kld
            alpha_vec /= np.max(alpha_vec)
            color_vec = np.full(alpha_vec.shape, (ts+1)/ts_max)
            
            self.viz.plot_alpha_grid(color_vec, alpha_vec, ax=ax, alpha_normalize=False, hmin=0, hmax=1, vmin=0, vmax=1, **kwargs);
            self.viz.plot_maze(ax=ax, plot_grid=False, plot_axis=False, neg_rew_cmap="Greys")

            return mplfig_to_npimage(fig)    

        self.viz.agent.load_history_policy()
        animation = VideoClip(make_frame, duration = duration)
        ax.clear()
        return animation
    

    def animate_gradient_colored(self, ts_interval=100, fps=5):
        fig, ax = plt.subplots()
        ts_max = self.viz.agent.theta_t.shape[0] -1
        duration = (ts_max / ts_interval) / fps

        def make_frame(s):
            ax.clear()

            ts = np.floor(s*fps*ts_interval).astype(int)
            self.viz.agent.load_history_policy(ts)
            g = self.viz.agent.gradient()
            alpha_vec = np.sqrt((g**2).sum(axis=1))
            alpha_vec /= np.max(alpha_vec)
            color_vec = np.full(alpha_vec.shape, (ts+1)/ts_max)
            
            self.viz.plot_alpha_grid(color_vec, alpha_vec, hmin=0, vmin=0, hmax=1, vmax=1, alpha_normalize=False, ax=ax, plot_axis=False);
            self.viz.plot_maze(ax=ax, plot_grid=False, plot_axis=False)

            return mplfig_to_npimage(fig)    
        
        self.viz.agent.load_history_policy()
        animation = VideoClip(make_frame, duration = duration)
        ax.clear()
        return animation
    