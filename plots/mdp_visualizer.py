from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx 
from utils.policy_functions import *


def draw_world(
            agent,
            state_weights=None,
            transition_weights=None,
            state_sets=[], 
            state_set_colors=[], 
            state_action_sets=[], 
            state_action_set_colors=[], 
            show_only_edge_sets=False,
            max_alpha_node=1,
            min_alpha_node=0.05,
            max_alpha_edge=1,
            min_alpha_edge=0.025,
            node_color_default="grey",
            edge_color_default="black",
            node_size=100,
            edge_size=2,
            arrow_size=10, 
            ax=None,
            **kwargs):
    
    if ax is None:
        ax = plt.gca()
        
    if len(state_sets) == 0 and hasattr(agent.env, 'state_sets') and hasattr(agent.env, 'state_set_colors'):
        state_sets = agent.env.state_sets
        state_set_colors = agent.env.state_set_colors

    P = agent.env.P
    T = agent.T
    T[agent.env.states_terminal, :] = 0
    
    # Build Graph
    S = np.arange(T.shape[0])
    E = np.argwhere(T)
    G = nx.DiGraph()
    G.add_nodes_from(S)
    G.add_edges_from(E)

    # Graph Weights
    if state_weights is None:
        state_weights = np.ones(len(S))
    if transition_weights is None:
        transition_weights = np.ones(len(E))

    # Graph topology
    node_pos = nx.kamada_kawai_layout(G)
    node_alpha = state_weights * (max_alpha_node - min_alpha_node) + min_alpha_node
    edge_alpha = transition_weights  * (max_alpha_edge - min_alpha_edge) + min_alpha_edge

    # Color Nodes
    node_color = np.array([node_color_default for _ in S], dtype=object)
    for ss, c in zip(state_sets, state_set_colors):
        for s in ss:
            node_color[s] = c

    # Color Edges
    edge_color = [edge_color_default for _ in E]
    edges_colored = np.full(len(E), False, dtype=bool)
    for sas, c in zip(state_action_sets, state_action_set_colors):
        for s, a in sas:
            s2s = np.argwhere(P[s, a]).flatten()
            for s2 in s2s:
                for i, e in enumerate(E):
                    if e.tolist() == [s, s2]:
                        edge_color[i] = c
                        edges_colored[i] = True

    if show_only_edge_sets:
        edge_alpha[~edges_colored] = 0

    # Node sets
    start_nodes = agent.env.states_start
    terminal_nodes = agent.env.states_terminal
    transient_nodes = np.setdiff1d(agent.env.states_transient, agent.env.states_start)

    nx.draw_networkx_nodes(G, node_pos, node_shape='s', nodelist=start_nodes, alpha=node_alpha[start_nodes], node_color=node_color[start_nodes], ax=ax, node_size=node_size)
    nx.draw_networkx_nodes(G, node_pos, node_shape='*', nodelist=terminal_nodes, alpha=node_alpha[terminal_nodes], node_color=node_color[terminal_nodes], ax=ax, node_size=node_size)
    nx.draw_networkx_nodes(G, node_pos, nodelist=transient_nodes, alpha=node_alpha[transient_nodes], node_color=node_color[transient_nodes], ax=ax, node_size=node_size)
    nx.draw_networkx_edges(G, node_pos, arrows=True, arrowstyle="->", arrowsize=arrow_size, edge_color=edge_color, width=edge_size, alpha=edge_alpha, ax=ax)
    plt.axis("off")


def draw_gradients(agent, policy_gradient=False, **kwargs):
    # gradient
    g = agent.policy_gradient(**kwargs) if policy_gradient else agent.gradient(**kwargs)
    g /= np.max(g)
    g_norm = np.sqrt((g**2).sum(axis=1))

    # gradient projection in state space
    v = np.einsum("sa, saz -> sz", g, agent.env.P)

    # weights
    T = agent.T
    T[agent.env.states_terminal, :] = 0
    signed_weights = v[T>0] / np.max(np.abs(v[T>0]))
    transition_weights = np.abs(signed_weights)
    state_weights = g_norm
    state_weights /= np.max(state_weights)

    # state colors 
    state_action_sets = [np.argwhere(g>=0)]
    state_action_colors = ["green"]
    draw_world(agent, state_weights, transition_weights, state_action_sets=state_action_sets, state_action_set_colors=state_action_colors, show_only_edge_sets=True, **kwargs)


def draw_policy(agent, **kwargs):
    T = agent.T
    T[agent.env.states_terminal, :] = 0
    transition_weights = (T[T>0]) / np.max(T[T>0])
    state_weights = (agent.p0 @ agent.SR)

    state_weights[agent.env.states_terminal] = 0
    state_weights[agent.env.states_terminal] = np.max(state_weights)
    
    state_weights /= np.max(state_weights)
    draw_world(agent, state_weights, transition_weights, **kwargs)


def mdp_animation(agent, func, func_args, ts_interval=100, fps=5):
    fig, ax = plt.subplots()
    ts_max = agent.theta_t.shape[0] -1
    duration = (ts_max / ts_interval) / fps

    def make_frame(s):
        ax.clear()
        ts = np.floor(s*fps*ts_interval).astype(int)
        agent.load_history_policy(ts)
        func(agent, ax=ax, **func_args)
        return mplfig_to_npimage(fig)    
    
    animation = VideoClip(make_frame, duration = duration)
    return animation