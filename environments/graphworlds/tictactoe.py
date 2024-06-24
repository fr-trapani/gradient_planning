from environments.world import World
from os import path
from functools import reduce
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np

STATES_FILE = path.join("__storage", "worlds", "tictactoe", "configs", "states.csv")
ACTIONS_FILE = path.join("__storage", "worlds", "tictactoe", "configs", "actions.csv")


class TicTacToe(World):
    """
    MDP implemementation of the Tic Tac Toe game
    """

    def __init__(self, play_first=True, opponent_policy=None, name="tictactoe", **kwargs):

        # Load tables with game structure
        states_table = np.genfromtxt(STATES_FILE, delimiter=',', dtype=None, encoding=None, names=True)
        transitions_table = np.genfromtxt(ACTIONS_FILE, delimiter=',', dtype=None, encoding=None, names=True)

        n_state_board = states_table.shape[0]
        n_action = 9

        # Create mapping between states and board configuration
        self.board_configurations = np.zeros([n_state_board, n_action], dtype=int)
        for i in range(n_action):
            mask = states_table["BoardStates_" + str(i + 1)]
            self.board_configurations[mask == 1, i] = 1
            self.board_configurations[mask == 0, i] = -1

        # Load mappings between source and target states for each valid transition
        transition_sources = transitions_table["SourceState"] - 1  # state IDs should be in range [0 n) - and not (0 n] 
        transition_targets = transitions_table["TargetState"] - 1  # state IDs should be in range [0 n) - and not (0 n]

        # The state space changes depending on whether we are playing first or second
        if play_first:
            self.player_id = "X"
            opponent_id = "O"
            self.states_player = np.argwhere(np.sum(self.board_configurations, axis=1) == 0).flatten() 
            self.states_opponent = np.argwhere(np.sum(self.board_configurations, axis=1) != 0).flatten() 
        else:
            self.player_id = "O"
            opponent_id = "X"
            self.states_player = np.argwhere(np.sum(self.board_configurations, axis=1) != 0).flatten()
            self.states_opponent = np.argwhere(np.sum(self.board_configurations, axis=1) == 0).flatten()
            
        # Identify all the important board configurations
        board_configs_start = np.argwhere(states_table["start" + self.player_id]).flatten()
        board_configs_win = np.argwhere(states_table["goal" + self.player_id]).flatten()
        board_configs_lose = np.argwhere(states_table["goal" + opponent_id]).flatten()
        board_configs_full = np.argwhere(np.sum(np.abs(self.board_configurations), axis=1) == 9).flatten()
        board_configs_terminal = reduce(np.union1d, (board_configs_win, board_configs_lose, board_configs_full))
        board_configs_draw = np.setdiff1d(board_configs_terminal, np.union1d(board_configs_win, board_configs_lose))

        self.states_player = reduce(np.union1d, (self.states_player, board_configs_terminal))
        self.states_opponent = reduce(np.union1d, (self.states_opponent, board_configs_terminal))
        n_state_player = len(self.states_player)      

        S0 = np.argwhere([s in board_configs_start for s in self.states_player]).flatten()
        S_win = np.argwhere([s in board_configs_win for s in self.states_player]).flatten()
        S_lose = np.argwhere([s in board_configs_lose for s in self.states_player]).flatten()
        S_draw = np.argwhere([s in board_configs_draw for s in self.states_player]).flatten()
        S_fork_lose = np.array([], dtype=int)
        S_to_win = np.array([], dtype=int)

        # Identify important events
        E_to_win = np.ndarray([0, 2], dtype=int)
        E_fork_win = np.ndarray([0, 2], dtype=int)
        E_to_lose = np.ndarray([0, 2], dtype=int)
        E_fork_lose = np.ndarray([0, 2], dtype=int)

        # Fill the MDP matrices
        A = np.full([n_state_player, n_action], False, dtype=bool)
        P = np.zeros([n_state_player, n_action, n_state_player], dtype=float)
        R = np.zeros([n_state_player, n_action, n_state_player], dtype=int)


        for s1_player, s2_player in zip(transition_sources, transition_targets):
            if (s1_player in self.states_player) and (s1_player not in board_configs_terminal):
                
                # Check outcome of the player's move
                board_before = self.board_configurations[s1_player]
                board_after = self.board_configurations[s2_player]

                # Indexing of states in [0 n_state_player)
                idx_a1_player = self.get_board_action(board_before, board_after)
                idx_s1_player = self.states_player == s1_player

                A[idx_s1_player, idx_a1_player] = True

                # If the target state is terminal, no reason to check for opponent's next move
                if s2_player in board_configs_terminal:
                    idx_s2_player = self.states_player == s2_player
                    P[idx_s1_player, idx_a1_player, idx_s2_player] = 1

                if s2_player in board_configs_win:
                    s_towin = np.argwhere(idx_s1_player).flatten()
                    a_towin = np.argwhere(idx_a1_player).flatten()
                    E_to_win = np.vstack([E_to_win, np.array([s_towin, a_towin]).T])
                    S_to_win = np.append(S_to_win, s_towin)
                    R[idx_s1_player, idx_a1_player, idx_s2_player] = +100
                
                else:
                    # Check all possible moves from opponent
                    s1_opponent = s2_player
                    possible_moves_opponent = transition_targets[transition_sources == s1_opponent]

                    if np.any([ s2_opponent in board_configs_lose for s2_opponent in possible_moves_opponent ]):
                        s_tolose = np.argwhere(idx_s1_player).flatten()
                        a_tolose = np.argwhere(idx_a1_player).flatten()
                        E_to_lose = np.vstack([E_to_lose, np.array([s_tolose, a_tolose]).T])
                    
                    for s2_opponent in possible_moves_opponent:

                        # Check outcome of the opponent's move
                        board_before = self.board_configurations[s1_opponent]
                        board_after = self.board_configurations[s2_opponent]

                        # Indexing of states in [0 n_state_player)
                        idx_a1_opponent = self.get_board_action(board_before, board_after)
                        idx_s1_opponent = self.states_opponent == s1_opponent

                        # If there is no opponent policy available, assume random policy
                        if opponent_policy is None:
                            p_O = 1/len(possible_moves_opponent)
                        else:
                            p_O = opponent_policy[idx_s1_opponent, idx_a1_opponent]

                        idx_s3_player = self.states_player == s2_opponent
                        P[idx_s1_player, idx_a1_player, idx_s3_player] = p_O

                        if s2_opponent in board_configs_lose:
                            R[idx_s1_player, idx_a1_player, idx_s3_player] = -100 

        # Once you are in a terminal state, no rewards should be assigned anymore
        for state_terminal in board_configs_terminal:
            idx_sT_player = self.states_player == state_terminal
            A[idx_sT_player, :] = True
            P[idx_sT_player, :, :] = 0
            P[idx_sT_player, :, idx_sT_player] = 1
            R[idx_sT_player, :, :] = 0

        # Indentify all fork states
        for s in range(n_state_player):
            for a in np.argwhere(A[s]).flatten():
                s2s = np.argwhere(P[s, a] > 0).flatten()
                if np.all([s2 in E_to_win[:, 0] for s2 in s2s]) and not (s in E_to_win[:, 0]):
                    E_fork_win = np.vstack([E_fork_win, np.array([s, a]).T])

                if not (s in E_to_lose[:, 0]):
                    if np.any([    np.all([    [s2, a2] in E_to_lose.tolist()    for a2 in np.argwhere(A[s2]).flatten() ])    for s2 in s2s ]):
                        E_fork_lose = np.vstack([E_fork_lose, np.array([s, a]).T])
                    for s2 in s2s:
                        if np.all([    [s2, a2] in E_to_lose.tolist()    for a2 in np.argwhere(A[s2]).flatten() ]):
                            S_fork_lose = np.append(S_fork_lose, s2)

        self.states_lose=S_lose
        self.states_win=S_win
        self.states_draw=S_draw
        self.states_to_win=S_to_win
        self.states_fork_lose=S_fork_lose
        self.events_to_win=E_to_win
        self.events_to_lose=E_to_lose
        self.events_fork_win=E_fork_win
        self.events_fork_lose=E_fork_lose

        super().__init__(A, P, R, S0, 
                        n_state=n_state_player, 
                        n_action=n_action, 
                        name=name, 
                        **kwargs)

    
    def get_board_action(self, board_before, board_after):
        board_before = np.reshape(board_before, [3, 3])
        board_after = np.reshape(board_after, [3, 3])
        for k in range(8):
            board_after_equivalent = np.rot90(board_after, k)
            if k >= 4:
                board_after_equivalent = board_after_equivalent.T
            action = np.abs(board_after_equivalent - board_before)
            if np.sum(action) == 1:
                return action.flatten().astype(bool)
        raise Exception("action not found - board configurations before and after transition are not compatible") 


    def draw(self, state, action=None, show_moves=True, show_move_numbers=False, ax=None):

        if ax is None:
            ax = plt.gca()

        # get board configuration
        board_config = self.board_configurations[self.states_player[state]]
        board = np.reshape(board_config, [3, 3])
        moves = np.reshape(self.A[state, :], [3, 3]) if state not in self.states_terminal else np.full([3, 3], False, dtype=bool)

        # draw  grid
        plt.hlines([0.5, 1.5], xmin=-0.5, xmax=2.5, linewidth=2, colors="black")
        plt.vlines([0.5, 1.5], ymin=-0.5, ymax=2.5, linewidth=2, colors="black")
        
        i_action = 0
        for i_row in range(3):
            for i_col in range(3):

                if action == i_action:
                    if self.player_id == "X":
                        plt.scatter(action % 3, action // 3, s=300, c="green", marker='x', clip_on=False)
                    else:
                        plt.scatter(action % 3, action // 3, s=300, c="green", marker='o', clip_on=False)

                i_action += 1

                if moves[i_row, i_col] and show_moves:
                    shape = Rectangle(np.array([i_col, i_row]) - 0.4, 0.8, 0.8, facecolor="green", edgecolor=None, alpha=0.3)  
                    if show_move_numbers:
                        plt.text(i_col, i_row, i_action)
                    ax.add_patch(shape)

                if board[i_row, i_col] == +1:
                    c = "red" if self.player_id == "X" else "black"
                    plt.scatter(i_col, i_row, s=300, c=c, marker='x', clip_on=False)

                if board[i_row, i_col] == -1:
                    c = "red" if self.player_id == "O" else "black"
                    plt.scatter(i_col, i_row, s=300, c=c, marker='o', clip_on=False)  
        
        ax.set_aspect('equal', 'box')
        ax.axis("off")
        return ax


class TicTacToeInteractive(TicTacToe):

    def __init__(self, play_first=True, opponent_policy=None, **kwargs):
        super().__init__(play_first, opponent_policy, **kwargs)


    def play_human(self, inline=False):

        state = np.random.choice(self.states_start)
        game_over = False
        outcome = 0

        ax = plt.gca()
        while not game_over:
            if inline:
                ax = plt.gca()
            ax.clear()
            ax = self.draw(state, show_moves=True, show_move_numbers=True, ax=ax)
            plt.ion()
            plt.show()

            available_actions = np.argwhere(self.A[state]).flatten() + 1
            action = int(input("type the action index ({}):\n".format(available_actions)))
            while action not in available_actions:
                action = int(input())
            state, outcome, game_over = self.step(state, action - 1)
        
        if inline:
            ax = plt.gca()
        ax.clear()
        ax = self.draw(state, show_moves=True, show_move_numbers=True, ax=ax)
        plt.ion()
        plt.show()
        
        if outcome > 0:
            message = "you win"
        elif outcome < 0:
            message = "you lose"
        else:
            message = "draw"
        print(message)

        if not inline:
            plt.waitforbuttonpress()

    
    def play_agent(self, agent, inline=False):

        state = np.random.choice(self.states_start)
        game_over = False
        outcome = 0

        ax = plt.gca()
        while not game_over:
            if inline:
                ax = plt.gca()
            ax.clear()
            ax = self.draw(state, ax=ax)
            plt.ion()
            plt.show()
            if inline:
                plt.pause(0.5)
            else:
                plt.pause(5)

            action = agent.choose_action(state)
            state, outcome, game_over = self.step(state, action)
        
        if outcome > 0:
            message = "you win"
        elif outcome < 0:
            message = "you lose"
        else:
            message = "draw"

        if inline:
            ax = plt.gca()
        ax.clear()
        ax = self.draw(state, ax=ax)
        plt.title(message)
        plt.ion()
        plt.show()
        if inline:
            plt.pause(0.5)
        else:
            plt.pause(5)