from environments.world import World
import numpy as np


class GraphWorld(World):

    def __init__(self, A, W, S0, **kwargs):
        n_state = A.shape[0]
        D = np.eye(n_state)
        P = np.einsum("ij, jk -> ijk", A, D)
        R = np.einsum("ij, jk -> ijk", W, D)
        super().__init__(A, P, R, S0, **kwargs)
        self.W = W


class ThreePentagonsWorld(GraphWorld):

    def __init__(self, **kwargs):
        D = np.eye(15, dtype=bool)
        
        A = np.full([15, 15], False, dtype=bool)
        # Create three fully connected pentagons 
        A[0:5, 0:5] = True
        A[5:10, 5:10] = True
        A[10:15, 10:15] = True
        # For each pentagon, break one edge
        A[0, 4] = False
        A[4, 0] = False
        A[5, 9] = False
        A[9, 5] = False
        A[10, 14] = False
        A[14, 10] = False
        # Connect the three pentagons
        A[4, 5] = True
        A[5, 4] = True
        A[9, 10] = True
        A[10, 9] = True
        A[14, 0] = True
        A[0, 14] = True
        # Remove all self connections
        A[D] = False

        W = A * -1.0
        S0 = np.array([1])

        # Terminals 
        A[12] = D[12]
        A[7] = D[7]

        super().__init__(A, W, S0, **kwargs)

        # state_sets
        self.state_sets = []
        self.state_set_colors = []
        self.state_set_labels = []

        self.state_sets.append(np.arange(1, 4))
        self.state_set_colors.append("blue")
        self.state_set_labels.append("room1")

        self.state_sets.append(np.arange(5, 10))
        self.state_set_colors.append("red")
        self.state_set_labels.append("room2")

        self.state_sets.append(np.arange(10, 15))
        self.state_set_colors.append("orange")
        self.state_set_labels.append("room3")

        self.state_sets.append(np.array([0,4]))
        self.state_set_colors.append("purple")
        self.state_set_labels.append("doors")


class ButterflyWorld(GraphWorld):

    def __init__(self, **kwargs):
        D = np.eye(19, dtype=bool)
        
        A = np.full([19, 19], False, dtype=bool)
        # Create two grid rooms 
        for i in range(9):
            if (i+1) % 3:
                A[i, i+1] = True
                A[i+1, i] = True
            if i < 6:
                A[i, i+3] = True
                A[i+3, i] = True
        for i in range(10, 19):
            if i % 3:
                A[i, i+1] = True
                A[i+1, i] = True
            if i < 16:
                A[i, i+3] = True
                A[i+3, i] = True
        # Connect the two rooms
        A[6, 9] = True
        A[9, 6] = True
        A[8, 9] = True
        A[9, 8] = True
        A[10, 9] = True
        A[9, 10] = True
        A[12, 9] = True
        A[9, 12] = True

        # Remove all self connections
        A[D] = False

        W = A * -1.0
        S0 = np.array([1])

        # Terminals 
        A[17] = D[17]

        super().__init__(A, W, S0, **kwargs)

        # state_sets
        self.state_sets = []
        self.state_set_colors = []
        self.state_set_labels = []

        self.state_sets.append(np.arange(0, 9))
        self.state_set_colors.append("blue")
        self.state_set_labels.append("room1")

        self.state_sets.append(np.array([9]))
        self.state_set_colors.append("red")
        self.state_set_labels.append("bridge")

        self.state_sets.append(np.arange(10, 19))
        self.state_set_colors.append("orange")
        self.state_set_labels.append("room2")
