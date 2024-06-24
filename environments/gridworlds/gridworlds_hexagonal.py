from environments.gridworlds.gridworld import GridWorld
from environments.gridworlds.gridactions import HexaActionSet
import numpy as np

class TowerOfHanoi(GridWorld):
    def __init__(self, **kwargs):
                
        nx = 8
        ny = 8

        init_locs = np.array([[2, 2]])
        term_locs = np.array([[0, 7]])

        # Walls
        valid_map = np.full([nx, ny], True, dtype=bool)        
        valid_map = np.triu(valid_map).astype(bool)
        valid_map[1, 6] = False
        valid_map[3, 6] = False
        valid_map[5, 6] = False
        valid_map[1, 4] = False
        valid_map[1, 2] = False

        valid_map[2, 4] = False
        valid_map[2, 5] = False
        valid_map[3, 4] = False
        valid_map[3, 5] = False
        walls = np.argwhere(~valid_map)

        super().__init__(nx, ny, init_locs, term_locs=term_locs, walls=walls, actions=HexaActionSet(), **kwargs)

        # state_sets
        self.state_sets = []
        self.state_set_colors = []
        self.state_set_labels = []

        room_1 = np.full([nx, ny], False, dtype=bool)
        room_1[:4, :4] = True
        room_1[0, 3] = False  # Door
        room_1_valid = room_1 * valid_map
        
        room_2 = np.full([nx, ny], False, dtype=bool)   
        room_2[:4, 4:] = True
        room_2_valid = room_2 * valid_map

        room_3 = np.full([nx, ny], False, dtype=bool)
        room_3[4:, 4:] = True
        room_3_valid = room_3 * valid_map

        self.state_sets.append(self.encode(np.argwhere(room_1_valid)))
        self.state_set_colors.append("blue")
        self.state_set_labels.append("room1")

        self.state_sets.append(self.encode(np.argwhere(room_2_valid)))
        self.state_set_colors.append("red")
        self.state_set_labels.append("room2")

        self.state_sets.append(self.encode(np.argwhere(room_3_valid)))
        self.state_set_colors.append("orange")
        self.state_set_labels.append("room3")

        self.state_sets.append(self.encode([np.array([0, 3])]))
        self.state_set_colors.append("magenta")
        self.state_set_labels.append("door")       