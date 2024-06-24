import numpy as np
from environments.mutantworlds.mutantworld import MutantWorld


class FosterWorld(MutantWorld):

    def __init__(self, goal_reward=100, doors_cost=-100, small_version=False, pillars_as_cost=False, **kwargs):

        if small_version:
            doors, starts, goal, outer_pillars, inner_pillars, wells = getFosterWorldStructureSmall()
        else:
            doors, starts, goal, outer_pillars, inner_pillars, wells = getFosterWorldStructure()

        if pillars_as_cost:
            mutations_sR = (doors + inner_pillars) * doors_cost + goal * goal_reward 
            walls = np.ndarray([0, 2])
        else:
            mutations_sR = doors * doors_cost + goal * goal_reward
            walls = np.argwhere(inner_pillars)
        mutations_s0 = starts

        super().__init__(mutations_s0, mutations_sR, walls=walls, name="foster_maze", **kwargs)
        self.state_wells = self.encode(np.argwhere(wells))
        self.labels_wells = ["well Top-L", "well Mid-L", "well Bot-L", 
                             "well Top-C", "well Mid-C", "well Bot-C",
                             "well Top-R", "well Mid-R", "well Bot-R"]


def getFosterWorldStructure():
        """
        Return a [n_mutations * n_x * n_y] matrix representing the reward value of each state across mutations
        """        
        
        doors = np.full([10, 13, 13], False)

        doors[1, 1:4, 8] = True
        doors[1, 4, 9:12] = True
        doors[1, 4, 5:8] = True
        doors[1, 8, 1:4] = True
        doors[1, 9:12, 4] = True
        doors[1, 5:8, 4] = True

        doors[2, 1:4, 4] = True
        doors[2, 4, 1:4] = True
        doors[2, 4, 5:8] = True
        doors[2, 8, 1:4] = True
        doors[2, 8, 5:8] = True
        doors[2, 8, 9:12] = True

        doors[3, 4, 1:4] = True
        doors[3, 4, 9:12] = True
        doors[3, 8, 1:4] = True
        doors[3, 8, 5:8] = True
        doors[3, 5:8, 4] = True       
        doors[3, 9:12, 4] = True       

        doors[4, 1:4, 4] = True
        doors[4, 5:8, 4] = True
        doors[4, 9:12, 4] = True
        doors[4, 1:4, 8] = True
        doors[4, 5:8, 8] = True
        doors[4, 8, 5:8] = True

        doors[5, 1:4, 4] = True
        doors[5, 9:12, 4] = True
        doors[5, 5:8, 8] = True
        doors[5, 9:12, 8] = True
        doors[5, 4, 5:8] = True
        doors[5, 8, 9:12] = True

        doors[6, 4, 1:4] = True
        doors[6, 4, 5:8] = True
        doors[6, 4, 9:12] = True
        doors[6, 8, 1:4] = True
        doors[6, 5:8, 8] = True
        doors[6, 9:12, 4] = True

        doors[7, 4, 9:12] = True
        doors[7, 8, 9:12] = True
        doors[7, 1:4, 8] = True
        doors[7, 5:8, 4] = True
        doors[7, 9:12, 4] = True
        doors[7, 9:12, 8] = True

        doors[8, 1:4, 4] = True
        doors[8, 4, 1:4] = True
        doors[8, 4, 9:12] = True
        doors[8, 1:4, 8] = True
        doors[8, 5:8, 8] = True
        doors[8, 9:12, 8] = True

        doors[9, 1:4, 4] = True
        doors[9, 1:4, 8] = True
        doors[9, 9:12, 8] = True
        doors[9, 8, 1:4] = True
        doors[9, 8, 5:8] = True
        doors[9, 8, 9:12] = True

        goal = np.full([10, 13, 13], False)
        goal[0, 10, 6] = True
        goal[1, 6, 6] = True
        goal[2, 10, 6] = True
        goal[3, 6, 2] = True
        goal[4, 6, 6] = True
        goal[5, 10, 2] = True
        goal[6, 2, 6] = True
        goal[7, 10, 10] = True
        goal[8, 2, 10] = True
        goal[9, 2, 6] = True

        wells = np.full([13, 13], False)
        for i in range(2, 13, 4):
             for j in range(2, 13, 4):
                  wells[i, j] = True
 
        starts_idxs = [2, 2, 1, 2, 7, 5, 7, 6, 7, 6]
        starts_locs = np.argwhere(wells)
        starts = np.zeros([10, 13, 13])
        for i, start_idx in enumerate(starts_idxs):
             x, y = starts_locs[start_idx] + np.array([-1, 1])
             starts[i, x, y] = 1

        inner_pillars = np.full([13, 13], False)
        outer_pillars = np.full([13, 13], False)
        for i in (4, 8):
            for j in (4, 8):
                inner_pillars[i, j] = True
        for i in (0, 12):
            for j in (0, 12):
                outer_pillars[i, j] = True

        return doors, starts, goal, outer_pillars, inner_pillars, wells



def getFosterWorldStructureSmall():
        """
        Return a [n_mutations * n_x * n_y] matrix representing the reward value of each state across mutations
        """        
        
        doors = np.full([10, 7, 7], False)

        doors[1, 1, 4] = True
        doors[1, 2, 5] = True
        doors[1, 2, 3] = True
        doors[1, 4, 1] = True
        doors[1, 5, 2] = True
        doors[1, 3, 2] = True

        doors[2, 1, 2] = True
        doors[2, 2, 1] = True
        doors[2, 2, 3] = True
        doors[2, 4, 1] = True
        doors[2, 4, 3] = True
        doors[2, 4, 5] = True

        doors[3, 2, 1] = True
        doors[3, 2, 5] = True
        doors[3, 4, 1] = True
        doors[3, 4, 3] = True
        doors[3, 3, 2] = True       
        doors[3, 5, 2] = True       

        doors[4, 1, 2] = True
        doors[4, 3, 2] = True
        doors[4, 5, 2] = True
        doors[4, 1, 4] = True
        doors[4, 3, 4] = True
        doors[4, 4, 3] = True

        doors[5, 1, 2] = True
        doors[5, 5, 2] = True
        doors[5, 3, 4] = True
        doors[5, 5, 4] = True
        doors[5, 2, 3] = True
        doors[5, 4, 5] = True

        doors[6, 2, 1] = True
        doors[6, 2, 3] = True
        doors[6, 2, 5] = True
        doors[6, 4, 1] = True
        doors[6, 3, 4] = True
        doors[6, 5, 2] = True

        doors[7, 2, 5] = True
        doors[7, 4, 5] = True
        doors[7, 1, 4] = True
        doors[7, 5, 2] = True
        doors[7, 5, 2] = True
        doors[7, 5, 4] = True

        doors[8, 1, 2] = True
        doors[8, 2, 1] = True
        doors[8, 2, 5] = True
        doors[8, 1, 4] = True
        doors[8, 3, 4] = True
        doors[8, 5, 4] = True

        doors[9, 1, 2] = True
        doors[9, 1, 4] = True
        doors[9, 5, 4] = True
        doors[9, 4, 1] = True
        doors[9, 4, 3] = True
        doors[9, 4, 5] = True

        goal = np.full([10, 7, 7], False)
        goal[0, 3, 3] = True
        goal[1, 3, 3] = True
        goal[2, 5, 3] = True
        goal[3, 3, 1] = True
        goal[4, 3, 3] = True
        goal[5, 5, 1] = True
        goal[6, 1, 3] = True
        goal[7, 5, 5] = True
        goal[8, 1, 5] = True
        goal[9, 1, 3] = True

        wells = np.full([7, 7], False)
        for i in range(1, 7, 2):
             for j in range(1, 7, 2):
                  wells[i, j] = True
 
        starts_idxs = [2, 2, 1, 2, 7, 5, 7, 6, 7, 6]
        starts_locs = np.argwhere(wells)
        starts = np.zeros([10, 7, 7])
        for i, start_idx in enumerate(starts_idxs):
             x, y = starts_locs[start_idx]
             starts[i, x, y] = 1

        inner_pillars = np.full([7, 7], False)
        inner_pillars[2, 2] = True
        inner_pillars[2, 4] = True
        inner_pillars[4, 2] = True
        inner_pillars[4, 4] = True

        outer_pillars = np.full([7, 7], False)
        outer_pillars[0, 0] = True
        outer_pillars[0, 6] = True
        outer_pillars[6, 0] = True
        outer_pillars[6, 6] = True

        return doors, starts, goal, outer_pillars, inner_pillars, wells