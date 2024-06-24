import numpy as np
from environments.gridworlds.gridworld import *
from environments.gridworlds.gridactions import *


class MutantWorld(GridWorld):

    def __init__(self, 
                 mutations_s0, 
                 mutations_sR, 
                 walls=np.ndarray([0,2]),
                 actions=BasicActionSet(),
                 action_cost = -1,
                 name="mutant_world"):        
        
        assert(mutations_s0.shape == mutations_sR.shape)

        self.n_mutations, self.nx, self.ny = mutations_s0.shape
        self.mutations_s0 = mutations_s0
        self.mutations_sR = mutations_sR
        self.current_mutation = 0
        
        self.walls=walls
        self.actions=actions
        self.action_cost=action_cost
        self.name=name

        self.mutate(0)


    def mutate(self, i_m=None):        
        if i_m is None:
            i_m = (self.current_mutation + 1) % self.n_mutations

        m_sR = self.mutations_sR[i_m]
        m_s0 = self.mutations_s0[i_m]

        m_locsT = np.argwhere(m_sR >  0)
        m_locsR = np.argwhere(m_sR != 0)
        m_locs0 = np.argwhere(m_s0 != 0)

        m_r = m_sR[*m_locsR.T]
        m_p0 = m_s0[*m_locs0.T]

        super().__init__(
            self.nx, 
            self.ny, 
            init_locs=m_locs0,
            p_init_locs=m_p0,
            term_locs=m_locsT, 
            walls=self.walls, 
            actions=self.actions,
            action_cost=self.action_cost,
            name=self.name)
        
        self.add_grid_rewards(m_r, m_locsR)
        self.current_mutation = i_m


def freeze_mutant_world(world, temperature=-100, i_m=None):
    if i_m is None:
        i_m = world.current_mutation
    
        m_sR = world.mutations_sR[i_m]
        m_s0 = world.mutations_s0[i_m]

        m_locsT = np.argwhere(m_sR > 0)
        m_locsR = np.argwhere(m_sR > temperature)
        m_locsW = np.argwhere(m_sR <= temperature)
        m_locs0 = np.argwhere(m_s0 != 0)

        m_r = m_sR[*m_locsR.T]
        m_p0 = m_s0[*m_locs0.T]

        walls = np.vstack([world.walls, m_locsW]).astype(int)

        frozen_world = GridWorld(
            world.nx, 
            world.ny, 
            init_locs=m_locs0,
            p_init_locs=m_p0,
            term_locs=m_locsT, 
            walls=walls, 
            actions=world.actions,
            name=world.name)
        
        frozen_world.add_grid_rewards(m_r, m_locsR)
        return frozen_world