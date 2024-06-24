import numpy as np

from environments.mutantworlds.mutantworld import MutantWorld
from environments.gridworlds.gridworlds_classic import *


class MutantCross(MutantWorld):

    def __init__(self):

        nx = 10
        ny = 10

        mutations_s0 = np.zeros([5, nx, ny])
        mutations_s0[:, 2, 2] += .8
        mutations_s0[:, 1, 1] += .2
        
        mutations_sR = np.zeros([5, nx, ny])
        mutations_sR[:, 2, 8] += -50
        mutations_sR[:, 8, 2] += -50
        mutations_sR[:, 8, 8] += +100

        mutations_sR[0, 4, 3:6] += -100
        mutations_sR[0, 2:6, 4] += -100
        mutations_sR[1, 5, 2:8] += -100
        mutations_sR[1, 2:8, 5] += -100
        mutations_sR[2, 7, 3:7] += -100
        mutations_sR[2, 3:7, 7] += -100
        mutations_sR[3, 7, 3:7] += -100
        mutations_sR[3, 5:7, 7] += -100
        mutations_sR[4, 7, 3:7] += -100
        mutations_sR[4, 5:7, 7] += -100
        mutations_sR[4, 4,   8] += +20

        super().__init__(mutations_s0, mutations_sR, name="mutant_cross")