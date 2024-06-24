from environments.poseworlds.poseworld import *
from environments.gridworlds.gridworlds_classic import *


class OneDoorIdio(OneDoor, PoseWorld):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FourRoomIdio(FourRoom, PoseWorld):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)