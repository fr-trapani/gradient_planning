import numpy as np


## Grid Action Class

class GridActionSet:

    @ property
    def n_action(self):
        return len(self.action_names)
    
    def __init__(self, action_dict, set_name="action_set"):
        self.set_name = set_name
        self.action_names = np.array(list(action_dict.keys()))
        self.action_values = np.array(list(action_dict.values()))

    def get_action_index(self, name):
        """
        name : the label (str) of the action to retrieve
        returns the corresponding action index
        """
        idx_logic = self.action_names == name
        if np.sum(idx_logic) != 1:
            raise Exception("action not found")
        idx = np.argwhere(idx_logic).flatten()[0]
        return idx

    def get_action_name(self, idx=None):
        """
        idx: the index (int) of the action to retrieve
        returns the corresponding action name
        """
        if idx is None:
            return self.action_names
        else:
            return self.action_names[idx]
    
    def get_action(self, id):
        """
        id: the name (str) or index (int) of the action to retrieve
        returns the corresponding action
        """
        if isinstance(id, str):
            id = self.get_action_index(id)
        action = self.action_values[id]
        return action


##  Grid Action Sets

basic_actions =        {"north":        np.array([+0, -1], dtype=int), 
                        "east":         np.array([+1, +0], dtype=int), 
                        "south":        np.array([+0, +1], dtype=int),
                        "west":         np.array([-1, +0], dtype=int)}

idle_actions =         {"north":        np.array([+0, -1], dtype=int), 
                        "east":         np.array([+1, +0], dtype=int), 
                        "south":        np.array([+0, +1], dtype=int),
                        "west":         np.array([-1, +0], dtype=int), 
                        "stay":         np.array([+0, +0], dtype=int)}

hexa_actions =         {"north":        np.array([+0, -1], dtype=int), 
                        "north-west":   np.array([-1, -1], dtype=int), 
                        "east":         np.array([+1, +0], dtype=int), 
                        "south":        np.array([+0, +1], dtype=int),
                        "south-east":   np.array([+1, +1], dtype=int), 
                        "west":         np.array([-1, +0], dtype=int)}

king_actions =         {"north":        np.array([+0, -1], dtype=int), 
                        "north-east":   np.array([+1, -1], dtype=int), 
                        "east":         np.array([+1, +0], dtype=int), 
                        "south-east":   np.array([+1, +1], dtype=int), 
                        "south":        np.array([+0, +1], dtype=int),
                        "south-west":   np.array([-1, +1], dtype=int), 
                        "west":         np.array([-1, +0], dtype=int), 
                        "north-west":   np.array([-1, -1], dtype=int)}

king_idle_actions =    {"north":        np.array([+0, -1], dtype=int), 
                        "north-east":   np.array([+1, -1], dtype=int), 
                        "east":         np.array([+1, +0], dtype=int), 
                        "south-east":   np.array([+1, +1], dtype=int), 
                        "south":        np.array([+0, +1], dtype=int),
                        "south-west":   np.array([-1, +1], dtype=int), 
                        "west":         np.array([-1, +0], dtype=int), 
                        "north-west":   np.array([-1, -1], dtype=int),
                        "stay":         np.array([+0, +0], dtype=int)}

knight_actions =       {"WNW":          np.array([-2, -1], dtype=int), 
                        "WSW":          np.array([-2, +1], dtype=int), 
                        "ENE":          np.array([+2, -1], dtype=int), 
                        "ESE":          np.array([+2, +1], dtype=int), 
                        "NNW":          np.array([-1, -2], dtype=int),
                        "SSW":          np.array([-1, +2], dtype=int), 
                        "NNE":          np.array([+1, -2], dtype=int), 
                        "SSE":          np.array([+1, +2], dtype=int)}

knight_idle_actions =  {"WNW":          np.array([-2, -1], dtype=int), 
                        "WSW":          np.array([-2, +1], dtype=int), 
                        "ENE":          np.array([+2, -1], dtype=int), 
                        "ESE":          np.array([+2, +1], dtype=int), 
                        "NNW":          np.array([-1, -2], dtype=int),
                        "SSW":          np.array([-1, +2], dtype=int), 
                        "NNE":          np.array([+1, -2], dtype=int), 
                        "SSE":          np.array([+1, +2], dtype=int),
                        "stay":         np.array([+0, +0], dtype=int)}

##  Idiothetic Action Sets

idio_basic_actions =   {"forward":      np.array([+0, +1], dtype=int), 
                        "left":         np.array([-1, +1], dtype=int), 
                        "right":        np.array([+1, +1], dtype=int)}

idio_turning_actions = {"forward":      np.array([+0, +1], dtype=int), 
                        "left":         np.array([-1, +0], dtype=int), 
                        "right":        np.array([+1, +0], dtype=int)}


       

## Action Subclasses
        
class BasicActionSet(GridActionSet):
    def __init__(self, set_name="basic_action_set", action_dict=basic_actions):
        super().__init__(action_dict, set_name)

class IdleActionSet(GridActionSet):
    def __init__(self, set_name="idle_action_set", action_dict=idle_actions):
        super().__init__(action_dict, set_name)

class HexaActionSet(GridActionSet):
    def __init__(self, set_name="hexa_action_set", action_dict=hexa_actions):
        super().__init__(action_dict, set_name)

class KingActionSet(GridActionSet):
    def __init__(self, set_name="king_action_set", action_dict=king_actions):
        super().__init__(action_dict, set_name)

class KingIdleActionSet(GridActionSet):
    def __init__(self, set_name="king_idle_actions", action_dict=king_idle_actions):
        super().__init__(action_dict, set_name)

class PoseActionSet(GridActionSet):
    def __init__(self, set_name="pose_action_set", action_dict=idio_basic_actions):
        super().__init__(action_dict, set_name)

class TurningActionSet(GridActionSet):
    def __init__(self, set_name="turning_action_set", action_dict=idio_turning_actions):
        super().__init__(action_dict, set_name)