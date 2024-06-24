import os
from os import path
import pickle


def load_model(model_name, world_name):
    model_file = get_storage_path(model_name, world_name)

    if not path.exists(model_file): 
        return None

    with open(model_file, 'rb') as file:
        return pickle.load(file)


def save_model(model):
    model.reset(clear_params=False, clear_history=True)
    model_file = get_storage_path(model.name, model.env.name)
    with open(model_file, 'wb') as file:
        pickle.dump(model, file)


def get_storage_path(model_name, world_name):

    world_path = path.join("__storage", "worlds", world_name)
    if not path.exists(world_path): 
        os.makedirs(world_path) 
    
    models_path = path.join(world_path, "models")
    if not path.exists(models_path): 
        os.makedirs(models_path) 

    model_file = path.join(models_path, model_name)
    return model_file
