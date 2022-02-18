from sqlitedict import SqliteDict

import os
import shutil
import logging
import json

# Que element class
# Saves the model element and the type of operation that the trainer 
# is supposed to do with it
# possible op types: train, continue, evaluate, predict, predict_text
class QueueElement():

    # When initializing model_id and operation type have to be given.
    # text would be a list of documents for evaluate and predict
    # or a sentence string for predict_text
    # it is used to store additional information if needed
    def __init__(self, model_id, op_type, text = None):
        self._op_type = op_type
        self._model_id = model_id
        self._text = text

    # Returns a tuple (model_id, op_type)
    def get_info(self):
        return (self._model_id, self._op_type)

    # return the text
    def get_text(self):
        return self._text

# SAves the interrupt state
class InterruptState():

    def __init__(self):
        self._stop = 0

    # get the state
    def get_state(self):
        return self._stop

    # set the state
    def set_state(self, x):
        self._stop = x

# Saves current model if
class CurrentModel():

    def __init__(self):
        self._id = ''

    # set current model id
    def set_id(self, id):
        self._id = id 

    # get current model id
    def get_id(self):
        return self._id


# Methods that removes checkpoints for model if checkpoints exist.
def remove_checkpoints(model_id, config):
    # check if checkpoint directory exists
    if os.path.isdir(f'{config["checkpoints"]}{model_id}'):
        # if yes, delete it and its contents
        shutil.rmtree(f'{config["checkpoints"]}{model_id}')
        log('Removed checkpoints', 'DEBUG')

# Deletes the model with model_id
# Tries to delete saved model, checkpoint, and the kv-store entry.
# The method returns the model_info (kv-store entry) of the deleted model.
def delete_model(model_id, config):
    log(f'DELETE {model_id}')

    model_info = ''

    # remove the model from kv store
    with SqliteDict(config['kv']) as db:
        model_info = db.pop(model_id)
        db.commit()

    log(f'Deleted model {model_id} from kv-store', 'DEBUG')

    # delete model file if it exists
    if os.path.isfile(f'{config["models"]}{model_id}.pth'):
        os.remove(f'{config["models"]}{model_id}.pth')

        log(f'Deleted model {model_id} from harddrive', 'DEBUG')

    # remove the checkpoints
    remove_checkpoints(model_id, config)

    return model_info


# Check if model state matches files on hard drive.
# If that is not the case, delete the model
# Returns true if state is correct, else false is returned
def check_model(model_id, config):
    model_info = SqliteDict(config['kv'])[model_id]

    good = True

    # Model is trained => check for saved weights
    if (model_info['status'] == 'trained'):
        good = os.path.isfile(f'{config["models"]}{model_id}.pth')

    # Model was interrupted => check for checkpoints
    # check point file has to include atleast one checkpoint for an
    # interrupted model
    if (model_info['status'] == 'interrupted'):
        good = (os.path.isdir(f'{config["checkpoints"]}{model_id}') 
            and len(os.listdir(f'{config["checkpoints"]}{model_id}')) > 0)
        log(good)
        log(model_id)
        log(model_info)
        log(os.path.isdir(f'{config["checkpoints"]}{model_id}'))
        log(os.listdir(f'{config["checkpoints"]}{model_id}'))

    # if model is faulty => delete it
    if not good:
        delete_model(model_id, config)

    # return False if model is faulty, true if it is fine
    return good


# Logging method
# edit this to log to logging service or somewhere else
def log(message, log_type = 'INFO'):
    #normal logging
    if log_type == 'INFO':
        logging.info(message)
    elif log_type == 'DEBUG':
        logging.debug(message)
    elif log_type == 'ERROR':
        logging.error(message)
    elif log_type == 'WARNING':
        logging.warning(message)
    else:
        logging.error('Unkwon log type.')


# Method reading and returning the configuration file
# also read the right startup file and retrieves the default values
def get_config(path, port):
    # read the config file
    with open('./config.json') as json_file:
        config = json.load(json_file)

    #set path and port
    config['path'] = path
    config['port'] = port

    # read start up file
    with open(f'./{config["path"]}/startup.json') as json_file:
        startup = json.load(json_file)

    # add default values to config file
    config['defaults'] = {}
    for param in startup['hyper_parameters']:
        config['defaults'][param['name']] = param['default']

    # read classifier config file
    with open(f'./{config["path"]}/model.json') as json_file:
        cls_config = json.load(json_file)

    # add the values to the config file
    for k, v in cls_config.items():
        config[k] = v

    # add a few more things for convenience
    config['kv'] = f'./{config["path"]}/kv.sqlite'
    config['que'] = f'./{config["path"]}/que.obj'
    config['checkpoints'] = f'./{config["path"]}/checkpoints/'
    config['models'] = f'./{config["path"]}/models/'
    config['startup'] = f'./{config["path"]}/startup.json'
    config['address'] = (f'{config["host"]}:{config["port"]}'
        f'/{config["path"]}')
 
    log(config)

    # return the config file
    return config