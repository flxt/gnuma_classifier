from sqlitedict import SqliteDict

import os
import shutil
import logging

# Que element class
# Saves the model element and the type of operation that the trainer 
# is supposed to do with it
# Possible types:
# train = Train new model
# continue = Continue training a model
# evaluate = Evaluate model
# predict_text = Predict with model
# predict
class QueueElement():

    # When initializing model_id and operation type have to be given.
    def __init__(self, model_id, op_type, text = None):
        self._op_type = op_type
        self._model_id = model_id
        self._text = text

    # Returns a tuple (model_id, op_type)
    def get_info(self):
        return (self._model_id, self._op_type)

    def get_text(self):
        return self._text

# SAves the interrupt state
class InterruptState():

    def __init__(self):
        self._stop = 0

    def get_state(self):
        return self._stop

    def set_state(self, x):
        self._stop = x


# Methods that removes checkpoints for model if checkpoints exist.
def remove_checkpoints(model_id):
    if os.path.isdir(f'./checkpoints/{model_id}'):
        shutil.rmtree('./checkpoints/' + model_id)
        log('Removed checkpoints', 'DEBUG')

# Deletes the model with model_id
# Tries to delete saved model, checkpoint, and the kv-store entry.
# The method returns the model_info (kv-store entry) of the deleted model.
def delete_model(model_id):
    model_info = ''

    # remove the model from db
    with SqliteDict('./distilBERT.sqlite') as db:
        model_info = db.pop(model_id)
        db.commit()

    log(f'Deleted model {model_id} from kv-store', 'DEBUG')

    # delete model file
    if os.path.isfile(f'models/{model_id}.pth'):
        os.remove(f'models/{model_id}.pth')

        log(f'Deleted model {model_id} from harddrive', 'DEBUG')

    # remove the checkpoints
    if os.path.isdir(f'./checkpoints/{model_id}'):
        shutil.rmtree(f'./checkpoints/{model_id}')

        log(f'Deleted model {model_id} checkpoints', 'DEBUG')

    return model_info


# Check if model state matches files on hard drive.
# If that is not the case, delete the model
# Returns true if state is correct, else false is returned
def check_model(model_id):
    model_info = SqliteDict('./distilBERT.sqlite')[model_id]

    good = True

    # Model is trained => check for saved weights
    if (model_info['status'] == 'trained'):
        good = os.path.isfile(f'models/{model_id}.pth')

    # Model was interrupted => check for checkpoints
    if (model_info['status'] == 'interrupted'):
        good = os.path.isdir(f'./checkpoints/{model_id}')

    if not good:
        delete_model(model_id)

    return good


# Logging method
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