from sqlitedict import SqliteDict

import os
import shutil
import logging

# Que element class
# Saves the model element and the type of operation that the trainer is supposed to do with it
# Possible types:
# train = Train new model
# continue = Continue training a model
# evaluate = Evaluate model
# predict = Predict with model
class QueueElement():

    # When initializing model_id and operation type have to be given.
    def __init__(self, model_id, op_type):
        self._op_type = op_type
        self._model_id = model_id

    # Returns a tuple (model_id, op_type)
    def get_info(self):
        return (self._model_id, self._op_type)

# SAves the interrupt state
class InterruptState():

    def __init__(self):
        self._stop = 0

    def get_state(self):
        return self._stop

    def set_state(self, x):
        self._stop = x


# Methods that removes checkpoints for model with model_id if checkpoints exist.
def remove_checkpoints(model_id):
    if os.path.isdir(f'./checkpoints/{model_id}'):
        shutil.rmtree('./checkpoints/' + model_id)
        logging.debug('Removed checkpoints')

# Deletes the model with model_id
# Tries to delete saved model, checkpoint, and the kv-store entry.
# The method returns the model_info (kv-store entry) of the deleted model.
def delete_model(model_id):
    model_info = ''

    # remove the model from db
    with SqliteDict('./distilBERT.sqlite') as db:
        model_info = db.pop(model_id)
        db.commit()

    logging.debug(f'Deleted model {model_id} from kv-store')

    # delete model file
    if os.path.isfile(f'models/{model_id}.pth'):
        os.remove(f'models/{model_id}.pth')

        logging.debug(f'Deleted model {model_id} from harddrive')

    # remove the checkpoints
    if os.path.isdir(f'./checkpoints/{model_id}'):
        shutil.rmtree(f'./checkpoints/{model_id}')

        logging.debug(f'Deleted model {model_id} checkpoints')

    return model_info