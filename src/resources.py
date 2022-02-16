from flask import request
from flask_restful import Resource, abort

from sqlitedict import SqliteDict
import uuid

from queue import Queue
import os
import dill

from src.utils import InterruptState, QueueElement, delete_model, check_model
from src.utils import log, CurrentModel


# Abort if a json file is expected, but not part of the request
def abort_not_json():
    abort(400, 
        message='Only accepting requests with mime type application/json.')
    log('Only accepting requests with mime type application/json.', 'WARNING')

# Abort if expected parameter is missing from the request.
def abort_missing_parameter(parameter_name: str):
    abort(400, 
        message=f'Expected "{parameter_name}" to be part of the request body.')
    log(f'Expected "{parameter_name}" to be part of the request body.', 
        'WARNING')

# Abort if specified model doesnt exist.
def abort_wrong_model_id(model_id: str):
    abort(400, message=f'No model with ÍD "{model_id}" exists.')
    log(f'No model with ÍD "{model_id}" exists.', 'WARNING')

# Abort if model is corrupted.
def abort_faulty_model(model_id: str):
    abort(400, message=f'Model "{model_id}" corrupted. Delete the model.')
    log(f'Model "{model_id}" corrupted. Delete the model.', 'WARNING')

# Abort wrong model for operation
def abort_wrong_op_type(model_id: str, op_type: str, status: str):
    abort(400, 
        message = f'Cant {op_type} for model {model_id} with status {status}.')
    log(f'Can not {status} for model {model_id} with status {status}.', 
        'WARNING')

# API enpoint where only a model ID is given
class Base(Resource):

    #init ressource
    def __init__(self, current_model_id: CurrentModel, config, que):
        self._current_model_id = current_model_id
        self._config = config
        self._q = q

    # Return the decription and more info for the model with the given id
    def get(self, model_id: str):
        # check if model exists
        if not model_id in SqliteDict(self._config['kv']).keys():
            abort_wrong_model_id(model_id)

        # check if model is in good state
        if not check_model(model_id, self._config):
            abort_faulty_model(model_id)

        # get info for model from kv store
        model_info = SqliteDict(self._config['kv'])[model_id]

        return model_info

    # Delete the model with the specified id
    def delete(self, model_id: str):
        # check if model exists
        if not model_id in SqliteDict(self._config['kv']).keys():
            abort_wrong_model_id(model_id)

        # check if model is currently in use
        if self._current_model_id.get_id() == model_id:
            abort(404, 
                message = f'Can not delete model {model_id} cause it is'
                ' currently getting trained.')

        # delete all instances in que
        old_q = self._q
        self._q = Queue()

        while not old_q.empty():
            ele = old_q.get()
            if (ele.get_info()[0] != model_id):
                self._q.put(ele)

        # save que to disk
        with open(self._config['que'],'wb') as queue_save_file:
            dill.dump(self._q, queue_save_file)

        # delete the model
        delete_model(model_id, self._config)

        return


class Continue(Resource):

    # init the resource
    def __init__(self, que: Queue, config):
        self._q = que
        self._op_type = 'continue'
        self._config = config

    # Continue the training of the classifiers with the specified id.
    def post(self, model_id: str):
        # Check if model exists
        if not model_id in SqliteDict(self._config['kv']).keys():
            abort_wrong_model_id(model_id)

        # abort if model is faulty
        if not check_model(model_id, self._config):
            abort_faulty_model(model_id)

        # Check if model was interruptd
        if (SqliteDict(self._config['kv'])[model_id]['status'] 
            != 'interrupted'):
            abort_wrong_op_type(model_id, self._op_type, 
                SqliteDict(self._config['kv'])[model_id][status])

        # put training request in the que
        self._q.put(QueueElement(model_id, self._op_type))

        # save que to disk
        with open(self._config['que'],'wb') as queue_save_file:
            dill.dump(self._q, queue_save_file)

        log(f'Put model {model_id} in queue to continue training')

        return


# API endpoint for interrupting the training to continue later
class Pause(Resource):

    # init the resource
    def __init__(self, stop: InterruptState, current_model_id: CurrentModel, 
        config, bux, que):
        self._q = que
        self._stop = stop
        self._current_model_id = current_model_id
        self._config = config
        self._bux = bux

    # Interrupt the training and save the model to continue it later
    def patch(self, model_id: str):
        # check if model exists
        if not model_id in SqliteDict(self._config['kv']).keys():
            abort_wrong_model_id(model_id)

        # delete all instances in que
        old_q = self._q
        self._q = Queue()

        while not old_q.empty():
            ele = old_q.get()
            if (ele.get_info()[0] != model_id):
                self._q.put(ele)

        # save que to disk
        with open(self._config['que'],'wb') as queue_save_file:
            dill.dump(self._q, queue_save_file)

        # If current model the specified one, send interuption
        if model_id == self._current_model_id.get_id():
            self._stop.set_state(1)
        else:
            bux.deliver_interrupt_message(model_id, True)
        return


# API endpoint for interrupting the training
class Interrupt(Resource):

    # init the resource
    def __init__(self, stop: InterruptState, current_model_id: CurrentModel, 
        config, bux, que):
        self._q = que
        self._bux = bux
        self._stop = stop
        self._current_model_id = current_model_id
        self._config = config

    # Interrupt the Training and discard the model.
    def delete(self, model_id: str):
        # check if model exists
        if not model_id in SqliteDict(self._config['kv']).keys():
            abort_wrong_model_id(model_id)

        # delete all instances in que
        old_q = self._q
        self._q = Queue()

        while not old_q.empty():
            ele = old_q.get()
            if (ele.get_info()[0] != model_id):
                self._q.put(ele)

        # save que to disk
        with open(self._config['que'],'wb') as queue_save_file:
            dill.dump(self._q, queue_save_file)

        # If current model the specified one, send interuption
        if model_id == self._current_model_id.get_id():
            self._stop.set_state(2)
        else:
            bux.deliver_interrupt_message(model_id, False)
        return


# API endpoint for classifying data wiht a specified model
class PredictText(Resource):

    # Init the resource
    def __init__(self, que: Queue, config):
        self._q = que
        self._op_type = 'predict_text'
        self._config = config

    # Predict data wiht a specified model
    def post(self, model_id: str):
        # check if model exists
        if not model_id in SqliteDict(self._config['kv']).keys():
            abort_wrong_model_id(model_id)

        # abort if model is faulty
        if not check_model(model_id, self._config):
            abort_faulty_model(model_id)

        # abort if no json
        if not request.is_json:
            return abort_not_json()

        req = request.json

        # abort if required parameter is not in json
        if 'text' not in req:
            abort_missing_parameter('text')

        # put prediction request in the que
        self._q.put(QueueElement(model_id, self._op_type, req['text']))

        # save que to disk
        with open(self._config['que'], 'wb') as queue_save_file:
            dill.dump(self._q, queue_save_file)
        log(f'Put model {model_id} in queue for text prediction')

        return

# API endpoint for classifying data wiht a specified model
class Predict(Resource):

    # Init the resource
    def __init__(self, que: Queue, config):
        self._q = que
        self._op_type = 'predict'
        self._config = config

    # Predict data wiht a specified model
    def post(self, model_id: str):
        # check if model exists
        if not model_id in SqliteDict(self._config['kv']).keys():
            abort_wrong_model_id(model_id)

        # check if model is faulty
        if not check_model(model_id, self._config):
            abort_faulty_model(model_id)

        # check if request has json body
        if not request.is_json:
            return abort_not_json()

        req = request.json

        # check if required param is in json
        if 'doc_ids' not in req:
            abort_missing_parameter('doc_ids')

        # put prediction request in the que
        self._q.put(QueueElement(model_id, self._op_type, req['doc_ids']))

        # save que to disk
        with open(self._config['que'],'wb') as queue_save_file:
            dill.dump(self._q, queue_save_file)

        log(f'Put model {model_id} in queue for prediction')

        return


# API endpoint for given some labeled data for testing to the model.
class Evaluate(Resource):

    # Init the resource
    def __init__(self, que: Queue, config):
        self._q = que
        self._op_type = 'evaluate'
        self._config = config

    # Evaluate the model with the given data and return some performance info
    def post(self, model_id: str):
        # check if model exists
        if not model_id in SqliteDict(self._config['kv']).keys():
            abort_wrong_model_id(model_id)

        # check if model is faulty
        if not check_model(model_id, self._config):
            abort_faulty_model(model_id)

        # check for json body
        if not request.is_json:
            return abort_not_json()

        req = request.json

        # check if required parameter in json
        if 'doc_ids' not in req:
            abort_missing_parameter('doc_ids')

        # Put evaluation request in que
        self._q.put(QueueElement(model_id, self._op_type, req['doc_ids']))

        # save que to disk
        with open(self._config['que'],'wb') as queue_save_file:
            dill.dump(self._q, queue_save_file)

        log(f'Put model {model_id} in queue for evaluation')

        return

# API endpoint for returning a list of all saved models.
class List(Resource):

    # Init the resource
    def __init__(self, config):
        self._config = config

    # Return a list of all saved models
    def get(self):
        model_list = []

        # occupy kv to get all info
        # should be fast enough
        # build list of all classifiers
        with SqliteDict(self._config['kv']) as db:
            for model_id in db.keys():
                model_list.append({'model_id': model_id, 
                    'model_name': db[model_id]['model_name'], 
                    'dataset_id': db[model_id]['dataset_id'], 
                    'status': db[model_id]['status']})

        return model_list


# API endpoint for training a new model.
class Train(Resource):

    # init the resource
    def __init__(self, que: Queue, config):
        self._q = que
        self._op_type = 'train'
        self._config = config

    # Train a new Classifier
    def post(self):
        # check for json file
        if not request.is_json:
            return abort_not_json()

        req = request.json

        # check if all required params are in json
        if 'model_name' not in req:
            abort_missing_parameter('model_name')

        if 'dataset_id' not in req:
            abort_missing_parameter('dataset_id')

        if 'train_ids' not in req:
            abort_missing_parameter('train_ids')
            
        if 'val_ids' not in req:
            abort_missing_parameter('val_ids') 

        if 'label_mapping' not in req:
            abort_missing_parameter('label_mapping') 

        # Generate a random model id
        model_id = str(uuid.uuid4())

        # No duplicate model ids :(
        while (model_id in SqliteDict(self._config['kv'])):
            model_id = str(uuid.uuid4())

        # model id and status to model info
        req['model_id'] = model_id
        req['status'] = 'in_que'

        #remove later. moch up for now
        req['label_mapping'] = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 
            'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}

        # Save the model info
        with SqliteDict(self._config['kv']) as db:
            db[model_id] = req
            db.commit()

        # Put training request in the que
        self._q.put(QueueElement(model_id, self._op_type))

        # save que to disk
        with open(self._config['que'],'wb') as queue_save_file:
            dill.dump(self._q, queue_save_file)

        log(f'Put model {model_id} in queue for training')

        # return id of new model
        return {'model_id':model_id}