from flask import request
from flask_restful import Resource, abort

from sqlitedict import SqliteDict
from queue import Queue

import os
import logging

import uuid

# Abort if a json file is expected, but not part of the request
def abort_not_json():
    abort(400, message='Only accepting requests with mime type application/json.')
    logging.error('Only accepting requests with mime type application/json.')

# Abort if expected parameter is missing from the request.
def abort_missing_parameter(parameter_name: str):
    abort(400, message=f'Expected "{parameter_name}" to be part of the request body.')
    logging.error(f'Expected "{parameter_name}" to be part of the request body.')

# Abort if specified model doesnt exist.
def abort_wrong_model_id(model_id: str):
	abort(400, message=f'No model with ÍD "{model_id}" exists.')
	logging.error('No model with ÍD "{model_id}" exists.')

# API enpoint where only a model ID is given
class Base(Resource):
	# Return the decription and more info for the model with the given id
	def get(self, model_id: str):
		# check if model exists
		if not model_id in SqliteDict('./distilBERT.sqlite').keys():
			abort_wrong_model_id(model_id)

		# get info for model from db
		model_info = SqliteDict('./distilBERT.sqlite')[model_id]

		logging.debug('Model from kv-store returned')

		return model_info

	# Delete the model with the specified id
	def delete(self, model_id: str):
		# check if model exists
		if not model_id in SqliteDict('./distilBERT.sqlite').keys():
			abort_wrong_model_id(model_id)

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

	# Continue the training of the classifiers with the specified id.
	def put(self, model_id: str):
		# check if model exists
		if not model_id in SqliteDict('./distilBERT.sqlite').keys():
			abort_wrong_model_id(model_id)

		#todo
		return 'TODO: Continue the training of the specified model'

# API endpoint for interrupting the training
class Interrupt(Resource):

	# init the resource
	def __init__(self, stop: bool):
		self._stop = stop

	# Interrupt the training and save the model to continue it later
	def put(self):
		self._stop = 1
		return 'Interrupted training'

	# Interrupt the Training and discard the model.
	def delete(self):
		self._stop = 2
		return 'Interrupted training and deleted the model'

# API endpoint for classifying data wiht a specified model
class Classify(Resource):
	# classifying data wiht a specified model
	def get(self, model_id: str):
		#todo
		return 'TODO: Classify data with specified model'

# API endpoint for given some labeled data for testing to the model.
class Test(Resource):
	# Test the model with the given data and return some performance information
	def get(self, model_id: str):
		#todo
		return 'TODO: Test the model with given data'

# API endpoint for returning a list of all saved models.
class List(Resource):
	# Return a list of all saved models
	def get(self):
		model_list = {}

		with SqliteDict('./distilBERT.sqlite') as db:
			for model_id in db.keys():
				if 'Description' in db[model_id]:
					model_list[model_id] = db[model_id]['Description']
				else:
					model_list[model_id] = 'no description'

		logging.debug('Returned a list of all models')

		return model_list

# API endpoint for training a new model.
class Train(Resource):

	# init the resource
	def __init__(self, que: Queue):
		self._q = que

	# Train a new Classifier
	def post(self):
		# check for json file
		if not request.is_json:
			return abort_not_json()

		# generate a random model id
		model_id = str(uuid.uuid4())

		# save the model info
		with SqliteDict('./distilBERT.sqlite') as db:
			req = request.json
			req['model_id'] = model_id
			req['trainend'] = False
			db[model_id] = req
			db.commit()

		logging.debug(f'Put model {model_id} in kv-store')

		# put training request in the que
		self._q.put(model_id)

		logging.info(f'Put model {model_id} in training queue')

		return {'status':'in_list', 'model_id':model_id}