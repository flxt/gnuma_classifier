from flask import request
from flask_restful import Resource, abort

from sqlitedict import SqliteDict
from queue import Queue

import uuid

# Abort if a json file is expected, but not part of the request
def abort_not_json():
    abort(400, message='Only accepting requests with mime type application/json.')

# Abort if expected parameter is missing from the request.
def abort_missing_parameter(parameter_name: str):
    abort(400, message=f'Expected "{parameter_name}" to be part of the request body.')

# Abort if specified model doesnt exist.
def abort_wrong_model_id(model_id: str):
	abort(400, message=f'No model with ÍD "{model_id}" exists.')

# API enpoint where only a model ID is given
class Base(Resource):
	# Return the decription and more info for the model with the given id
	def get(self, model_id: str):
		# check if model exists
		if not model_id in SqliteDict('./distilBERT.sqlite').keys():
			abort_wrong_model_id(model_id)

		# get info for model from db
		model_info = SqliteDict('./distilBERT.sqlite')[model_id]

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

		# todo remove model file

		return model_info

	# Continue the training of the classifiers with the specified id.
	def put(self, model_id: str):
		#todo
		return 'TODO: Continue the training of the specified model'

# API endpoint for interrupting the training
class Interrupt(Resource):
	# Interrupt the training and save the model to continue it later
	def put(self):
		#todo
		return 'TODO: Interrupt training and save model'

	# Interrupt the Training and discard the model.
	def delete(self):
		#todo
		return 'TODO: Interrupt training and yeet model'

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
			db[model_id] = request.json
			db[model_id]['model_id'] = model_id
			db.commit()

		# put training request in the que
		self._q.put(model_id)

		return {'status':'put in list', 'model_id':model_id}