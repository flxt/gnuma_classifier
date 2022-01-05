from transformers import Trainer
import torch

from sqlitedict import SqliteDict

from queue import Queue
import time
import logging

from src.training_help import *


# method that should be run as thread for training models
# it is given the q with the models that are supposed to be trained
def training_thread(q: Queue, stop: InterruptState):
	logging.debug('Training thread alive')
	while True:
		# If queue is empty: wait a second and check again
		if q.empty():
			time.sleep(1)
		else:
			# reset stop
			stop.set_state(0)

			# Get the model id and op type from the first element in the queue.
			model_id, op_type = q.get().get_info()

			logging.info(f'Got model {model_id} with operation type {op_type} from the queue')

			if (op_type == 'train'):
				train_new_model(model_id, stop)
			elif (op_type == 'continue'):
				continue_training_model(model_id, stop)
			elif (op_type == 'evaluate'):
				evaluate_model(model_id, stop)
			elif (op_type == 'predict'):
				predict_with_model(model_id, stop)
			else:
				logging.error(f'Wrong operation type {op_type} for model {model_id}')

# Call this method to train a new model
def train_new_model(model_id, stop):
	# Check if default values are needed and set them accordingly
	update_model_info(model_id)

	# Get the training Arguments
	training_args = get_training_args(model_id)

	# Get the data
	dh = DataHelper()
	data, num_labels = dh.get_data(model_id)

	# Define a new model
	model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels = num_labels)

	# Define the trainer
	trainer = Trainer(
			model = model,
			args = training_args,
			train_dataset = data['train'],
			eval_dataset = data['test'],
			data_collator = dh.data_collator,
			tokenizer = dh.tokenizer,
			callbacks = [InterruptCallback(stop)]
			)

	# Update the model info that the model is training
	with SqliteDict('./distilBERT.sqlite') as db:
		model_info = db[model_id]
		model_info['status'] = 'training'
		model_info['num_labels'] = num_labels
		db[model_id] = model_info
		db.commit()

	# Start training the model if no interruption
	logging.info(f'Starting the training for model {model_id}')
	trainer.train()

	# Training done
	# Case: Training finished normally
	if (stop.get_state() == 0):
		# Save the model
		torch.save(model.state_dict(), f'models/{model_id}.pth')

		# Remove the checkpoints because either the best model is already loaded or the final model was the goal
		remove_checkpoints(model_id)

		# Update the model info is trained
		with SqliteDict('./distilBERT.sqlite') as db:
			model_info = db[model_id]
			model_info['status'] = 'trained'
			db[model_id] = model_info
			db.commit()

		logging.info(f'Training for model {model_id} finished.')

	# Case: Training was interrupted
	elif (stop.get_state() == 1): 
		# Update the model info that the model was interrupted
		with SqliteDict('./distilBERT.sqlite') as db:
			model_info = db[model_id]
			model_info['status'] = 'interrupted'
			db[model_id] = model_info
			db.commit()

		logging.info(f'Training of model {model_id} was interrupted.')

	# Case: Training interrupted and model to be deleted
	else:
		delete_model(model_id)

		logging.info(f'Training of model {model_id} was interrupted and the model was deleted.')

# Call this method to continue the training of a model.
def continue_training_model(model_id, stop):
	# Get a list of all checkpoints
	cp_list = os.listdir(f'./checkpoints/{model_id}')
	# Sort the list in a way that the last checkpoint is in the first spot.
	cp_list.sort(reverse = True)

	# If there are no checkpoints something went wrong.
	# The training can't be continued.
	if (len(cp_list) == 0):
		logging.error(f'Training for model {model_id} can not be continued.')

	# If there is at least one check point continue the training.
	else:
		# Get the data
		dh = DataHelper()
		data, num_labels = dh.get_data(model_id)

		# Define a new model
		model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels = num_labels)

		# Define the trainer
		trainer = Trainer(
				model = model,
				args = training_args,
				train_dataset = data['train'],
				eval_dataset = data['test'],
				data_collator = dh.data_collator,
				tokenizer = dh.tokenizer,
				callbacks = [InterruptCallback(stop)]
				)

		# Update the model info that the model is training
		with SqliteDict('./distilBERT.sqlite') as db:
			model_info = db[model_id]
			model_info['status'] = 'training'
			model_info['num_labels'] = num_labels
			db[model_id] = model_info
			db.commit()

		# Continue training the model if no interruption
		logging.info(f'Continueing the training for model {model_id}')
		trainer.train(f'./checkpoints/{cp_list[0]}')

		# Training done
		# Case: Training finished normally
		if (stop.get_state() == 0):
			# Save the model
			torch.save(model.state_dict(), f'models/{model_id}.pth')

			# Remove the checkpoints because either the best model is already loaded or the final model was the goal
			remove_checkpoints(model_id)

			# Update the model info is trained
			with SqliteDict('./distilBERT.sqlite') as db:
				model_info = db[model_id]
				model_info['status'] = 'trained'
				db[model_id] = model_info
				db.commit()

			logging.info(f'Training for model {model_id} finished.')

		# Case: Training was interrupted
		elif (stop.get_state() == 1): 
			# Update the model info that the model was interrupted
			with SqliteDict('./distilBERT.sqlite') as db:
				model_info = db[model_id]
				model_info['status'] = 'interrupted'
				db[model_id] = model_info
				db.commit()

			logging.info(f'Training of model {model_id} was interrupted.')

		# Case: Training interrupted and model to be deleted
		else:
			delete_model(model_id)

			logging.info(f'Training of model {model_id} was interrupted and the model was deleted.')


# Call this method evaluate a model
def evaluate_model(model_id, stop):
	pass


# Call this method to predict with a model
def predict_with_model(model_id, stop):
	pass

			




		

