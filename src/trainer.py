from transformers import Trainer
import torch

from queue import Queue
import time
import os
import shutil
from sqlitedict import SqliteDict

import logging

from src.training_help import *


# method that should be run as thread for training models
# it is given the q with the models that are supposed to be trained
def training_thread(q: Queue, stop: InterruptState):
	logging.debug('Training thread alive')
	while True:
		# if queue is empty: wait a second and check again
		#ugly. change!
		if q.empty():
			time.sleep(1)
		else:
			# reset stop
			logging.debug(f'Trainer stop: {id(stop)}')
			stop.set_state(0)

			# get the model id from the q
			model_id = q.get()

			logging.info(f'Got model {model_id} from the training queue')

			# update the model id with default values for missing values from request
			update_model_info(model_id)

			# get training arguments
			training_args = get_training_args(model_id)

			# get the data
			dh = DataHelper()
			data, num_labels = dh.get_data(model_id)

			# define the model
			model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels = num_labels)

			logging.debug('Model defined')

			# define the Trainer
			trainer = Trainer(
				model = model,
				args = training_args,
				train_dataset = data['train'],
				eval_dataset = data['test'],
				data_collator = dh.data_collator,
				tokenizer = dh.tokenizer,
				callbacks = [InterruptCallback(stop)]
				)

			logging.debug('Trainer defined')

			with SqliteDict('./distilBERT.sqlite') as db:
				model_info = db[model_id]
				model_info['status'] = 'training'
				model_info['num_labels'] = num_labels
				db[model_id] = model_info
				db.commit()

			logging.debug('Set model status to training.')
			logging.info(f'Starting training for model {model_id}')

			# train
			if (stop.get_state() == 0):
				#check if training is getting continued
				if (len(os.listdir(f'./checkpoints/{model_id}')) > 0):
					cp_list = os.listdir(f'./checkpoints/{model_id}')
					cp_list.sort(reverse = True)
					
					logging.debug(f'Starting to continue the traing for model {model_id} from checkpoint {cp_list[0]}')
					trainer.train(f'./checkpoints/{model_id}/{cp_list[0]}')

				#continue normally
				else:
					logging.info(f'Starting training for model {model_id}')
					trainer.train()
			else:
				logging.debug('training stopped before it started')

			#Training completed normally
			if (stop.get_state() == 0):
				logging.debug(f'Training done for model {model_id}')

				# save the model
				torch.save(model.state_dict(), f'models/{model_id}.pth')

				logging.debug(f'Saved model {model_id}')

				# set the model as trained
				with SqliteDict('./distilBERT.sqlite') as db:
					model_info = db[model_id]
					model_info['status'] = 'trained'
					db[model_id] = model_info
					db.commit()

				logging.debug('Set model status to trained')

				# remove the checkpoints
				if os.path.isdir(f'./checkpoints/{model_id}'):
					shutil.rmtree(f'./checkpoints/{model_id}')

					logging.debug('Removed checkpoints')

				logging.info(f'Training for model {model_id} finished')

			# Training interrupted
			elif (stop.get_state() == 1):
				# set the model as interrupted
				with SqliteDict('./distilBERT.sqlite') as db:
					model_info = db[model_id]
					model_info['status'] = 'interrupted'
					db[model_id] = model_info
					db.commit()

				logging.debug('Set model status to interrupted')

				logging.debug(f'model {model_id} interrupted')

			# Training interruptd and model to be deleted
			else:
				# remove the model from db
				with SqliteDict('./distilBERT.sqlite') as db:
					model_info = db.pop(model_id)
					db.commit()

				logging.debug(f'Deleted model {model_id} from kv-store')

				# remove the checkpoints
				if os.path.isdir(f'./checkpoints/{model_id}'):
					shutil.rmtree('./checkpoints/' + model_id)

					logging.debug('Removed checkpoints')

				logging.debug(f'model {model_id} interrupted and deleted')

