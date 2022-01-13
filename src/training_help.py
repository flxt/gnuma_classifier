from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments
from transformers import TrainerCallback

from sqlitedict import SqliteDict

import os
import shutil
import logging

# If not all needed infos where in training request
# Update key value model info with default values
def update_model_info(model_id):
	model_info = SqliteDict('./distilBERT.sqlite')[model_id]

	if 'learning_rate' not  in model_info:
		model_info['learning_rate'] = defaults.learning_rate

	if 'batch_size' not in model_info:
		model_info['batch_size'] = defaults.batch_size

	if 'epochs' not in model_info:
		model_info['epochs'] = defaults.epochs

	if 'warmupsteps' not in model_info:
		model_info['warmupsteps'] = defaults.warmupsteps

	if 'weight_decay' not in model_info:
		model_info['weight_decay'] = defaults.weight_decay

	if 'best_model' not in model_info:
		model_info['best_model'] = True

	#save to key value store
	with SqliteDict('./distilBERT.sqlite') as db:
		db[model_id] = model_info
		db.commit()

	logging.debug(f'Updated the info for model {model_id} with default values if necessary')


# class storing the default values
class defaults():
	learning_rate = 2e-5
	batch_size = 16
	epochs = 3
	warmupsteps = 400
	weight_decay = 0.0

# returns the training arguments. values are taken from from model_info
def get_training_args(model_id):
	model_info = SqliteDict('./distilBERT.sqlite')[model_id]

	logging.debug(f'Returning training arguments based on kv-store info for model {model_id}')

	return TrainingArguments(
		output_dir = './checkpoints/' + model_id,
		learning_rate = model_info['learning_rate'],
		per_device_train_batch_size = model_info['batch_size'],
		per_device_eval_batch_size = model_info['batch_size'],
		num_train_epochs = model_info['epochs'],
		weight_decay = model_info['weight_decay'],
		warmup_steps = model_info['warmupsteps'],
		load_best_model_at_end = model_info['best_model'],
		evaluation_strategy = 'steps')

class DataHelper():

	def __init__(self):
		self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
		self.data_collator = DataCollatorForTokenClassification(self.tokenizer)
		logging.debug('Set up tokenizer and data collector')

	# get the data and prepare it for 
	# for now it only loads the wnut_17 data set
	def get_data(self, model_id):
		wnut = load_dataset('wnut_17')

		data = wnut.map(self.tokenize_and_align_labels, batched=True)
		num_labels = len(wnut["train"].features[f"ner_tags"].feature.names)

		logging.debug(f'Prepared data for model {model_id}')

		return data, num_labels


	# method to tokenize and allign labels
	# taken from the hugging face documentation
	def tokenize_and_align_labels(self, examples):
		tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

		labels = []
		for i, label in enumerate(examples[f"ner_tags"]):
			word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
			previous_word_idx = None
			label_ids = []
			for word_idx in word_ids:                            # Set the special tokens to -100.
				if word_idx is None:
					label_ids.append(-100)
				elif word_idx != previous_word_idx:              # Only label the first token of a given word.
					label_ids.append(label[word_idx])

			labels.append(label_ids)

		tokenized_inputs["labels"] = labels

		logging.debug('Tokenizeda and alligned labels')
		return tokenized_inputs

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

#callback that check if training is supposed to be interrupted
class InterruptCallback(TrainerCallback):

	def __init__(self, stop: InterruptState):
		self._stop = stop

	#check after every training step if training should stop
	def on_step_end(self, args, state, control, **kwargs):
		if (self._stop.get_state() > 0):
			if (self._stop.get_state() == 1):
				control.should_evaluate = True
				control.should_save = True 
			control.should_training_stop = True 

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
