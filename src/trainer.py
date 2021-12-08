from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

import time
from sqlitedict import SqliteDict


# method that should be run as thread for training models
# it is given the q with the models that are supposed to be trained
def training_thread(q):
	while True:
		# if queue is empty: wait a second and check again
		#ugly. change!
		if q.empty():
			time.sleep(1)
		else:
			# get the model id from the q
			model_id = q.get()

			# update the model id with default values for missing values from request
			update_model_info(model_id)

			# get training arguments
			training_args = get_training_args(model_id)

			# get the data
			dh = DataHelper()
			data, num_labels = dh.get_data(model_id)

			# define the model
			model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels = num_labels)

			# define the Trainer
			trainer = Trainer(
				model = model,
				args = training_args,
				train_dataset = data['train'],
				eval_dataset = data['test'],
				data_collator = dh.data_collator,
				tokenizer = dh.tokenizer,
				)

			# train
			trainer.train()

# if not all needed infos where in training request
# update key value model info with default values
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

	#save to key value store
	with SqliteDict('./distilBERT.sqlite') as db:
		db[model_id] = model_info
		db.commit()

# class storing the default values
class defaults():
	learning_rate = 2e-5
	batch_size = 16
	epochs = 3
	warmupsteps = 400
	weight_decay = 0.01

# returns the training arguments. values are taken from from model_info
def get_training_args(model_id):
	model_info = SqliteDict('./distilBERT.sqlite')[model_id]

	return TrainingArguments(
		output_dir = './models/' + model_id,
		evaluation_strategy = 'epoch',
		learning_rate = model_info['learning_rate'],
		per_device_train_batch_size = model_info['batch_size'],
		per_device_eval_batch_size = model_info['batch_size'],
		num_train_epochs = model_info['epochs'],
		weight_decay = model_info['weight_decay'],
		warmup_steps = model_info['warmupsteps'])

class DataHelper():

	def __init__(self):
		self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
		self.data_collator = DataCollatorForTokenClassification(self.tokenizer)

	# get the data and prepare it for 
	# for now it only loads the wnut_17 data set
	def get_data(self, model_id):
		wnut = load_dataset('wnut_17')

		data = wnut.map(self.tokenize_and_align_labels, batched=True)
		num_labels = len(wnut["train"].features[f"ner_tags"].feature.names)

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
		return tokenized_inputs
