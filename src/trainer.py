from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

import time
from sqlitedict import SqliteDict


# method that should be run as thread for training models
# it is given the q with the models that are supposed to be trained
def training_tread(q):
	while True:
		# if queue is empty: wait a second and check again
		#ugly. change!
		if q.empty():
			time.sleep(1)
		else:
			# init model trainer
			mt = ModelTrainer()

			#update the model indo with default values
			mt.update_model_info()




class ModelTrainer():
	# default values as static variables
	default_learning_rate = 2e-5
	default_batch_size = 16
	default_epochs = 3
	default_warmupsteps = 400
	default_weight_decay = 0.01

	# define the class
	# model id is given
	# set up tokenizer
	def __init__(self, model_id):
		self._model_id = model_id
		self._tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
		self._data_collator = DataCollatorForTokenClassification(tokenizer)

		self.update_model_info()
		self._training_args = self.get_training_args()

		self._data, self.num_labels = self.get_data()

		self._model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=self.num_labels)

		self._trainer = Trainer(
			model = self._model,
			args = self.get_training_args,
			train_dataset = self._data['train'],
			eval_dataset = self._data['test'],
			data_collator = self._data_collator,
			tokenizer = self._tokenizer,
			)

	# if not all needed infos where in training request
	# update key value model info with default values
	def update_model_info()
		model_info = SqliteDict('./distilBERT.sqlite')[model_id]

		if 'learning_rate' not  inin model_info:
			model_info['learning_rate'] = default_learning_rate

		if 'batch_size' not in in model_info:
			model_info['batch_size'] = default_batch_size

		if 'epochs' not in model_info:
			model_info['epochs'] = default_epochs

		if 'warmupsteps' not in model_info:
			model_info['warmupsteps'] = default_warmupsteps

		if 'weight_decay' not in model_info:
			model_info['weight_decay'] = default_weight_decay

		#save to key value store
		with SqliteDict('./distilBERT.sqlite') as db:
			db[model_id] = request.json
			db.commit()

	# returns the training arguments. values are taken from from model_info
	def get_training_args():
		model_info = SqliteDict('./distilBERT.sqlite')[model_id]

		return TrainingArguments(
			output_dir = './models/' + self._model_id,
			evaluation_strategy = 'epoch',
			learning_rate = model_info['learning_rate'],
			per_device_train_batch_size = model_info['batch_size'],
			per_device_eval_batch_size = model_info['batch_size'],
			num_train_epochs = model_info['epochs'],
			weight_decay = model_info['weight_decay'],
			warmup_steps = model_info['warmupsteps'])


	# get the data and prepare it for 
	# for now it only loads the wnut_17 data set
	def get_data():
		wnut = load_dataset('wnut_17')

		data = wnut.map(tokenize_and_align_labels, batched=True)
		num_labels = len(wnut["train"].features[f"ner_tags"].feature.names)

		return data, num_labels

	# method to tokenize and allign labels
	# taken from the hugging face documentation
	def tokenize_and_align_labels(examples):
		tokenized_inputs = self._tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

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

	def train():
		self._trainer.train()