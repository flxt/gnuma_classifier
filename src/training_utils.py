from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments
from transformers import TrainerCallback

from sqlitedict import SqliteDict
import numpy as np
import json
import requests

from src.bunny import BunnyPostalService
from src.utils import InterruptState, log

# returns the training arguments. values are taken from from model_info
def get_training_args(model_id, config):
    # get hyper parameters from model info
    hyper_parameters = SqliteDict(config['kv'])[model_id]['hyper_parameters']

    log(f'Returning training arguments based on kv-store info for model '
        f'{model_id}', 'DEBUG')

    # return training arguments based on those hyper parameters
    return TrainingArguments(
        output_dir = f'{config["checkpoints"]}{model_id}',
        learning_rate = hyper_parameters['learning_rate'],
        per_device_train_batch_size = hyper_parameters['batch_size'],
        per_device_eval_batch_size = hyper_parameters['batch_size'],
        num_train_epochs = hyper_parameters['epochs'],
        weight_decay = hyper_parameters['adam_weigth_decay'],
        warmup_ratio = hyper_parameters['warmup_ratio'],
        load_best_model_at_end = hyper_parameters['best_model'],
        evaluation_strategy = 'steps',
        save_steps = hyper_parameters['steps'],
        eval_steps = hyper_parameters['steps'],
        adam_beta1 = hyper_parameters['adam_beta1'],
        adam_beta2 = hyper_parameters['adam_beta2'],
        adam_epsilon = hyper_parameters['adam_epsilon'],
        # take train error every x steps
        logging_steps = 10,
        )

# class for loading documents from the doc service
# and preparing them fro training
class DataHelper():

    # initialize the class
    # takes model id and config
    def __init__(self, model_id, config):
        # set up tokenizer and data clollator
        # the add prefix part is necessary for roberta
        self.tokenizer = AutoTokenizer.from_pretrained(config['model'], 
            add_prefix_space=True)
        self.data_collator = DataCollatorForTokenClassification(
            self.tokenizer)
        log('Set up tokenizer and data collector', 'DEBUG')

        # get dictionary for converting labels from config
        self._tags = SqliteDict(config['kv'])[model_id]['label_mapping']

    # methods gets a document from the document service
    # takes a single document
    def get_doc(self, doc_id):
        # get the document from the service
        response = requests.get(doc_id)

        # get the sentences 
        sentences = response.json()['sentences']

        # convert them into a format that can be converted into a
        # hugging face data set easily
        tokens = []
        ner_tags = []
        ids = []

        for sentence in sentences:
            tok_temp = []
            ner_temp = []
            for token in sentence['tokens']:
                tok_temp.append(token['token'])
                ner_temp.append(self._convert_labels(token['nerTag']))

            tokens.append(tok_temp)
            ner_tags.append(ner_temp)
            ids.append(sentence['id'])

        return tokens, ner_tags, ids

    # get the data and prepare it for 
    # takes a list of documents
    def get_data(self, doc_ids):
        tokens = []
        ner_tags = []
        ids = []
        #rotate through all documents to build a data set
        for doc_id in doc_ids:
            tok_temp, ner_temp, id_temp = self.get_doc(doc_id)
            tokens += tok_temp
            ner_tags += ner_temp
            ids += id_temp

        # create the data set
        ds = Dataset.from_dict({'id': ids, 'tokens': tokens, 
            'ner_tags': ner_tags})

        # tokenize the data set and allign the labels
        data = ds.map(self.tokenize_and_align_labels)

        # return the data set
        return data

    # get the data and prepare it for 
    # for prediction => no labels
    # takes a single doc address
    def get_data_pred(self, doc_ids):
        # get doc and extract sentences
        response = requests.get(doc_id)
        sentences = response.json()['sentences']

        # convert them to create a data set
        tokens = []
        ids = []
        
        #
        for sentence in sentences:
            tok_temp = []
            for token in sentence['tokens']:
                tok_temp.append(token['token'])

            tokens.append(tok_temp)
            ids.append(sentence['id'])

        # create a data set
        ds = Dataset.from_dict({'id': ids, 'tokens': tokens})

        # tokenize the data set
        data = ds.map(self.tokenize)

        # return the data set
        return data

    #t okenize for predict data
    def tokenize(self, examples):
        return self.tokenizer(examples["tokens"], truncation=True, 
            is_split_into_words=True)

    # method to tokenize and allign labels
    # partly taken from the hugging face documentation
    def tokenize_and_align_labels(self, examples):
        #tokenize the input
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, 
            is_split_into_words=True)

        # allign the labels
        label_ids = []
        word_ids = tokenized_inputs.word_ids()  
        previous_word_idx = None

        for word_idx in word_ids:
            # Set the special tokens to -100.                          
            if word_idx is None:
                label_ids.append(-100)
            # Only label the first token of a given word.
            elif word_idx != previous_word_idx:
          
                label_ids.append(examples['ner_tags'][word_idx])


        # add alligned labels
        tokenized_inputs["labels"] = label_ids

        # return result
        return tokenized_inputs

    # convert str tag to int
    def _convert_labels(self, ner_tag):
        return self._tags[ner_tag]


#callback that check if training is supposed to be interrupted
class InterruptCallback(TrainerCallback):

    # init the callback. needs an interrupt state
    def __init__(self, stop: InterruptState):
        self._stop = stop

    # check after every training step if training should stop
    def on_step_end(self, args, state, control, **kwargs):
        # check if training should stop
        if (self._stop.get_state() > 0):
            # save a checkpoint if pause
            if (self._stop.get_state() == 1):
                control.should_evaluate = False
                control.should_save = True 
            # set the stop flag
            control.should_training_stop = True

# callback that check if training is supposed to be interrupted
class EvaluateCallback(TrainerCallback):

    # callback that send regular progress updates during training
    # needs the bunny postal service and a model id
    def __init__(self, bux: BunnyPostalService, model_id: str):
        self._bux = bux
        self._model_id = model_id
        self._metrics = {'train_loss': -1, 'eval_loss': -1, 
            'eval_accuracy': -1, 'eval_f1': -1}
        self._finished = False

    # on evaluate up date the metrics for evaluation
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self._metrics['eval_loss'] = metrics['eval_loss']
        self._metrics['eval_accuracy'] = metrics['eval_accuracy']
        self._metrics['eval_f1'] = metrics['eval_f1']

    # on log update the train loss
    def on_log(self, args, state, control, logs, **kwargs):
        # Not always in to log file
        # My guess is it is not there if there was an evaluation step
        if 'loss' in logs:
            self._metrics['train_loss'] = logs['loss']

    # send updates on a regular basis
    def on_step_begin(self, args, state, control, **kwargs):
        # update every 10 steps
        if(state.global_step % 10 == 0):
            self._bux.give_update(self._model_id, self._finished, 
                state.global_step, state.max_steps, state.epoch, 
                self._metrics)

        if(state.global_step == state.max_steps - 1):
            control.should_evaluate = True
            self._finished = True

    # training ended => set finished flag
    def on_train_end(self, args, state, control, **kwargs):
        self._bux.give_update(self._model_id, self._finished, 
            state.global_step, state.max_steps, state.epoch, self._metrics)


# method computing the metrics
def compute_metrics(pred):
    #flatten the arrays
    labels = pred.label_ids.flatten()
    preds = pred.predictions.argmax(-1).flatten()

    #remove the padding
    preds = preds[labels != -100]
    labels = labels[labels != -100]

    #calculate the accuracy
    acc = accuracy_score(labels, preds)
    
    # calculate f1 score
    f1 = f1_score(labels, preds, average = 'macro')

    return {
        'accuracy': acc,
        'f1': f1
    }