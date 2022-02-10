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
def get_training_args(model_id):
    model_info = SqliteDict('./distilBERT.sqlite')[model_id]

    log(f'Returning training arguments based on kv-store info for model '
        f'{model_id}', 'DEBUG')

    return TrainingArguments(
        output_dir = f'./checkpoints/{model_id}',
        learning_rate = model_info['learning_rate'],
        per_device_train_batch_size = model_info['batch_size'],
        per_device_eval_batch_size = model_info['batch_size'],
        num_train_epochs = model_info['epochs'],
        weight_decay = model_info['adam_weigth_decay'],
        warmup_ratio = model_info['warmup_ratio'],
        load_best_model_at_end = model_info['best_model'],
        evaluation_strategy = 'steps',
        save_steps = model_info['steps'],
        eval_steps = model_info['steps'],
        adam_beta1 = model_info['adam_beta1'],
        adam_beta2 = model_info['adam_beta2'],
        adam_epsilon = model_info['adam_epsilon'],
        )

class DataHelper():

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased")
        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)
        log('Set up tokenizer and data collector', 'DEBUG')

        with open('service_address.json') as json_file:
            dat = json.load(json_file)
            self._doc_address = dat['doc_address']

        log(self._doc_address)

    # methods gets a document from the document service
    def get_doc(self, doc_id):
        response = requests.get(f'http://{self._doc_address}/api/v1/'
            f'documents/{doc_id}')
        sentences = response.json()['sentences']

        tokens = []
        ner_tags = []
        ids = []

        for sentence in sentences:
            tok_temp = []
            ner_temp = []
            for token in sentence['tokens']:
                tok_temp.append(token['token'])
                ner_temp.append(get_int_labels(token['nerTag']))

            tokens.append(tok_temp)
            ner_tags.append(ner_temp)
            ids.append(sentence['id'])

        return tokens, ner_tags, ids

    # get the data and prepare it for 
    # for now it only loads the wnut_17 data set
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


        ds = Dataset.from_dict({'id': ids, 'tokens': tokens, 
            'ner_tags': ner_tags})

        data = ds.map(self.tokenize_and_align_labels)

        return data


    # method to tokenize and allign labels
    # taken from the hugging face documentation
    def tokenize_and_align_labels(self, examples):
        log(examples['ner_tags'])

        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, 
            is_split_into_words=True)

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


        tokenized_inputs["labels"] = label_ids

        log('Tokenized and alligned labels', 'DEBUG')
        log(tokenized_inputs)
        return tokenized_inputs

#callback that check if training is supposed to be interrupted
class InterruptCallback(TrainerCallback):

    def __init__(self, stop: InterruptState):
        self._stop = stop

    # check after every training step if training should stop
    def on_step_end(self, args, state, control, **kwargs):
        if (self._stop.get_state() > 0):
            if (self._stop.get_state() == 1):
                control.should_evaluate = True
                control.should_save = True 
            control.should_training_stop = True

# callback that check if training is supposed to be interrupted
class EvaluateCallback(TrainerCallback):

    def __init__(self, bux: BunnyPostalService, model_id: str):
        self._bux = bux
        self._model_id = model_id
        self._metrics = {'eval_loss': -1, 'eval_accuracy': -1, 'eval_f1': -1}
        self._finished = False

    # check after every training step if training should stop
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self._metrics['eval_loss'] = metrics['eval_loss']
        self._metrics['eval_accuracy'] = metrics['eval_accuracy']
        self._metrics['eval_f1'] = metrics['eval_f1']

        self._bux.give_update(self._model_id, self._finished, 
            state.global_step, state.max_steps, state.epoch, self._metrics)

    # first update when the training starts
    def on_train_begin(self, args, state, control, **kwargs):
        self._bux.give_update(self._model_id, self._finished, 
            state.global_step, state.max_steps, state.epoch, self._metrics)

    def on_train_end(self, args, state, control, **kwargs):
        self._finished = True


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

def get_int_labels(ner_tag):
     tags = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
     return tags[ner_tag]