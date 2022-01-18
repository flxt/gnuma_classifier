from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments
from transformers import TrainerCallback

from sqlitedict import SqliteDict
import numpy as np

import logging

from src.bunny import BunnyPostalService
from src.utils import InterruptState

# returns the training arguments. values are taken from from model_info
def get_training_args(model_id):
    model_info = SqliteDict('./distilBERT.sqlite')[model_id]

    logging.debug(f'Returning training arguments based on kv-store info for model {model_id}')

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

        self._bux.give_update(self._model_id, self._finished, state.global_step, state.max_steps, state.epoch, self._metrics)

    # first update when the training starts
    def on_train_begin(self, args, state, control, **kwargs):
        self._bux.give_update(self._model_id, self._finished, state.global_step, state.max_steps, state.epoch, self._metrics)

    def on_train_end(self, args, state, control, **kwargs):
        self._finished = True


# method computing the metrics
def compute_metrics(pred):
    labels = pred.label_ids.flatten()

    preds = pred.predictions.argmax(-1).flatten()
    
    for idx, val in enumerate(labels):
        if val == -100:
            preds[idx] = -100


    logging.info(preds)
    logging.info(labels)
    f1 = f1_score(labels.flatten(), preds.flatten(), average = 'macro')
    acc = accuracy_score(labels.flatten(), preds.flatten())
    return {
        'accuracy': acc,
        'f1': f1
    }