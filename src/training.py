from transformers import Trainer, AutoModelForTokenClassification, AutoTokenizer
import torch

from sqlitedict import SqliteDict

from queue import Queue
import time
import logging
import os
import json

from src.training_utils import DataHelper, get_training_args, InterruptCallback, EvaluateCallback, compute_metrics
from src.utils import InterruptState, remove_checkpoints, delete_model, check_model
from src.bunny import BunnyPostalService

# method that should be run as thread for training models
# it is given the q with the models that are supposed to be trained
def training_thread(q: Queue, stop: InterruptState, bux: BunnyPostalService, current_model_id):
    logging.debug('Training thread alive')
    while True:
        # If queue is empty: wait a second and check again
        if q.empty():
            time.sleep(1)
        else:
            # reset stop
            stop.set_state(0)

            # Get the model id and op type from the first element in the queue.
            ele = q.get()
            model_id, op_type = ele.get_info()

            # set current model id
            current_model_id = model_id

            logging.info(f'Got model {model_id} with operation type {op_type} from the queue')

            if (op_type == 'train'):
                train_new_model(model_id, stop, bux)
            elif (op_type == 'continue'):
                continue_training_model(model_id, stop, bux)
            elif (op_type == 'evaluate'):
                data_id = ele.get_text()
                evaluate_model(model_id, stop, bux, data_id)
            elif (op_type == 'predict_text'):
                text = ele.get_text() 
                predict_text(model_id, stop, bux, text)
            elif (op_type == 'predict'):
                doc_id =  ele.get_text()
                predict_data(model_id, stop, bux, doc_id)
            else:
                logging.error(f'Wrong operation type {op_type} for model {model_id}')

            current_model_id = None

# Call this method to train a new model
def train_new_model(model_id: str, stop: InterruptState, bux: BunnyPostalService):
    # Check if default values are needed and set them accordingly
    model_info = SqliteDict('./distilBERT.sqlite')[model_id]
    
    with open('./defaults.json') as json_file:
        defaults = json.load(json_file)

    for key in defaults.keys():
        if key not in model_info:
            model_info[key] = defaults[key]

    #save to key value store
    with SqliteDict('./distilBERT.sqlite') as db:
        db[model_id] = model_info
        db.commit()

    logging.debug(f'Updated the info for model {model_id} with default values if necessary')

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
            callbacks = [InterruptCallback(stop), EvaluateCallback(bux, model_id)],
            compute_metrics = compute_metrics
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

        trainer.evaluate()

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
def continue_training_model(model_id: str, stop: InterruptState, bux: BunnyPostalService):
    # Get a list of all checkpoints
    cp_list = os.listdir(f'./checkpoints/{model_id}')
    # Sort the list in a way that the last checkpoint is in the first spot.
    cp_list.sort(reverse = True)

    #check for correct status
    if SqliteDict('./distilBERT.sqlite')[model_id]['status'] != 'interrupted' or not check_model(model_id):
        logging.error(f'model {model_id} cant be continued')
        # todo: some error message
        return

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
            callbacks = [InterruptCallback(stop), EvaluateCallback(bux, model_id)],
            compute_metrics = compute_metrics
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

        # Run final evaluation
        trainer.evaluate()

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
def evaluate_model(model_id: str, stop: InterruptState, bux: BunnyPostalService, data_id: str):
    #check for correct status
    if SqliteDict('./distilBERT.sqlite')[model_id]['status'] != 'trained' or not check_model(model_id):
        logging.error(f'model {model_id} cant be evaluated')
        # todo: some error message
        return

    # Get the training Arguments
    training_args = get_training_args(model_id)

    # Get the evaluation data
    dh = DataHelper()
    data, num_labels = dh.get_data(model_id)

    # Define a new model
    model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels = num_labels)

    # Load trained weights
    model.load_state_dict(torch.load(f'models/{model_id}.pth'))
    model.eval()

    # Define the trainer
    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = dh.data_collator,
        tokenizer = dh.tokenizer,
        compute_metrics = compute_metrics
        )

    # Run the Evaluation
    logging.info(f'Beginning evaluation for model {model_id}')
    out = trainer.evaluate(eval_dataset = data['train'])

    metrics = {}
    metrics['eval_loss'] = out['eval_loss']
    metrics['eval_accuracy'] = out['eval_accuracy']
    metrics['eval_f1'] = out['eval_f1']


    bux.deliver_eval_results(model_id, metrics)

    logging.info(f'Evaluated model {model_id}.')


# Call this method to predict text with a model
def predict_text(model_id: str, stop: InterruptState, bux: BunnyPostalService, sequence: str):
    #check for correct status
    if SqliteDict('./distilBERT.sqlite')[model_id]['status'] != 'trained' or not check_model(model_id):
        logging.error(f'model {model_id} cant be predicted')
        # todo: some error message
        return

    # Define a new model
    model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels = SqliteDict('./distilBERT.sqlite')[model_id]['num_labels'])

    # Load trained weights
    model.load_state_dict(torch.load(f'models/{model_id}.pth'))
    model.eval()

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # convert input to tokens
    inputs = tokenizer(sequence, return_tensors="pt")
    tokens = inputs.tokens()

    # get model output
    outputs = model(**inputs).logits

    # get actual predictions
    predictions = torch.argmax(outputs, dim=2)

    logging.info(f'Text prediction for model {model_id} done.')

    # convert labeled data to a list of tuples
    out_list = []
    for token, prediction in zip(tokens, predictions[0].numpy()):
        out_list.append((token, model.config.id2label[prediction]))

    # bux send results to rabbit mq
    bux.deliver_text_prediction(model_id, out_list)


# Call this method to predict text with a model
def predict_data(model_id: str, stop: InterruptState, bux: BunnyPostalService, doc_id: str):
    #check for correct status
    if SqliteDict('./distilBERT.sqlite')[model_id]['status'] != 'trained' or not check_model(model_id):
        logging.error(f'model {model_id} cant be predicted')
        # todo: some error message
        return

    # Define a new model
    model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels = SqliteDict('./distilBERT.sqlite')[model_id]['num_labels'])

    # Load trained weights
    model.load_state_dict(torch.load(f'models/{model_id}.pth'))
    model.eval()

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # todo