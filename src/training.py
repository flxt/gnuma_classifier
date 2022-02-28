from transformers import Trainer, AutoModelForTokenClassification
from transformers import AutoTokenizer
import torch
import numpy as np

from sqlitedict import SqliteDict

from queue import Queue
import time
import os
import json
import dill

from src.training_utils import DataHelper, get_training_args 
from src.training_utils import InterruptCallback
from src.training_utils import EvaluateCallback, compute_metrics
from src.utils import InterruptState, remove_checkpoints, delete_model
from src.utils import check_model, log, CurrentModel
from src.bunny import BunnyPostalService

# method that should be run as thread for training models
def training_thread(q: Queue, stop: InterruptState, 
    bux: BunnyPostalService, current_model_id: CurrentModel, config):

    log('Training thread alive', 'DEBUG')
    
    while True:
        # If queue is empty: wait a second and check again
        if q.empty():
            time.sleep(1)
        else:
            # Get the model id and op type from the first element in the queue
            ele = q.get()

            # save que to disk
            with open(config['que'],'wb') as queue_save_file:
                dill.dump(q, queue_save_file)

            # get model id and operation type
            model_id, op_type = ele.get_info()

            # set current model id
            current_model_id.set_id(model_id)

            log(f'Got model {model_id} with operation type'
                f'{op_type} from the queue')

            # If an error happens during traing => stop it and go to next
            # element in que
            try:
                # choose the correct method to run based on operation type
                if (op_type == 'train'):
                    train_new_model(model_id, stop, bux, config)
                elif (op_type == 'continue'):
                    continue_training_model(model_id, stop, bux, config)
                elif (op_type == 'evaluate'):
                    data_id = ele.get_text()
                    evaluate_model(model_id, stop, bux, data_id, config)
                elif (op_type == 'predict_text'):
                    text = ele.get_text() 
                    predict_text(model_id, stop, bux, text, config)
                elif (op_type == 'predict'):
                    doc_id =  ele.get_text()
                    predict_data(model_id, stop, bux, doc_id, config)
                else:
                    # this should never happen
                    log(f'Wrong operation type {op_type} for '
                        f'model {model_id}', 'ERROR')

                    bux.deliver_error_message(model_id, 
                        f'Wrong operation type {op_type} '
                        f'for model {model_id}')
            except Exception as e:
                # Very rudementary for now
                # Error Occurs => cancel training 
                log(f'Excpetion occured during training: {e}', 'ERROR')
                
                bux.deliver_error_message(model_id, e)

            # reset current model
            current_model_id.set_id('')

            # send interrupt message to rabbit mq if training was interrupted
            if (stop.get_state() == 1):
                bux.deliver_interrupt_message(model_id, True)
            elif (stop.get_state() == 2):
                bux.deliver_interrupt_message(model_id, False)

            # reset stop state
            stop.set_state(0)


# Call this method to train a new model
def train_new_model(model_id: str, stop: InterruptState, 
    bux: BunnyPostalService, config):

    # get model info from kv store
    model_info = SqliteDict(config['kv'])[model_id]

    # Check if default values are needed and set them accordingly
    for k, v in config['defaults'].items():
        if k not in model_info['hyper_parameters']:
            model_info['hyper_parameters'][k] = v

    #save to key value store
    with SqliteDict(config['kv']) as db:
        db[model_id] = model_info
        db.commit()

    log(f'Updated the info for model {model_id} with default' 
        f'values if necessary')

    # Get the training Arguments
    training_args = get_training_args(model_id, config)

    # Get the data
    dh = DataHelper(model_id, config)
    train_data = dh.get_data(model_info['train_ids'])
    val_data = dh.get_data(model_info['val_ids'])

    # calculate the number of labels
    num_labels = len(model_info['label_mapping'])

    # get pretrained model
    model = AutoModelForTokenClassification.from_pretrained(
        config['model'], num_labels = num_labels)

    # Define the trainer
    trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset = train_data,
            eval_dataset = val_data,
            data_collator = dh.data_collator,
            tokenizer = dh.tokenizer,
            # callbacks for interrupting the training
            # and for sending progressupdates
            callbacks = [InterruptCallback(stop), 
            EvaluateCallback(bux, model_id)],
            compute_metrics = compute_metrics
            )

    # Update the model info that the model is training
    model_info['status'] = 'training'
    model_info['num_labels'] = num_labels

    with SqliteDict(config['kv']) as db:
        db[model_id] = model_info
        db.commit()

    # Start training the training
    log(f'Starting the training for model {model_id}')
    trainer.train()

    # Training done
    # Case: Training finished normally
    if (stop.get_state() == 0):
        # Save the model
        torch.save(model.state_dict(), f'{config["models"]}{model_id}.pth')

        # Remove the checkpoints because either the best model is already 
        # loaded or the final model was the goal
        remove_checkpoints(model_id, config)

        # Update the model info is trained
        model_info['status'] = 'trained'

        with SqliteDict(config['kv']) as db:
            db[model_id] = model_info
            db.commit()

        log(f'Training for model {model_id} finished.')

    # Case: Training was paused
    elif (stop.get_state() == 1): 
        # Update the model info that the model was interrupted
        model_info['status'] = 'paused'

        with SqliteDict(config['kv']) as db:
            db[model_id] = model_info
            db.commit()

        log(f'Training of model {model_id} was interrupted.')

    # Case: Training interrupted
    else:
        delete_model(model_id, config)

        log(f'Training of model {model_id} was interrupted and the model' 
            f'was deleted.')

# Call this method to continue the training of a model.
def continue_training_model(model_id: str, stop: InterruptState, 
    bux: BunnyPostalService, config):
    
    # Get a list of all checkpoints
    cp_list = os.listdir(f'{config["checkpoints"]}{model_id}')

    # get model info
    model_info = SqliteDict(config['kv'])[model_id]

    #check for correct status
    if (model_info['status'] != 'paused' or not 
        check_model(model_id, config)):
        # if not stop and send error message
        log(f'model {model_id} cant be continued', 'ERROR')
        bux.deliver_error_message(model_id, f'Model {model_id} with status'
            f'{status}cant be continued.')
        return

    # find newest checkpoint
    cp_val = -1
    for cp in cp_list:
        cp_val_temp = int(cp.split('-')[1])
        if cp_val_temp > cp_val:
            cp_val = cp_val_temp

    # Get the data
    dh = DataHelper(model_id, config)
    train_data = dh.get_data(model_info['train_ids'])
    val_data = dh.get_data(model_info['val_ids'])

    # Get the training Arguments
    training_args = get_training_args(model_id, config)

    # get pretrained model
    model = AutoModelForTokenClassification.from_pretrained(
        config['model'], num_labels = model_info['num_labels'])

    # Define the trainer
    trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset = train_data,
            eval_dataset = val_data,
            data_collator = dh.data_collator,
            tokenizer = dh.tokenizer,
            # callbacks for interrupting the training
            # and for sending progressupdates
            callbacks = [InterruptCallback(stop), 
            EvaluateCallback(bux, model_id)],
            compute_metrics = compute_metrics
            )

    # Update the model info that the model is training
    model_info['status'] = 'training'

    with SqliteDict(config['kv']) as db:
        db[model_id] = model_info
        db.commit()

    # Continue training the model
    log(f'Continueing the training for model {model_id}')
    trainer.train(f'{config["checkpoints"]}/{model_id}/checkpoint-{cp_val}')

    # Training done
    # Case: Training finished normally
    if (stop.get_state() == 0):
        # Save the model
        torch.save(model.state_dict(), f'{config["models"]}{model_id}.pth')

        # Remove the checkpoints because either the best model is 
        # already loaded or the final model was the goal
        remove_checkpoints(model_id, config)

        # Update the model info is trained
        model_info['status'] = 'trained'

        with SqliteDict(config['kv']) as db:
            db[model_id] = model_info
            db.commit()

        log(f'Training for model {model_id} finished.')

    # Case: Training was paused
    elif (stop.get_state() == 1): 
        # Update the model info that the model was interrupted
        model_info['status'] = 'paused'

        with SqliteDict(config['kv']) as db:
            db[model_id] = model_info
            db.commit()

        log(f'Training of model {model_id} was interrupted.')

    # Case: Training interrupted
    else:
        # delete model
        delete_model(model_id, config)

        log(f'Training of model {model_id} was interrupted and the model was' 
            f' deleted.')


# Call this method evaluate a model
def evaluate_model(model_id: str, stop: InterruptState, 
    bux: BunnyPostalService, data_id: str, config):
    
    #get model info
    model_info = SqliteDict(config['kv'])[model_id]

    #check for correct status
    if (model_info['status'] != 'trained' or not 
        check_model(model_id, config)):
        # else stop and send error message
        log(f'model {model_id} cant be evaluated', 'ERROR')
        bux.deliver_error_message(model_id, f'Model {model_id} with status'
            f'{status}cant be evaluated.')
        return

    # Get the training Arguments
    training_args = get_training_args(model_id, config)

    # Get the evaluation data
    dh = DataHelper(model_id, config)
    data = dh.get_data(data_id)

    # get pretrained model
    model = AutoModelForTokenClassification.from_pretrained(
        config['model'], num_labels = model_info['num_labels'])

    # Load trained weights
    model.load_state_dict(torch.load(f'{config["models"]}{model_id}.pth'))
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
    log(f'Beginning evaluation for model {model_id}')
    out = trainer.evaluate(eval_dataset = data)

    # get metrics for output
    metrics = {}
    metrics['eval_precission'] = out['eval_precission']
    metrics['eval_recall'] = out['eval_recall']
    metrics['eval_f1'] = out['eval_f1']

    # send the results
    bux.deliver_eval_results(model_id, metrics)

    log(f'Evaluated model {model_id}.')


# Call this method to predict text with a model
def predict_text(model_id: str, stop: InterruptState, 
    bux: BunnyPostalService, sequence: str, config):
    
    #get model info
    model_info = SqliteDict(config['kv'])[model_id]

    #check for correct status
    if (model_info['status'] != 'trained' or not 
        check_model(model_id, config)):
        # else stop and send error message
        log(f'model {model_id} cant be predicted', 'ERROR')
        bux.deliver_error_message(model_id, f'Model {model_id} with status'
            f'{status}cant be predicted.')
        return

    # get pretrained model
    model = AutoModelForTokenClassification.from_pretrained(
        config['model'], num_labels = model_info['num_labels'])

    # Load trained weights
    model.load_state_dict(torch.load(f'{config["models"]}{model_id}.pth'))
    model.eval()

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # convert input to tokens
    inputs = tokenizer(sequence, return_tensors="pt")
    tokens = inputs.tokens()

    # number of tokens without [CLS] and [SEP]
    num_tokens =  len(tokens) - 2

    # if you find a sentence longer than this im sorry
    # Supposed to only work for a sentence.
    # less than a 100 tokens should work for any transformer model
    # does not actually check if only one sentence or even a sentence was sent
    if num_tokens > 100:
        log(f'Sentence: {sequence} with {num_tokens} tokens is too long.', 
            'ERROR')
        bux.deliver_error_message(model_id, f'Sentence: {sequence} with ' 
            f'{num_tokens} tokens is too long.')
        return

    # get model output
    outputs = model(**inputs).logits

    # get actual predictions
    predictions = torch.argmax(outputs, dim=2)

    #convert to list
    predictions = predictions.tolist()[0]

    #define stuff for changing back to original labels
    tags = SqliteDict(config['kv'])[model_id]['label_mapping']
    reverse_tags = {v: k for k, v in tags.items()}

    def conv_labels(tag):
        return reverse_tags[tag]

    # change labels to strings
    predictions = list(map(conv_labels, predictions))

    log(f'Text prediction for model {model_id} done.')

    # bux send results to rabbit mq
    bux.deliver_text_prediction(model_id, tokens[1:num_tokens+1], 
        predictions[1:num_tokens+1])

    log(f'Prediction finished for model {model_id}')


# Call this method to predict text with a model
def predict_data(model_id: str, stop: InterruptState, bux: BunnyPostalService, 
    doc_id: str, config):
    
    #get model info from kv store
    model_info = SqliteDict(config['kv'])[model_id]

    #check for correct status
    if (model_info['status'] != 'trained' or not 
        check_model(model_id, config)):
        log(f'model {model_id} cant be predicted', 'ERROR')
        bux.deliver_error_message(model_id, f'Model {model_id} with status'
            f'{status}cant be predicted.')
        return

    # get pretrained model
    model = AutoModelForTokenClassification.from_pretrained(
        config['model'], 
        num_labels = model_info['num_labels'])

    # Load trained weights
    model.load_state_dict(torch.load(f'{config["models"]}{model_id}.pth'))
    model.eval()

    # Get the training Arguments
    training_args = get_training_args(model_id, config)

    # init data helper
    dh = DataHelper(model_id, config)

    # Define the trainer
    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = dh.data_collator,
        tokenizer = dh.tokenizer,
        compute_metrics = compute_metrics
        )

    # if doc_id is a string and not a list, convert it.
    if not isinstance(doc_id, list):
        doc_id = [doc_id]

    #define stuff for changing back to original labels
    tags = SqliteDict(config['kv'])[model_id]['label_mapping']
    reverse_tags = {v: k for k, v in tags.items()}

    def conv_labels(tag):
        return reverse_tags[tag]

    # predict the documents one after another
    for doc in doc_id:

        #get the data
        data = dh.get_data_pred([doc])

        # the data tokens
        token_data = data['tokens']

        # get predictions from trainer
        results = trainer.predict(data)
        preds = np.argmax(results[0], 2)

        # remove the padding. and convert the labels
        pred_data = []
        for i, val in enumerate(token_data):
            # first: select remove the [CLS], [SEP] and [PAD] tokens
            # second: map the values to ints
            # third: convert those to string labels
            pred_data.append(list(
                map(conv_labels, map(int, preds[i][1:len(val) + 1]))))

        # send results
        bux.deliver_data_prediction(model_id, token_data, pred_data, doc)

    log(f'Prediction finished for model {model_id}')