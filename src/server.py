from flask import Flask
from flask_restful import Api
from sqlitedict import SqliteDict

from queue import Queue
from threading import Thread
import logging
import json
import sys
import os
import dill
from pathlib import Path
import argparse

from src.resources import Base, Interrupt, Pause, PredictText, Evaluate 
from src.resources import Continue, List, Train, Predict
from src.training import training_thread
from src.utils import InterruptState, check_model, delete_model
from src.utils import log, CurrentModel, get_config
from src.bunny import BunnyPostalService, bunny_listening_thread
from src.bunny import bunny_alive_thread

def main(path, port):
    print("Starting server. Press ctrl + C to quit.")

    # get config
    config = get_config(path, port)

    # make sure models directory exists, so save model does not crash
    Path(config['models']).mkdir(parents=True, exist_ok=True)

    # set logging lvl
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    # Delete all models that are in a faulty state
    keys = SqliteDict(config['kv']).keys()
    for model_id in keys:
        if check_model(model_id, config):
            # delete model with status training unless checkpoints where saved.
            # In that case change its status to interrupted, so training
            # can be continued.
            if SqliteDict(config['kv'])[model_id]['status'] == 'training':
                if os.path.isdir(f'./checkpoints/{model_id}'):
                    #log(SqliteDict(config['kv'])[model_id]['status'])
                    with SqliteDict(config['kv']) as db:
                        model_info = db[model_id]
                        model_info['status'] = 'interrupted'
                        db[model_id] = model_info
                        db.commit()
                    #log(SqliteDict(config['kv'])[model_id]['status'])
                else:
                    delete_model(model_id, config)

    # set interrupt vairable
    stop = InterruptState()

    # variable storing the id of the currently trained model
    current_model_id = CurrentModel()

    # init the bunny postal service
    bux = BunnyPostalService(config)

    # Init Flask App and API
    app = Flask(__name__)
    api = Api(app)

    # init the
    q = Queue()

    # check if que file exists
    try:
        if os.path.isfile(config['que']):
            with open(config['que'],'rb') as queue_save_file:
                q = dill.load(queue_save_file)
    except:
        log('Loading the que went went wrong. Deleting saved que.', 'ERROR')
        q = Queue()
        os.remove(config['que'])

    # if the que is emtpy => delete all models with status 'in_que'
    if q.empty():
        keys = SqliteDict(config['kv']).keys()
        for model_id in keys:
            if (SqliteDict(config['kv'])[model_id]['status'] 
                == 'in_que'):
                delete_model(model_id, config)

    # Add the RestFULL Resources to the api
    api.add_resource(Base, f'/{config["path"]}/models/<model_id>', 
        resource_class_kwargs ={'current_model_id': current_model_id, 
        'config': config, 'que': q})
    api.add_resource(Interrupt, f'/{config["path"]}/interrupt/<model_id>', 
        resource_class_kwargs ={'stop' : stop, 'bux': bux, 'que' : q,
        'current_model_id': current_model_id, 'config': config})
    api.add_resource(Pause, f'/{config["path"]}/pause/<model_id>', 
        resource_class_kwargs ={'stop' : stop, 'bux': bux, 'que' : q,
        'current_model_id': current_model_id, 'config': config})
    api.add_resource(PredictText, f'/{config["path"]}/predict/text/'
        f'<model_id>', resource_class_kwargs ={'que' : q, 'config': config})
    api.add_resource(Predict, f'/{config["path"]}/predict/data/<model_id>', 
        resource_class_kwargs ={'que' : q, 'config': config})
    api.add_resource(Evaluate, f'/{config["path"]}/evaluate/<model_id>', 
        resource_class_kwargs ={'que' : q, 'config': config})
    api.add_resource(Continue, f'/{config["path"]}/continue/<model_id>', 
        resource_class_kwargs ={'que' : q, 'config': config})
    api.add_resource(List, f'/{config["path"]}/models',
        resource_class_kwargs ={'config': config})
    api.add_resource(Train, f'/{config["path"]}/train', 
        resource_class_kwargs ={'que' : q, 'config': config})

    #start listening
    t2 = Thread(target = bunny_listening_thread, args = (bux, config,))
    t2.daemon = True
    t2.start()

    # Say Hello to RabbitMQ
    bux.say_hello()

    # i am alive
    t3 = Thread(target = bunny_alive_thread, args = (bux,))
    t3.daemon = True
    t3.start()

    # start thread for running model
    t = Thread(target = training_thread, 
        args=(q, stop, bux, current_model_id, config))
    t.daemon = True
    t.start()

    # Start the APP
    try:
        app.run(debug=False, use_reloader=False, port = config['port'], 
            host = config['host'])
    except Exception as e:
        log('flask crashed', 'ERROR')
        bux.deliver_error_message('', e)

        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Parse path and port')
    parser.add_argument('path', type = str,
                    help='The folder name of the classifier')
    parser.add_argument('port', type=int,
                    help='The port number')
    
    args = parser.parse_args()

    try:
        main(args.path, args.port)
    except (KeyboardInterrupt, SystemExit):
        print('Received KeyboardInterrupt. Shutting Down.')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)