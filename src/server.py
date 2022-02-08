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

from src.resources import Base, Interrupt, Pause, PredictText, Evaluate, Continue, List, Train, Predict
from src.training import training_thread
from src.utils import InterruptState, check_model, delete_model
from src.bunny import BunnyPostalService, bunny_listening_thread

def main():
    print("Starting server. Press ctrl + C to quit.")

    # Delete all models that are in a faulty state
    keys = SqliteDict('./distilBERT.sqlite').keys()
    for model_id in keys:
        check_model(model_id)

    # set logging lvl
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    # set interrupt vairable
    stop = InterruptState()

    # variable storing the id of the currently trained model
    current_model_id = None

    # init the bunny postal service
    bux = BunnyPostalService()

    # Init Flask App and API
    app = Flask(__name__)
    api = Api(app)

    q = Queue()

    # check if que file exists
    try:
        if os.path.isfile('que.obj'):
            with open('que.obj','rb') as queue_save_file:
                q = dill.load(queue_save_file)
    except:
        log('Loading the que went went wrong. Deleting saved que.', 'ERROR')
        q = Queue()
        os.remove('que.obj')

    # Add the RestFULL Resources to the api
    api.add_resource(Base, '/distilbert/models/<model_id>', resource_class_kwargs ={'current_model_id': current_model_id})
    api.add_resource(Interrupt, '/distilbert/interrupt', resource_class_kwargs ={'stop' : stop})
    api.add_resource(Pause, '/distilbert/pause', resource_class_kwargs ={'stop' : stop})
    api.add_resource(PredictText, '/distilbert/predict/text/<model_id>', resource_class_kwargs ={'que' : q})
    api.add_resource(Predict, '/distilbert/predict/data/<model_id>', resource_class_kwargs ={'que' : q})
    api.add_resource(Evaluate, '/distilbert/evaluate/<model_id>', resource_class_kwargs ={'que' : q})
    api.add_resource(Continue, '/distilbert/continue/<model_id>', resource_class_kwargs ={'que' : q})
    api.add_resource(List, '/distilbert/models')
    api.add_resource(Train, '/distilbert/train', resource_class_kwargs ={'que' : q})

    #start listening
    t2 = Thread(target = bunny_listening_thread, args = (bux,))
    t2.daemon = True
    t2.start()

    # Say Hello to RabbitMQ
    bux.say_hello()

    # start thread for running model
    t = Thread(target = training_thread, args=(q, stop, bux, current_model_id,))
    t.daemon = True
    t.start()

    # Start the APP
    app.run(debug=True, use_reloader=False, port = 4793)

if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        print('Received KeyboardInterrupt. Shutting Down.')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)