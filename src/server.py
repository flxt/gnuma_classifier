from flask import Flask
from flask_restful import Api

from queue import Queue
from threading import Thread
import logging

from src.resources import Base, Interrupt, Predict, Evaluate, List, Train
from src.training import training_thread
from src.training_help import InterruptState

if __name__ == '__main__':
    #set logging lvl
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    #global interrupt vairable
    stop = InterruptState()

    # Init Flask App and API
    app = Flask(__name__)
    api = Api(app)

    # init que
    q = Queue()

    # start thread for running model
    t = Thread(target = training_thread, args=(q,stop,))
    t.start()

    # Add the RestFULL REsources to the api
    api.add_resource(Base, '/distilbert/models/<model_id>', resource_class_kwargs ={'que' : q})
    api.add_resource(Interrupt, '/distilbert/interrupt', resource_class_kwargs ={'stop' : stop})
    api.add_resource(Predict, '/distilbert/classify/<model_id>', resource_class_kwargs ={'que' : q})
    api.add_resource(Evaluate, '/distilbert/test/<model_id>', resource_class_kwargs ={'que' : q})
    api.add_resource(List, '/distilbert/models')
    api.add_resource(Train, '/distilbert/train', resource_class_kwargs ={'que' : q})

    # Start the APP
    app.run(debug=True, use_reloader=False)