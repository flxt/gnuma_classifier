from flask import Flask
from flask_restful import Api

from queue import Queue
from threading import Thread
import logging

from src.resources import *
from src.training import training_thread
from src.training_help import InterruptState
from src.bunny import BunnyPostalService

if __name__ == '__main__':
    # set logging lvl
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    # global interrupt vairable
    stop = InterruptState()

    # init the bunny postal service
    bux = BunnyPostalService()

    # Init Flask App and API
    app = Flask(__name__)
    api = Api(app)

    # init que
    q = Queue()

    # start thread for running model
    t = Thread(target = training_thread, args=(q, stop, bux, ))
    t.start()

    # Add the RestFULL Resources to the api
    api.add_resource(Base, '/distilbert/models/<model_id>', resource_class_kwargs ={'que' : q})
    api.add_resource(Interrupt, '/distilbert/interrupt', resource_class_kwargs ={'stop' : stop})
    api.add_resource(Pause, '/distilbert/pause', resource_class_kwargs ={'stop' : stop})
    api.add_resource(Predict, '/distilbert/predict/<model_id>', resource_class_kwargs ={'que' : q})
    api.add_resource(Evaluate, '/distilbert/evaluate/<model_id>', resource_class_kwargs ={'que' : q})
    api.add_resource(Continue, '/distilbert/continue/<model_id>', resource_class_kwargs ={'que' : q})
    api.add_resource(List, '/distilbert/models')
    api.add_resource(Train, '/distilbert/train', resource_class_kwargs ={'que' : q})

    # Say Hello to RabbitMQ
    # TODO

    # Start the APP
    app.run(debug=True, use_reloader=False)