#from configparser import ConfigParser

from flask import Flask
from flask_restful import Api

from queue import Queue
from threading import Thread

from src.resources import Base, Interrupt, Classify, Test, List, Train
from src.trainer import training_thread

import logging

if __name__ == '__main__':
    #set logging lvl
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    #global interrupt vairable
    stop = 0

    # Init Flask App and API
    app = Flask(__name__)
    api = Api(app)

    logging.debug('Flask app initiated')

    # init que
    q = Queue()

    logging.debug('Queue initiated')

    # start thread for running model
    t = Thread(target = training_thread, args=(q,stop,))
    t.start()

    logging.debug('Training thread started')

    # Add the RestFULL REsources to the api
    api.add_resource(Base, '/distilbert/models/<model_id>', resource_class_kwargs ={'que' : q})
    api.add_resource(Interrupt, '/distilbert/interrupt', resource_class_kwargs ={'stop' : stop})
    api.add_resource(Classify, '/distilbert/classify/<model_id>')
    api.add_resource(Test, '/distilbert/test/<model_id>')
    api.add_resource(List, '/distilbert/models')
    api.add_resource(Train, '/distilbert/train', resource_class_kwargs ={'que' : q})

    logging.debug('Resources added to api')
    logging.debug('Starting falsk app')

    # Start the APP
    app.run(debug=True, use_reloader=False)