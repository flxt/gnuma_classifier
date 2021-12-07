#from configparser import ConfigParser

from flask import Flask
from flask_restful import Api

from queue import Queue
from threading import Thread

from src.resources import Base, Interrupt, Classify, Test, List, Train

if __name__ == '__main__':
    # Init Flask App and API
    app = Flask(__name__)
    api = Api(app)

    # init que
    q = Queue()

    # start thread for running model

    # Add the RestFULL REsources to the api
    api.add_resource(Base, '/distilbert/models/<model_id>')
    api.add_resource(Interrupt, '/distilbert/interrupt')
    api.add_resource(Classify, '/distilbert/classify/<model_id>')
    api.add_resource(Test, '/distilbert/test/<model_id>')
    api.add_resource(List, '/distilbert/models')
    api.add_resource(Train, '/distilbert/train', resource_class_kwargs ={'que' : q})

    # Start the APP
    app.run(debug=True, use_reloader=False)