import pika
from pika.exchange_type import ExchangeType

import json
from src.utils import log

import time

# The Bunny postal service is responsible to send messages to rabbit mq
class BunnyPostalService():

    # Bunny says hello to rabbit.
    # Aka you init the postal service
    def __init__(self, config):
        #get credits from config file
        creds = config['rabbit_mq']

        # load the start up file and save it in var to use it later
        # goal is to load it only once, it is not big so it does not matter
        # that it is in memory
        with open(config['startup']) as json_file:
            self._startup = json.load(json_file)

        # init the connection parameters
        self._connection_params = pika.ConnectionParameters(
            host=creds['host'], port=creds['port'],
            credentials=pika.PlainCredentials(creds['user'], creds['pass']))

        # init connection and channel
        self._connection = pika.BlockingConnection(self._connection_params)
        self._channel = self._connection.channel()

        # define exchange and routing key
        self._exchange = creds['exchange']
        self._routing_key = 'needed?'

        # get own address from config file
        self._address = config['address']

        # todo remove after testing
        self._channel.exchange_declare(exchange = self._exchange, 
            exchange_type='fanout')

    # The BunnyPostalServices delivers your message to the distilbert exchange
    def send_message(self, message, event):
        # Add address for identification
        message['address'] = self._address
        log(f'Send {event} message to exchange: {message}')

        # publish to exchange
        self._channel.basic_publish(exchange = self._exchange, 
            routing_key = self._routing_key, body = json.dumps(message), 
            properties = pika.BasicProperties(headers={'event': event}))

    # The BunnyPostalService introduces the Microservice to the exchange
    def say_hello(self):
        self.send_message(self._startup, 'ClassifierStart')

    # The BunnyPostalService gives an update about training
    def give_update(self, model_id, finished, current_step, total_steps, 
        epoch, metrics):
        metric_list = []

        # convert dict to list
        for key, value in metrics.items():
            metric_list.append({'key':key, 'value':value})

        # define the message
        message = {'classifier_id': 'distilbert', 'model_id': model_id, 
        'finished': finished, 'current_step': current_step, 
        'total_steps': total_steps, 'epoch': epoch, 'metrics': metric_list}

        self.send_message(message, 'TrainingUpdate')

    # The BunnyPostalService delivers the results of an ecvaluation
    def deliver_eval_results(self, model_id, metrics):
        metric_list = []

        # convert dict to list 
        for key, value in metrics.items():
            metric_list.append({'key':key, 'value':value})

        # define message
        message = {'classifier_id': 'distilbert', 'model_id': model_id, 
        'metrics': metric_list}

        self.send_message(message, 'EvaluationFinished')

    # The BunnyPostalService delivers the results of an prediction on a text
    def deliver_text_prediction(self, model_id, tokens, labels):
        prediction_list = []

        # define the message
        message = {'classifier_id': 'distilbert', 'model_id': model_id, 
        'tokens': tokens, 'labels': labels}

        self.send_message(message, 'PredictionFinishedText')

    # The BunnyPostalService delivers the results of a 
    # prediction on a document
    def deliver_data_prediction(self, model_id, tokens, labels, doc_id):
        prediction_list = []

        # define the message
        message = {'classifier_id': 'distilbert', 'model_id': model_id, 
        'tokens': tokens, 'labels': labels, 'doc_id': doc_id}

        self.send_message(message, 'PredictionFinishedData')

    # The BunnyPostalService delivers an error message
    def deliver_error_message(self, model_id, message):
        msg_dict = {'model_id': model_id, 'error_message': message}
        self.send_message(msg_dict, 'ClassifierError')

    # The BunnyPostalService delivers that the training was interrupted
    def deliver_interrupt_message(self, model_id, pause):
        message = {'model_id': model_id, 'pause': pause}
        self.send_message(message, 'ClassifierInterrupt')


# Listening to rabbit to check if supposed to say hello
def bunny_listening_thread(bux: BunnyPostalService, config):
    # get credentials from config file
    creds = config['rabbit_mq']

    # define name for que
    q_name = config['path']

    # conntect to rabbit mq
    connection_params = pika.ConnectionParameters(
        host=creds['host'], port=creds['port'], 
        credentials = pika.PlainCredentials(creds['user'], creds['pass']))

    # define connection, channle and exchange
    connection = pika.BlockingConnection(connection_params)
    channel = connection.channel()
    exchange = creds['exchange']

    # define a que and bind it to the exchange
    channel.queue_declare(queue = q_name)
    channel.queue_bind(q_name, exchange)

    # callback that send the startup message to the exchange if 
    # ExperimentStart up is received
    def bunny_callback(ch, method, properties, body):
        #check if supposed to send hello message
        if properties.headers['event'] == 'ExperimentStart':
            bux.say_hello()

    # consume message from que
    channel.basic_consume(queue=q_name, 
        on_message_callback=bunny_callback, auto_ack=True)

    # catch keyboard interrupts
    try:
        #start listening aka consuming
        channel.start_consuming()
    except (KeyboardInterrupt, SystemExit):
        print('Shutting down bunny listening thread.')


# send status update that still alive in intervals
def bunny_alive_thread(bux: BunnyPostalService):
    while True:
        # catch keyboard interrupts
        try:
            # cute bunny sleeps for 10 seconds
            time.sleep(10)

            # wakes up to say hello
            bux.send_message({}, 'ClassifierAlive')
        except (KeyboardInterrupt, SystemExit):
            print('Shutting down bunny alive thread.')