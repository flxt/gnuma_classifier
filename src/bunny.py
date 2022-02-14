import pika
from pika.exchange_type import ExchangeType

import json
from src.utils import log

import time


# read credentials from json on disk
def read_credentials(): 
    with open('./rabbitmq_creds.json') as json_file:
        creds = json.load(json_file)

    return creds

class BunnyPostalService():

    # Bunny says hello to rabbit.
    def __init__(self):
        creds = read_credentials()

        self._credentials = pika.PlainCredentials(creds['rabbit_mq_user'], 
            creds['rabbit_mq_pass'])
        self._connection_params = pika.ConnectionParameters(
            host=creds['rabbit_mq_host'], port=creds['rabbit_mq_port'],
            credentials=self._credentials)
        self._connection = pika.BlockingConnection(self._connection_params)
        self._channel = self._connection.channel()
        self._exchange = creds['rabbit_mq_exchange']
        self._routing_key = creds['rabbit_mq_routing_key']

        self._address = 'tba'

        self._channel.exchange_declare(exchange = self._exchange, 
            exchange_type='fanout')

    # The BunnyPostalServices delivers your message to the distilbert exchange.
    def send_message(self, message, event):
        log(f'Send {event} message to exchange: {message}')
        message['address'] = slef._address
        self._channel.basic_publish(exchange = self._exchange, 
            routing_key = self._routing_key, body = json.dumps(message), 
            properties = pika.BasicProperties(headers={'event': event}))

    # The BunnyPostalService introduces the Microservice to the exchange
    def say_hello(self):
        with open('./distilbert_startup.json') as json_file:
            startup = json.load(json_file)

        self.send_message(startup, 'ClassifierStart')

    # The BunnyPostalService gives an update about training
    def give_update(self, model_id, finished, current_step, total_steps, epoch, 
        metrics):
        metric_list = []

        for key, value in metrics.items():
            metric_list.append({'key':key, 'value':value})

        message = {'classifier_id': 'distilbert', 'model_id': model_id, 
        'finished': finished, 'current_step': current_step, 
        'total_steps': total_steps, 'epoch': epoch, 'metrics': metric_list}

        self.send_message(message, 'TrainingUpdate')

    # The BunnyPostalService delivers the results of an ecvaluation
    def deliver_eval_results(self, model_id, metrics):
        metric_list = []

        for key, value in metrics.items():
            metric_list.append({'key':key, 'value':value})

        message = {'classifier_id': 'distilbert', 'model_id': model_id, 
        'metrics': metric_list}

        self.send_message(message, 'EvaluationFinished')

    # The BunnyPostalService delivers the results of an prediction on a text
    def deliver_text_prediction(self, model_id, tokens, labels):
        prediction_list = []

        message = {'classifier_id': 'distilbert', 'model_id': model_id, 
        'tokens': tokens, 'labels': labels}

        self.send_message(message, 'PredictionFinishedText')

    # The BunnyPostalService delivers the results of a 
    # prediction on a document
    def deliver_data_prediction(self, model_id, tokens, labels):
        prediction_list = []

        message = {'classifier_id': 'distilbert', 'model_id': model_id, 
        'tokens': tokens, 'labels': labels}

        self.send_message(message, 'PredictionFinishedData')

    # The BunnyPostalService delivers an error message
    def deliver_error_message(self, model_id, message):
        msg_dict = {'model_id': model_id, 'error_message': message}
        self.send_message(msg_dict, 'ClassifierError')


# Listening to rabbit to check if supposed to say hello
def bunny_listening_thread(bux: BunnyPostalService):
    creds = read_credentials()

    # conntect to bunny
    credentials = pika.PlainCredentials(creds['rabbit_mq_user'], 
        creds['rabbit_mq_pass'])
    connection_params = pika.ConnectionParameters(host=creds['rabbit_mq_host'], 
        port=creds['rabbit_mq_port'], credentials = credentials)
    connection = pika.BlockingConnection(connection_params)
    channel = connection.channel()
    exchange = creds['rabbit_mq_exchange']
    routing_key = creds['rabbit_mq_routing_key']
    channel.queue_declare(queue = 'distilbert_listen')
    channel.queue_bind('distilbert_listen', exchange)

    def bunny_callback(ch, method, properties, body):
        #check if supposed to send hello message
        if properties.headers['event'] == 'ExperimentStart':
            bux.say_hello()

    channel.basic_consume(queue='distilbert_listen', 
        on_message_callback=bunny_callback, auto_ack=False)

    # catch keyboard interrupts
    try:
        channel.start_consuming()
    except (KeyboardInterrupt, SystemExit):
        print('Shutting down bunny listening thread.')


# send status update that still alive in intervals
def bunny_alive_thread(bux: BunnyPostalService):
    while True:
        # cute bunny sleeps for 10 seconds
        time.sleep(10)

        # wakes up to say hello
        bux.send_message({}, 'ClassifierAlive')
