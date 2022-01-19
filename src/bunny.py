import pika
from pika.exchange_type import ExchangeType

import logging
import json


class BunnyCredentials():
    rabbit_mq_host = 'h2826957.stratoserver.net'
    rabbit_mq_port = 5672
    rabbit_mq_user = 'rabbitmqtest'
    rabbit_mq_pass = 'rabbitmqtest'
    rabbit_mq_exchange = 'test'
    rabbit_mq_routing_key = 'Classifier.Distilbert'

class BunnyPostalService():

    # Bunny says hello to rabbit.
    def __init__(self):
        self._credentials = pika.PlainCredentials(BunnyCredentials.rabbit_mq_user, BunnyCredentials.rabbit_mq_pass)
        self._connection_params = pika.ConnectionParameters(host=BunnyCredentials.rabbit_mq_host, port=BunnyCredentials.rabbit_mq_port, credentials=self._credentials)
        self._connection = pika.BlockingConnection(self._connection_params)
        self._channel = self._connection.channel()
        self._exchange = BunnyCredentials.rabbit_mq_exchange
        self._routing_key = BunnyCredentials.rabbit_mq_routing_key

        self._channel.exchange_declare(exchange = self._exchange, exchange_type='fanout')

    # The BunnyPostalServices delivers your message to the distilbert exchange.
    def send_message(self, message):
        logging.info(f'Send to exchange: {message}')
        self._channel.basic_publish(exchange = self._exchange, routing_key = self._routing_key, body = json.dumps(message))

    # The BunnyPostalService introduces the Microservice to the exchange
    def say_hello(self):
        with open('./distilbert_startup.json') as json_file:
            startup = json.load(json_file)

        self.send_message(startup)

    # The BunnyPostalService gives an update about training
    def give_update(self, model_id, finished, current_step, total_steps, epoch, metrics):
        metric_list = []

        for key, value in metrics.items():
            metric_list.append({'key':key, 'value':value})

        message = {'classifier_id': 'distilbert', 'model_id': model_id, 'finished': finished, 'current_step': current_step, 'total_steps': total_steps, 'epoch': epoch, 'metrics': metric_list}

        self.send_message(message)

    def deliver_eval_results(self, model_id, metrics):
        metric_list = []

        for key, value in metrics.items():
            metric_list.append({'key':key, 'value':value})

        message = {'classifier_id': 'distilbert', 'model_id': model_id, 'metrics': metric_list}

        self.send_message(message)

    def deliver_text_prediction(self, model_id, result_list):
        prediction_list = []

        for token, label in result_list:
            prediction_list.append({'token': token, 'label': label})

        message = {'classifier_id': 'distilbert', 'model_id': model_id, 'predictions': prediction_list}

        self.send_message(message)
