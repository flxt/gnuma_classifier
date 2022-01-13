import pika
from pika.exchange_type import ExchangeType

import logging
import json


class BunnyCredentials():
    rabbit_mq_host = 'h2826957.stratoserver.net'
    rabbit_mq_port = 5672
    rabbit_mq_user = 'rabbitmqtest'
    rabbit_mq_pass = 'rabbitmqtest'
    rabbit_mq_exchange = 'GNUMAExchange'
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

    # The BunnyPostalServices delivers your message to the distilbert exchange.
    def send_message(self, message):
        logging.info(f'Send a message to the distilbert exchange.')
        self._channel.basic_publish(exchange = self._exchange, routing_key = self._routing_key, body = json.dumps(message))

    # The BunnyPostalService introduces the Microservice to the exchange
    def say_hello(self):
        with open('./distilbert_startup.json') as json_file:
            startup = json.load(json_file)

        self.send_message(startup)