# kafka/producer.py
import json
import random
import time
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

TOPIC = 'transactions'

def random_tx(feature_columns):
    return {col: float(random.gauss(0,1) if col != "Amount" else random.uniform(1, 500)) for col in feature_columns}

if __name__ == "__main__":
    # load features
    import os, json
    with open("../models/feature_columns.json") as f:
        cols = json.load(f)
    while True:
        tx = random_tx(cols)
        producer.send(TOPIC, tx)
        print("Sent:", tx)
        time.sleep(0.2)
