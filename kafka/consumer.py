# kafka/consumer.py
import json
import joblib
import pandas as pd
from kafka import KafkaConsumer

pipeline = joblib.load("../models/fraud_pipeline.pkl")
with open("../models/feature_columns.json") as f:
    feature_columns = json.load(f)
consumer = KafkaConsumer(
    'transactions',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='fraud-consumer-group'
)

for msg in consumer:
    tx = msg.value
    # prepare df
    row = {col: float(tx.get(col, 0.0)) for col in feature_columns}
    df = pd.DataFrame([row], columns=feature_columns)
    prob = float(pipeline.predict_proba(df)[:,1][0])
    if prob > 0.8:
        print("ALERT!: Fraud probability:", prob, "tx:", tx)
    else:
        print("OK:", prob)
