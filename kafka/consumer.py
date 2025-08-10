import json
import joblib
from kafka import KafkaConsumer
import asyncio
import websockets
import os

# Load the trained model pipeline
model_path = os.path.join("..", "models", "fraud_pipeline.pkl")
feature_columns_path = os.path.join("..", "models", "feature_columns.json")

pipeline = joblib.load(model_path)
with open(feature_columns_path, "r") as f:
    feature_columns = json.load(f)

# Kafka consumer configuration
consumer = KafkaConsumer(
    "transactions",
    bootstrap_servers=["localhost:9092"],
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="fraud-detection-group",
    value_deserializer=lambda x: json.loads(x.decode("utf-8"))
)

# Async function to send alert to WebSocket dashboard
async def send_alert(alert_data):
    uri = "ws://localhost:8001/ws"
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps(alert_data))
    except Exception as e:
        print("Failed to send alert:", e)

def run_consumer():
    print("Kafka consumer started, listening for transactions...")

    for message in consumer:
        transaction = message.value  # this is a dict with features

        # Prepare feature vector for prediction
        features = [transaction.get(col, 0) for col in feature_columns]

        # Predict probability of fraud
        fraud_prob = pipeline.predict_proba([features])[0][1]
        is_fraud = fraud_prob > 0.5  # threshold, adjust as needed

        print(f"Transaction: {transaction}")
        print(f"Fraud probability: {fraud_prob:.4f} --> {'FRAUD' if is_fraud else 'LEGIT'}")

        if is_fraud:
            alert = {
                "transaction": transaction,
                "fraud_probability": fraud_prob,
                "message": "Fraudulent transaction detected!"
            }
            # Send alert asynchronously to dashboard
            asyncio.run(send_alert(alert))

if __name__ == "__main__":
    run_consumer()
