# api/main.py
import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from pydantic import BaseModel
from typing import Dict, Any

MODEL_PATH = "../models/fraud_pipeline.pkl"  # adjust if running from project root
FEATURES_PATH = "../models/feature_columns.json"
THRESH_PATH = "../config/threshold.json"

# If running api from root move paths:
if os.path.exists("models/fraud_pipeline.pkl"):
    MODEL_PATH = "models/fraud_pipeline.pkl"
    FEATURES_PATH = "models/feature_columns.json"
    THRESH_PATH = "config/threshold.json"

print("Loading model:", MODEL_PATH)
pipeline = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "r") as f:
    feature_columns = json.load(f)
with open(THRESH_PATH, "r") as f:
    threshold = json.load(f).get("threshold", 0.5)

app = FastAPI(title="Credit Card Fraud Detection API")

class Transaction(BaseModel):
    data: Dict[str, float]  # flexible schema: map of feature -> value

def make_dataframe_from_input(data: Dict[str, float]):
    # Ensure all required columns present in correct order
    # missing columns filled with 0
    row = {col: float(data.get(col, 0.0)) for col in feature_columns}
    df = pd.DataFrame([row], columns=feature_columns)
    return df

@app.post("/predict")
async def predict(payload: Transaction):
    try:
        df = make_dataframe_from_input(payload.data)
        prob = float(pipeline.predict_proba(df)[:, 1][0])
        is_fraud = bool(prob >= threshold)
        return {"fraud_probability": prob, "is_fraud": is_fraud, "threshold_used": threshold}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_raw")
async def predict_raw(request: Request):
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object mapping feature->value")
        df = make_dataframe_from_input(payload)
        prob = float(pipeline.predict_proba(df)[:, 1][0])
        is_fraud = bool(prob >= threshold)
        return {"fraud_probability": prob, "is_fraud": is_fraud, "threshold_used": threshold}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

# Optional: endpoint to get metadata
@app.get("/metadata")
async def metadata():
    return {"n_features": len(feature_columns), "feature_columns": feature_columns, "threshold": threshold}
