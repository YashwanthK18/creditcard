# explain.py
import os
import json
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

MODEL_PATH = "models/fraud_pipeline.pkl"
FEATURES_PATH = "models/feature_columns.json"

pipeline = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "r") as f:
    feature_columns = json.load(f)

# Function to get transformed features and underlying model
def get_transformed_X(X):
    # pipeline is scaler->smote->clf; imblearn's SMOTE is not used on transform, so pipeline[:-1] gives scaler
    # But imblearn.Pipeline does not support slicing the same way; we will transform manually using scaler stored inside pipeline
    # This assumes the pipeline steps are named 'scaler' then 'smote' then 'clf'
    scaler = None
    for name, step in pipeline.named_steps.items():
        if "scaler" in name:
            scaler = step
            break
    if scaler is None:
        raise RuntimeError("Scaler not found in pipeline")
    X_scaled = scaler.transform(X)
    return X_scaled

def explain_sample(sample_dict):
    df = pd.DataFrame([sample_dict], columns=feature_columns)
    X_scaled = get_transformed_X(df)
    clf = pipeline.named_steps.get("clf") or pipeline.named_steps[list(pipeline.named_steps.keys())[-1]]
    # TreeExplainer right for tree models
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_scaled)
    # shap_values shape: (n_classes, n_samples, n_features) for sklearn API? For XGBoost TreeExplainer it returns array
    # For binary classification shap_values[1] is relevant
    if isinstance(shap_values, list) and len(shap_values) == 2:
        vals = shap_values[1][0]
    else:
        vals = shap_values[0]
    # summary bar for single sample
    shap.summary_plot([vals], X_scaled, feature_names=feature_columns, show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_sample.png", dpi=150)
    print("Saved reports/shap_sample.png")

if __name__ == "__main__":
    # example usage: create a dummy sample of zeros
    sample = {c: 0.0 for c in feature_columns}
    explain_sample(sample)
