# train.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, average_precision_score, roc_curve
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb

# Paths
DATA_PATH = "data/creditcard.csv"
MODEL_DIR = "models"
CONFIG_DIR = "config"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def build_and_train(X_train, y_train):
    # Build pipeline: scaling -> SMOTE -> XGBoost
    pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("clf", xgb.XGBClassifier(
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42
        ))
    ])

    # Basic param search (small)
    param_dist = {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__subsample": [0.6, 0.8, 1.0]
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=8,
        scoring="average_precision",  # PR AUC is useful for imbalanced
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    return search.best_estimator_

def evaluate(pipeline, X_test, y_test):
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    print("Classification report (threshold=0.5):")
    print(classification_report(y_test, y_pred, digits=4))
    roc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    print(f"ROC AUC: {roc:.4f}, PR AUC (AP): {ap:.4f}")

    # Save ROC and PR curves
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC={roc:.4f}")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
    plt.legend()
    plt.savefig("reports/roc_curve.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(recalls, precisions, label=f"AP={ap:.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig("reports/pr_curve.png", dpi=150)
    plt.close()

    return y_proba, precisions, recalls, thresholds

def choose_threshold(precisions, recalls, thresholds, target_recall=0.90):
    # Choose threshold where recall >= target_recall and precision is maximized at that recall
    idxs = np.where(recalls >= target_recall)[0]
    if len(idxs) == 0:
        # fallback: pick threshold at maximum F1-like point
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
        best_idx = np.nanargmax(f1_scores)
        chosen = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        return float(chosen)
    # choose the index with maximum precision among those
    best_idx = idxs[np.argmax(precisions[idxs])]
    chosen = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return float(chosen)

def main():
    os.makedirs("reports", exist_ok=True)
    df = load_data()
    print("Loaded data shape:", df.shape)
    if "Class" not in df.columns:
        raise ValueError("Expected 'Class' column in CSV (0=legit,1=fraud)")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # simple train/test split with stratify to preserve imbalance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train model pipeline
    pipeline = build_and_train(X_train, y_train)

    # Evaluate
    y_proba, precisions, recalls, thresholds = evaluate(pipeline, X_test, y_test)

    # Choose threshold for desired recall (business decision) — default target recall 0.9
    chosen_threshold = choose_threshold(precisions, recalls, thresholds, target_recall=0.9)
    print("Chosen probability threshold:", chosen_threshold)

    # Save pipeline and metadata
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "fraud_pipeline.pkl"))
    # Save feature columns order for API to ensure correct DataFrame columns
    feature_cols = X.columns.tolist()
    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)
    # Save threshold
    with open(os.path.join(CONFIG_DIR, "threshold.json"), "w") as f:
        json.dump({"threshold": chosen_threshold}, f)

    print("Saved pipeline and metadata to:", MODEL_DIR, CONFIG_DIR)

if __name__ == "__main__":
    main()
