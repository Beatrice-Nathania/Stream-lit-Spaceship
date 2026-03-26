"""
Session 04 – Step 4: Evaluation
Loads model from MLflow run, evaluates on test set, and logs metrics.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

from config.config import MLFLOW_TRACKING_URI

def evaluate(valid_prepro, run_id):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    x_test = valid_prepro.drop("Transported", axis = 1)
    y_test = valid_prepro["Transported"]

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model_logreg")

    preds = model.predict(x_test)
    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro")
    rec  = recall_score(y_test, preds, average="macro")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall",    rec)

    print(f"Evaluation | Accuracy={acc:.3f} | Precision={prec:.3f} | Recall={rec:.3f}")
    return acc, prec, rec


if __name__ == "__main__":
    evaluate(None, None)
