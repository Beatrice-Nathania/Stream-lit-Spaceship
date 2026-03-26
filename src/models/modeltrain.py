"""
Session 04 – Step 3: Training
Trains a Random Forest classifier and logs to MLflow.
"""

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from src.pipelines.sklearn_pipeline import model_pipeline
from src.utils.io import save_artifact
from config.config import (
    LG_C, LG_MAXITER,
    MLFLOW_TRACKING_URI, MLFLOW_EXP_PIPELINE, ARTIFACT_PIPELINE,
)

def train(pipeline, train_prepro):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Streamlit--Pipeline")

    X_train = train_prepro.drop("Transported", axis = 1)
    y_train = train_prepro["Transported"]

    with mlflow.start_run() as run:
        mlflow.log_param("max_iter", LG_MAXITER)
        mlflow.log_param("C", LG_C)

        pipeline.fit(X_train, y_train)
        mlflow.sklearn.log_model(pipeline, name="model_logreg", registered_model_name="Space_titan")
        save_artifact(pipeline, ARTIFACT_PIPELINE)
        print(f"Pipeline trained. Run ID: {run.info.run_id}")
        return run.info.run_id

