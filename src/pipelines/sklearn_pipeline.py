from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config.config import LG_C, LG_MAXITER
from src.features.pre_processing import preprocess


def model_pipeline(df) -> Pipeline:
    preprocessor = preprocess(df, True)
    lg = LogisticRegression(C= LG_C, max_iter= LG_MAXITER)
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", lg),
    ])
    return pipeline
