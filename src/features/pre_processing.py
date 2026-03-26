"""
Session 04 – Step 2: Preprocessing
Reads ingested data, splits, scales, and saves the preprocessor artifact.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from config.config import ARTIFACTS_DIR

def preprocess(df, is_train):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Define features to use
    categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Age_group']
    numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                         'Cabin_num', 'Group_size', 'Solo', 'Family_size', 'TotalSpending',
                         'HasSpending', 'NoSpending', 'Age_missing', 'CryoSleep_missing'] + \
                        [col for col in df.columns if '_ratio' in col]
    
    
    numeric_preprocess = Pipeline([
        ("num_imputer", SimpleImputer(strategy="median")),
    ])

    categorical_preprocess = Pipeline([
        ("cat_imputer", SimpleImputer(strategy="most_frequent")),
        ("cat_encoder", OrdinalEncoder()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numPreprocess", numeric_preprocess, numerical_features),
            ("catPreprocess", categorical_preprocess, categorical_features),
        ],
        remainder="drop",
    )

    # Select features
    feature_columns = categorical_features + numerical_features
    
    if is_train:
        return preprocessor
    else:
        return feature_columns
