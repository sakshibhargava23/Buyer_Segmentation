"""Step 2 & 3 – Feature encoding and scaling."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


CATEGORICAL_ONEHOT = ["client_type", "region", "referral_channel", "country"]
CATEGORICAL_LABEL = ["acquisition_purpose", "gender"]
NUMERIC_FEATURES = [
    "age",
    "satisfaction_score",
    "num_properties",
    "total_investment",
    "avg_sale_price",
    "avg_floor_area",
    "max_sale_price",
    "price_per_sqft",
    "is_investor",
    "loan_applied_flag",
]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_ONEHOT,
            ),
            ("label", "passthrough", CATEGORICAL_LABEL),
        ],
        remainder="drop",
    )


def encode_label_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Apply label encoding to selected categorical columns."""
    encoded = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in CATEGORICAL_LABEL:
        le = LabelEncoder()
        encoded[col] = le.fit_transform(encoded[col].astype(str))
        encoders[col] = le

    return encoded, encoders


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, ColumnTransformer, dict[str, LabelEncoder]]:
    """Encode and scale features for clustering."""
    encoded_df, label_encoders = encode_label_columns(df)
    preprocessor = build_preprocessor()
    feature_matrix = preprocessor.fit_transform(encoded_df)
    return feature_matrix, preprocessor, label_encoders


def save_preprocessor(preprocessor: ColumnTransformer, label_encoders: dict, path: str) -> None:
    joblib.dump({"preprocessor": preprocessor, "label_encoders": label_encoders}, path)


def load_preprocessor(path: str) -> tuple[ColumnTransformer, dict[str, LabelEncoder]]:
    bundle = joblib.load(path)
    return bundle["preprocessor"], bundle["label_encoders"]
