import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.model_io import save_object
from src.config.config import (
    MODEL_PATH,
    SCALER_PATH,
    ENCODER_PATH,
    MODEL_DIR,
    REPORT_DIR,
    PROCESSED_DATA_DIR,
    X_TRAIN_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_TEST_PATH,
    METRICS_PATH,
)

NUMERICAL_COLS = [
    "tenure_months",
    "monthly_charges",
    "usage_minutes",
    "support_calls",
    "payment_delay_days",
    "num_services",
    "satisfaction_score"
]

CATEGORICAL_COLS = [
    "contract_type",
    "marketing_segment",
    "signup_channel"]


def split_data(df):
    X = df.drop("churn", axis=1)
    y = df["churn"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


def encode_train_features(X_train, X_test):

    encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False
    )

    X_train_encoded = encoder.fit_transform(
        X_train[CATEGORICAL_COLS]
    )

    X_test_encoded = encoder.transform(
        X_test[CATEGORICAL_COLS]
    )

    encoded_cols = encoder.get_feature_names_out(CATEGORICAL_COLS)

    X_train_encoded = pd.DataFrame(
        X_train_encoded,
        columns=encoded_cols,
        index=X_train.index
    )

    X_test_encoded = pd.DataFrame(
        X_test_encoded,
        columns=encoded_cols,
        index=X_test.index
    )

    X_train = X_train.drop(columns=CATEGORICAL_COLS)
    X_test = X_test.drop(columns=CATEGORICAL_COLS)

    X_train = pd.concat([X_train, X_train_encoded], axis=1)
    X_test = pd.concat([X_test, X_test_encoded], axis=1)

    return X_train, X_test, encoder


def scale_train_features(X_train, X_test):

    scaler = StandardScaler()

    X_train[NUMERICAL_COLS] = scaler.fit_transform(
        X_train[NUMERICAL_COLS]
    )

    X_test[NUMERICAL_COLS] = scaler.transform(
        X_test[NUMERICAL_COLS]
    )

    return X_train, X_test, scaler


def transform_features(df, encoder, scaler):
    """
    Transform new data using saved encoder and scaler.
    """

    encoded = encoder.transform(
        df[CATEGORICAL_COLS]
    )

    encoded = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(CATEGORICAL_COLS),
        index=df.index
    )

    df = df.drop(columns=CATEGORICAL_COLS)

    df = pd.concat(
        [df, encoded],
        axis=1
    )

    df[NUMERICAL_COLS] = scaler.transform(
        df[NUMERICAL_COLS]
    )

    return df


def save_preprocessing_objects(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    encoder
):

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    X_train.to_csv(
        X_TRAIN_PATH,
        index=False
    )

    X_test.to_csv(
        X_TEST_PATH,
        index=False
    )

    y_train.to_csv(
        Y_TRAIN_PATH,
        index=False
    )

    y_test.to_csv(
    Y_TEST_PATH,
    index=False
    )

    save_object(
    scaler,
    SCALER_PATH
    )

    save_object(
        encoder,
        ENCODER_PATH
    )


def feature_engineering(df):

    X_train, X_test, y_train, y_test = split_data(df)

    X_train, X_test, encoder = encode_train_features(
        X_train,
        X_test
    )

    X_train, X_test, scaler = scale_train_features(
        X_train,
        X_test
    )

    save_preprocessing_objects(
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        encoder
    )

    return X_train, X_test, y_train, y_test