from src.data.ingestion import load_data
from src.data.preprocessing import clean_data
from src.features.feature_engineering import feature_engineering
from src.models.train import train_pipeline
from src.models.evaluate import evaluate_model

import sys

from src.exception.exception import CustomException
from src.logger.logger import logger
from src.config.config import (
    RAW_DATA_PATH,
    MODEL_PATH,
    SCALER_PATH,
    ENCODER_PATH,
    X_TRAIN_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_TEST_PATH,
)

def run_training_pipeline():
    logger.info('started training pipeline...💀💀')
    df = load_data(RAW_DATA_PATH)

    df = clean_data(df)

    X_train, X_test, y_train, y_test = feature_engineering(df)

    model = train_pipeline(X_train, y_train)

    evaluate_model(model, X_test, y_test)
    logger.info('Finally done with training🫡🫡')


if __name__ == "__main__":
    run_training_pipeline()