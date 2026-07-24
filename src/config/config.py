import os

ROOT_DIR = os.getcwd()

DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
FUTURE_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "future_data")

MODEL_DIR = os.path.join(ROOT_DIR, "models")
REPORT_DIR = os.path.join(ROOT_DIR, "reports")

RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "data.csv")

X_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "X_train.csv")
X_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "X_test.csv")
Y_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "y_train.csv")
Y_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "y_test.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "adaboost_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "onehot_encoder.pkl")

METRICS_PATH = os.path.join(REPORT_DIR, "metrics.csv")


