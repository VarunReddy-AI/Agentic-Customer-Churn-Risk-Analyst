from src.data.ingestion import load_data
from src.data.preprocessing import clean_data
from src.features.feature_engineering import feature_engineering
from src.models.train import train_pipeline
from src.models.evaluate import evaluate_model


def run_training_pipeline():

    df = load_data(r"D:\Desktop\mlops\customer-churn-monitoring-mlops\data\raw\data.csv")

    df = clean_data(df)

    X_train, X_test, y_train, y_test = feature_engineering(df)

    model = train_pipeline(X_train, y_train)

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    run_training_pipeline()