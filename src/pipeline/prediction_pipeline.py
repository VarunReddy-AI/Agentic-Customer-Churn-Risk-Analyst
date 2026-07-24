import os
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from src.data.ingestion import load_data
from src.data.preprocessing import clean_data
from src.features.feature_engineering import transform_features
from src.models.predict import (
    load_model,
    predict,
    predict_probability
)
from src.utils.model_io import load_object


def prediction_pipeline(csv_path):
    """
    Predict churn for new customer data.
    If the dataset contains the target column (churn),
    evaluate the model and save metrics.
    """

    # -------------------------------
    # Load and clean data
    # -------------------------------
    df = load_data(csv_path)

    df = clean_data(df)

    # -------------------------------
    # Save true labels (if available)
    # -------------------------------
    y_true = None

    if "churn" in df.columns:
        y_true = df["churn"].copy()
        df = df.drop(columns="churn")

    # -------------------------------
    # Load preprocessing objects
    # -------------------------------
    encoder = load_object("models/onehot_encoder.pkl")

    scaler = load_object("models/scaler.pkl")

    # -------------------------------
    # Transform features
    # -------------------------------
    X = transform_features(
        df,
        encoder,
        scaler
    )

    # -------------------------------
    # Load model
    # -------------------------------
    model = load_model()

    # -------------------------------
    # Predictions
    # -------------------------------
    y_pred = predict(model, X)

    y_prob = predict_probability(model, X)

    # -------------------------------
    # Save predictions
    # -------------------------------
    os.makedirs("reports", exist_ok=True)

    prediction_results = df.copy()

    # Add actual label if available
    if y_true is not None:
        prediction_results["actual"] = y_true.values

    # Model prediction
    prediction_results["prediction"] = y_pred

    # Prediction probability
    prediction_results["probability"] = y_prob[:, 1]

    # Correct or incorrect prediction
    if y_true is not None:
        prediction_results["correct_prediction"] = (
            prediction_results["actual"] ==
            prediction_results["prediction"]
        )

    prediction_results.to_csv(
        f"reports/{os.path.basename(csv_path).replace('.csv', '_predictions.csv')}",
        index=False
    )

    print("Predictions saved successfully.")

    # -------------------------------
    # Evaluate if labels exist
    # -------------------------------
    if y_true is not None:

        accuracy = accuracy_score(y_true, y_pred)

        precision = precision_score(y_true, y_pred)

        recall = recall_score(y_true, y_pred)

        f1 = f1_score(y_true, y_pred)

        roc_auc = roc_auc_score(
            y_true,
            y_prob[:, 1]
        )

        print("\nAccuracy")
        print(accuracy)

        print("\nClassification Report")
        print(classification_report(y_true, y_pred))

        print("\nConfusion Matrix")
        print(confusion_matrix(y_true, y_pred))

        metrics = pd.DataFrame({
            "file": [os.path.basename(csv_path)],
            "accuracy": [accuracy],
            "precision": [precision],
            "recall": [recall],
            "f1_score": [f1],
            "roc_auc": [roc_auc]
        })

        metrics_path = "reports/metrics.csv"

        if os.path.exists(metrics_path):

            old_metrics = pd.read_csv(metrics_path)

            metrics = pd.concat(
                [old_metrics, metrics],
                ignore_index=True
            )

        metrics.to_csv(
            metrics_path,
            index=False
        )

        print("Metrics saved successfully.")

    return prediction_results


if __name__ == "__main__":

    prediction_pipeline(
        "data/processed/future_data/day_002.csv"
    )