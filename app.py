from flask import Flask, request, jsonify

from src.models.predict import predict_customer

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Customer Churn Prediction API",
        "status": "Running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    prediction, probability = predict_customer(data)

    return jsonify({
        "prediction": int(prediction),
        "probability": float(probability)
    })


if __name__ == "__main__":
    app.run(debug=True)