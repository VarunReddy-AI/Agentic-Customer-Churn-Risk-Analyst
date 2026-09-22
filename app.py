import os
import json

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from openai import OpenAI

from src.models.predict import predict_customer


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")


gemini = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=GOOGLE_API_KEY
)


app = Flask(__name__)


# =========================================================
# HOME
# =========================================================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# =========================================================
# EXISTING ML PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    prediction, probability = predict_customer(data)

    return jsonify({
        "prediction": int(prediction),
        "probability": float(probability)
    })


# =========================================================
# AGENT
# =========================================================

@app.route("/agent", methods=["POST"])
def agent():

    data = request.get_json()

    if not data:
        return jsonify({
            "error": "No customer data provided"
        }), 400

    # -----------------------------------------------------
    # STEP 1: ML MODEL
    # -----------------------------------------------------

    prediction, probability = predict_customer(data)

    prediction = int(prediction)
    probability = float(probability)

    risk = "High" if prediction == 1 else "Low"

    ml_result = {
        "prediction": prediction,
        "churn_probability": probability,
        "risk_level": risk
    }

    # -----------------------------------------------------
    # STEP 2: GEMINI AGENT
    # -----------------------------------------------------

    prompt = f"""
You are a customer churn analysis assistant.

CUSTOMER DATA:
{json.dumps(data, indent=2)}

ML PREDICTION:
{json.dumps(ml_result, indent=2)}

Return ONLY HTML using exactly this structure:

<h2>Why?</h2>

<ul>
<li>Short reason</li>
<li>Short reason</li>
<li>Short reason</li>
</ul>

<h2>How to Improve</h2>

<ul>
<li>Short practical action</li>
<li>Short practical action</li>
<li>Short practical action</li>
</ul>

Rules:
- Do NOT repeat the risk or churn probability.
- Keep the response short.
- Give 2-4 reasons.
- Give 2-3 practical actions.
- Use only information provided.
- Do not invent information.
- Return only HTML.
"""

    response = gemini.chat.completions.create(
        model="gemini-3.5-flash-lite",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    explanation = response.choices[0].message.content

    # -----------------------------------------------------
    # FINAL RESPONSE
    # -----------------------------------------------------

    return jsonify({
        "customer": data,
        "ml_result": ml_result,
        "llm_explanation": explanation
    })


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)