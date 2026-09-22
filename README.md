Customer Churn Monitoring & Agentic Analysis

An end-to-end customer churn prediction system that combines a machine learning model with a Gemini-powered analysis agent.

What it does

Predicts customer churn risk using a trained ML model.

Returns churn probability.

Uses Gemini to explain the main reasons and suggest practical actions.

Provides a simple web UI through Flask.

Architecture

┌──────────────────┐
│   Customer Data  │
│   (Web UI / JSON)│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Flask API      │
│  /predict /agent │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Preprocessing    │
│ Encoder + Scaler │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   ML Model       │
│  Churn Prediction│
└────────┬─────────┘
         │
         ├──────────────► Churn Risk + Probability
         │
         ▼
┌──────────────────┐
│ Gemini Analysis  │
│   Agent          │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Explanation +    │
│ Actions to Improve│
└──────────────────┘

Tech Stack

Python

Flask

Scikit-learn

Pandas

Gemini API

HTML / CSS / JavaScript

Joblib

Run locally

Install dependencies:

pip install flask pandas scikit-learn joblib python-dotenv openai

Create .env:

GOOGLE_API_KEY=your_api_key

Start the application:

python app.py

Then open the local URL shown by Flask in your terminal, usually:

http://127.0.0.1:5000

Example Input

{
  "customer_id": 1042,
  "tenure_months": 36,
  "monthly_charges": 45,
  "usage_minutes": 500,
  "support_calls": 1,
  "payment_delay_days": 0,
  "autopay": 1,
  "num_services": 5,
  "satisfaction_score": 9,
  "contract_type": "one-year",
  "marketing_segment": "A",
  "signup_channel": "organic"
}

Example Output

ML Prediction

Risk: Low
Churn Probability: 41.25%

Gemini Agent Analysis

Why?

One-year contract

Moderate monthly charges

Moderate churn probability

How to Improve

Encourage longer-term renewal

Maintain autopay and payment reliability

Continue customer engagement


Note

The ML model handles the prediction. Gemini is used to generate a short, human-readable analysis based on the customer data and ML result.
