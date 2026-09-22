Agentic Customer Churn Risk Analyst

An ML-powered customer churn prediction system extended with a Gemini LLM analysis layer and Flask API.

Overview

This project predicts whether a customer is likely to churn and estimates churn probability using a trained machine-learning model. The prediction and customer information are then passed to Gemini, which produces a short explanation of the main risk factors and practical actions that may help reduce churn risk.

Architecture

Customer Data
      |
      v
   Flask App
      |
      v
ML Preprocessing
      |
      v
Trained Churn Model
      |
      +----> Churn Prediction
      |      Churn Probability
      |
      v
    Gemini
      |
      v
Risk Explanation
      |
      +----> Why?
      |
      +----> How to Improve

Features

Machine Learning

Customer churn classification

Churn probability prediction

One-hot encoding for categorical features

StandardScaler for numerical features

Saved model, encoder, and scaler artifacts

Class balancing using sample weights during training

Gemini Analysis

Gemini receives the customer information and ML prediction and generates:

A short explanation of the main risk factors

A few practical actions to improve retention

Tech Stack

Python

Flask

scikit-learn

Pandas

NumPy

CatBoost

Gemini API

HTML/CSS/JavaScript

Joblib

Git/GitHub

Project Structure

customer-churn-monitoring-mlops/
|
├── src/
│   ├── config/
│   ├── data/
│   ├── exception/
│   ├── features/
│   ├── logger/
│   ├── models/
│   ├── pipeline/
│   └── utils/
|
├── templates/
│   └── index.html
|
├── reports/
├── tests/
|
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore

Setup

1. Clone the repository

git clone <your-repository-url>
cd customer-churn-monitoring-mlops

2. Create a virtual environment

Windows:

python -m venv venv
.\venv\Scripts\Activate.ps1

3. Install dependencies

python -m pip install -r requirements.txt

4. Configure Gemini

Create a .env file in the project root:

GOOGLE_API_KEY=your_gemini_api_key

Never commit .env or your API key to GitHub.

5. Run the application

python app.py

Open:

http://127.0.0.1:5000/

API Endpoints

Home

GET /

Loads the customer churn analysis interface.

ML Prediction

POST /predict

Returns the ML prediction and churn probability.

Agent Analysis

POST /agent

Runs the customer data through the ML prediction pipeline and sends the prediction and customer information to Gemini for concise risk analysis.

Example Input

{
  "customer_id": "1042",
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

Example Result

Prediction

Risk: Low
Churn Probability: 41.25%

Why?
- Moderate churn probability despite low risk classification
- One-year contract
- Moderate monthly charges

How to Improve
- Encourage longer-term renewal
- Maintain payment reliability
- Continue customer engagement

Important Notes

Use a compatible scikit-learn version when loading saved model/preprocessing artifacts.

Categorical values supplied during prediction must match categories expected by the saved encoder.

customer_id is treated as an identifier and is not used as an ML feature.

Gemini explains the ML result; it does not replace the trained churn model.

The current implementation uses an LLM analysis layer. It can later be extended with genuine function/tool calling.

Future Improvements

Add Gemini function/tool calling

Add customer statistics tools

Add model explainability

Add churn monitoring and drift detection

Add automated retention recommendations

Add authentication and production WSGI deployment

Add automated CI/CD testing

Containerize and deploy the application

Disclaimer

This project is for educational and demonstration purposes. Model predictions should be validated before being used for real customer decisions.