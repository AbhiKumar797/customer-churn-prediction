from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import os

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API to predict customer churn using Machine Learning",
    version="1.0.0"
)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'models', 'churn_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Define the input data structure
class CustomerData(BaseModel):
    gender: str = "Male"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = 12
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "DSL"
    OnlineSecurity: str = "Yes"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 50.0
    TotalCharges: float = 600.0
    
    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 50.0,
                "TotalCharges": 600.0
            }
        }

# Define the response structure
class PredictionResponse(BaseModel):
    churn_prediction: str
    churn_probability: float
    retention_probability: float
    risk_level: str

# Root endpoint
@app.get("/")
def root():
    return {
        "message": "Welcome to Customer Churn Prediction API",
        "docs": "/docs",
        "health": "/health"
    }

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    """
    Predict customer churn based on customer data.
    
    - **churn_prediction**: Whether the customer will churn (Yes/No)
    - **churn_probability**: Probability of churning (0-1)
    - **retention_probability**: Probability of staying (0-1)
    - **risk_level**: Risk category (Low/Medium/High)
    """
    # Convert input to DataFrame
    input_data = pd.DataFrame([customer.model_dump()])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    # Determine risk level
    churn_prob = probabilities[1]
    if churn_prob < 0.3:
        risk_level = "Low"
    elif churn_prob < 0.6:
        risk_level = "Medium"
    else:
        risk_level = "High"
    
    return PredictionResponse(
        churn_prediction="Yes" if prediction == 1 else "No",
        churn_probability=round(probabilities[1], 4),
        retention_probability=round(probabilities[0], 4),
        risk_level=risk_level
    )

# Batch prediction endpoint
@app.post("/predict/batch")
def predict_batch(customers: list[CustomerData]):
    """
    Predict churn for multiple customers at once.
    """
    results = []
    for customer in customers:
        input_data = pd.DataFrame([customer.model_dump()])
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        results.append({
            "churn_prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(probabilities[1], 4)
        })
    
    return {"predictions": results, "total": len(results)}
