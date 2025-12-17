# 📊 Customer Churn Prediction - End-to-End ML Project

A complete machine learning project that predicts customer churn for a telecom company. This project demonstrates the full ML lifecycle from data preprocessing to model deployment.

---

## 🎯 Project Overview

Customer churn prediction is a critical business problem. This project:
- Trains ML models to predict which customers are likely to leave
- Provides a **Streamlit web application** for easy predictions
- Exposes a **FastAPI REST API** for integration with other systems

---

## 🎬 Project Demo

[![Customer Churn Prediction Demo](https://img.youtube.com/vi/Yd_SA581lSI/maxresdefault.jpg)](https://www.youtube.com/watch?v=Yd_SA581lSI)

▶️ **[Watch Full Project Walkthrough on YouTube](https://www.youtube.com/watch?v=Yd_SA581lSI)**

---

## 🏗️ Project Structure

```
customer-churn-prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
├── models/
│   └── churn_model.pkl                         # Trained model
├── src/
│   └── train_model.py                          # Training script
├── Customer_Churn_Prediction_using_ML.ipynb    # Jupyter Notebook (EDA)
├── app.py                                       # Streamlit web app
├── api.py                                       # FastAPI REST API
├── requirements.txt                             # Dependencies
└── README.md                                    # Documentation
```

---

## 🚀 Features

### ✅ Machine Learning Pipeline
- **Data Cleaning:** Handles missing values and data type conversions
- **Feature Engineering:** One-Hot Encoding for categorical variables
- **Class Imbalance:** SMOTE oversampling (applied correctly AFTER train-test split to prevent data leakage)
- **Model Training:** Random Forest Classifier with scikit-learn pipeline

### ✅ Web Application (Streamlit)
- Interactive form to input customer details
- Real-time churn prediction
- Visual probability gauge

### ✅ REST API (FastAPI)
- `/predict` - Single customer prediction
- `/predict/batch` - Batch predictions
- Auto-generated API documentation at `/docs`

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.10+ |
| **ML Libraries** | scikit-learn, imbalanced-learn, XGBoost |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Web App** | Streamlit |
| **API** | FastAPI, Uvicorn, Pydantic |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 77.4% |
| **Precision (Churn)** | 57% |
| **Recall (Churn)** | 59% |
| **F1-Score (Churn)** | 58% |

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model:**
   ```bash
   python src/train_model.py
   ```

---

## 🎮 Usage

### Run the Streamlit Web App
```bash
streamlit run app.py
```
Then open your browser to `http://localhost:8501`

### Run the FastAPI Server
```bash
uvicorn api:app --reload
```
Then open your browser to `http://localhost:8000/docs` for the interactive API documentation.

### Example API Request
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "gender": "Male",
           "SeniorCitizen": 0,
           "Partner": "Yes",
           "Dependents": "No",
           "tenure": 12,
           "PhoneService": "Yes",
           "MultipleLines": "No",
           "InternetService": "DSL",
           "OnlineSecurity": "No",
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
         }'
```

---

## 📚 Key Learnings

1. **Data Leakage Prevention:** SMOTE is applied only to training data, not the entire dataset
2. **Proper Encoding:** One-Hot Encoding for nominal variables instead of Label Encoding
3. **Pipeline Architecture:** Using scikit-learn Pipeline ensures consistent preprocessing
4. **API Design:** RESTful API with proper documentation and data validation

---

## 📌 Future Enhancements

- Deploy the API on cloud (AWS/GCP/Azure)
- Add Docker containerization
- Implement CI/CD pipeline
- Add model monitoring and retraining

---

## 👨‍💻 Author

**Abhishek**
⭐ If you found this project helpful, please give it a star!
