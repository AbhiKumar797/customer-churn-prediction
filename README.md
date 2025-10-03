# 📊 Customer Churn Prediction using Machine Learning

This project predicts customer churn (whether a customer will leave or stay) for a telecom company using **machine learning models**. The dataset used is the popular **Telco Customer Churn Dataset**, which contains customer demographics, services subscribed, and account information.

---

## 🚀 Features

* Data cleaning and preprocessing
* Handling categorical variables with encoding
* Balancing imbalanced dataset using **SMOTE**
* Training multiple ML models:

  * Decision Tree
  * Random Forest
  * XGBoost
* Model evaluation using accuracy, confusion matrix, and classification report
* Saving trained models with **Pickle**

---

## 📂 Dataset

* **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
* **Features:** Customer demographics, service details, billing info
* **Target Variable:** `Churn` (Yes/No)

---

## 🛠️ Technologies Used

* **Python** (Data Science & ML)
* **Libraries:**

  * `numpy`, `pandas` → Data manipulation
  * `matplotlib`, `seaborn` → Data visualization
  * `scikit-learn` → ML models, preprocessing, evaluation
  * `imbalanced-learn (SMOTE)` → Oversampling minority class
  * `xgboost` → Gradient boosting model
  * `pickle` → Model persistence

---

## 📊 Model Training & Evaluation

The following models were trained and evaluated:

* **Decision Tree Classifier**
* **Random Forest Classifier**
* **XGBoost Classifier**

Each model was tested using **accuracy score, confusion matrix, and classification report** to determine the best-performing approach.

---

## 📁 Project Structure

```
├── Customer_Churn_Prediction_using_ML.ipynb   # Jupyter Notebook with full workflow
├── WA_Fn-UseC_-Telco-Customer-Churn.csv       # Dataset
├── README.md                                  # Project documentation
└── models/                                    # (Optional) Saved trained models
```

---

## ⚡ How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/customer-churn-prediction.git
   cd customer-churn-prediction
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Open Jupyter Notebook:

   ```bash
   jupyter notebook Customer_Churn_Prediction_using_ML.ipynb
   ```

---

## 📌 Future Enhancements

* Deploy the model using **Flask/Django API**
* Build an interactive **dashboard (Streamlit/Plotly Dash)**
* Hyperparameter tuning for better accuracy

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

👨‍💻 Developed by **Abhishek Kumar**
