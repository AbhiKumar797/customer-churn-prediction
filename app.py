import streamlit as st
import pandas as pd
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Churn Predictor | TelcoAI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Black & Red Theme
st.markdown("""
<style>
    /* Hide sidebar completely */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0d0d0d 0%, #1a1a1a 100%);
    }
    
    /* Navbar styling */
    .navbar {
        background: linear-gradient(90deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 1rem 2rem;
        border-bottom: 2px solid #dc3545;
        margin: -6rem -4rem 2rem -4rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .navbar-brand {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 1px;
    }
    
    .navbar-brand span {
        color: #dc3545;
    }
    
    .navbar-links {
        display: flex;
        gap: 2rem;
    }
    
    .navbar-links a {
        color: #b0b0b0;
        text-decoration: none;
        font-size: 0.9rem;
        font-weight: 500;
        transition: color 0.3s;
    }
    
    .navbar-links a:hover {
        color: #dc3545;
    }
    
    /* Card styling */
    .card {
        background: #1e1e1e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .card-header {
        color: #dc3545;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #333;
    }
    
    /* Form labels */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #e0e0e0 !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }
    
    /* Input styling */
    .stSelectbox > div > div {
        background-color: #2d2d2d !important;
        border: 1px solid #444 !important;
        color: #fff !important;
        font-size: 0.85rem !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #dc3545 0%, #b02a37 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.5px !important;
        border-radius: 6px !important;
        transition: all 0.3s !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #e04555 0%, #c03545 100%) !important;
        box-shadow: 0 4px 15px rgba(220, 53, 69, 0.4) !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #888 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
    }
    
    /* Result boxes */
    .result-high-risk {
        background: linear-gradient(135deg, #2d1f1f 0%, #1a1a1a 100%);
        border: 1px solid #dc3545;
        border-left: 4px solid #dc3545;
        border-radius: 8px;
        padding: 1.5rem;
    }
    
    .result-low-risk {
        background: linear-gradient(135deg, #1f2d1f 0%, #1a1a1a 100%);
        border: 1px solid #28a745;
        border-left: 4px solid #28a745;
        border-radius: 8px;
        padding: 1.5rem;
    }
    
    .risk-label {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .risk-high { color: #dc3545; }
    .risk-low { color: #28a745; }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #dc3545 !important;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Section headers */
    .section-title {
        color: #ffffff;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.75rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #333;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'churn_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Navbar
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">⚡ Telco<span>AI</span></div>
    <div class="navbar-links">
        <a href="#">Dashboard</a>
        <a href="#">Analytics</a>
        <a href="#">Predictions</a>
        <a href="#">Settings</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Page Title
st.markdown("<h2 style='color: #fff; font-weight: 600; margin-bottom: 0.5rem;'>Customer Churn Prediction</h2>", unsafe_allow_html=True)
st.markdown("<p style='color: #888; font-size: 0.9rem; margin-bottom: 2rem;'>Enter customer information to predict churn probability</p>", unsafe_allow_html=True)

# Create 3 columns for the form
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='card-header'>Customer Profile</div>", unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

with col2:
    st.markdown("<div class='card-header'>Services</div>", unsafe_allow_html=True)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col3:
    st.markdown("<div class='card-header'>Billing & Contract</div>", unsafe_allow_html=True)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges))

# Hidden services (using defaults for cleaner UI)
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="dp", label_visibility="collapsed") if False else "No"
tech_support = "No"
streaming_tv = "No"
streaming_movies = "No"

# Create input dataframe
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [senior_citizen],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'PhoneService': [phone_service],
    'MultipleLines': [multiple_lines],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

st.markdown("<br>", unsafe_allow_html=True)

# Predict button centered
col_left, col_center, col_right = st.columns([1, 1, 1])
with col_center:
    predict_clicked = st.button("Analyze Customer Risk", use_container_width=True)

# Results section
if predict_clicked:
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    churn_prob = probability[1] * 100
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Results in columns
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-high-risk">
                <div class="risk-label risk-high">⚠ HIGH CHURN RISK</div>
                <p style="color: #ccc; font-size: 0.85rem; margin: 0;">
                    This customer has a <strong style="color: #dc3545;">{churn_prob:.1f}%</strong> probability of leaving.
                    Immediate retention actions are recommended.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-low-risk">
                <div class="risk-label risk-low">✓ LOW CHURN RISK</div>
                <p style="color: #ccc; font-size: 0.85rem; margin: 0;">
                    This customer has a <strong style="color: #28a745;">{probability[0]*100:.1f}%</strong> probability of staying.
                    Continue standard engagement.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with res_col2:
        st.markdown("<div class='card-header'>Risk Analysis</div>", unsafe_allow_html=True)
        
        # Risk meter
        risk_color = "#dc3545" if churn_prob > 50 else "#ffc107" if churn_prob > 30 else "#28a745"
        st.progress(probability[1])
        
        # Key metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Churn Risk", f"{churn_prob:.0f}%")
        with m2:
            st.metric("Monthly Value", f"${monthly_charges:.0f}")
        with m3:
            risk_level = "High" if churn_prob > 50 else "Medium" if churn_prob > 30 else "Low"
            st.metric("Risk Level", risk_level)

# Footer
st.markdown("""
<div class="footer">
    TelcoAI Churn Prediction System • Built with Machine Learning • © 2024
</div>
""", unsafe_allow_html=True)
