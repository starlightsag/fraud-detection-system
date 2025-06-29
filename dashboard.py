# Modified version of dashboard.py to always auto-generate transactions automatically

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import time
import random
from faker import Faker
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="\U0001F3E6 Fraud Detection Dashboard",
    page_icon="\U0001F6A8",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .medium-risk {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    .low-risk {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
if 'fraud_alerts' not in st.session_state:
    st.session_state.fraud_alerts = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

st.markdown('<h1 class="main-header">\U0001F3E6 Real-Time Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Grant Thornton Bharat - Summer Internship Project</p>', unsafe_allow_html=True)

st.sidebar.header("\U0001F527 Control Panel")

@st.cache_resource
def load_models():
    try:
        model = joblib.load('fraud_detection_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        return model, scaler, True
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run Part 2 first.")
        return None, None, False

model, scaler, models_loaded = load_models()
if not models_loaded:
    st.stop()

st.sidebar.success("✅ AI Models Loaded Successfully!")

def detect_fraud_realtime(transaction_data, model, feature_cols):
    features = [transaction_data.get(col, 0) for col in feature_cols]
    features = np.array(features).reshape(1, -1)
    fraud_prob = model.predict_proba(features)[0, 1]
    is_fraud = fraud_prob > 0.5
    if fraud_prob > 0.8:
        risk_level = "HIGH"; risk_color = "\U0001F534"
    elif fraud_prob > 0.5:
        risk_level = "MEDIUM"; risk_color = "\U0001F7E1"
    elif fraud_prob > 0.2:
        risk_level = "LOW"; risk_color = "\U0001F7E2"
    else:
        risk_level = "VERY LOW"; risk_color = "\U0001F7E2"
    return {'fraud_probability': fraud_prob, 'is_fraud': is_fraud, 'risk_level': risk_level, 'risk_color': risk_color}

def generate_random_transaction():
    fake = Faker()
    customer_id = f"CUST_{random.randint(1, 1000):06d}"
    merchant_categories = ['Grocery', 'Gas Station', 'Restaurant', 'Online Shopping', 'ATM', 'Pharmacy', 'Department Store', 'Hotel', 'Airlines']
    merchant_category = random.choice(merchant_categories)
    is_suspicious = random.random() < 0.1
    if is_suspicious:
        amount = random.choice([round(random.uniform(2000, 5000), 2), round(random.uniform(1, 5), 2)])
        hour = random.choice([2, 3, 4, 5])
        location = random.choice(["Unknown Location", "Foreign Country", "High-Risk Area"])
    else:
        amount = round(random.uniform(20, 300), 2)
        hour = random.randint(8, 22)
        location = fake.city() + ", " + fake.state()
    transaction = {
        'transaction_id': f"TXN_{random.randint(10000000, 99999999)}",
        'customer_id': customer_id,
        'timestamp': datetime.now(),
        'amount': amount,
        'merchant_category': merchant_category,
        'location': location,
        'hour': hour,
        'day_of_week': datetime.now().weekday(),
        'is_weekend': 1 if datetime.now().weekday() >= 5 else 0,
        'is_night': 1 if hour <= 6 or hour >= 22 else 0,
        'amount_log': np.log1p(amount),
        'is_round_amount': 1 if amount % 1 == 0 else 0,
        'customer_avg_amount': round(random.uniform(50, 200), 2),
        'customer_amount_std': round(random.uniform(10, 50), 2),
        'customer_transaction_count': random.randint(5, 100),
        'amount_deviation': abs(amount - random.uniform(50, 200)) / (random.uniform(10, 50) + 1),
        'merchant_risk_score': random.uniform(0.01, 0.15)
    }
    return transaction

feature_columns = [
    'amount', 'amount_log', 'hour', 'day_of_week', 'is_weekend', 'is_night',
    'is_round_amount', 'customer_avg_amount', 'customer_amount_std',
    'customer_transaction_count', 'amount_deviation', 'merchant_risk_score'
]

# Auto-Generate always enabled
auto_generate = True
st.sidebar.checkbox("\U0001F501 Auto-Generate Transactions", value=True, disabled=True)
generation_speed = st.sidebar.slider("⚡ Generation Speed (seconds)", 1, 10, 3)

# Track time of last generation
if 'last_generated_time' not in st.session_state:
    st.session_state.last_generated_time = datetime.now() - timedelta(seconds=generation_speed)

now = datetime.now()
elapsed = (now - st.session_state.last_generated_time).total_seconds()

if auto_generate and elapsed >= generation_speed:
    new_transaction = generate_random_transaction()
    fraud_result = detect_fraud_realtime(new_transaction, model, feature_columns)
    new_transaction.update(fraud_result)
    st.session_state.transactions.append(new_transaction)

    if fraud_result['risk_level'] in ['HIGH', 'MEDIUM']:
        st.session_state.fraud_alerts.append({
            'timestamp': new_transaction['timestamp'],
            'transaction_id': new_transaction['transaction_id'],
            'amount': new_transaction['amount'],
            'risk_level': fraud_result['risk_level'],
            'fraud_probability': fraud_result['fraud_probability']
        })

    st.session_state.last_generated_time = now
    st.rerun()

# Clear data
if st.sidebar.button("\U0001F5D1️ Clear All Data"):
    st.session_state.transactions = []
    st.session_state.fraud_alerts = []
    st.session_state.last_generated_time = datetime.now() - timedelta(seconds=generation_speed)
    st.rerun()

# You can append the remaining dashboard metrics/plots logic below this point.
