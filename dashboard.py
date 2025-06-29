# Fraud Detection System - Part 3: Real-Time Dashboard
# Summer Internship Project - Grant Thornton Bharat

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
    page_title="🏦 Fraud Detection Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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

# Header
st.markdown('<h1 class="main-header">🏦 Real-Time Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Grant Thornton Bharat - Summer Internship Project</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🔧 Control Panel")

# Load model function
@st.cache_resource
def load_models():
    """Load the trained models"""
    try:
        model = joblib.load('fraud_detection_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        return model, scaler, True
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run Part 2 first.")
        return None, None, False

# Load models
model, scaler, models_loaded = load_models()

if not models_loaded:
    st.stop()

st.sidebar.success("✅ AI Models Loaded Successfully!")

# Fraud detection function
def detect_fraud_realtime(transaction_data, model, feature_cols):
    """Real-time fraud detection function"""
    # Prepare features in the correct order
    features = []
    for col in feature_cols:
        if col in transaction_data:
            features.append(transaction_data[col])
        else:
            features.append(0)
    
    features = np.array(features).reshape(1, -1)
    
    # Get fraud probability
    fraud_prob = model.predict_proba(features)[0, 1]
    is_fraud = fraud_prob > 0.5
    
    # Risk level
    if fraud_prob > 0.8:
        risk_level = "HIGH"
        risk_color = "🔴"
    elif fraud_prob > 0.5:
        risk_level = "MEDIUM"
        risk_color = "🟡"
    elif fraud_prob > 0.2:
        risk_level = "LOW"
        risk_color = "🟢"
    else:
        risk_level = "VERY LOW"
        risk_color = "🟢"
    
    return {
        'fraud_probability': fraud_prob,
        'is_fraud': is_fraud,
        'risk_level': risk_level,
        'risk_color': risk_color
    }

# Generate random transaction
def generate_random_transaction():
    """Generate a random transaction for simulation"""
    fake = Faker()
    
    # Basic transaction info
    customer_id = f"CUST_{random.randint(1, 1000):06d}"
    
    # Merchant categories
    merchant_categories = ['Grocery', 'Gas Station', 'Restaurant', 'Online Shopping', 
                          'ATM', 'Pharmacy', 'Department Store', 'Hotel', 'Airlines']
    merchant_category = random.choice(merchant_categories)
    
    # Generate suspicious or normal transaction
    is_suspicious = random.random() < 0.1  # 10% chance of suspicious transaction
    
    if is_suspicious:
        # Suspicious patterns
        amount = random.choice([
            round(random.uniform(2000, 5000), 2),  # High amount
            round(random.uniform(1, 5), 2)         # Very small amount
        ])
        hour = random.choice([2, 3, 4, 5])  # Very early morning
        location = random.choice(["Unknown Location", "Foreign Country", "High-Risk Area"])
    else:
        # Normal transaction
        if merchant_category == 'Grocery':
            amount = round(random.uniform(20, 200), 2)
        elif merchant_category == 'Gas Station':
            amount = round(random.uniform(30, 80), 2)
        elif merchant_category == 'Restaurant':
            amount = round(random.uniform(15, 150), 2)
        else:
            amount = round(random.uniform(10, 300), 2)
        
        hour = random.randint(8, 22)  # Normal business hours
        location = fake.city() + ", " + fake.state()
    
    # Create transaction
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

# Feature columns (same as in Part 2)
feature_columns = [
    'amount', 'amount_log', 'hour', 'day_of_week', 'is_weekend', 'is_night',
    'is_round_amount', 'customer_avg_amount', 'customer_amount_std',
    'customer_transaction_count', 'amount_deviation', 'merchant_risk_score'
]

# Sidebar controls
st.sidebar.subheader("🎮 Simulation Controls")

# Auto-generate transactions
auto_generate = st.sidebar.checkbox("🔄 Auto-Generate Transactions", value=False)
generation_speed = st.sidebar.slider("⚡ Generation Speed (seconds)", 1, 10, 3)

# Manual transaction generator
if st.sidebar.button("🎲 Generate Random Transaction"):
    new_transaction = generate_random_transaction()
    
    # Detect fraud
    fraud_result = detect_fraud_realtime(new_transaction, model, feature_columns)
    
    # Add results to transaction
    new_transaction.update(fraud_result)
    
    # Add to session state
    st.session_state.transactions.append(new_transaction)
    
    # Add to alerts if high risk
    if fraud_result['risk_level'] in ['HIGH', 'MEDIUM']:
        st.session_state.fraud_alerts.append({
            'timestamp': new_transaction['timestamp'],
            'transaction_id': new_transaction['transaction_id'],
            'amount': new_transaction['amount'],
            'risk_level': fraud_result['risk_level'],
            'fraud_probability': fraud_result['fraud_probability']
        })

# Auto-generation
if auto_generate:
    # Create placeholder for auto-generated transactions
    placeholder = st.empty()
    
    # Auto-generate transactions
    if len(st.session_state.transactions) == 0 or \
       (datetime.now() - st.session_state.transactions[-1]['timestamp']).seconds >= generation_speed:
        
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
        
        time.sleep(generation_speed)
        st.rerun()

# Clear data
if st.sidebar.button("🗑️ Clear All Data"):
    st.session_state.transactions = []
    st.session_state.fraud_alerts = []
    st.rerun()

# Main dashboard
if len(st.session_state.transactions) > 0:
    
    # Convert to DataFrame
    df_transactions = pd.DataFrame(st.session_state.transactions)
    
    # Key Metrics
    st.subheader("📊 Real-Time Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_transactions = len(df_transactions)
        st.metric("🔢 Total Transactions", total_transactions)
    
    with col2:
        fraud_count = len(df_transactions[df_transactions['is_fraud'] == True])
        st.metric("🚨 Fraud Alerts", fraud_count)
    
    with col3:
        fraud_rate = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0
        st.metric("📈 Fraud Rate", f"{fraud_rate:.1f}%")
    
    with col4:
        total_amount = df_transactions['amount'].sum()
        st.metric("💰 Total Amount", f"${total_amount:,.2f}")
    
    # Recent Transactions
    st.subheader("🕒 Recent Transactions")
    
    # Display last 10 transactions
    recent_transactions = df_transactions.tail(10).copy()
    recent_transactions['Risk'] = recent_transactions.apply(
        lambda x: f"{x['risk_color']} {x['risk_level']}", axis=1
    )
    
    display_cols = ['timestamp', 'transaction_id', 'amount', 'merchant_category', 'Risk', 'fraud_probability']
    st.dataframe(
        recent_transactions[display_cols].sort_values('timestamp', ascending=False),
        use_container_width=True
    )
    
    # Fraud Alerts
    if len(st.session_state.fraud_alerts) > 0:
        st.subheader("🚨 Fraud Alerts")
        
        for alert in st.session_state.fraud_alerts[-5:]:  # Show last 5 alerts
            risk_class = "high-risk" if alert['risk_level'] == 'HIGH' else "medium-risk"
            
            st.markdown(f"""
            <div class="alert-box {risk_class}">
                <strong>⚠️ {alert['risk_level']} RISK TRANSACTION</strong><br>
                ID: {alert['transaction_id']}<br>
                Amount: ${alert['amount']:,.2f}<br>
                Fraud Probability: {alert['fraud_probability']:.1%}<br>
                Time: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            """, unsafe_allow_html=True)
    
    # Charts
    st.subheader("📈 Analytics Dashboard")
    
    # Create charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Fraud probability distribution
        fig_dist = px.histogram(
            df_transactions, 
            x='fraud_probability',
            nbins=20,
            title='Fraud Probability Distribution',
            labels={'fraud_probability': 'Fraud Probability', 'count': 'Number of Transactions'}
        )
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Transactions by risk level
        risk_counts = df_transactions['risk_level'].value_counts()
        fig_risk = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title='Transactions by Risk Level'
        )
        fig_risk.update_layout(height=400)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    # Time series chart
    if len(df_transactions) > 1:
        df_time = df_transactions.copy()
        df_time['minute'] = df_time['timestamp'].dt.floor('min')
        time_series = df_time.groupby('minute').agg({
            'amount': 'sum',
            'is_fraud': 'sum'
        }).reset_index()
        
        fig_time = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Transaction Volume Over Time', 'Fraud Alerts Over Time'),
            shared_xaxes=True
        )
        
        fig_time.add_trace(
            go.Scatter(x=time_series['minute'], y=time_series['amount'], 
                      mode='lines+markers', name='Transaction Amount'),
            row=1, col=1
        )
        
        fig_time.add_trace(
            go.Scatter(x=time_series['minute'], y=time_series['is_fraud'], 
                      mode='lines+markers', name='Fraud Count', line=dict(color='red')),
            row=2, col=1
        )
        
        fig_time.update_layout(height=500, title_text="Transaction Timeline")
        st.plotly_chart(fig_time, use_container_width=True)

else:
    # Welcome message
    st.info("👋 Welcome to the Fraud Detection Dashboard! Click 'Generate Random Transaction' to start the simulation.")
    
    # Feature explanation
    st.subheader("🧠 How Our AI Detects Fraud")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔍 Key Fraud Indicators:**
        - 💰 Unusual transaction amounts
        - 🕐 Transactions at odd hours (2-6 AM)
        - 📍 Suspicious locations
        - 🔄 Rapid successive transactions
        - 📊 Deviation from customer's normal behavior
        """)
    
    with col2:
        st.markdown("""
        **⚡ Real-time Features:**
        - Instant fraud scoring (< 1 second)
        - Risk level classification
        - Automated alerts
        - Live dashboard updates
        - Transaction monitoring
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    🏦 <strong>Fraud Detection System</strong>
    Powered by Machine Learning & Real-time Analytics
</div>
""", unsafe_allow_html=True)

# Instructions
st.sidebar.markdown("---")
st.sidebar.markdown("""
**📋 Instructions:**
1. Click "Generate Random Transaction" to create test data
2. Enable "Auto-Generate" for continuous simulation
3. Watch the dashboard update in real-time
4. Monitor fraud alerts and analytics
5. Adjust generation speed as needed
""")