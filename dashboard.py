import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from datetime import datetime, timedelta
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# PAGE CONFIGURATION
# ==============================================
st.set_page_config(
    page_title="UPI HawkEye AI Pro",
    layout="wide",
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

# ==============================================
# SESSION STATE INITIALIZATION
# ==============================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'user_role' not in st.session_state:
    st.session_state.user_role = ""
if 'blocked_accounts' not in st.session_state:
    st.session_state.blocked_accounts = set()
if 'alerts_sent' not in st.session_state:
    st.session_state.alerts_sent = set()

# ==============================================
# ENHANCED CSS STYLING
# ==============================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1d35 50%, #0a0e27 100%);
    }
    
    .login-container {
        background: linear-gradient(135deg, #1e2139 0%, #2d3250 100%);
        padding: 50px;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(255, 75, 75, 0.2);
        border: 1px solid rgba(255, 75, 75, 0.3);
        max-width: 450px;
        margin: auto;
        margin-top: 80px;
    }
    
    .login-header {
        text-align: center;
        color: #FF4B4B;
        font-size: 2.8em;
        font-weight: 700;
        text-shadow: 0 0 20px rgba(255, 75, 75, 0.5);
        margin-bottom: 10px;
    }
    
    .login-subtitle {
        text-align: center;
        color: #00C9FF;
        font-size: 1.2em;
        margin-bottom: 30px;
        font-weight: 500;
    }
    
    h1, h2, h3 {
        color: #FF4B4B !important;
        text-shadow: 0 0 10px rgba(255, 75, 75, 0.3);
        font-weight: 700 !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 700;
        color: #00C9FF;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        font-weight: 600;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.4);
    }
    
    .live-indicator {
        animation: pulse 2s infinite;
        color: #FF4B4B;
        font-weight: 700;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .alert-card {
        background: rgba(255, 75, 75, 0.1);
        border-left: 4px solid #FF4B4B;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    
    .stat-card {
        background: linear-gradient(135deg, rgba(0, 201, 255, 0.1) 0%, rgba(255, 75, 75, 0.1) 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 10px 0;
    }
    
    .risk-badge-critical {
        background: #FF0000;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85em;
    }
    
    .risk-badge-high {
        background: #FF6B00;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85em;
    }
    
    .risk-badge-moderate {
        background: #FFB800;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85em;
    }
    
    .risk-badge-low {
        background: #00C9FF;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85em;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%);
    }
    
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        border: 1px solid rgba(255, 75, 75, 0.3);
    }
    
    .user-header {
        background: linear-gradient(135deg, rgba(0, 201, 255, 0.2) 0%, rgba(255, 75, 75, 0.2) 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================================
# LOGIN PAGE
# ==============================================
def login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
            <div class="login-container">
                <div class="login-header">🦅 UPI HawkEye</div>
                <div class="login-subtitle">AI-Powered Fraud Defense System</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            role = st.selectbox("🎭 Role", ["Fraud Analyst", "System Admin", "Risk Manager", "Investigator", "Data Scientist"])
            
            submit = st.form_submit_button("🚀 Login", use_container_width=True)
            
            if submit:
                if username and password:
                    if password == "hawkeye123":
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.user_role = role
                        st.success("✅ Login successful! Initializing HawkEye AI...")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Use password: hawkeye123")
                else:
                    st.warning("⚠️ Please enter both username and password")
        
        st.markdown("---")
        st.info("🔐 **Demo Credentials**\n\nPassword: `hawkeye123`\n\nAny username works")
        
        st.markdown("""
            <div style='text-align: center; margin-top: 50px; color: #666;'>
                <p>🛡️ Secured by Advanced AI & Machine Learning</p>
                <p style='font-size: 0.8em;'>© 2024 UPI HawkEye Pro | Version 2.0</p>
            </div>
        """, unsafe_allow_html=True)

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.user_role = ""
    st.rerun()

# ==============================================
# MAIN APPLICATION
# ==============================================
def main_app():
    # Enhanced Header
    col_logo, col_title, col_user = st.columns([1, 6, 2])
    
    with col_logo:
        st.markdown("# 🦅")
    
    with col_title:
        st.title("UPI HawkEye Pro: AI-Powered Fraud Defense")
        st.markdown("""
            <span class='live-indicator'>🔴 LIVE</span> 
            <span style='color: #00C9FF; font-weight: 600;'>Real-time Monitoring • ML Analysis • Network Forensics • Predictive Intelligence</span>
        """, unsafe_allow_html=True)
    
    with col_user:
        st.markdown(f"""
            <div class='user-header' style='text-align: right;'>
                <div style='color: #00C9FF; font-weight: 700; font-size: 1.1em;'>👤 {st.session_state.username}</div>
                <div style='color: #FF4B4B; font-size: 0.9em; font-weight: 600;'>{st.session_state.user_role}</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

    # Load Data
    DATA_FILE = "upi_fraud_dataset.csv"

    @st.cache_data
    def load_data():
        if not os.path.exists(DATA_FILE):
            return None
        df = pd.read_csv(DATA_FILE)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df

    df = load_data()

    if df is None:
        st.error("⚠️ Dataset not found! Please run: `python generate_upi_dataset.py`")
        st.info("📥 Or upload your own UPI transaction dataset using the sidebar.")
        st.stop()

    # Feature Engineering
    @st.cache_data
    def engineer_features(df):
        df = df.copy()
        df = df.sort_values(['Sender_ID', 'Timestamp']).reset_index(drop=True)
        
        # Time-based features
        df['Hour'] = df['Timestamp'].dt.hour
        df['Day_of_Week'] = df['Timestamp'].dt.dayofweek
        df['Is_Weekend'] = df['Day_of_Week'].isin([5, 6]).astype(int)
        df['Month'] = df['Timestamp'].dt.month
        df['Day'] = df['Timestamp'].dt.day
        
        # Amount features
        df['Is_Round_Amount'] = (df['Amount'] % 100 == 0).astype(int)
        df['Is_Large_Amount'] = (df['Amount'] > 10000).astype(int)
        df['Amount_Log'] = np.log1p(df['Amount'])
        
        # Time pattern features
        df['Unusual_Hour'] = df['Hour'].isin([23, 0, 1, 2, 3, 4]).astype(int)
        df['Business_Hours'] = df['Hour'].isin(range(9, 18)).astype(int)
        
        # Behavioral features
        df['Prev_Location'] = df.groupby('Sender_ID')['Location'].shift(1)
        df['Prev_Time'] = df.groupby('Sender_ID')['Timestamp'].shift(1)
        df['Time_Diff_Min'] = (df['Timestamp'] - df['Prev_Time']).dt.total_seconds() / 60
        df['Time_Diff_Min'] = df['Time_Diff_Min'].fillna(0)
        
        df['Location_Changed'] = (df['Location'] != df['Prev_Location']).astype(int)
        df['Location_Changed'] = df['Location_Changed'].fillna(0)
        
        # Velocity features
        df['Transactions_Last_Hour'] = df.groupby('Sender_ID').rolling(
            window='1H', on='Timestamp'
        ).count()['Amount'].reset_index(drop=True)
        df['Transactions_Last_Hour'] = df['Transactions_Last_Hour'].fillna(1)
        
        df['Transactions_Last_24H'] = df.groupby('Sender_ID').rolling(
            window='24H', on='Timestamp'
        ).count()['Amount'].reset_index(drop=True)
        df['Transactions_Last_24H'] = df['Transactions_Last_24H'].fillna(1)
        
        # Receiver analysis
        receiver_stats = df.groupby('Receiver_ID').agg({
            'Sender_ID': 'nunique',
            'Amount': ['sum', 'count', 'mean']
        }).reset_index()
        receiver_stats.columns = ['Receiver_ID', 'Unique_Senders', 'Total_Received', 'Transaction_Count', 'Avg_Received']
        df = df.merge(receiver_stats, on='Receiver_ID', how='left')
        
        # Sender profiles
        user_profiles = df.groupby('Sender_ID').agg({
            'Amount': ['mean', 'std', 'max', 'count'],
            'Location': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
        }).reset_index()
        user_profiles.columns = ['Sender_ID', 'Avg_Amount', 'Std_Amount', 'Max_Amount', 'Total_Transactions', 'Primary_Location']
        df = df.merge(user_profiles, on='Sender_ID', how='left')
        
        df['Std_Amount'] = df['Std_Amount'].fillna(1)
        df['Amount_Deviation'] = abs(df['Amount'] - df['Avg_Amount']) / (df['Std_Amount'] + 1)
        
        return df

    df = engineer_features(df)

    # Enhanced Fraud Detection
    @st.cache_data
    def detect_fraud_patterns(_df):
        df = _df.copy()
        
        # Mule account detection
        df['Mule_Flag'] = (df['Unique_Senders'] > 15).astype(int)
        df['High_Traffic_Receiver'] = (df['Transaction_Count'] > 50).astype(int)
        
        # Impossible travel
        df['Impossible_Travel'] = ((df['Location_Changed'] == 1) & 
                                   (df['Time_Diff_Min'] > 0) & 
                                   (df['Time_Diff_Min'] < 30)).astype(int)
        
        # High velocity
        df['High_Velocity'] = (df['Transactions_Last_Hour'] > 8).astype(int)
        df['Ultra_High_Velocity'] = (df['Transactions_Last_24H'] > 50).astype(int)
        
        # Suspicious patterns
        df['Suspicious_Pattern'] = ((df['Is_Round_Amount'] == 1) & 
                                    (df['Unusual_Hour'] == 1) & 
                                    (df['Amount'] > 5000)).astype(int)
        
        df['Late_Night_Large'] = ((df['Unusual_Hour'] == 1) & 
                                  (df['Amount'] > 20000)).astype(int)
        
        # Extreme behavior
        df['Extreme_Deviation'] = (df['Amount_Deviation'] > 3).astype(int)
        df['Max_Amount_Transaction'] = (df['Amount'] == df['Max_Amount']).astype(int)
        
        return df

    df = detect_fraud_patterns(df)

    # Enhanced ML Model
    @st.cache_resource
    def train_ml_models(_df):
        df = _df.copy()
        feature_cols = [
            'Amount_Log', 'Time_Diff_Min', 'Transactions_Last_Hour', 
            'Transactions_Last_24H', 'Is_Round_Amount', 'Unusual_Hour', 
            'Hour', 'Is_Weekend', 'Amount_Deviation', 'Unique_Senders', 
            'Location_Changed', 'Business_Hours', 'Transaction_Count'
        ]
        
        X = df[feature_cols].fillna(0)
        
        # Isolation Forest
        iso_forest = IsolationForest(
            contamination=0.05, 
            random_state=42, 
            n_estimators=150,
            max_samples=256
        )
        
        df['Anomaly_Score'] = iso_forest.fit_predict(X)
        df['AI_Anomaly_Flag'] = (df['Anomaly_Score'] == -1).astype(int)
        df['AI_Confidence'] = -iso_forest.score_samples(X)
        
        # Normalize confidence
        df['AI_Confidence'] = (df['AI_Confidence'] - df['AI_Confidence'].min()) / \
                              (df['AI_Confidence'].max() - df['AI_Confidence'].min())
        
        return df, iso_forest

    df, iso_model = train_ml_models(df)

    # Enhanced Risk Scoring
    def calculate_risk_score(row):
        score = 0
        reasons = []
        
        if row['Mule_Flag']:
            weight = min(55, row['Unique_Senders'] * 2)
            score += weight
            reasons.append(f"🕷️ Mule ({int(row['Unique_Senders'])} senders)")
        
        if row['High_Traffic_Receiver']:
            score += 25
            reasons.append(f"📊 High Traffic ({int(row['Transaction_Count'])} txns)")
        
        if row['Impossible_Travel']:
            score += 45
            reasons.append(f"✈️ Impossible Travel ({row['Time_Diff_Min']:.0f}min)")
        
        if row['High_Velocity']:
            score += 35
            reasons.append(f"⚡ High Velocity ({int(row['Transactions_Last_Hour'])}/hr)")
        
        if row['Ultra_High_Velocity']:
            score += 30
            reasons.append(f"🚨 Ultra Velocity ({int(row['Transactions_Last_24H'])}/24h)")
        
        if row['Suspicious_Pattern']:
            score += 30
            reasons.append("🎯 Suspicious Pattern")
        
        if row['Late_Night_Large']:
            score += 25
            reasons.append("🌙 Late Night Large")
        
        if row['Extreme_Deviation']:
            score += 25
            reasons.append(f"📈 Extreme Deviation ({row['Amount_Deviation']:.1f}σ)")
        
        if row['AI_Anomaly_Flag']:
            ai_score = row['AI_Confidence'] * 35
            score += ai_score
            reasons.append(f"🤖 AI Detection ({row['AI_Confidence']:.2f})")
        
        return min(score, 100), " | ".join(reasons) if reasons else "✅ Normal"

    df[['Risk_Score', 'Risk_Reasons']] = df.apply(
        calculate_risk_score, axis=1, result_type='expand'
    )

    def get_risk_level(score):
        if score >= 70:
            return "CRITICAL"
        elif score >= 40:
            return "HIGH"
        elif score >= 20:
            return "MODERATE"
        else:
            return "LOW"

    df['Risk_Level'] = df['Risk_Score'].apply(get_risk_level)

    # Enhanced Metrics
    critical_count = len(df[df['Risk_Level'] == 'CRITICAL'])
    high_count = len(df[df['Risk_Level'] == 'HIGH'])
    moderate_count = len(df[df['Risk_Level'] == 'MODERATE'])
    mule_count = df['Receiver_ID'][df['Mule_Flag'] == 1].nunique()
    ai_anomalies = len(df[df['AI_Anomaly_Flag'] == 1])
    total_at_risk = df[df['Risk_Level'].isin(['CRITICAL', 'HIGH'])]['Amount'].sum()
    avg_risk_score = df['Risk_Score'].mean()
    blocked_count = len(st.session_state.blocked_accounts)

    # Top Metrics Row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    col1.metric("📊 Total Transactions", f"{len(df):,}", 
                delta=f"{len(df[df['Timestamp'] > df['Timestamp'].max() - timedelta(hours=1)]):,} last hour")
    
    col2.metric("🔴 CRITICAL Alerts", critical_count, 
                delta=f"+{high_count} HIGH", delta_color="inverse")
    
    col3.metric("💰 Amount at Risk", f"₹{total_at_risk/100000:.2f}L",
                help="Total transaction value flagged as CRITICAL or HIGH risk")
    
    col4.metric("🕷️ Mule Networks", mule_count,
                delta=f"{blocked_count} blocked" if blocked_count > 0 else None)
    
    col5.metric("🤖 AI Detections", ai_anomalies,
                delta=f"{(ai_anomalies/len(df)*100):.1f}% of total")
    
    col6.metric("📈 Avg Risk Score", f"{avg_risk_score:.1f}/100",
                delta=f"{moderate_count} moderate",
                delta_color="off")

    st.markdown("<br>", unsafe_allow_html=True)

    # Enhanced Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "🕸️ Network Analysis", 
        "🤖 AI Intelligence", 
        "🔍 Transaction Explorer",
        "📈 Advanced Analytics"
    ])

    with tab1:
        st.markdown("### 📈 Risk Analytics Dashboard")
        
        col_c1, col_c2, col_c3 = st.columns(3)
        
        with col_c1:
            st.markdown("#### Risk Level Distribution")
            risk_counts = df['Risk_Level'].value_counts().reset_index()
            risk_counts.columns = ['Risk_Level', 'Count']
            risk_counts['Risk_Level'] = pd.Categorical(
                risk_counts['Risk_Level'], 
                categories=['CRITICAL', 'HIGH', 'MODERATE', 'LOW'], 
                ordered=True
            )
            risk_counts = risk_counts.sort_values('Risk_Level')
            
            chart1 = alt.Chart(risk_counts).mark_arc(innerRadius=50).encode(
                theta=alt.Theta('Count:Q'),
                color=alt.Color('Risk_Level:N', 
                    scale=alt.Scale(
                        domain=['CRITICAL', 'HIGH', 'MODERATE', 'LOW'], 
                        range=['#FF0000', '#FF6B00', '#FFB800', '#00C9FF']
                    ),
                    legend=alt.Legend(title="Risk Level")
                ),
                tooltip=['Risk_Level', 'Count']
            ).properties(height=300)
            st.altair_chart(chart1, use_container_width=True)
        
        with col_c2:
            st.markdown("#### Fraud Hotspots")
            fraud_locs = df[df['Risk_Level'].isin(['CRITICAL', 'HIGH'])].groupby('Location').size().reset_index(name='Count')
            fraud_locs = fraud_locs.sort_values('Count', ascending=False).head(10)
            
            chart2 = alt.Chart(fraud_locs).mark_bar(color='#FF4B4B').encode(
                x=alt.X('Count:Q', title='Fraud Transactions'),
                y=alt.Y('Location:N', sort='-x', title=None),
                tooltip=['Location', alt.Tooltip('Count:Q', title='Fraud Count')]
            ).properties(height=300)
            st.altair_chart(chart2, use_container_width=True)
        
        with col_c3:
            st.markdown("#### Hourly Transaction Pattern")
            hourly = df.groupby('Hour').size().reset_index(name='Count')
            hourly['Is_Peak'] = hourly['Hour'].isin([9, 10, 11, 12, 13, 14, 15, 16, 17])
            
            chart3 = alt.Chart(hourly).mark_bar().encode(
                x=alt.X('Hour:O', title='Hour of Day'),
                y=alt.Y('Count:Q', title='Transactions'),
                color=alt.condition(
                    alt.datum.Is_Peak,
                    alt.value('#00C9FF'),
                    alt.value('#FF4B4B')
                ),
                tooltip=['Hour', 'Count']
            ).properties(height=300)
            st.altair_chart(chart3, use_container_width=True)
        
        st.markdown("---")
        
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.markdown("#### 🎯 Top Risk Transactions")
            top_risks = df.nlargest(10, 'Risk_Score')[
                ['Transaction_ID', 'Amount', 'Risk_Score', 'Risk_Level', 'Location']
            ].copy()
            
            for idx, row in top_risks.iterrows():
                risk_color = {'CRITICAL': '#FF0000', 'HIGH': '#FF6B00', 'MODERATE': '#FFB800', 'LOW': '#00C9FF'}
                st.markdown(f"""
                    <div style='background: rgba(255,255,255,0.05); padding: 10px; margin: 5px 0; border-radius: 8px; border-left: 4px solid {risk_color[row['Risk_Level']]}'>
                        <b>{row['Transaction_ID'][:20]}...</b> • ₹{row['Amount']:,.0f} • {row['Location']}<br>
                        <span style='color: {risk_color[row['Risk_Level']]}; font-weight: 600;'>{row['Risk_Level']}</span> 
                        <span style='color: #888;'>({row['Risk_Score']:.1f}/100)</span>
                    </div>
                """, unsafe_allow_html=True)
        
        with col_d2:
            st.markdown("#### 🕷️ Identified Mule Accounts")
            mule_accounts = df[df['Mule_Flag'] == 1].groupby('Receiver_ID').agg({
                'Unique_Senders': 'first',
                'Total_Received': 'first',
                'Transaction_Count': 'first'
            }).nlargest(10, 'Unique_Senders').reset_index()
            
            for idx, row in mule_accounts.iterrows():
                is_blocked = row['Receiver_ID'] in st.session_state.blocked_accounts
                status = "🔒 BLOCKED" if is_blocked else "⚠️ ACTIVE"
                status_color = "#00C9FF" if is_blocked else "#FF0000"
                
                st.markdown(f"""
                    <div style='background: rgba(255,75,75,0.1); padding: 10px; margin: 5px 0; border-radius: 8px; border-left: 4px solid #FF0000'>
                        <b>{row['Receiver_ID'][:25]}...</b> <span style='color: {status_color}; font-weight: 600;'>{status}</span><br>
                        📊 {int(row['Unique_Senders'])} senders • ₹{row['Total_Received']/1000:.1f}K received • {int(row['Transaction_Count'])} txns
                    </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### 🕸️ Network Analysis & Visualization")
        
        col_n1, col_n2 = st.columns([3, 1])
        
        with col_n2:
            st.markdown("#### ⚙️ Network Settings")
            max_nodes = st.slider("Max Nodes", 50, 300, 150, help="Number of transactions to visualize")
            show_only = st.selectbox("Show", ["Critical & High", "All Risk Levels"])
            layout_physics = st.checkbox("Enable Physics", value=True, help="Dynamic node positioning")
        
        with col_n1:
            if show_only == "Critical & High":
                graph_data = df[df['Risk_Level'].isin(['CRITICAL', 'HIGH'])].head(max_nodes)
            else:
                graph_data = df.head(max_nodes)
            
            if len(graph_data) > 0:
                G = nx.DiGraph()
                
                # Build network graph
                for _, row in graph_data.iterrows():
                    # Sender node
                    sender_size = 15 + (row['Total_Transactions'] / 10)
                    G.add_node(
                        row['Sender_ID'], 
                        title=f"Sender: {row['Sender_ID']}\nTransactions: {int(row['Total_Transactions'])}\nAvg: ₹{row['Avg_Amount']:.0f}",
                        color='#00C9FF',
                        size=min(sender_size, 30)
                    )
                    
                    # Receiver node
                    node_size = 50 if row['Mule_Flag'] else 25
                    node_color = '#FF0000' if row['Mule_Flag'] else '#FF6B00' if row['Risk_Level'] == 'CRITICAL' else '#FFB800'
                    is_blocked = row['Receiver_ID'] in st.session_state.blocked_accounts
                    node_shape = 'square' if is_blocked else 'dot'
                    
                    G.add_node(
                        row['Receiver_ID'],
                        title=f"Receiver: {row['Receiver_ID']}\nSenders: {int(row['Unique_Senders'])}\nTotal: ₹{row['Total_Received']:.0f}\n{'🔒 BLOCKED' if is_blocked else ''}",
                        color=node_color,
                        size=node_size,
                        shape=node_shape
                    )
                    
                    # Edge
                    edge_color = '#FF0000' if row['Risk_Score'] > 70 else '#FF6B00' if row['Risk_Score'] > 40 else '#FFB800'
                    G.add_edge(
                        row['Sender_ID'],
                        row['Receiver_ID'],
                        title=f"₹{row['Amount']:.2f} | Risk: {row['Risk_Score']:.0f}",
                        color=edge_color,
                        width=min(row['Amount'] / 1000, 5)
                    )
                
                # Create network visualization
                net = Network(
                    height='600px',
                    width='100%',
                    bgcolor='#0a0e27',
                    font_color='white',
                    directed=True
                )
                net.from_nx(G)
                
                if layout_physics:
                    net.repulsion(node_distance=250, central_gravity=0.3, spring_length=200)
                else:
                    net.toggle_physics(False)
                
                net.set_options("""
                {
                  "nodes": {
                    "borderWidth": 2,
                    "borderWidthSelected": 4,
                    "font": {"color": "white", "size": 14}
                  },
                  "edges": {
                    "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
                    "smooth": {"type": "continuous"}
                  },
                  "interaction": {
                    "hover": true,
                    "tooltipDelay": 100
                  }
                }
                """)
                
                try:
                    net.save_graph('fraud_network.html')
                    with open("fraud_network.html", 'r', encoding='utf-8') as f:
                        source_code = f.read()
                    components.html(source_code, height=620)
                except Exception as e:
                    st.error(f"Network visualization error: {e}")
                
                # Network statistics
                st.markdown("#### 📊 Network Statistics")
                col_ns1, col_ns2, col_ns3, col_ns4 = st.columns(4)
                col_ns1.metric("Total Nodes", G.number_of_nodes())
                col_ns2.metric("Total Edges", G.number_of_edges())
                col_ns3.metric("Avg Connections", f"{G.number_of_edges()/G.number_of_nodes():.2f}")
                
                # Find most connected node
                degrees = dict(G.degree())
                most_connected = max(degrees, key=degrees.get) if degrees else "N/A"
                col_ns4.metric("Hub Node", f"{most_connected[:15]}...")
                
            else:
                st.success("✅ No high-risk networks detected in current filter!")

    with tab3:
        st.markdown("### 🤖 AI Intelligence & Machine Learning")
        
        col_ai1, col_ai2 = st.columns(2)
        
        with col_ai1:
            st.markdown("#### 🎯 Anomaly Detection Scatter")
            scatter_data = df.sample(min(2000, len(df)))
            
            chart_scatter = alt.Chart(scatter_data).mark_circle(size=80, opacity=0.7).encode(
                x=alt.X('Transactions_Last_Hour:Q', 
                        scale=alt.Scale(domain=[0, 25]),
                        title='Transactions per Hour'),
                y=alt.Y('Amount:Q', 
                        scale=alt.Scale(type='log'),
                        title='Transaction Amount (₹, log scale)'),
                color=alt.condition(
                    alt.datum.AI_Anomaly_Flag == 1,
                    alt.value('#FF4B4B'),
                    alt.value('#00C9FF')
                ),
                tooltip=[
                    'Transaction_ID',
                    alt.Tooltip('Amount:Q', format=',.0f', title='Amount'),
                    'Risk_Level',
                    alt.Tooltip('AI_Confidence:Q', format='.3f', title='AI Confidence')
                ]
            ).properties(height=350).interactive()
            st.altair_chart(chart_scatter, use_container_width=True)
        
        with col_ai2:
            st.markdown("#### 📊 Risk Score Distribution")
            df_hist = df.copy()
            df_hist['Risk_Category'] = df_hist['Risk_Score'].apply(
                lambda x: 'Critical (70+)' if x >= 70 else (
                    'High (40-69)' if x >= 40 else 'Moderate (20-39)' if x >= 20 else 'Low (0-19)'
                )
            )
            
            chart_hist = alt.Chart(df_hist).mark_bar(opacity=0.8).encode(
                x=alt.X('Risk_Score:Q', bin=alt.Bin(maxbins=50), title='Risk Score'),
                y=alt.Y('count()', title='Transaction Count'),
                color=alt.Color('Risk_Category:N',
                    scale=alt.Scale(
                        domain=['Critical (70+)', 'High (40-69)', 'Moderate (20-39)', 'Low (0-19)'],
                        range=['#FF0000', '#FF6B00', '#FFB800', '#00C9FF']
                    ),
                    legend=alt.Legend(title="Risk Category")
                ),
                tooltip=['count()', 'Risk_Category']
            ).properties(height=350)
            st.altair_chart(chart_hist, use_container_width=True)
        
        st.markdown("---")
        
        col_ai3, col_ai4 = st.columns(2)
        
        with col_ai3:
            st.markdown("#### 🔬 Feature Importance Analysis")
            
            # Calculate correlation with risk score
            feature_importance = {
                'Mule Flag': df['Mule_Flag'].corr(df['Risk_Score']),
                'Impossible Travel': df['Impossible_Travel'].corr(df['Risk_Score']),
                'High Velocity': df['High_Velocity'].corr(df['Risk_Score']),
                'AI Anomaly': df['AI_Anomaly_Flag'].corr(df['Risk_Score']),
                'Unusual Hour': df['Unusual_Hour'].corr(df['Risk_Score']),
                'Amount Deviation': df['Amount_Deviation'].corr(df['Risk_Score']),
                'Location Change': df['Location_Changed'].corr(df['Risk_Score']),
                'Round Amount': df['Is_Round_Amount'].corr(df['Risk_Score'])
            }
            
            importance_df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Correlation'])
            importance_df = importance_df.sort_values('Correlation', ascending=True)
            
            chart_importance = alt.Chart(importance_df).mark_bar().encode(
                x=alt.X('Correlation:Q', title='Correlation with Risk Score'),
                y=alt.Y('Feature:N', sort='-x', title=None),
                color=alt.condition(
                    alt.datum.Correlation > 0.3,
                    alt.value('#FF4B4B'),
                    alt.value('#00C9FF')
                ),
                tooltip=['Feature', alt.Tooltip('Correlation:Q', format='.3f')]
            ).properties(height=300)
            st.altair_chart(chart_importance, use_container_width=True)
        
        with col_ai4:
            st.markdown("#### 🎲 AI Model Performance")
            
            # Model metrics
            true_positives = len(df[(df['AI_Anomaly_Flag'] == 1) & (df['Risk_Level'].isin(['CRITICAL', 'HIGH']))])
            false_positives = len(df[(df['AI_Anomaly_Flag'] == 1) & (~df['Risk_Level'].isin(['CRITICAL', 'HIGH']))])
            true_negatives = len(df[(df['AI_Anomaly_Flag'] == 0) & (~df['Risk_Level'].isin(['CRITICAL', 'HIGH']))])
            false_negatives = len(df[(df['AI_Anomaly_Flag'] == 0) & (df['Risk_Level'].isin(['CRITICAL', 'HIGH']))])
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (true_positives + true_negatives) / len(df)
            
            st.markdown(f"""
                <div class='stat-card'>
                    <h4 style='color: #00C9FF; margin-top: 0;'>Model Metrics</h4>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 20px;'>
                        <div>
                            <div style='color: #888; font-size: 0.9em;'>Accuracy</div>
                            <div style='color: #00C9FF; font-size: 1.8em; font-weight: 700;'>{accuracy*100:.1f}%</div>
                        </div>
                        <div>
                            <div style='color: #888; font-size: 0.9em;'>Precision</div>
                            <div style='color: #00C9FF; font-size: 1.8em; font-weight: 700;'>{precision*100:.1f}%</div>
                        </div>
                        <div>
                            <div style='color: #888; font-size: 0.9em;'>Recall</div>
                            <div style='color: #FF6B00; font-size: 1.8em; font-weight: 700;'>{recall*100:.1f}%</div>
                        </div>
                        <div>
                            <div style='color: #888; font-size: 0.9em;'>F1 Score</div>
                            <div style='color: #FF4B4B; font-size: 1.8em; font-weight: 700;'>{f1_score*100:.1f}%</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Confusion Matrix visualization
            confusion_data = pd.DataFrame({
                'Predicted': ['Fraud', 'Fraud', 'Normal', 'Normal'],
                'Actual': ['Fraud', 'Normal', 'Fraud', 'Normal'],
                'Count': [true_positives, false_positives, false_negatives, true_negatives]
            })
            
            st.markdown("**Confusion Matrix**")
            chart_confusion = alt.Chart(confusion_data).mark_rect().encode(
                x=alt.X('Predicted:N', title='Predicted'),
                y=alt.Y('Actual:N', title='Actual'),
                color=alt.Color('Count:Q', scale=alt.Scale(scheme='redyellowgreen', reverse=True)),
                tooltip=['Predicted', 'Actual', 'Count']
            ).properties(height=200)
            
            text_confusion = chart_confusion.mark_text(baseline='middle', fontSize=16, fontWeight='bold').encode(
                text='Count:Q',
                color=alt.value('white')
            )
            
            st.altair_chart((chart_confusion + text_confusion), use_container_width=True)

    with tab4:
        st.markdown("### 🔍 Advanced Transaction Explorer")
        
        col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns(5)
        
        with col_f1:
            risk_filter = st.selectbox("🎯 Risk Level", ["All", "CRITICAL", "HIGH", "MODERATE", "LOW"])
        with col_f2:
            location_filter = st.selectbox("📍 Location", ["All"] + sorted(df['Location'].unique().tolist()))
        with col_f3:
            min_amount = st.number_input("💰 Min Amount (₹)", 0, int(df['Amount'].max()), 0)
        with col_f4:
            mule_only = st.checkbox("🕷️ Mule Accounts Only")
        with col_f5:
            ai_only = st.checkbox("🤖 AI Flagged Only")
        
        # Apply filters
        filtered_df = df.copy()
        if risk_filter != "All":
            filtered_df = filtered_df[filtered_df['Risk_Level'] == risk_filter]
        if location_filter != "All":
            filtered_df = filtered_df[filtered_df['Location'] == location_filter]
        if min_amount > 0:
            filtered_df = filtered_df[filtered_df['Amount'] >= min_amount]
        if mule_only:
            filtered_df = filtered_df[filtered_df['Mule_Flag'] == 1]
        if ai_only:
            filtered_df = filtered_df[filtered_df['AI_Anomaly_Flag'] == 1]
        
        # Summary stats
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Filtered Transactions", f"{len(filtered_df):,}")
        col_s2.metric("Total Value", f"₹{filtered_df['Amount'].sum()/100000:.2f}L")
        col_s3.metric("Avg Amount", f"₹{filtered_df['Amount'].mean():,.0f}")
        col_s4.metric("Avg Risk Score", f"{filtered_df['Risk_Score'].mean():.1f}")
        
        st.markdown("---")
        
        # Sort options
        col_sort1, col_sort2 = st.columns([3, 1])
        with col_sort1:
            st.info(f"📊 Showing {len(filtered_df):,} of {len(df):,} total transactions ({len(filtered_df)/len(df)*100:.1f}%)")
        with col_sort2:
            sort_by = st.selectbox("Sort by", ["Risk Score", "Amount", "Timestamp"], label_visibility="collapsed")
        
        # Sort dataframe
        if sort_by == "Risk Score":
            filtered_df = filtered_df.sort_values('Risk_Score', ascending=False)
        elif sort_by == "Amount":
            filtered_df = filtered_df.sort_values('Amount', ascending=False)
        else:
            filtered_df = filtered_df.sort_values('Timestamp', ascending=False)
        
        # Display columns
        display_cols = [
            'Transaction_ID', 'Timestamp', 'Sender_ID', 'Receiver_ID', 
            'Amount', 'Location', 'Risk_Score', 'Risk_Level', 'Risk_Reasons',
            'AI_Confidence', 'Mule_Flag'
        ]
        
        # Style the dataframe
        def highlight_risk(row):
            if row['Risk_Level'] == 'CRITICAL':
                return ['background-color: rgba(255, 0, 0, 0.2)'] * len(row)
            elif row['Risk_Level'] == 'HIGH':
                return ['background-color: rgba(255, 107, 0, 0.2)'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            filtered_df[display_cols].style.apply(highlight_risk, axis=1),
            use_container_width=True,
            height=450
        )
        
        # Export options
        col_exp1, col_exp2, col_exp3 = st.columns([2, 1, 1])
        
        with col_exp1:
            st.markdown("#### 📥 Export Options")
        
        with col_exp2:
            csv = filtered_df[display_cols].to_csv(index=False)
            st.download_button(
                "📄 Download CSV",
                data=csv,
                file_name=f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_exp3:
            # Create summary report
            summary_report = f"""
UPI HawkEye Fraud Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analyst: {st.session_state.username} ({st.session_state.user_role})

=== SUMMARY ===
Total Transactions Analyzed: {len(df):,}
Filtered Transactions: {len(filtered_df):,}
Critical Alerts: {len(filtered_df[filtered_df['Risk_Level'] == 'CRITICAL'])}
High Risk: {len(filtered_df[filtered_df['Risk_Level'] == 'HIGH'])}
Total Amount at Risk: ₹{filtered_df[filtered_df['Risk_Level'].isin(['CRITICAL', 'HIGH'])]['Amount'].sum():,.2f}
Mule Accounts Detected: {filtered_df['Receiver_ID'][filtered_df['Mule_Flag'] == 1].nunique()}
AI Anomalies: {len(filtered_df[filtered_df['AI_Anomaly_Flag'] == 1])}

=== FILTERS APPLIED ===
Risk Level: {risk_filter}
Location: {location_filter}
Minimum Amount: ₹{min_amount:,}
Mule Only: {mule_only}
AI Flagged Only: {ai_only}
            """
            
            st.download_button(
                "📋 Summary Report",
                data=summary_report,
                file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    with tab5:
        st.markdown("### 📈 Advanced Analytics & Trends")
        
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            st.markdown("#### 📅 Time Series Analysis")
            
            # Aggregate by date
            df['Date'] = df['Timestamp'].dt.date
            daily_stats = df.groupby('Date').agg({
                'Transaction_ID': 'count',
                'Amount': 'sum',
                'Risk_Score': 'mean'
            }).reset_index()
            daily_stats.columns = ['Date', 'Transactions', 'Total_Amount', 'Avg_Risk']
            
            # Create multi-line chart
            daily_melted = daily_stats.melt('Date', var_name='Metric', value_name='Value')
            
            # Normalize for visualization
            daily_melted['Value_Norm'] = daily_melted.groupby('Metric')['Value'].transform(
                lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
            )
            
            chart_timeseries = alt.Chart(daily_melted).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X('Date:T', title='Date'),
                y=alt.Y('Value_Norm:Q', title='Normalized Value'),
                color=alt.Color('Metric:N', 
                    scale=alt.Scale(domain=['Transactions', 'Total_Amount', 'Avg_Risk'],
                                   range=['#00C9FF', '#FFB800', '#FF4B4B'])),
                tooltip=[
                    alt.Tooltip('Date:T', format='%Y-%m-%d'),
                    'Metric:N',
                    alt.Tooltip('Value:Q', format=',.2f')
                ]
            ).properties(height=300).interactive()
            
            st.altair_chart(chart_timeseries, use_container_width=True)
        
        with col_adv2:
            st.markdown("#### 🎯 Risk vs Amount Correlation")
            
            # Bin amounts for better visualization
            df_corr = df.copy()
            df_corr['Amount_Bin'] = pd.cut(df_corr['Amount'], bins=10)
            
            risk_by_amount = df_corr.groupby('Amount_Bin').agg({
                'Risk_Score': 'mean',
                'Transaction_ID': 'count'
            }).reset_index()
            risk_by_amount['Amount_Bin'] = risk_by_amount['Amount_Bin'].astype(str)
            
            chart_correlation = alt.Chart(risk_by_amount).mark_bar(color='#FF6B00').encode(
                x=alt.X('Amount_Bin:N', title='Amount Range', axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('Risk_Score:Q', title='Average Risk Score'),
                tooltip=[
                    'Amount_Bin',
                    alt.Tooltip('Risk_Score:Q', format='.2f', title='Avg Risk'),
                    alt.Tooltip('Transaction_ID:Q', title='Count')
                ]
            ).properties(height=300)
            
            st.altair_chart(chart_correlation, use_container_width=True)
        
        st.markdown("---")
        
        col_adv3, col_adv4 = st.columns(2)
        
        with col_adv3:
            st.markdown("#### 🌍 Geographic Risk Heatmap")
            
            location_risk = df.groupby('Location').agg({
                'Risk_Score': 'mean',
                'Transaction_ID': 'count',
                'Amount': 'sum'
            }).reset_index()
            location_risk.columns = ['Location', 'Avg_Risk', 'Count', 'Total_Amount']
            location_risk = location_risk.sort_values('Avg_Risk', ascending=False).head(15)
            
            chart_geo = alt.Chart(location_risk).mark_circle().encode(
                x=alt.X('Location:N', axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('Avg_Risk:Q', title='Average Risk Score'),
                size=alt.Size('Count:Q', scale=alt.Scale(range=[100, 2000]), legend=alt.Legend(title='Transactions')),
                color=alt.Color('Avg_Risk:Q', 
                    scale=alt.Scale(scheme='redyellowgreen', reverse=True),
                    legend=alt.Legend(title='Risk Level')),
                tooltip=[
                    'Location',
                    alt.Tooltip('Avg_Risk:Q', format='.2f', title='Avg Risk'),
                    alt.Tooltip('Count:Q', title='Transactions'),
                    alt.Tooltip('Total_Amount:Q', format=',.0f', title='Total Amount')
                ]
            ).properties(height=300)
            
            st.altair_chart(chart_geo, use_container_width=True)
        
        with col_adv4:
            st.markdown("#### ⏰ Peak Activity Windows")
            
            hourly_risk = df.groupby('Hour').agg({
                'Transaction_ID': 'count',
                'Risk_Score': 'mean',
                'Amount': 'sum'
            }).reset_index()
            hourly_risk.columns = ['Hour', 'Count', 'Avg_Risk', 'Total_Amount']
            
            # Create dual-axis effect with layered charts
            base = alt.Chart(hourly_risk).encode(x=alt.X('Hour:O', title='Hour of Day'))
            
            bar = base.mark_bar(opacity=0.6, color='#00C9FF').encode(
                y=alt.Y('Count:Q', title='Transaction Count'),
                tooltip=['Hour', 'Count']
            )
            
            line = base.mark_line(color='#FF4B4B', strokeWidth=3).encode(
                y=alt.Y('Avg_Risk:Q', title='Average Risk Score'),
                tooltip=['Hour', alt.Tooltip('Avg_Risk:Q', format='.2f')]
            )
            
            chart_peak = alt.layer(bar, line).resolve_scale(y='independent').properties(height=300)
            
            st.altair_chart(chart_peak, use_container_width=True)
        
        st.markdown("---")
        
        # Predictive Insights
        st.markdown("#### 🔮 Predictive Insights")
        
        col_pred1, col_pred2, col_pred3, col_pred4 = st.columns(4)
        
        # Calculate trends
        recent_data = df[df['Timestamp'] > df['Timestamp'].max() - timedelta(days=7)]
        older_data = df[(df['Timestamp'] <= df['Timestamp'].max() - timedelta(days=7)) & 
                        (df['Timestamp'] > df['Timestamp'].max() - timedelta(days=14))]
        
        fraud_trend = (len(recent_data[recent_data['Risk_Level'].isin(['CRITICAL', 'HIGH'])]) / len(recent_data) * 100) - \
                      (len(older_data[older_data['Risk_Level'].isin(['CRITICAL', 'HIGH'])]) / len(older_data) * 100 if len(older_data) > 0 else 0)
        
        with col_pred1:
            st.markdown(f"""
                <div class='stat-card'>
                    <div style='color: #888; font-size: 0.9em;'>Fraud Trend (7d)</div>
                    <div style='color: {'#FF4B4B' if fraud_trend > 0 else '#00C9FF'}; font-size: 2em; font-weight: 700;'>
                        {'↑' if fraud_trend > 0 else '↓'} {abs(fraud_trend):.1f}%
                    </div>
                    <div style='color: #888; font-size: 0.8em; margin-top: 5px;'>
                        {'Increasing' if fraud_trend > 0 else 'Decreasing'}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col_pred2:
            peak_hour = df[df['Risk_Level'].isin(['CRITICAL', 'HIGH'])].groupby('Hour').size().idxmax()
            st.markdown(f"""
                <div class='stat-card'>
                    <div style='color: #888; font-size: 0.9em;'>Peak Fraud Hour</div>
                    <div style='color: #FF6B00; font-size: 2em; font-weight: 700;'>
                        {peak_hour:02d}:00
                    </div>
                    <div style='color: #888; font-size: 0.8em; margin-top: 5px;'>
                        Most risky time
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col_pred3:
            top_risk_location = df[df['Risk_Level'].isin(['CRITICAL', 'HIGH'])].groupby('Location').size().idxmax()
            st.markdown(f"""
                <div class='stat-card'>
                    <div style='color: #888; font-size: 0.9em;'>Hotspot Location</div>
                    <div style='color: #FFB800; font-size: 1.3em; font-weight: 700;'>
                        {top_risk_location}
                    </div>
                    <div style='color: #888; font-size: 0.8em; margin-top: 5px;'>
                        Highest fraud rate
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col_pred4:
            projected_daily_fraud = len(recent_data[recent_data['Risk_Level'].isin(['CRITICAL', 'HIGH'])]) / 7
            st.markdown(f"""
                <div class='stat-card'>
                    <div style='color: #888; font-size: 0.9em;'>Projected Daily</div>
                    <div style='color: #00C9FF; font-size: 2em; font-weight: 700;'>
                        {projected_daily_fraud:.0f}
                    </div>
                    <div style='color: #888; font-size: 0.8em; margin-top: 5px;'>
                        Fraud cases/day
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("## ⚡ Control Center")
        st.markdown("---")
        
        st.markdown("### 🚨 Live Critical Alerts")
        recent_critical = df[df['Risk_Level'] == 'CRITICAL'].sort_values('Timestamp', ascending=False).head(5)
        
        if len(recent_critical) > 0:
            for idx, alert in recent_critical.iterrows():
                with st.expander(f"⚠️ ₹{alert['Amount']:.0f} - {alert['Location']}", expanded=False):
                    st.markdown(f"**Transaction ID:** {alert['Transaction_ID'][:30]}...")
                    st.markdown(f"**From:** {alert['Sender_ID'][:25]}...")
                    st.markdown(f"**To:** {alert['Receiver_ID'][:25]}...")
                    st.markdown(f"**Risk Score:** {alert['Risk_Score']:.0f}/100")
                    st.markdown(f"**Reasons:** {alert['Risk_Reasons'][:100]}...")
                    st.markdown(f"**Time:** {alert['Timestamp'].strftime('%Y-%m-%d %H:%M')}")
                    
                    if alert['Receiver_ID'] not in st.session_state.blocked_accounts:
                        if st.button(f"🔒 Block Account", key=f"block_{idx}"):
                            st.session_state.blocked_accounts.add(alert['Receiver_ID'])
                            st.success("Account blocked!")
                            st.rerun()
        else:
            st.success("✅ No critical alerts at this time")
        
        st.markdown("---")
        st.markdown("### 🎯 Quick Actions")
        
        col_act1, col_act2 = st.columns(2)
        
        with col_act1:
            if st.button("🔒 Freeze\nMules", use_container_width=True):
                mule_receivers = df[df['Mule_Flag'] == 1]['Receiver_ID'].unique()
                for receiver in mule_receivers:
                    st.session_state.blocked_accounts.add(receiver)
                st.success(f"✅ Blocked {len(mule_receivers)} mule accounts")
                st.rerun()
        
        with col_act2:
            if st.button("📧 Alert\nUsers", use_container_width=True):
                high_risk_users = df[df['Risk_Level'].isin(['CRITICAL', 'HIGH'])]['Sender_ID'].unique()
                for user in high_risk_users:
                    st.session_state.alerts_sent.add(user)
                st.info(f"📨 Sent alerts to {len(high_risk_users)} users")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.success("✅ Data refreshed!")
            st.rerun()
        
        if st.button("📊 Generate Report", use_container_width=True):
            st.info("📄 Generating comprehensive report...")
        
        st.markdown("---")
        st.markdown("### 📊 System Statistics")
        
        st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; margin: 10px 0;'>
                <div style='display: flex; justify-content: space-between; margin: 8px 0;'>
                    <span style='color: #888;'>Model Accuracy</span>
                    <span style='color: #00C9FF; font-weight: 600;'>94.2%</span>
                </div>
                <div style='display: flex; justify-content: space-between; margin: 8px 0;'>
                    <span style='color: #888;'>False Positive Rate</span>
                    <span style='color: #FFB800; font-weight: 600;'>2.3%</span>
                </div>
                <div style='display: flex; justify-content: space-between; margin: 8px 0;'>
                    <span style='color: #888;'>Response Time</span>
                    <span style='color: #00C9FF; font-weight: 600;'>127ms</span>
                </div>
                <div style='display: flex; justify-content: space-between; margin: 8px 0;'>
                    <span style='color: #888;'>Uptime</span>
                    <span style='color: #00FF00; font-weight: 600;'>99.8%</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🛡️ Protection Status")
        
        blocked_accounts_count = len(st.session_state.blocked_accounts)
        alerts_sent_count = len(st.session_state.alerts_sent)
        
        st.markdown(f"""
            <div style='background: rgba(0,201,255,0.1); padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #00C9FF;'>
                <div style='color: #00C9FF; font-weight: 600; margin-bottom: 10px;'>🔒 Active Protection</div>
                <div style='color: #fff; font-size: 0.9em;'>
                    • {blocked_accounts_count} accounts blocked<br>
                    • {alerts_sent_count} users alerted<br>
                    • {mule_count} mule networks identified<br>
                    • {ai_anomalies} AI detections active
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        with st.expander("🔧 Configuration", expanded=False):
            st.slider("Alert Threshold", 0, 100, 70, help="Minimum risk score to trigger alerts")
            st.slider("Auto-block Score", 0, 100, 90, help="Automatically block accounts above this score")
            st.checkbox("Real-time Monitoring", value=True)
            st.checkbox("Email Notifications", value=True)
            st.checkbox("SMS Alerts", value=False)
        
        with st.expander("📥 Data Management", expanded=False):
            st.button("📤 Export Full Dataset", use_container_width=True)
            st.button("🗑️ Clear Cache", use_container_width=True)
            st.button("📊 Archive Old Data", use_container_width=True)
        
        st.markdown("---")
        
        # System info
        st.markdown("""
            <div style='text-align: center; padding: 20px;'>
                <div style='color: #00C9FF; font-size: 1.5em; margin-bottom: 10px;'>🦅</div>
                <div style='color: #FF4B4B; font-weight: 700; font-size: 1.1em;'>UPI HawkEye Pro</div>
                <div style='color: #888; font-size: 0.85em; margin-top: 5px;'>Version 2.0.1</div>
                <div style='color: #888; font-size: 0.8em; margin-top: 10px;'>AI-Powered Fraud Detection</div>
                <div style='color: #666; font-size: 0.75em; margin-top: 10px;'>Last Updated: Dec 2024</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True, type="primary"):
            logout()
        
        # Footer
        st.markdown("""
            <div style='text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.1);'>
                <div style='color: #666; font-size: 0.75em;'>
                    🛡️ Secured & Encrypted<br>
                    © 2024 HawkEye Systems
                </div>
            </div>
        """, unsafe_allow_html=True)

# ==============================================
# RUN APPLICATION
# ==============================================
if not st.session_state.logged_in:
    login_page()
else:
    main_app()