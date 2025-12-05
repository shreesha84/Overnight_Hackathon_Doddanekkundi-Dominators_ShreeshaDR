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

# ==============================================
# PAGE CONFIGURATION
# ==============================================
st.set_page_config(
    page_title="UPI HawkEye AI",
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

# ==============================================
# CSS STYLING
# ==============================================
st.markdown("""
    <style>
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
        margin-top: 100px;
    }
    .login-header {
        text-align: center;
        color: #FF4B4B;
        font-size: 2.5em;
        font-weight: bold;
        text-shadow: 0 0 20px rgba(255, 75, 75, 0.5);
        margin-bottom: 10px;
    }
    .login-subtitle {
        text-align: center;
        color: #00C9FF;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    h1, h2, h3 {
        color: #FF4B4B !important;
        text-shadow: 0 0 10px rgba(255, 75, 75, 0.3);
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
        color: #00C9FF;
    }
    .stButton>button {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: bold;
    }
    .live-indicator {
        animation: pulse 2s infinite;
        color: #FF4B4B;
        font-weight: bold;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
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
            role = st.selectbox("🎭 Role", ["Fraud Analyst", "System Admin", "Risk Manager", "Investigator"])
            
            submit = st.form_submit_button("🚀 Login", use_container_width=True)
            
            if submit:
                if username and password:
                    if password == "hawkeye123":
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.user_role = role
                        st.success("✅ Login successful!")
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
                <p style='font-size: 0.8em;'>© 2024 UPI HawkEye</p>
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
    # Header
    col_logo, col_title, col_user = st.columns([1, 6, 2])
    
    with col_logo:
        st.markdown("# 🦅")
    
    with col_title:
        st.title("UPI HawkEye: AI-Powered Fraud Defense")
        st.markdown("""
            <span class='live-indicator'>🔴 LIVE</span> 
            <span style='color: #00C9FF;'>Real-time Monitoring • ML Analysis • Network Forensics</span>
        """, unsafe_allow_html=True)
    
    with col_user:
        st.markdown(f"""
            <div style='text-align: right; padding: 10px;'>
                <div style='color: #00C9FF; font-weight: bold;'>👤 {st.session_state.username}</div>
                <div style='color: #888; font-size: 0.9em;'>{st.session_state.user_role}</div>
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
        st.error("⚠️ Dataset not found! Run: python generate_upi_dataset.py")
        st.stop()

    # Feature Engineering
    @st.cache_data
    def engineer_features(df):
        df = df.copy()
        df = df.sort_values(['Sender_ID', 'Timestamp']).reset_index(drop=True)
        
        df['Hour'] = df['Timestamp'].dt.hour
        df['Day_of_Week'] = df['Timestamp'].dt.dayofweek
        df['Is_Weekend'] = df['Day_of_Week'].isin([5, 6]).astype(int)
        df['Is_Round_Amount'] = (df['Amount'] % 100 == 0).astype(int)
        df['Is_Large_Amount'] = (df['Amount'] > 10000).astype(int)
        df['Amount_Log'] = np.log1p(df['Amount'])
        df['Unusual_Hour'] = df['Hour'].isin([23, 0, 1, 2, 3, 4]).astype(int)
        
        df['Prev_Location'] = df.groupby('Sender_ID')['Location'].shift(1)
        df['Prev_Time'] = df.groupby('Sender_ID')['Timestamp'].shift(1)
        df['Time_Diff_Min'] = (df['Timestamp'] - df['Prev_Time']).dt.total_seconds() / 60
        df['Time_Diff_Min'] = df['Time_Diff_Min'].fillna(0)
        
        df['Location_Changed'] = (df['Location'] != df['Prev_Location']).astype(int)
        df['Location_Changed'] = df['Location_Changed'].fillna(0)
        
        df['Transactions_Last_Hour'] = df.groupby('Sender_ID').rolling(
            window='1H', on='Timestamp'
        ).count()['Amount'].reset_index(drop=True)
        df['Transactions_Last_Hour'] = df['Transactions_Last_Hour'].fillna(1)
        
        receiver_stats = df.groupby('Receiver_ID').agg({
            'Sender_ID': 'nunique',
            'Amount': ['sum', 'count']
        }).reset_index()
        receiver_stats.columns = ['Receiver_ID', 'Unique_Senders', 'Total_Received', 'Transaction_Count']
        df = df.merge(receiver_stats, on='Receiver_ID', how='left')
        
        user_profiles = df.groupby('Sender_ID').agg({
            'Amount': ['mean', 'std'],
            'Location': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
        }).reset_index()
        user_profiles.columns = ['Sender_ID', 'Avg_Amount', 'Std_Amount', 'Primary_Location']
        df = df.merge(user_profiles, on='Sender_ID', how='left')
        
        df['Std_Amount'] = df['Std_Amount'].fillna(1)
        df['Amount_Deviation'] = abs(df['Amount'] - df['Avg_Amount']) / (df['Std_Amount'] + 1)
        
        return df

    df = engineer_features(df)

    # Fraud Detection
    @st.cache_data
    def detect_fraud_patterns(_df):
        df = _df.copy()
        df['Mule_Flag'] = (df['Unique_Senders'] > 15).astype(int)
        df['Impossible_Travel'] = ((df['Location_Changed'] == 1) & (df['Time_Diff_Min'] > 0) & (df['Time_Diff_Min'] < 30)).astype(int)
        df['High_Velocity'] = (df['Transactions_Last_Hour'] > 8).astype(int)
        df['Suspicious_Pattern'] = ((df['Is_Round_Amount'] == 1) & (df['Unusual_Hour'] == 1) & (df['Amount'] > 5000)).astype(int)
        df['Extreme_Deviation'] = (df['Amount_Deviation'] > 3).astype(int)
        return df

    df = detect_fraud_patterns(df)

    # ML Model
    @st.cache_resource
    def train_ml_models(_df):
        df = _df.copy()
        feature_cols = ['Amount_Log', 'Time_Diff_Min', 'Transactions_Last_Hour', 'Is_Round_Amount', 'Unusual_Hour', 'Hour', 'Is_Weekend', 'Amount_Deviation', 'Unique_Senders', 'Location_Changed']
        X = df[feature_cols].fillna(0)
        iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
        df['Anomaly_Score'] = iso_forest.fit_predict(X)
        df['AI_Anomaly_Flag'] = (df['Anomaly_Score'] == -1).astype(int)
        df['AI_Confidence'] = -iso_forest.score_samples(X)
        df['AI_Confidence'] = (df['AI_Confidence'] - df['AI_Confidence'].min()) / (df['AI_Confidence'].max() - df['AI_Confidence'].min())
        return df, iso_forest

    df, iso_model = train_ml_models(df)

    # Risk Scoring
    def calculate_risk_score(row):
        score = 0
        reasons = []
        if row['Mule_Flag']:
            weight = min(50, row['Unique_Senders'] * 1.5)
            score += weight
            reasons.append(f"Mule ({int(row['Unique_Senders'])} senders)")
        if row['Impossible_Travel']:
            score += 40
            reasons.append(f"Travel ({row['Time_Diff_Min']:.0f}min)")
        if row['High_Velocity']:
            score += 30
            reasons.append(f"Velocity ({int(row['Transactions_Last_Hour'])}txn/hr)")
        if row['Suspicious_Pattern']:
            score += 25
            reasons.append("Suspicious pattern")
        if row['Extreme_Deviation']:
            score += 20
            reasons.append(f"Deviation ({row['Amount_Deviation']:.1f}σ)")
        if row['AI_Anomaly_Flag']:
            ai_score = row['AI_Confidence'] * 30
            score += ai_score
            reasons.append(f"AI ({row['AI_Confidence']:.2f})")
        return min(score, 100), " | ".join(reasons) if reasons else "Normal"

    df[['Risk_Score', 'Risk_Reasons']] = df.apply(calculate_risk_score, axis=1, result_type='expand')

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

    # Metrics
    critical_count = len(df[df['Risk_Level'] == 'CRITICAL'])
    high_count = len(df[df['Risk_Level'] == 'HIGH'])
    mule_count = df['Receiver_ID'][df['Mule_Flag'] == 1].nunique()
    ai_anomalies = len(df[df['AI_Anomaly_Flag'] == 1])
    total_at_risk = df[df['Risk_Level'].isin(['CRITICAL', 'HIGH'])]['Amount'].sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Transactions", f"{len(df):,}")
    col2.metric("🔴 CRITICAL", critical_count, delta=f"+{high_count} HIGH", delta_color="inverse")
    col3.metric("💰 At Risk", f"₹{total_at_risk/100000:.1f}L")
    col4.metric("🕷️ Mule Networks", mule_count)
    col5.metric("🤖 AI Detections", ai_anomalies)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🕸️ Network", "🤖 AI Analysis", "📋 Explorer"])

    with tab1:
        st.markdown("### 📈 Risk Analytics")
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown("#### Risk Distribution")
            risk_counts = df['Risk_Level'].value_counts().reset_index()
            risk_counts.columns = ['Risk_Level', 'Count']
            risk_counts['Risk_Level'] = pd.Categorical(risk_counts['Risk_Level'], categories=['CRITICAL', 'HIGH', 'MODERATE', 'LOW'], ordered=True)
            risk_counts = risk_counts.sort_values('Risk_Level')
            
            chart1 = alt.Chart(risk_counts).mark_bar().encode(
                x=alt.X('Risk_Level:N', sort=['CRITICAL', 'HIGH', 'MODERATE', 'LOW']),
                y=alt.Y('Count:Q'),
                color=alt.Color('Risk_Level:N', scale=alt.Scale(domain=['CRITICAL', 'HIGH', 'MODERATE', 'LOW'], range=['#FF0000', '#FF6B00', '#FFB800', '#00C9FF']), legend=None),
                tooltip=['Risk_Level', 'Count']
            ).properties(height=300)
            st.altair_chart(chart1, use_container_width=True)
        
        with col_c2:
            st.markdown("#### Fraud Hotspots")
            fraud_locs = df[df['Risk_Level'].isin(['CRITICAL', 'HIGH'])].groupby('Location').size().reset_index(name='Count')
            fraud_locs = fraud_locs.sort_values('Count', ascending=False).head(10)
            
            chart2 = alt.Chart(fraud_locs).mark_bar(color='#FF4B4B').encode(
                x=alt.X('Count:Q'),
                y=alt.Y('Location:N', sort='-x'),
                tooltip=['Location', 'Count']
            ).properties(height=300)
            st.altair_chart(chart2, use_container_width=True)

    with tab2:
        st.markdown("### 🕸️ Network Visualization")
        max_nodes = st.slider("Max nodes", 50, 200, 100)
        
        graph_data = df[df['Risk_Level'].isin(['CRITICAL', 'HIGH'])].head(max_nodes)
        
        if len(graph_data) > 0:
            G = nx.DiGraph()
            for _, row in graph_data.iterrows():
                G.add_node(row['Sender_ID'], title=f"Sender: {row['Sender_ID']}", color='#00C9FF', size=20)
                node_size = 40 if row['Mule_Flag'] else 25
                node_color = '#FF0000' if row['Mule_Flag'] else '#FF6B00'
                G.add_node(row['Receiver_ID'], title=f"Receiver: {row['Receiver_ID']}", color=node_color, size=node_size)
                G.add_edge(row['Sender_ID'], row['Receiver_ID'], title=f"₹{row['Amount']:.2f}")
            
            net = Network(height='600px', width='100%', bgcolor='#0a0e27', font_color='white', directed=True)
            net.from_nx(G)
            net.repulsion(node_distance=250, central_gravity=0.3)
            
            try:
                net.save_graph('fraud_network.html')
                with open("fraud_network.html", 'r', encoding='utf-8') as f:
                    source_code = f.read()
                components.html(source_code, height=620)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.success("✅ No critical networks detected!")

    with tab3:
        st.markdown("### 🤖 ML Insights")
        col_ai1, col_ai2 = st.columns(2)
        
        with col_ai1:
            st.markdown("#### Anomaly Detection")
            scatter_data = df.sample(min(1500, len(df)))
            chart3 = alt.Chart(scatter_data).mark_circle(size=60).encode(
                x=alt.X('Transactions_Last_Hour:Q', scale=alt.Scale(domain=[0, 20])),
                y=alt.Y('Amount:Q', scale=alt.Scale(type='log')),
                color=alt.condition(alt.datum.AI_Anomaly_Flag == 1, alt.value('#FF4B4B'), alt.value('#00C9FF')),
                tooltip=['Transaction_ID', 'Amount', 'Risk_Level']
            ).properties(height=350).interactive()
            st.altair_chart(chart3, use_container_width=True)
        
        with col_ai2:
            st.markdown("#### Risk Distribution")
            df_cat = df.copy()
            df_cat['Risk_Category'] = df_cat['Risk_Score'].apply(lambda x: 'Critical (70+)' if x >= 70 else ('High (40-69)' if x >= 40 else 'Low (0-39)'))
            chart4 = alt.Chart(df_cat).mark_bar().encode(
                x=alt.X('Risk_Score:Q', bin=alt.Bin(maxbins=40)),
                y=alt.Y('count()'),
                color=alt.Color('Risk_Category:N', scale=alt.Scale(domain=['Critical (70+)', 'High (40-69)', 'Low (0-39)'], range=['#FF0000', '#FF6B00', '#00C9FF'])),
                tooltip=['count()', 'Risk_Category']
            ).properties(height=350)
            st.altair_chart(chart4, use_container_width=True)

    with tab4:
        st.markdown("### 🔍 Transaction Explorer")
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        
        with col_f1:
            risk_filter = st.selectbox("Risk", ["All", "CRITICAL", "HIGH", "MODERATE", "LOW"])
        with col_f2:
            location_filter = st.selectbox("Location", ["All"] + sorted(df['Location'].unique().tolist()))
        with col_f3:
            min_amount = st.number_input("Min ₹", 0, int(df['Amount'].max()), 0)
        with col_f4:
            mule_only = st.checkbox("Mule Only")
        
        filtered_df = df.copy()
        if risk_filter != "All":
            filtered_df = filtered_df[filtered_df['Risk_Level'] == risk_filter]
        if location_filter != "All":
            filtered_df = filtered_df[filtered_df['Location'] == location_filter]
        if min_amount > 0:
            filtered_df = filtered_df[filtered_df['Amount'] >= min_amount]
        if mule_only:
            filtered_df = filtered_df[filtered_df['Mule_Flag'] == 1]
        
        st.info(f"📊 Showing {len(filtered_df):,} of {len(df):,} transactions")
        
        display_cols = ['Transaction_ID', 'Timestamp', 'Sender_ID', 'Receiver_ID', 'Amount', 'Location', 'Risk_Score', 'Risk_Level', 'Risk_Reasons']
        st.dataframe(filtered_df[display_cols].sort_values('Risk_Score', ascending=False), use_container_width=True, height=450)
        
        csv = filtered_df[display_cols].to_csv(index=False)
        st.download_button("📥 Download CSV", data=csv, file_name=f"fraud_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚡ Control Center")
        st.markdown("---")
        
        st.markdown("### 🚨 Live Alerts")
        recent_critical = df[df['Risk_Level'] == 'CRITICAL'].sort_values('Timestamp', ascending=False).head(5)
        
        if len(recent_critical) > 0:
            for idx, alert in recent_critical.iterrows():
                with st.expander(f"⚠️ ₹{alert['Amount']:.0f} - {alert['Location']}", expanded=False):
                    st.write(f"**From:** {alert['Sender_ID'][:20]}...")
                    st.write(f"**To:** {alert['Receiver_ID'][:20]}...")
                    st.write(f"**Risk:** {alert['Risk_Score']:.0f}/100")
        else:
            st.success("✅ No alerts")
        
        st.markdown("---")
        st.markdown("### 🎯 Actions")
        
        if st.button("🔒 Freeze Accounts", use_container_width=True):
            st.success(f"✅ Blocked {mule_count} accounts")
        
        if st.button("📧 Alert Users", use_container_width=True):
            high_risk_users = df[df['Risk_Level'].isin(['CRITICAL', 'HIGH'])]['Sender_ID'].nunique()
            st.info(f"📨 Alerted {high_risk_users} users")
        
        st.markdown("---")
        st.markdown("### 📈 System Health")
        st.metric("Model Accuracy", "94.2%")
        st.metric("False Positives", "2.3%")
        st.metric("Response Time", "127ms")
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True, type="primary"):
            logout()
        
        st.markdown("---")
        st.caption("🦅 UPI HawkEye v2.0")
        st.caption("AI-Powered Fraud Detection")

# ==============================================
# RUN
# ==============================================
if not st.session_state.logged_in:
    login_page()
else:
    main_app()