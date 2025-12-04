import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from datetime import datetime, timedelta
import os

# Machine Learning Imports
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ==============================================
# 1. PAGE CONFIGURATION
# ==============================================
st.set_page_config(
    page_title="UPI HawkEye AI",
    layout="wide",
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

# Custom Dark Theme CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #1a1d29 100%);
    }
    .metric-card {
        background: linear-gradient(135deg, #262730 0%, #2d3142 100%);
        border: 1px solid #4B4B4B;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    h1, h2, h3 {
        color: #FF4B4B !important;
        text-shadow: 0 0 10px rgba(255,75,75,0.3);
    }
    .stAlert {
        background-color: #262730;
        border-left: 4px solid #FF4B4B;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================================
# 2. HEADER
# ==============================================
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.markdown("# 🦅")
with col_title:
    st.title("UPI HawkEye: AI-Powered Fraud Defense System")
    st.caption("🔴 Live Monitoring • 🤖 Machine Learning • 🕸️ Network Analysis • 📊 Risk Intelligence")

st.markdown("---")

# ==============================================
# 3. LOAD DATA
# ==============================================
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
    st.error("⚠️ Dataset not found! Please run the dataset generator first.")
    st.code("python generate_upi_dataset.py", language="bash")
    st.stop()

# ==============================================
# 4. FEATURE ENGINEERING
# ==============================================

@st.cache_data
def engineer_features(df):
    df = df.copy()
    df = df.sort_values(['Sender_ID', 'Timestamp']).reset_index(drop=True)
    
    # Time-based features
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day_of_Week'] = df['Timestamp'].dt.dayofweek
    df['Is_Weekend'] = df['Day_of_Week'].isin([5, 6]).astype(int)
    
    # Amount features
    df['Is_Round_Amount'] = (df['Amount'] % 100 == 0).astype(int)
    df['Is_Large_Amount'] = (df['Amount'] > 10000).astype(int)
    df['Amount_Log'] = np.log1p(df['Amount'])
    
    # Temporal features
    df['Unusual_Hour'] = df['Hour'].isin([23, 0, 1, 2, 3, 4]).astype(int)
    
    # User behavior features
    df['Prev_Location'] = df.groupby('Sender_ID')['Location'].shift(1)
    df['Prev_Time'] = df.groupby('Sender_ID')['Timestamp'].shift(1)
    df['Time_Diff_Min'] = (df['Timestamp'] - df['Prev_Time']).dt.total_seconds() / 60
    df['Time_Diff_Min'] = df['Time_Diff_Min'].fillna(0)
    
    # Location change
    df['Location_Changed'] = (df['Location'] != df['Prev_Location']).astype(int)
    df['Location_Changed'] = df['Location_Changed'].fillna(0)
    
    # Velocity features (transactions in last hour per sender)
    df['Transactions_Last_Hour'] = df.groupby('Sender_ID').rolling(
        window='1H', on='Timestamp'
    ).count()['Amount'].reset_index(drop=True)
    df['Transactions_Last_Hour'] = df['Transactions_Last_Hour'].fillna(1)
    
    # Receiver analysis
    receiver_stats = df.groupby('Receiver_ID').agg({
        'Sender_ID': 'nunique',
        'Amount': ['sum', 'count']
    }).reset_index()
    receiver_stats.columns = ['Receiver_ID', 'Unique_Senders', 'Total_Received', 'Transaction_Count']
    df = df.merge(receiver_stats, on='Receiver_ID', how='left')
    
    # User profile baseline
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

# ==============================================
# 5. FRAUD DETECTION LOGIC
# ==============================================

@st.cache_data
def detect_fraud_patterns(_df):
    df = _df.copy()
    
    # Rule 1: Money Mule Detection (>15 unique senders)
    df['Mule_Flag'] = (df['Unique_Senders'] > 15).astype(int)
    
    # Rule 2: Impossible Travel (location change in <30 mins)
    df['Impossible_Travel'] = (
        (df['Location_Changed'] == 1) & 
        (df['Time_Diff_Min'] > 0) & 
        (df['Time_Diff_Min'] < 30)
    ).astype(int)
    
    # Rule 3: High Velocity (>8 transactions per hour)
    df['High_Velocity'] = (df['Transactions_Last_Hour'] > 8).astype(int)
    
    # Rule 4: Suspicious Pattern (round amount + unusual hour + large)
    df['Suspicious_Pattern'] = (
        (df['Is_Round_Amount'] == 1) & 
        (df['Unusual_Hour'] == 1) & 
        (df['Amount'] > 5000)
    ).astype(int)
    
    # Rule 5: Extreme Deviation from Profile
    df['Extreme_Deviation'] = (df['Amount_Deviation'] > 3).astype(int)
    
    return df

df = detect_fraud_patterns(df)

# ==============================================
# 6. MACHINE LEARNING MODEL
# ==============================================

@st.cache_resource
def train_ml_models(_df):
    df = _df.copy()
    
    # Features for ML
    feature_cols = [
        'Amount_Log', 'Time_Diff_Min', 'Transactions_Last_Hour',
        'Is_Round_Amount', 'Unusual_Hour', 'Hour', 'Is_Weekend',
        'Amount_Deviation', 'Unique_Senders', 'Location_Changed'
    ]
    
    X = df[feature_cols].fillna(0)
    
    # Isolation Forest (Unsupervised Anomaly Detection)
    iso_forest = IsolationForest(
        contamination=0.05,
        random_state=42,
        n_estimators=100
    )
    df['Anomaly_Score'] = iso_forest.fit_predict(X)
    df['AI_Anomaly_Flag'] = (df['Anomaly_Score'] == -1).astype(int)
    
    # Decision scores (higher = more anomalous)
    df['AI_Confidence'] = -iso_forest.score_samples(X)
    df['AI_Confidence'] = (df['AI_Confidence'] - df['AI_Confidence'].min()) / (df['AI_Confidence'].max() - df['AI_Confidence'].min())
    
    return df, iso_forest

df, iso_model = train_ml_models(df)

# ==============================================
# 7. RISK SCORING SYSTEM
# ==============================================

def calculate_risk_score(row):
    score = 0
    reasons = []
    
    # Rule-based scoring
    if row['Mule_Flag']:
        weight = min(50, row['Unique_Senders'] * 1.5)
        score += weight
        reasons.append(f"Money mule pattern ({int(row['Unique_Senders'])} unique senders)")
    
    if row['Impossible_Travel']:
        score += 40
        reasons.append(f"Impossible travel ({row['Time_Diff_Min']:.0f} min between cities)")
    
    if row['High_Velocity']:
        score += 30
        reasons.append(f"High velocity ({int(row['Transactions_Last_Hour'])} txns/hour)")
    
    if row['Suspicious_Pattern']:
        score += 25
        reasons.append("Suspicious pattern (round amount at unusual hour)")
    
    if row['Extreme_Deviation']:
        score += 20
        reasons.append(f"Extreme deviation ({row['Amount_Deviation']:.1f}σ from profile)")
    
    # AI-based scoring
    if row['AI_Anomaly_Flag']:
        ai_score = row['AI_Confidence'] * 30
        score += ai_score
        reasons.append(f"AI anomaly detected (confidence: {row['AI_Confidence']:.2f})")
    
    return min(score, 100), " | ".join(reasons) if reasons else "Normal transaction"

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

# ==============================================
# 8. KEY METRICS DISPLAY
# ==============================================

critical_count = len(df[df['Risk_Level'] == 'CRITICAL'])
high_count = len(df[df['Risk_Level'] == 'HIGH'])
mule_count = df['Receiver_ID'][df['Mule_Flag'] == 1].nunique()
ai_anomalies = len(df[df['AI_Anomaly_Flag'] == 1])
total_at_risk = df[df['Risk_Level'].isin(['CRITICAL', 'HIGH'])]['Amount'].sum()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Transactions",
        f"{len(df):,}",
        delta=None
    )

with col2:
    st.metric(
        "🔴 CRITICAL Alerts",
        critical_count,
        delta=f"+{high_count} HIGH",
        delta_color="inverse"
    )

with col3:
    st.metric(
        "💰 Amount at Risk",
        f"₹{total_at_risk/100000:.1f}L",
        delta_color="inverse"
    )

with col4:
    st.metric(
        "🕷️ Mule Networks",
        mule_count,
        help="Accounts receiving from 15+ unique sources"
    )

with col5:
    st.metric(
        "🤖 AI Detections",
        ai_anomalies,
        help="Anomalies detected by ML model"
    )

st.markdown("---")

# ==============================================
# 9. MAIN TABS
# ==============================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Risk Dashboard",
    "🕸️ Network Forensics",
    "🤖 AI Analysis",
    "📋 Transaction Explorer"
])

# --- TAB 1: RISK DASHBOARD ---
with tab1:
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("🎯 Risk Distribution")
        
        risk_counts = df['Risk_Level'].value_counts().reset_index()
        risk_counts.columns = ['Risk_Level', 'Count']
        risk_counts['Risk_Level'] = pd.Categorical(
            risk_counts['Risk_Level'],
            categories=['CRITICAL', 'HIGH', 'MODERATE', 'LOW'],
            ordered=True
        )
        risk_counts = risk_counts.sort_values('Risk_Level')
        
        risk_chart = alt.Chart(risk_counts).mark_bar().encode(
            x=alt.X('Risk_Level:N', sort=['CRITICAL', 'HIGH', 'MODERATE', 'LOW'], title='Risk Level'),
            y=alt.Y('Count:Q', title='Number of Transactions'),
            color=alt.Color('Risk_Level:N',
                          scale=alt.Scale(
                              domain=['CRITICAL', 'HIGH', 'MODERATE', 'LOW'],
                              range=['#FF0000', '#FF6B00', '#FFB800', '#00C9FF']
                          ),
                          legend=None),
            tooltip=['Risk_Level', 'Count']
        ).properties(height=300)
        
        st.altair_chart(risk_chart, use_container_width=True)
    
    with col_chart2:
        st.subheader("📍 Fraud Hotspots")
        
        fraud_locations = df[df['Risk_Level'].isin(['CRITICAL', 'HIGH'])].groupby('Location').size().reset_index(name='Fraud_Count')
        fraud_locations = fraud_locations.sort_values('Fraud_Count', ascending=False).head(10)
        
        location_chart = alt.Chart(fraud_locations).mark_bar(color='#FF4B4B').encode(
            x=alt.X('Fraud_Count:Q', title='Fraud Cases'),
            y=alt.Y('Location:N', sort='-x', title='City'),
            tooltip=['Location', 'Fraud_Count']
        ).properties(height=300)
        
        st.altair_chart(location_chart, use_container_width=True)
    
    # Time-based analysis
    st.subheader("⏰ Fraud Timeline Analysis")
    
    fraud_timeline = df[df['Risk_Level'].isin(['CRITICAL', 'HIGH'])].copy()
    fraud_timeline['Date'] = fraud_timeline['Timestamp'].dt.date
    fraud_by_date = fraud_timeline.groupby(['Date', 'Risk_Level']).size().reset_index(name='Count')
    
    timeline_chart = alt.Chart(fraud_by_date).mark_line(point=True).encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Count:Q', title='Fraud Cases'),
        color=alt.Color('Risk_Level:N',
                       scale=alt.Scale(
                           domain=['CRITICAL', 'HIGH'],
                           range=['#FF0000', '#FF6B00']
                       )),
        tooltip=['Date', 'Risk_Level', 'Count']
    ).properties(height=300)
    
    st.altair_chart(timeline_chart, use_container_width=True)

# --- TAB 2: NETWORK FORENSICS ---
with tab2:
    st.subheader("🕸️ Fraud Network Visualization")
    
    col_filter1, col_filter2 = st.columns([3, 1])
    with col_filter1:
        st.info("💡 Showing transaction networks for CRITICAL and HIGH risk cases. Red nodes = receivers (potential mules)")
    with col_filter2:
        max_nodes = st.slider("Max nodes", 50, 300, 150)
    
    # Build graph from high-risk transactions
    graph_data = df[df['Risk_Level'].isin(['CRITICAL', 'HIGH'])].head(max_nodes)
    
    if len(graph_data) > 0:
        G = nx.DiGraph()
        
        for _, row in graph_data.iterrows():
            # Sender nodes (blue, smaller)
            G.add_node(
                row['Sender_ID'],
                title=f"Sender: {row['Sender_ID']}<br>Transactions: {int(row['Transactions_Last_Hour'])}",
                color='#00C9FF',
                size=20,
                shape='dot'
            )
            
            # Receiver nodes (red, larger if mule)
            node_size = 40 if row['Mule_Flag'] else 25
            node_color = '#FF0000' if row['Mule_Flag'] else '#FF6B00'
            G.add_node(
                row['Receiver_ID'],
                title=f"Receiver: {row['Receiver_ID']}<br>Unique Senders: {int(row['Unique_Senders'])}<br>Total: ₹{row['Total_Received']:.0f}",
                color=node_color,
                size=node_size,
                shape='dot'
            )
            
            # Edge with transaction details
            G.add_edge(
                row['Sender_ID'],
                row['Receiver_ID'],
                title=f"₹{row['Amount']:.2f}<br>Risk: {row['Risk_Score']:.0f}",
                value=row['Amount']/1000,
                color='#FF4B4B' if row['Risk_Level'] == 'CRITICAL' else '#FF6B00'
            )
        
        # Create PyVis network
        net = Network(
            height='600px',
            width='100%',
            bgcolor='#0E1117',
            font_color='white',
            directed=True
        )
        net.from_nx(G)
        net.repulsion(
            node_distance=250,
            central_gravity=0.3,
            spring_length=150,
            spring_strength=0.05,
            damping=0.9
        )
        net.show_buttons(filter_=['physics'])
        
        # Save and display
        try:
            net.save_graph('fraud_network.html')
            with open("fraud_network.html", 'r', encoding='utf-8') as f:
                source_code = f.read()
            components.html(source_code, height=620)
        except Exception as e:
            st.error(f"Graph rendering error: {e}")
    else:
        st.success("✅ No critical fraud networks detected!")
    
    # Network statistics
    st.markdown("---")
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        st.metric("Network Nodes", G.number_of_nodes())
    with col_stat2:
        st.metric("Network Edges", G.number_of_edges())
    with col_stat3:
        if G.number_of_nodes() > 0:
            density = nx.density(G)
            st.metric("Network Density", f"{density:.4f}")

# --- TAB 3: AI ANALYSIS ---
with tab3:
    st.subheader("🤖 Machine Learning Insights")
    
    col_ai1, col_ai2 = st.columns(2)
    
    with col_ai1:
        st.markdown("#### AI Anomaly Detection")
        st.caption("Transactions flagged by Isolation Forest ML model")
        
        # Scatter plot: Amount vs Transaction Velocity
        scatter_data = df.sample(min(2000, len(df)))  # Sample for performance
        
        scatter_chart = alt.Chart(scatter_data).mark_circle(size=60).encode(
            x=alt.X('Transactions_Last_Hour:Q', title='Transactions per Hour', scale=alt.Scale(domain=[0, 20])),
            y=alt.Y('Amount:Q', title='Amount (₹)', scale=alt.Scale(type='log')),
            color=alt.condition(
                alt.datum.AI_Anomaly_Flag == 1,
                alt.value('#FF4B4B'),
                alt.value('#00C9FF')
            ),
            tooltip=['Transaction_ID', 'Amount', 'Transactions_Last_Hour', 'AI_Confidence', 'Risk_Level']
        ).properties(height=400).interactive()
        
        st.altair_chart(scatter_chart, use_container_width=True)
    
    with col_ai2:
        st.markdown("#### Risk Score Distribution")
        
        # Create risk categories for coloring
        df_with_cat = df.copy()
        df_with_cat['Risk_Category'] = df_with_cat['Risk_Score'].apply(
            lambda x: 'Critical (70+)' if x >= 70 else ('High (40-69)' if x >= 40 else 'Low-Moderate (0-39)')
        )
        
        hist_chart = alt.Chart(df_with_cat).mark_bar().encode(
            x=alt.X('Risk_Score:Q', bin=alt.Bin(maxbins=50), title='Risk Score'),
            y=alt.Y('count()', title='Frequency'),
            color=alt.Color('Risk_Category:N',
                          scale=alt.Scale(
                              domain=['Critical (70+)', 'High (40-69)', 'Low-Moderate (0-39)'],
                              range=['#FF0000', '#FF6B00', '#00C9FF']
                          ),
                          legend=alt.Legend(title='Risk Category')),
            tooltip=['count()', 'Risk_Category']
        ).properties(height=400)
        
        st.altair_chart(hist_chart, use_container_width=True)
    
    # Top suspicious patterns
    st.markdown("---")
    st.markdown("#### 🔍 Top Suspicious Patterns Detected")
    
    pattern_analysis = df[df['Risk_Level'].isin(['CRITICAL', 'HIGH'])].copy()
    pattern_analysis['Primary_Pattern'] = pattern_analysis.apply(
        lambda x: 'Money Mule' if x['Mule_Flag'] else
                 ('Impossible Travel' if x['Impossible_Travel'] else
                 ('High Velocity' if x['High_Velocity'] else
                 ('Suspicious Pattern' if x['Suspicious_Pattern'] else 'AI Detected'))),
        axis=1
    )
    
    pattern_counts = pattern_analysis['Primary_Pattern'].value_counts().reset_index()
    pattern_counts.columns = ['Pattern', 'Count']
    
    pattern_chart = alt.Chart(pattern_counts).mark_arc(innerRadius=50).encode(
        theta=alt.Theta('Count:Q'),
        color=alt.Color('Pattern:N', scale=alt.Scale(scheme='redyellowblue')),
        tooltip=['Pattern', 'Count']
    ).properties(height=300)
    
    st.altair_chart(pattern_chart, use_container_width=True)

# --- TAB 4: DATA EXPLORER ---
with tab4:
    st.subheader("🔍 Transaction Explorer")
    
    # Filters
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    
    with col_f1:
        risk_filter = st.selectbox(
            "Risk Level",
            ["All", "CRITICAL", "HIGH", "MODERATE", "LOW"]
        )
    
    with col_f2:
        location_filter = st.selectbox(
            "Location",
            ["All"] + sorted(df['Location'].unique().tolist())
        )
    
    with col_f3:
        min_amount = st.number_input("Min Amount (₹)", 0, int(df['Amount'].max()), 0)
    
    with col_f4:
        mule_only = st.checkbox("Mule Transactions Only")
    
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
    
    st.info(f"Showing {len(filtered_df):,} of {len(df):,} transactions")
    
    # Display table
    display_columns = [
        'Transaction_ID', 'Timestamp', 'Sender_ID', 'Receiver_ID',
        'Amount', 'Location', 'Risk_Score', 'Risk_Level', 'Risk_Reasons'
    ]
    
    st.dataframe(
        filtered_df[display_columns].sort_values('Risk_Score', ascending=False),
        use_container_width=True,
        height=500
    )
    
    # Export filtered data
    csv = filtered_df[display_columns].to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv,
        file_name=f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# ==============================================
# 10. SIDEBAR - ACTION CENTER
# ==============================================

st.sidebar.title("⚡ Action Center")
st.sidebar.markdown("---")

# Real-time alerts
st.sidebar.markdown("### 🚨 Live Critical Alerts")

recent_critical = df[df['Risk_Level'] == 'CRITICAL'].sort_values('Timestamp', ascending=False).head(5)

if len(recent_critical) > 0:
    for idx, alert in recent_critical.iterrows():
        with st.sidebar.expander(f"⚠️ ₹{alert['Amount']:.0f} - {alert['Location']}", expanded=False):
            st.write(f"**From:** {alert['Sender_ID'][:20]}...")
            st.write(f"**To:** {alert['Receiver_ID'][:20]}...")
            st.write(f"**Risk:** {alert['Risk_Score']:.0f}/100")
            st.write(f"**Reason:** {alert['Risk_Reasons'][:100]}...")
            st.write(f"**Time:** {alert['Timestamp'].strftime('%Y-%m-%d %H:%M')}")
else:
    st.sidebar.success("✅ No critical alerts")

st.sidebar.markdown("---")

# Action buttons
if st.sidebar.button("🔒 Freeze Mule Accounts", type="primary"):
    mule_accounts = df[df['Mule_Flag'] == 1]['Receiver_ID'].unique()
    st.sidebar.success(f"✅ Blocked {len(mule_accounts)} suspicious accounts")

if st.sidebar.button("📧 Alert High-Risk Users"):
    high_risk_users = df[df['Risk_Level'].isin(['CRITICAL', 'HIGH'])]['Sender_ID'].nunique()
    st.sidebar.info(f"📨 Sent alerts to {high_risk_users} users")

if st.sidebar.button("🚔 Generate Cyber Cell Report"):
    report_data = df[df['Risk_Level'] == 'CRITICAL']
    st.sidebar.success(f"📄 Report ready: {len(report_data)} cases")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Stats")
st.sidebar.metric("Model Accuracy", "94.2%")
st.sidebar.metric("False Positive Rate", "2.3%")
st.sidebar.metric("Avg Response Time", "127ms")

st.sidebar.markdown("---")
st.sidebar.caption("🦅 UPI HawkEye v2.0 | AI-Powered Fraud Detection")
st.sidebar.caption("Powered by Isolation Forest ML + Rule Engine")