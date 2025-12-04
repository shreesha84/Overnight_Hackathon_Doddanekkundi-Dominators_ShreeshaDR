import streamlit as st
import pandas as pd
import os
import numpy as np
import altair as alt
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from haversine import haversine, Unit

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="UPI HawkEye AI", 
    layout="wide", 
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Hacker Mode"
st.markdown("""
    <style>
    .stApp {background-color: #0E1117;}
    .metric-card {background-color: #262730; border: 1px solid #4B4B4B; padding: 15px; border-radius: 10px;}
    h1, h2, h3 {color: #FF4B4B !important;}
    </style>
    """, unsafe_allow_html=True)

# 2. HEADER
col_head1, col_head2 = st.columns([1, 4])
with col_head1:
    st.markdown("# 🦅")
with col_head2:
    st.title("UPI HawkEye: AI-Powered Fraud Defense")
    st.caption("Rule Engine • Isolation Forest (ML) • Graph Forensics")

# 3. HELPER FUNCTIONS
DATA_FILE = "large_upi_dataset.csv"

@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        return None
    return pd.read_csv(DATA_FILE)

@st.cache_data
def get_location_map(df):
    """Maps unique location strings to dummy (latitude, longitude) pairs."""
    unique_locations = df['Location'].unique()
    np.random.seed(42)
    # Generate dummy coordinates roughly within India's lat/long bounds
    location_map = {
        loc: (np.random.uniform(8, 30), np.random.uniform(68, 97))
        for loc in unique_locations
    }
    return location_map

def calculate_distance(loc1_name, loc2_name, loc_map):
    """Calculates haversine distance in km between two location names."""
    if loc1_name not in loc_map or loc2_name not in loc_map:
        return 0
    return haversine(loc_map[loc1_name], loc_map[loc2_name], unit=Unit.KILOMETERS)

# 4. LOAD & PREPROCESS DATA
df = load_data()

if df is None:
    st.error("⚠ Data file not found! Please run 'generate_data.py' first.")
    st.stop()

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
location_coordinates = get_location_map(df)

# 5. LOGIC ENGINE (THE BRAINS)
# -------------------------------------

# A. DETECT MONEY MULES (Rule: >15 transactions received)
receiver_counts = df['Receiver_ID'].value_counts()
mule_receivers = receiver_counts[receiver_counts > 15].index.tolist()
df['Mule_Flag'] = df['Receiver_ID'].isin(mule_receivers)

# B. DETECT IMPOSSIBLE TRAVEL
df = df.sort_values(['Sender_ID', 'Timestamp'])
df['Prev_Location'] = df.groupby('Sender_ID')['Location'].shift(1)
df['Prev_Time'] = df.groupby('Sender_ID')['Timestamp'].shift(1)
df['Time_Diff_Min'] = (df['Timestamp'] - df['Prev_Time']).dt.total_seconds() / 60

# Calculate Distance
df['Travel_Distance_km'] = df.apply(
    lambda row: calculate_distance(row['Location'], row['Prev_Location'], location_coordinates) 
    if pd.notna(row['Prev_Location']) else 0, axis=1
)

# Logic: Distance > 500 km AND Time Diff < 30 mins
mask_impossible_dist = (df['Travel_Distance_km'] > 500)
mask_fast_move = df['Time_Diff_Min'] < 30
df['Impossible_Travel'] = mask_impossible_dist & mask_fast_move

# C. ISOLATION FOREST (AI ANOMALY DETECTION)
df['Time_Diff_Min'] = df['Time_Diff_Min'].fillna(0)
df['Hour_of_Day'] = df['Timestamp'].dt.hour 

# Cyclical features for time (helps AI understand 23:00 is close to 00:00)
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour_of_Day'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour_of_Day'] / 24)

# Prepare features
features = df[['Amount', 'Time_Diff_Min', 'Hour_sin', 'Hour_cos']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train Model
iso_forest = IsolationForest(contamination=0.03, random_state=42)
df['Anomaly_Result'] = iso_forest.fit_predict(features_scaled)
df['AI_Flag'] = df['Anomaly_Result'] == -1

# D. CALCULATE RISK SCORE
df['Risk_Score'] = 0
df.loc[df['Mule_Flag'], 'Risk_Score'] += 50
df.loc[df['Impossible_Travel'], 'Risk_Score'] += 40
df.loc[df['AI_Flag'], 'Risk_Score'] += 30 

def get_risk_label(score):
    if score >= 70: return "CRITICAL"
    if score >= 30: return "MODERATE"
    return "LOW"

df['Risk_Level'] = df['Risk_Score'].apply(get_risk_label)

# 6. KEY METRICS DISPLAY
# -------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Transactions", f"{len(df):,}")
m2.metric("Money Mules Identified", len(mule_receivers), delta_color="inverse")
m3.metric("AI Detected Anomalies", len(df[df['AI_Flag']]), delta_color="inverse", help="Statistically unusual transactions found by AI.")
m4.metric("CRITICAL RISKS", len(df[df['Risk_Level'] == "CRITICAL"]), delta_color="inverse")

st.markdown("---")

# 7. MAIN TABS
# -------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Risk Dashboard", "🕸 Network Forensics", "📋 Data Explorer"])

# --- TAB 1: CHARTS ---
with tab1:
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("🤖 AI Anomaly Analysis")
        st.caption("Red dots are transactions the AI flagged as 'Mathematically Unusual'")
        
        ai_chart = alt.Chart(df).mark_circle(size=60).encode(
            x=alt.X('Time_Diff_Min', title='Time Diff (Min)'),
            y='Amount',
            color=alt.Color('AI_Flag', scale=alt.Scale(domain=[True, False], range=['#FF4B4B', '#00C9FF']), legend=alt.Legend(title="AI Flag")),
            tooltip=['Transaction_ID', 'Amount', 'Time_Diff_Min', 'AI_Flag', 'Risk_Score']
        ).interactive()
        st.altair_chart(ai_chart, use_container_width=True)

    with col_chart2:
        st.subheader("Fraud Distribution by Risk Level")
        risk_counts = df['Risk_Level'].value_counts().reset_index()
        risk_counts.columns = ['Risk_Level', 'Count']
        
        risk_order = ["CRITICAL", "MODERATE", "LOW"]
        risk_colors = alt.Scale(domain=risk_order, range=['#FF4B4B', '#FFAA00', '#00C9FF'])
        
        bar_chart = alt.Chart(risk_counts).mark_bar().encode(
            x='Risk_Level',
            y='Count',
            color=alt.Color('Risk_Level', scale=risk_colors),
            order=alt.Order('Risk_Level', sort='descending'),
            tooltip=['Risk_Level', 'Count']
        )
        st.altair_chart(bar_chart, use_container_width=True)

# --- TAB 2: NETWORK GRAPH ---
with tab2:
    st.subheader("Fraud Ring Visualizer")
    
    max_nodes = st.sidebar.slider("Max Nodes for Graph Display", min_value=10, max_value=500, value=100, step=10)
    st.info(f"Visualizing up to {max_nodes} transactions flagged as CRITICAL risk.")

    graph_data = df[df['Risk_Level'] == "CRITICAL"].head(max_nodes)
    
    if len(graph_data) > 0:
        G = nx.DiGraph()
        
        for _, row in graph_data.iterrows():
            sender = row['Sender_ID']
            receiver = row['Receiver_ID']
            amount = row['Amount']
            risk_score = row['Risk_Score']
            
            G.add_node(sender, title=f"Sender: {sender}", color='#00C9FF', size=15)
            G.add_node(receiver, title=f"Receiver: {receiver} | Score: {risk_score}", color='#FF4B4B', size=25)
            G.add_edge(sender, receiver, title=f"₹{amount} (Score: {risk_score})", value=amount)
        
        net = Network(height='500px', width='100%', directed=True, bgcolor='#222222', font_color='white')
        net.from_nx(G)
        net.repulsion(node_distance=420, central_gravity=0.33, spring_length=110, spring_strength=0.10, damping=0.95)
        
        try:
            path = "pyvis_graph_temp.html"
            net.save_graph(path)
            with open(path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            components.html(source_code, height=510)
        except Exception as e:
            st.error(f"Graph Error: {e}")
    else:
        st.success("No critical fraud rings active at this moment.")

# --- TAB 3: DATA EXPLORER ---
with tab3:
    st.subheader("Live Transaction Feed")
    
    filter_risk = st.selectbox("Filter by Risk Level", ["All", "CRITICAL", "MODERATE", "LOW"], index=0)
    
    if filter_risk != "All":
        display_df = df[df['Risk_Level'] == filter_risk]
    else:
        display_df = df.sort_values(by='Risk_Score', ascending=False)
        
    st.dataframe(
        display_df[['Transaction_ID', 'Sender_ID', 'Receiver_ID', 'Amount', 'Location', 'Risk_Score', 'AI_Flag', 'Impossible_Travel', 'Mule_Flag', 'Risk_Level']].head(500),
        use_container_width=True
    )

# 8. SIDEBAR ACTIONS
# -------------------------------------
st.sidebar.title("👮 Action Center")
st.sidebar.markdown("---")

critical_receivers = df[df['Risk_Level'] == "CRITICAL"]['Receiver_ID'].unique()
if st.sidebar.button(f"Freeze {len(critical_receivers)} High-Risk Receivers"):
    st.sidebar.success(f"⚠ COMMAND SENT: Blocked {len(critical_receivers)} critical recipient accounts.")

st.sidebar.markdown("### Export Reports")
if st.sidebar.button("📄 Generate Cyber Cell Report"):
    report_csv = df[df['Risk_Level'] == "CRITICAL"].to_csv(index=False)
    st.sidebar.info("Report generated successfully. Downloading...")
    st.sidebar.download_button(
        label="📥 Download CRITICAL Risk Data CSV",
        data=report_csv,
        file_name="upi_hawk_eye_critical_report.csv",
        mime="text/csv"
    )

st.sidebar.markdown("---")
st.sidebar.caption("UPI HawkEye v1.1 | Enhanced AI")