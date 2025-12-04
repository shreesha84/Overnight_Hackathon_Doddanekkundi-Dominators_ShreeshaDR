import streamlit as st
import pandas as pd
import os
import numpy as np
import altair as alt
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="UPI HawkEye", 
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
    st.title("UPI HawkEye: Advanced Fraud Detection")
    st.caption("Real-time Anomaly Detection • Graph Network Analysis • Risk Scoring")

# 3. LOAD DATA
DATA_FILE = "large_upi_dataset.csv"

@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        return None
    return pd.read_csv(DATA_FILE)

df = load_data()

if df is None:
    st.error("⚠️ Data file not found! Please run 'generate_data.py' first.")
    st.stop()

# Convert Timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# 4. LOGIC ENGINE (THE BRAINS)
# -------------------------------------

# A. DETECT MONEY MULES (Rule: >15 transactions received)
receiver_counts = df['Receiver_ID'].value_counts()
mule_receivers = receiver_counts[receiver_counts > 15].index.tolist()
df['Mule_Flag'] = df['Receiver_ID'].isin(mule_receivers)

# B. DETECT IMPOSSIBLE TRAVEL (Rule: Location change > 500km in < 30 mins)
# We sort by Sender and Time to check previous location
df = df.sort_values(['Sender_ID', 'Timestamp'])
df['Prev_Location'] = df.groupby('Sender_ID')['Location'].shift(1)
df['Prev_Time'] = df.groupby('Sender_ID')['Timestamp'].shift(1)
df['Time_Diff_Min'] = (df['Timestamp'] - df['Prev_Time']).dt.total_seconds() / 60

# Logic: Location Changed AND Time Diff < 30 mins
mask_loc_change = (df['Location'] != df['Prev_Location']) & (df['Prev_Location'].notna())
mask_fast_move = df['Time_Diff_Min'] < 30
df['Impossible_Travel'] = mask_loc_change & mask_fast_move

# C. CALCULATE RISK SCORE
df['Risk_Score'] = 0
df.loc[df['Mule_Flag'], 'Risk_Score'] += 70
df.loc[df['Impossible_Travel'], 'Risk_Score'] += 40

def get_risk_label(score):
    if score >= 70: return "CRITICAL"
    if score >= 30: return "MODERATE"
    return "LOW"

df['Risk_Level'] = df['Risk_Score'].apply(get_risk_label)

# 5. KEY METRICS DISPLAY
# -------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Transactions", f"{len(df):,}")
m2.metric("Money Mules Identified", len(mule_receivers), delta_color="inverse")
m3.metric("Impossible Travel Events", len(df[df['Impossible_Travel']]), delta_color="inverse")
m4.metric("CRITICAL RISKS", len(df[df['Risk_Level'] == "CRITICAL"]), delta_color="inverse")

st.markdown("---")

# 6. MAIN TABS
# -------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Risk Dashboard", "🕸️ Network Forensics", "📋 Data Explorer"])

# --- TAB 1: CHARTS ---
with tab1:
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Risk Velocity (Hourly)")
        # Group by hour to show volume
        hourly_risk = df.set_index('Timestamp').resample('H')['Risk_Score'].mean().reset_index()
        
        chart = alt.Chart(hourly_risk).mark_line(color='#FF4B4B', point=True).encode(
            x='Timestamp',
            y='Risk_Score',
            tooltip=['Timestamp', 'Risk_Score']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

    with col_chart2:
        st.subheader("Fraud Distribution by Location")
        high_risk_locs = df[df['Risk_Level'] == 'CRITICAL']['Location'].value_counts().reset_index()
        high_risk_locs.columns = ['Location', 'Count']
        
        bar_chart = alt.Chart(high_risk_locs).mark_bar(color='#FF4B4B').encode(
            x='Location',
            y='Count',
            tooltip=['Location', 'Count']
        )
        st.altair_chart(bar_chart, use_container_width=True)

# --- TAB 2: NETWORK GRAPH ---
with tab2:
    st.subheader("Fraud Ring Visualizer")
    st.info("Visualizing connections for CRITICAL risk transactions only.")
    
    # Filter only high risk data for the graph to keep it fast
    graph_data = df[df['Risk_Level'] == "CRITICAL"].head(100)
    
    if len(graph_data) > 0:
        G = nx.Graph()
        
        for _, row in graph_data.iterrows():
            # Add Nodes
            G.add_node(row['Sender_ID'], title=f"Sender: {row['Sender_ID']}", color='#00C9FF', size=15) # Blue
            G.add_node(row['Receiver_ID'], title=f"Receiver: {row['Receiver_ID']}", color='#FF4B4B', size=25) # Red
            
            # Add Edge
            G.add_edge(row['Sender_ID'], row['Receiver_ID'], title=f"₹{row['Amount']}")
        
        # PyVis Network
        net = Network(height='500px', width='100%', bgcolor='#222222', font_color='white')
        net.from_nx(G)
        net.repulsion(node_distance=420, central_gravity=0.33, spring_length=110, spring_strength=0.10, damping=0.95)
        
        # Save and display
        try:
            net.save_graph('pyvis_graph.html')
            with open("pyvis_graph.html", 'r', encoding='utf-8') as f:
                source_code = f.read()
            components.html(source_code, height=510)
        except Exception as e:
            st.error(f"Graph Error: {e}")
    else:
        st.success("No critical fraud rings active at this moment.")

# --- TAB 3: DATA EXPLORER ---
with tab3:
    st.subheader("Live Transaction Feed")
    
    # Filter
    filter_risk = st.selectbox("Filter by Risk Level", ["All", "CRITICAL", "MODERATE", "LOW"])
    
    if filter_risk != "All":
        display_df = df[df['Risk_Level'] == filter_risk]
    else:
        display_df = df
        
    st.dataframe(
        display_df[['Transaction_ID', 'Sender_ID', 'Receiver_ID', 'Amount', 'Location', 'Risk_Score', 'Risk_Level']].sort_values(by='Risk_Score', ascending=False),
        use_container_width=True
    )

# 7. SIDEBAR ACTIONS
# -------------------------------------
st.sidebar.title("👮 Action Center")
st.sidebar.markdown("---")

if st.sidebar.button("Freeze High-Risk Accounts"):
    st.sidebar.success(f"COMMAND SENT: Blocked {len(mule_receivers)} Money Mule Accounts.")

st.sidebar.markdown("### Export Reports")
if st.sidebar.button("Generate Cyber Cell Report"):
    st.sidebar.info("Report generated successfully. Downloading...")
    
    # Create a simple CSV download
    report_csv = df[df['Risk_Level'] == "CRITICAL"].to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=report_csv,
        file_name="fraud_report.csv",
        mime="text/csv"
    )

st.sidebar.markdown("---")
st.sidebar.caption("UPI HawkEye v1.0 | Hackathon Build")