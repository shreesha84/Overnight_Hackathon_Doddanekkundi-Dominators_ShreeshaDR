import streamlit as st
import pandas as pd
import os
import numpy as np
import altair as alt

st.set_page_config(page_title="UPI Fraud Detection", layout="wide")

# ---------------------------
# HEADER
# ---------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#4B8BBE;'>
         UPI Fraud Detection Dashboard
    </h1>
    <p style='text-align:center; font-size:18px;'>
        Real-time anomaly monitoring • Risk scoring • Suspicious pattern detection
    </p>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# DATA LOADING
# ---------------------------

DATA_FILE = "large_upi_dataset.csv"   # your generator uses this file name

@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        st.error(f"Dataset not found: {DATA_FILE}")
        st.stop()
    return pd.read_csv(DATA_FILE)

df = load_data()


# ---------------------------
# PREPROCESSING
# ---------------------------

df['Timestamp'] = pd.to_datetime(df['Timestamp'])


# ---------------------------
# RULE 1 — MULE DETECTION (TOO MANY PAYMENTS TO SAME RECEIVER)
# ---------------------------

receiver_counts = df['Receiver_ID'].value_counts()
mule_receivers = receiver_counts[receiver_counts > 15].index.tolist()

df['Mule_Flag'] = df['Receiver_ID'].isin(mule_receivers)


# ---------------------------
# RULE 2 — IMPOSSIBLE LOCATION JUMP (UNREALISTIC TRAVEL)
# ---------------------------

df = df.sort_values(['Sender_ID', 'Timestamp'])
df['Prev_Location'] = df.groupby('Sender_ID')['Location'].shift(1)
df['Prev_Time'] = df.groupby('Sender_ID')['Timestamp'].shift(1)

def detect_impossible_travel(row):
    if pd.isna(row['Prev_Location']):
        return False
    time_diff = (row['Timestamp'] - row['Prev_Time']).total_seconds() / 60
    return row['Location'] != row['Prev_Location'] and time_diff < 30

df['Impossible_Travel'] = df.apply(detect_impossible_travel, axis=1)


# ---------------------------
# RISK SCORING ENGINE
# ---------------------------

df['Risk_Score'] = 0

df.loc[df['Mule_Flag'], 'Risk_Score'] += 70
df.loc[df['Impossible_Travel'], 'Risk_Score'] += 40

def risk_label(score):
    if score >= 70:
        return "High"
    elif score >= 30:
        return "Medium"
    return "Low"

df['Risk_Level'] = df['Risk_Score'].apply(risk_label)


# ---------------------------
# METRICS SECTION
# ---------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Total Transactions", f"{len(df):,}")
col2.metric("Suspicious Receivers (Mules)", len(mule_receivers))
col3.metric("High-Risk Transactions", len(df[df['Risk_Level'] == "High"]))


# ---------------------------
# VISUALIZATION — FRAUD RISK OVER TIME
# ---------------------------

risk_trend = (
    df.groupby(df['Timestamp'].dt.date)['Risk_Score']
    .mean()
    .reset_index()
    .rename(columns={'Timestamp': 'Date', 'Risk_Score': 'Avg_Risk'})
)

st.subheader("Daily Average Risk Score Trend")

risk_chart = (
    alt.Chart(risk_trend)
    .mark_line(point=True)
    .encode(
        x='Date:T',
        y='Avg_Risk:Q',
        tooltip=['Date', 'Avg_Risk']
    )
    .interactive()
)

st.altair_chart(risk_chart, use_container_width=True)


# ---------------------------
# ALERTS PANEL
# ---------------------------

st.markdown("## Fraud Alerts")

high_risk = df[df['Risk_Level'] == "High"]

if len(high_risk) == 0:
    st.success("No high-risk activity detected.")
else:
    st.warning(f"{len(high_risk)} high-risk transactions detected.")
    st.dataframe(high_risk[['Transaction_ID', 'Sender_ID', 'Receiver_ID',
                            'Amount', 'Location', 'Risk_Level', 'Risk_Score']])


# ---------------------------
# FILTER EXPLORER
# ---------------------------

st.markdown("## Explore Transactions")

unique_senders = df['Sender_ID'].unique().tolist()
selected_sender = st.selectbox("Filter by Sender ID (optional):", ["All"] + unique_senders)

if selected_sender != "All":
    filtered = df[df['Sender_ID'] == selected_sender]
else:
    filtered = df

st.dataframe(filtered.head(200))


# ---------------------------
# DOWNLOAD CLEANED DATA
# ---------------------------

st.download_button(
    label="Download Processed Dataset",
    data=df.to_csv(index=False),
    file_name="processed_upi_fraud_data.csv",
    mime="text/csv"
)
