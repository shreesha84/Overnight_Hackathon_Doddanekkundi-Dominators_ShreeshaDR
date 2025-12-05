UPI HawkEye AI: Multi-Layered Fraud Defense Dashboard (v1.3)

## Project Overview

UPI HawkEye AI is a real-time fraud detection system designed to combat the sophisticated, instant, and irreversible nature of fraud within the Unified Payments Interface (UPI) ecosystem.

Unlike traditional rule-based systems, this project utilizes a multi-layered, adaptive architecture combining deterministic rules, unsupervised machine learning (ML), and graph forensics to minimize false positives and detect novel scam patterns (e.g., refund scams, social engineering).

Problem Addressed UPI fraud cost Indians over ₹1,000 crores in 2024. The instant nature of UPI transactions means prevention is the only viable defense, which traditional systems fail to provide.


## Key Features

The system is built on three core analytical layers:

1.  Enhanced Rule Engine: High-confidence flags for known high-risk behavior (e.g., Money Mule Flag, Impossible Travel, Suspicious Refund Scam pattern).
2.  Adaptive AI (Isolation Forest): Uses Unsupervised ML to detect zero-day and novel anomalies based on statistical deviation from normal transaction behavior.
3.  Graph Forensics: Visualizes critical risk transactions as an interactive network graph (Fraud Rings) to allow human analysts to visually identify central mule accounts and complex schemes.
4.  Actionability: Provides a real-time dashboard to assign Risk Scores and Risk Levels (CRITICAL/MODERATE/LOW), and tools to initiate commands like Freezing High-Risk Receivers.

## Tech Stack

#Core Development & Deployment
1. Python 3.8+ (Primary language)
2.Streamlit (Dashboard Frontend/Web Framework)

#Data Science & Modeling
1.Pandas (Data Ingestion & Manipulation)
2.NumPy (Numerical Operations)
3.Scikit-learn (Machine Learning - Isolation Forest)
4.SciPy/Haversine (Geospatial Calculations)

#Visualization & Graph Theory
1.NetworkX (Graph Modeling)
2.PyVis (Interactive Graph Rendering)
3.Altair (Statistical Charting)

## Setup and Execution

### Prerequisites

You must have Python 3.8 or higher installed.

### 1. Install Dependencies

Install all necessary Python libraries using pip:

#pip install streamlit pandas numpy scikit-learn altair networkx pyvis haversine scipy

2. Data Preparation
This dashboard requires a transaction data file named large_upi_dataset.csv in the root directory.

3. Run the Dashboard
Execute the main Python script using Streamlit:

#streamlit run dashboard.py

The application will launch in your default web browser (usually at http://localhost:8501).

4.CustomizationRisk Thresholds: 
The risk scores and final CRITICAL threshold can be adjusted in the CALCULATE RISK SCORE section of the code.
Contamination Rate: The IsolationForest parameter contamination=0.03 (expected anomaly percentage) can be tuned to control the sensitivity of the AI model.
