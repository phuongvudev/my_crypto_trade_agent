"""
Simple test dashboard to verify Streamlit is working correctly.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Test Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("🚀 Simple Test Dashboard")

st.write("This is a test to verify Streamlit is working correctly.")

# Simple metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Test Metric 1", "100", delta="10")

with col2:
    st.metric("Test Metric 2", "200", delta="-5")

with col3:
    st.metric("Test Metric 3", "300", delta="15")

# Simple chart
st.subheader("📈 Test Chart")

# Generate sample data
dates = pd.date_range(start='2025-01-01', periods=100, freq='1h')
values = 100 + np.random.randn(100).cumsum()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=dates,
    y=values,
    mode='lines',
    name='Test Data'
))

fig.update_layout(
    title="Sample Time Series",
    xaxis_title="Time",
    yaxis_title="Value"
)

st.plotly_chart(fig, use_container_width=True)

# Simple data table
st.subheader("📊 Test Data Table")

df = pd.DataFrame({
    'Column A': np.random.randn(10),
    'Column B': np.random.randn(10),
    'Column C': np.random.choice(['X', 'Y', 'Z'], 10)
})

st.dataframe(df)

st.success("✅ If you can see this, Streamlit is working correctly!")
