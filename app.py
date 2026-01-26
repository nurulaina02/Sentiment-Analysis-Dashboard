import os
import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import clean_text
from utils.sentiment import predict_sentiment

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    layout="wide"
)

st.title("📊 Sentiment Analysis Dashboard")

# -----------------------------------
# FIX: ABSOLUTE PATH FOR DATASET
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "sentiment_clean.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# -----------------------------------
# SENTIMENT ANALYSIS
# -----------------------------------
sentiments = []
scores = []

with st.spinner("Analyzing sentiments..."):
    for text in df["text"]:
        cleaned = clean_text(text)
        label, score = predict_sentiment(cleaned)
        sentiments.append(label)
        scores.append(score)

df["sentiment"] = sentiments
df["confidence"] = scores

# -----------------------------------
# VISUALIZATION
# -----------------------------------
col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(
        df,
        x="sentiment",
        color="sentiment",
        title="Sentiment Distribution"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.box(
        df,
        x="sentiment",
        y="confidence",
        title="Confidence Score by Sentiment"
    )
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------
# USER INPUT
# -----------------------------------
st.subheader("🔍 Try Your Own Text")

user_text = st.text_area("Enter text")

if st.button("Analyze"):
    label, score = predict_sentiment(clean_text(user_text))
    st.success(f"Sentiment: {label}")
    st.info(f"Confidence Score: {score:.2f}")

# -----------------------------------
# SHOW RAW DATA
# -----------------------------------
with st.expander("📄 Show Dataset"):
    st.dataframe(df.head(20))
