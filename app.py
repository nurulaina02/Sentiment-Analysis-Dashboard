import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.express as px

from sentiment_analysis_dashboard.model.sentiment_model import predict_sentiment
from utils.preprocessing import clean_text

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

st.title("📊 Sentiment Analysis Dashboard")

# Load data
df = pd.read_csv("data/sentiment_clean.csv")

# Preprocess
df["clean_text"] = df["text"].apply(clean_text)

# Predict sentiment
sentiments = df["clean_text"].apply(predict_sentiment)
df["sentiment"] = sentiments.apply(lambda x: x[0])
df["confidence"] = sentiments.apply(lambda x: x[1])

# Show data preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Sentiment distribution
sentiment_count = df["sentiment"].value_counts().reset_index()
sentiment_count.columns = ["Sentiment", "Count"]

fig = px.pie(
    sentiment_count,
    names="Sentiment",
    values="Count",
    title="Sentiment Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# User input (LIVE prediction)
st.subheader("🔍 Live Sentiment Prediction")

user_text = st.text_area("Enter text to analyze sentiment:")

if st.button("Analyze"):
    clean_input = clean_text(user_text)
    label, score = predict_sentiment(clean_input)

    st.success(f"Sentiment: {label}")
    st.write(f"Confidence Score: {score:.2f}")
