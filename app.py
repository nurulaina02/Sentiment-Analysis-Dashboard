import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
# ===========================================

import streamlit as st
import pandas as pd
import plotly.express as px

from model.sentiment_model import predict_sentiment
from utils.preprocessing import clean_text

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    layout="wide"
)

st.title("📊 Sentiment Analysis Dashboard")

# ----------------- LOAD DATA -----------------
@st.cache_data
def load_data():
    return pd.read_csv("data/sentiment_clean.csv")

df = load_data()

# ----------------- PREPROCESS -----------------
df["clean_text"] = df["text"].astype(str).apply(clean_text)

# ----------------- SENTIMENT PREDICTION -----------------
@st.cache_resource
def run_sentiment(texts):
    return texts.apply(predict_sentiment)

sentiments = run_sentiment(df["clean_text"])
df["sentiment"] = sentiments.apply(lambda x: x[0])
df["confidence"] = sentiments.apply(lambda x: x[1])

# ----------------- DATA PREVIEW -----------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ----------------- VISUALIZATION -----------------
sentiment_count = df["sentiment"].value_counts().reset_index()
sentiment_count.columns = ["Sentiment", "Count"]

fig = px.pie(
    sentiment_count,
    names="Sentiment",
    values="Count",
    title="Sentiment Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# ----------------- LIVE PREDICTION -----------------
st.subheader("🔍 Live Sentiment Prediction")

user_text = st.text_area("Enter text to analyze sentiment")

if st.button("Analyze Sentiment"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_text)
        label, score = predict_sentiment(cleaned)

        st.success(f"Sentiment: **{label}**")
        st.write(f"Confidence Score: **{score:.2f}**")
