import streamlit as st
import pandas as pd
import plotly.express as px

from preprocessing import clean_text
from sentiment import predict_sentiment


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    layout="wide"
)

st.title("📊 Sentiment Analysis Dashboard")
st.write("Sentiment analysis using a dataset loaded from GitHub.")


# --------------------------------------------------
# Load dataset from GitHub (NO local file access)
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "nurulaina02/Sentiment-Analysis-Dashboard/"
    "main/sentiment_clean.csv"
)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

df = load_data()

if "text" not in df.columns:
    st.error("Dataset must contain a 'text' column.")
    st.stop()


# --------------------------------------------------
# Sentiment analysis
# --------------------------------------------------
@st.cache_data
def analyze_sentiment(dataframe):
    sentiments = []
    scores = []

    for text in dataframe["text"]:
        cleaned = clean_text(text)
        label, score = predict_sentiment(cleaned)
        sentiments.append(label)
        scores.append(score)

    df_out = dataframe.copy()
    df_out["sentiment"] = sentiments
    df_out["confidence"] = scores
    return df_out

with st.spinner("Running sentiment analysis..."):
    df = analyze_sentiment(df)


# --------------------------------------------------
# Visualizations
# --------------------------------------------------
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


# --------------------------------------------------
# User input
# --------------------------------------------------
st.subheader("🔍 Analyze Your Own Text")

user_text = st.text_area("Enter text")

if st.button("Analyze"):
    if user_text.strip():
        label, score = predict_sentiment(clean_text(user_text))
        st.success(f"Sentiment: {label}")
        st.info(f"Confidence: {score:.2f}")
    else:
        st.warning("Please enter some text.")


# --------------------------------------------------
# Dataset preview
# --------------------------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head(20))
