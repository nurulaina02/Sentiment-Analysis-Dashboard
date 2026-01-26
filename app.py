import streamlit as st
import pandas as pd
import plotly.express as px

from preprocessing import clean_text
from sentiment import predict_sentiment

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

st.title("📊 Sentiment Analysis Dashboard")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/sentiment_clean.csv")

df = load_data()

# Sidebar
st.sidebar.header("Options")
show_raw = st.sidebar.checkbox("Show raw data")

# Analyze sentiments
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

# Visualization
col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        df,
        x="sentiment",
        color="sentiment",
        title="Sentiment Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig2 = px.box(
        df,
        x="sentiment",
        y="confidence",
        title="Confidence Score by Sentiment"
    )
    st.plotly_chart(fig2, use_container_width=True)

# User input
st.subheader("🔍 Try Your Own Text")
user_text = st.text_area("Enter text here")

if st.button("Analyze"):
    label, score = predict_sentiment(clean_text(user_text))
    st.success(f"Sentiment: {label}")
    st.info(f"Confidence Score: {score:.2f}")

# Show raw data
if show_raw:
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(20))
