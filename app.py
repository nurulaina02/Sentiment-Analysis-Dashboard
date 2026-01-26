import os
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
st.write("Analyze and visualize sentiment using a Transformer-based model.")


# --------------------------------------------------
# Load dataset safely (works on Streamlit Cloud)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    file_path = os.path.join(BASE_DIR, "data", "sentiment_clean.csv")
    return pd.read_csv(file_path)

df = load_data()


# --------------------------------------------------
# Run sentiment analysis
# --------------------------------------------------
@st.cache_data
def analyze_sentiment(dataframe):
    sentiments = []
    scores = []

    for text in dataframe["text"]:
        cleaned_text = clean_text(text)
        label, score = predict_sentiment(cleaned_text)
        sentiments.append(label)
        scores.append(score)

    dataframe = dataframe.copy()
    dataframe["sentiment"] = sentiments
    dataframe["confidence"] = scores
    return dataframe

with st.spinner("Running sentiment analysis..."):
    df = analyze_sentiment(df)


# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("Dashboard Options")
show_data = st.sidebar.checkbox("Show dataset preview")


# --------------------------------------------------
# Visualizations
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    fig_sentiment = px.histogram(
        df,
        x="sentiment",
        color="sentiment",
        title="Sentiment Distribution"
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)

with col2:
    fig_confidence = px.box(
        df,
        x="sentiment",
        y="confidence",
        title="Confidence Score by Sentiment"
    )
    st.plotly_chart(fig_confidence, use_container_width=True)


# --------------------------------------------------
# User input sentiment analysis
# --------------------------------------------------
st.subheader("🔍 Analyze Your Own Text")

user_input = st.text_area(
    "Enter text below:",
    placeholder="Type a review, tweet, or comment..."
)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        label, score = predict_sentiment(clean_text(user_input))
        st.success(f"Sentiment: {label}")
        st.info(f"Confidence Score: {score:.2f}")


# --------------------------------------------------
# Show raw data
# --------------------------------------------------
if show_data:
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(20))
