import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    sentiment_df = pd.read_csv(
        "https://raw.githubusercontent.com/nurulaina02/Sentiment-Analysis-Dashboard/main/combined_sentiment_data.csv"
    )
    emotion_df = pd.read_csv(
        "https://raw.githubusercontent.com/nurulaina02/Sentiment-Analysis-Dashboard/main/combined_emotion_small.csv"
    )
    return sentiment_df, emotion_df


sentiment_df, emotion_df = load_data()

# ---------------- TITLE ----------------
st.title("📊 Sentiment Analysis Dashboard")

# ---------------- KPI METRICS ----------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Reviews", len(sentiment_df))
col2.metric(
    "Positive",
    (sentiment_df["sentiment"].str.lower() == "positive").sum()
)
col3.metric(
    "Negative",
    (sentiment_df["sentiment"].str.lower() == "negative").sum()
)

# ---------------- SENTIMENT DISTRIBUTION ----------------
st.subheader("Sentiment Distribution")

sentiment_counts = sentiment_df["sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["Sentiment", "Count"]

fig_sentiment = px.pie(
    sentiment_counts,
    names="Sentiment",
    values="Count",
    title="Sentiment Breakdown"
)

st.plotly_chart(fig_sentiment, use_container_width=True)

# ---------------- EMOTION DISTRIBUTION ----------------
st.subheader("Emotion Distribution")

emotion_counts = emotion_df["emotion"].value_counts().reset_index()
emotion_counts.columns = ["Emotion", "Count"]

fig_emotion = px.bar(
    emotion_counts,
    x="Emotion",
    y="Count",
    title="Emotion Frequency"
)

st.plotly_chart(fig_emotion, use_container_width=True)

# ---------------- TEXT SEARCH ----------------
st.subheader("Explore Reviews")

search_text = st.text_input("Search review text:")

if search_text:
    filtered_df = sentiment_df[
        sentiment_df["clean_text"].str.contains(
            search_text, case=False, na=False
        )
    ]
    st.dataframe(filtered_df)
else:
    st.dataframe(sentiment_df.head(50))
