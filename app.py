import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# Load data
sentiment_df = pd.read_csv(
    "https://raw.githubusercontent.com/nurulaina02/Sentiment-Analysis-Dashboard/main/combined_sentiment_data.csv"
)

emotion_df = pd.read_csv(
    "https://raw.githubusercontent.com/nurulaina02/Sentiment-Analysis-Dashboard/main/combined_emotion_small.csv"
)

st.title("📊 Sentiment Analysis Dashboard")

# ===== KPI METRICS =====
col1, col2, col3 = st.columns(3)
col1.metric("Total Reviews", len(sentiment_df))
col2.metric("Positive", (sentiment_df["sentiment"] == "Positive").sum())
col3.metric("Negative", (sentiment_df["sentiment"] == "Negative").sum())

# ===== SENTIMENT DISTRIBUTION =====
st.subheader("Sentiment Distribution")

sentiment_counts = (
    sentiment_df["sentiment"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "Sentiment", "sentiment": "Count"})
)

fig = px.pie(
    sentiment_counts,
    names="Sentiment",
    values="Count",
    title="Sentiment Breakdown"
)

st.plotly_chart(fig, use_container_width=True)

# ===== TEXT SEARCH =====
st.subheader("Explore Reviews")

keyword = st.text_input("Search text:")

if keyword:
    filtered_df = sentiment_df[
        sentiment_df["clean_text"].str.contains(keyword, case=False, na=False)
    ]
    st.dataframe(filtered_df)
else:
    st.dataframe(sentiment_df.head(50))
