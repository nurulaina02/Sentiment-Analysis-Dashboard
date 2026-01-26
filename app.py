import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# Load data
df = pd.read_csv("data/processed/sentiment_results.csv")

st.title("📊 Sentiment Analysis Dashboard")

# KPI metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Reviews", len(df))
col2.metric("Positive", (df["sentiment"] == "Positive").sum())
col3.metric("Negative", (df["sentiment"] == "Negative").sum())

# Sentiment distribution
st.subheader("Sentiment Distribution")
sentiment_counts = df["sentiment"].value_counts().reset_index()
fig = px.pie(sentiment_counts, names="index", values="sentiment")
st.plotly_chart(fig, use_container_width=True)

# Text filter
st.subheader("Explore Reviews")
keyword = st.text_input("Search text:")
if keyword:
    st.dataframe(df[df["clean_text"].str.contains(keyword, case=False)])
else:
    st.dataframe(df.head(50))
