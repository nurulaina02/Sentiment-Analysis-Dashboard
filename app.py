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
st.write("Upload a dataset and analyze sentiment using a Transformer-based model.")


# --------------------------------------------------
# Upload dataset
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV file (must contain a 'text' column)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to begin analysis.")
    st.stop()

df = pd.read_csv(uploaded_file)

if "text" not in df.columns:
    st.error("CSV must contain a column named 'text'")
    st.stop()


# --------------------------------------------------
# Run sentiment analysis
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

    dataframe = dataframe.copy()
    dataframe["sentiment"] = sentiments
    dataframe["confidence"] = scores
    return dataframe

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
# User input sentiment analysis
# --------------------------------------------------
st.subheader("🔍 Analyze Your Own Text")

user_text = st.text_area("Enter text")

if st.button("Analyze Sentiment"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        label, score = predict_sentiment(clean_text(user_text))
        st.success(f"Sentiment: {label}")
        st.info(f"Confidence Score: {score:.2f}")


# --------------------------------------------------
# Dataset preview
# --------------------------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head(20))
